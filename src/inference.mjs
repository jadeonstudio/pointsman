import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { StringDecoder } from 'node:string_decoder';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { readText, atomicWrite, ensureDir } from './storage.mjs';
import { ControlError, fail, isObject } from './constants.mjs';
import { normalizeResponse } from './contracts.mjs';
import { only, text, HASH, digest, fraction } from './training/schema.mjs';

// Issued d6 qualification binds these accepted heads; editing routing prompts must not move this boundary.
const LEGACY_ROUTE_QUESTION_HASH = '90488a1bf3dd3c99c37b5b26accc4271e6a6378ac706d395c56715c30187ede6';
const LEGACY_ROUTE_ORDER_HASH = 'cbb075c5d49612ca78dc1c3f843d57b76c6ae83261300708f1855474573f82f1';

export const MAX_PROVIDER_BYTES = 65536;
const SCOPE_FIELDS = ['purpose', 'familyId', 'familyRevision', 'stateBuilderRevision', 'questionId', 'questionType', 'questionHash', 'stateSchemaHash', 'statePolicyHash', 'threshold'];
function stateShape(value) {
  if (value === null || typeof value === 'boolean') return 'nullable-boolean';
  if (Array.isArray(value)) return { array: [...new Set(value.map(v => JSON.stringify(stateShape(v))))].sort() };
  if (isObject(value)) return Object.fromEntries(Object.keys(value).sort().map(k => [k, stateShape(value[k])]));
  return typeof value;
}
/** Hash actual payload structure and policy, not a caller's claimed family label. */
export function decisionScope(request, questionId) {
  const q = request.questions[questionId];
  return { purpose: request.purpose, questionId, questionType: q.type,
    questionHash: digest({ question: q, criterionOrder: q.type === 'choice' ? Object.keys(q.criteria) : null }),
    stateSchemaHash: digest(stateShape(request.state)), statePolicyHash: digest(isObject(request.state) ? request.state.policy ?? null : null) };
}
export function validateDecisionIdentity(identity, laya, policy) {
  only(identity, ['version', 'checkpoint', 'runtimeVersion', 'precision', 'inputFit', 'calibrationVersion', 'scopes'], ['version', 'checkpoint', 'runtimeVersion', 'precision', 'inputFit', 'calibrationVersion', 'scopes']);
  if (identity.version !== 1 || identity.checkpoint !== laya.checkpoint || identity.runtimeVersion !== laya.runtimeVersion ||
      identity.precision !== (laya.precision ?? 'fp32') || identity.inputFit !== (laya.inputFit ?? 'lossless') ||
      identity.calibrationVersion !== policy.calibrationVersion || !Array.isArray(identity.scopes) || identity.scopes.length > 128) fail('INVALID_PROVIDER_CONFIG');
  const seen = new Set();
  for (const scope of identity.scopes) {
    only(scope, SCOPE_FIELDS, SCOPE_FIELDS);
    if (!policy.purposes.includes(scope.purpose)) fail('INVALID_PROVIDER_CONFIG');
    for (const key of ['familyId', 'familyRevision', 'stateBuilderRevision', 'questionId']) text(scope[key], 200);
    for (const key of ['questionHash', 'stateSchemaHash', 'statePolicyHash']) if (!HASH.test(scope[key])) fail('INVALID_PROVIDER_CONFIG');
    if (!['choice','noul','score'].includes(scope.questionType) || scope.stateBuilderRevision !== `payload-state-schema-v1:${scope.stateSchemaHash}`) fail('INVALID_PROVIDER_CONFIG');
    if (!Number.isFinite(scope.threshold) || scope.threshold < .5 || scope.threshold > 1) fail('INVALID_PROVIDER_CONFIG');
    const signature = digest(Object.fromEntries(['purpose','questionId','questionHash','stateSchemaHash','statePolicyHash'].map(k => [k, scope[k]])));
    if (seen.has(signature)) fail('INVALID_PROVIDER_CONFIG'); seen.add(signature);
  }
  return identity;
}
function legacyRoute(request) {
  if (request.purpose !== 'route' || digest(request.questions) !== LEGACY_ROUTE_QUESTION_HASH ||
      digest(Object.fromEntries(Object.entries(request.questions).filter(([,q]) => q.type === 'choice').map(([id,q]) => [id,Object.keys(q.criteria)]))) !== LEGACY_ROUTE_ORDER_HASH ||
      !isObject(request.state) || typeof request.state.task !== 'string' || !isObject(request.state.context) ||
      Object.keys(request.state).sort().join() !== 'context,task') return false;
  const c = request.state.context;
  return Object.keys(c).sort().join() === 'complete,exhaustive,highImpact,modelLocked,previousFailures,scope' &&
    ['complete','exhaustive','highImpact','modelLocked'].every(k => typeof c[k] === 'boolean') &&
    Number.isInteger(c.previousFailures) && c.previousFailures >= 0 && c.previousFailures <= 100 &&
    ['local','repository','cross-module','unknown'].includes(c.scope);
}
function matchingScopes(policy, laya, request) {
  if (!policy?.purposes.includes(request.purpose)) return null;
  if (!policy.decisionIdentity) return legacyRoute(request) ? [] : null;
  validateDecisionIdentity(policy.decisionIdentity, laya, policy);
  const matches = Object.keys(request.questions).map(name => {
    const actual = decisionScope(request, name);
    return policy.decisionIdentity.scopes.find(s => Object.entries(actual).every(([k,v]) => s[k] === v));
  });
  return matches.every(Boolean) ? matches : null;
}

export function loadProviderConfig(home) {
  const source = readText(path.join(home, 'providers.json'), { optional: true, privateFile: true, maxBytes: MAX_PROVIDER_BYTES });
  return validateProviderConfig(source === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(source));
}
/** The single providers.json contract; the Laya lifecycle validates what it writes with this same function. */
export function validateProviderConfig(c) {
  if (Buffer.byteLength(JSON.stringify(c) ?? '') > MAX_PROVIDER_BYTES) fail('INVALID_PROVIDER_CONFIG');
  only(c, ['version', 'provider', 'laya'], ['version', 'provider']);
  if (c.version !== 1 || !['jev', 'laya'].includes(c.provider)) fail('INVALID_PROVIDER_CONFIG');
  c.laya ??= null;
  if (c.laya !== null) {
    const l = c.laya;
    only(l, ['python', 'modelPath', 'model', 'checkpoint', 'runtimeVersion', 'device', 'startupTimeoutMs', 'idleTimeoutMs', 'precision', 'qualification', 'serverIdleUnloadMs', 'inputFit'],
      ['python', 'modelPath', 'model', 'checkpoint', 'runtimeVersion', 'device']);
    if (!path.isAbsolute(l.python) || !path.isAbsolute(l.modelPath) || !HASH.test(l.checkpoint) || l.runtimeVersion !== '0.3.4' ||
        !['cpu', 'mps', 'cuda'].includes(l.device) || !/^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$/.test(l.model)) fail('INVALID_PROVIDER_CONFIG');
    for (const [name, fallback, min, max] of [['startupTimeoutMs', 120000, 100, 180000], ['idleTimeoutMs', 60000, 100, 300000], ['serverIdleUnloadMs', 1800000, 60000, 86400000]]) {
      l[name] ??= fallback; if (!Number.isInteger(l[name]) || l[name] < min || l[name] > max) fail('INVALID_PROVIDER_CONFIG');
    }
    // fp16 halves resident memory (measured: english MPS tensor 1,610 -> 810MiB) with a measured max
    // probability delta of 0.0074 vs fp32; qualification below is bound to precision because a precision
    // change shifts the answer distribution slightly.
    l.precision ??= 'fp32'; if (!['fp32', 'fp16'].includes(l.precision)) fail('INVALID_PROVIDER_CONFIG');
    // Default OFF (new feature): 'lossless' keeps today's assert_lossless refusal behavior;
    // 'task-head' opts into worker.fit_task_head() truncating only state.task (see workers/laya_worker.py).
    l.inputFit ??= 'lossless'; if (!['lossless', 'task-head'].includes(l.inputFit)) fail('INVALID_PROVIDER_CONFIG');
    if (l.qualification != null) {
      only(l.qualification, ['checkpoint', 'calibrationVersion', 'purposes', 'minConfidence', 'minChoiceProbability', 'noulCertainty', 'precision', 'routeGate', 'decisionIdentity'],
        ['checkpoint', 'calibrationVersion', 'purposes', 'minConfidence', 'minChoiceProbability', 'noulCertainty']);
      if (l.qualification.checkpoint !== l.checkpoint || !Array.isArray(l.qualification.purposes) ||
          l.qualification.purposes.some(x => !['route', 'select', 'retry', 'review', 'judge', 'escalate'].includes(x))) fail('INVALID_PROVIDER_CONFIG');
      if (l.qualification.precision != null && l.qualification.precision !== l.precision) fail('INVALID_PROVIDER_CONFIG');
      text(l.qualification.calibrationVersion, 80);
      for (const k of ['minConfidence', 'minChoiceProbability', 'noulCertainty']) { fraction(l.qualification[k]); if (l.qualification[k] < .5) fail('INVALID_PROVIDER_CONFIG'); }
      if (l.qualification.decisionIdentity) validateDecisionIdentity(l.qualification.decisionIdentity, l, l.qualification);
      if (l.qualification.routeGate != null) {
        const g = l.qualification.routeGate;
        only(g, ['method', 'tierCoverage', 'maxHostProbability'], ['method', 'tierCoverage', 'maxHostProbability']);
        if (g.method !== 'decision-v1' || !Number.isFinite(g.tierCoverage) || g.tierCoverage < 0.5 || g.tierCoverage > 0.999 ||
            !Number.isFinite(g.maxHostProbability) || g.maxHostProbability < 0 || g.maxHostProbability > 0.5) fail('INVALID_PROVIDER_CONFIG');
      }
    }
  }
  if (c.provider === 'laya' && !c.laya) fail('LAYA_NOT_CONFIGURED');
  // Writers use pretty JSON; reject before an owned write could create an unreadable config.
  if (Buffer.byteLength(JSON.stringify(c, null, 2) + '\n') > MAX_PROVIDER_BYTES) fail('INVALID_PROVIDER_CONFIG');
  return c;
}
export function selectProvider(home, provider) {
  if (!['jev', 'laya'].includes(provider)) fail('INVALID_PROVIDER_CONFIG');
  const c = loadProviderConfig(home);
  if (provider === 'laya' && !c.laya) fail('LAYA_NOT_CONFIGURED');
  ensureDir(home, true);
  const file = path.join(home, 'providers.json'), old = readText(file, { optional: true, privateFile: true });
  atomicWrite(file, JSON.stringify({ ...c, provider }, null, 2) + '\n', { expected: old });
  return { provider, modeUnchanged: true, downloaded: false };
}
export function layaReady(l) {
  if (!l) return false;
  try { return fs.statSync(l.python).isFile() && fs.statSync(l.modelPath).isDirectory() && fs.statSync(path.join(l.modelPath, 'model.safetensors')).isFile(); }
  catch { return false; }
}
export function normalizeInference(provider, raw, request, config, settings) {
  if (provider === 'jev') {
    const n = normalizeResponse(raw, request, config);
    return { ...n, provenance: { provider, model: n.model, model_version: n.model, checkpoint: n.model,
      runtime_version: 'typesafe-systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'provider-distribution-statistic' } };
  }
  const identity = raw?.identity, l = settings.laya;
  const expectedPrecision = { fp32: 'torch.float32', fp16: 'torch.float16' }[l.precision ?? 'fp32'];
  if (!identity || identity.checkpoint !== l.checkpoint || identity.runtime_version !== l.runtimeVersion || identity.model !== l.model ||
      typeof identity.device !== 'string' || identity.device.split(':')[0] !== l.device || identity.precision !== expectedPrecision) fail('LAYA_IDENTITY_MISMATCH');
  const policy = l.qualification;
  const scopes = matchingScopes(policy, l, request);
  const threshold = scopes?.length ? Math.min(...scopes.map(s => s.threshold)) : undefined;
  // Canonical shape validation is shared; probability meaning and acceptance are provider-specific.
  const n = normalizeResponse({ ...raw, model: identity.model }, request, { ...config,
    minConfidence: threshold ?? policy?.minConfidence ?? 1, minChoiceProbability: threshold ?? policy?.minChoiceProbability ?? 1, noulCertainty: threshold ?? policy?.noulCertainty ?? 1 });
  // The training decision record's provenance (src/training/schema.mjs validateProvenance) only
  // allows the listed keys with STRING values -- an `input_fit` object there would make every
  // captured laya decision fail INVALID_TRAINING_SCHEMA. The configured MODE ('lossless' vs
  // 'task-head', from settings.laya.inputFit) is instead expressed through preprocessing_version,
  // which is exactly what that field means; the per-REQUEST outcome (was this call truncated, by
  // how much) is carried outside provenance on the normalized result as `inputFit` below.
  const preprocessingVersion = (l.inputFit ?? 'lossless') === 'task-head'
    ? 'official-laya-0.3.4-task-head-v1' : 'official-laya-0.3.4-lossless-v1';
  const scopePasses = scopes?.every(s => {
    const a = n.answers[s.questionId], t = s.threshold;
    return a.type === 'choice' ? a.confidence >= t && a.selectedProbability >= t :
      a.type === 'noul' ? Math.max(a.probabilityTrue, 1 - a.probabilityTrue) >= t : a.confidence >= t;
  }) ?? false;
  return { ...n, eligible: scopes !== null && n.eligible && scopePasses,
    qualified: scopes !== null, inputFit: normalizeInputFit(raw.input_fit),
    provenance: { provider, model: identity.model,
      model_version: identity.checkpoint, checkpoint: identity.checkpoint, runtime_version: identity.runtime_version,
      preprocessing_version: preprocessingVersion, confidence_semantics: 'choice-score-normalized-entropy;noul-probability',
      device: identity.device, precision: identity.precision } };
}
/** Validates the worker's per-request `input_fit` (never persisted in provenance/training schema;
 * see normalizeInference above). A malformed or missing value is treated as untruncated rather
 * than propagated, since it never affects correctness -- only telemetry/metrics read it. */
function normalizeInputFit(inputFit) {
  if (!inputFit || typeof inputFit !== 'object' || inputFit.truncated !== true) return { truncated: false };
  const { original_chars: originalChars, kept_chars: keptChars } = inputFit;
  if (!Number.isInteger(originalChars) || originalChars < 0 || !Number.isInteger(keptChars) || keptChars < 0 || keptChars > originalChars) return { truncated: false };
  return { truncated: true, original_chars: originalChars, kept_chars: keptChars };
}

/** One optional warm Python process. No shell, API keys, downloaded code, HTTP listener or automatic training. */
export function createLayaClient({ spawnImpl = spawn } = {}) {
  let child, starting, readyIdentity, identityKey, buffer = '', idle, decoder = new StringDecoder('utf8'), resident = false, generation = 0;
  const pending = new Map(), tombstones = new Map();
  let readyResolve, readyReject, startupTimer;
  function stop(code = 'LAYA_WORKER_STOPPED') {
    clearTimeout(idle); clearTimeout(startupTimer);
    const old = child; child = null; starting = null; readyIdentity = null; resident = false; buffer = ''; decoder = new StringDecoder('utf8');
    readyReject?.(new ControlError(code)); readyReject = null; readyResolve = null;
    for (const p of pending.values()) { clearTimeout(p.timer); p.reject(new ControlError(code)); }
    pending.clear();
    for (const p of tombstones.values()) clearTimeout(p.timer);
    tombstones.clear();
    if (old) { old.stdin.destroy(); old.kill('SIGTERM'); const kill = setTimeout(() => old.kill('SIGKILL'), 500); kill.unref(); }
  }
  function armIdle(l) { clearTimeout(idle); if (!resident && !pending.size && !tombstones.size) { idle = setTimeout(() => stop(), l.idleTimeoutMs); idle.unref(); } }
  function retire(id, p, code, l, livenessMs) {
    pending.delete(id); clearTimeout(p.timer); p.reject(new ControlError(code));
    const expire = () => {
      const tombstone = tombstones.get(id); if (!tombstone) return;
      if (pending.size) { tombstone.timer = setTimeout(expire, 100); tombstone.timer.unref(); return; }
      stop('LAYA_LIVENESS_TIMEOUT');
    };
    const timer = setTimeout(expire, Math.max(1000, livenessMs));
    timer.unref(); tombstones.set(id, { timer }); armIdle(l);
  }
  function awaitUntil(promise, { deadline, signal }) {
    return new Promise((resolve, reject) => {
      let timer, done = false;
      const finish = (fn, value) => { if (done) return; done = true; clearTimeout(timer); signal?.removeEventListener('abort', cancelled); fn(value); };
      const cancelled = () => finish(reject, new ControlError('CANCELLED'));
      promise.then(value => finish(resolve, value), error => finish(reject, error));
      const remaining = deadline - Date.now();
      if (remaining <= 0) { finish(reject, new ControlError('TIMEOUT')); return; }
      if (signal?.aborted) { cancelled(); return; }
      signal?.addEventListener('abort', cancelled, { once: true });
      timer = setTimeout(() => finish(reject, new ControlError('TIMEOUT')), remaining);
    });
  }
  async function start(l, env) {
    const key = digest(l);
    if (child && identityKey !== key) stop('LAYA_CONFIG_CHANGED');
    if (readyIdentity) return readyIdentity;
    if (starting) return starting;
    if (!layaReady(l)) fail('LAYA_NOT_READY');
    identityKey = key;
    const script = fileURLToPath(new URL('../workers/laya_worker.py', import.meta.url));
    const allowedEnv = Object.fromEntries(['HOME', 'PATH', 'TMPDIR', 'SYSTEMROOT'].filter(k => typeof env[k] === 'string').map(k => [k, env[k]]));
    const promise = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
    starting = promise;
    const process = spawnImpl(l.python, ['-I', script], { shell: false, stdio: ['pipe', 'pipe', 'pipe'], env: {
      ...allowedEnv, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', HF_HUB_DISABLE_TELEMETRY: '1', PYTHONUNBUFFERED: '1', TOKENIZERS_PARALLELISM: 'false',
    } });
    child = process; generation++;
    process.stderr.on('data', () => {}); // Drain only. Dependency diagnostics can contain paths/data.
    process.stdin.on('error', () => { if (child === process) stop('LAYA_WORKER_IO'); });
    process.on('error', () => { if (child === process) stop('LAYA_WORKER_START'); });
    process.on('exit', () => { if (child === process) stop('LAYA_WORKER_EXIT'); });
    process.stdout.on('data', chunk => {
      if (child !== process) return;
      buffer += decoder.write(chunk);
      if (Buffer.byteLength(buffer) > 262144) { stop('LAYA_FRAME_TOO_LARGE'); return; }
      let end;
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        let msg; try { msg = JSON.parse(line); } catch { stop('LAYA_PROTOCOL_ERROR'); return; }
        if (msg.ready === true) { clearTimeout(startupTimer); readyIdentity = msg.identity; const resolve = readyResolve; readyResolve = readyReject = null; resolve?.(msg.identity); armIdle(l); }
        else if (msg.error && !msg.id) { stop('LAYA_STARTUP_REJECTED'); return; }
        else {
          const p = pending.get(msg.id);
          if (!p) {
            const tombstone = tombstones.get(msg.id);
            if (tombstone) { clearTimeout(tombstone.timer); tombstones.delete(msg.id); armIdle(l); continue; }
            stop('LAYA_PROTOCOL_ERROR'); return;
          }
          pending.delete(msg.id); clearTimeout(p.timer);
          if (msg.error) p.reject(new ControlError(/^[A-Z_]{1,64}$/.test(msg.error) ? msg.error : 'LAYA_ERROR'));
          else p.resolve(msg.result);
          armIdle(l);
        }
      }
    });
    startupTimer = setTimeout(() => stop('LAYA_STARTUP_TIMEOUT'), l.startupTimeoutMs);
    process.stdin.write(JSON.stringify({ init: { modelPath: l.modelPath, model: l.model, checkpoint: l.checkpoint, runtimeVersion: l.runtimeVersion, device: l.device, precision: l.precision ?? 'fp32' } }) + '\n');
    return promise;
  }
  async function infer(request, settings, { timeoutMs, signal, env = process.env } = {}) {
    const l = settings.laya;
    if (!l) fail('LAYA_NOT_CONFIGURED');
    if (signal?.aborted) fail('CANCELLED');
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1) fail('INVALID_TIMEOUT');
    const deadline = Date.now() + timeoutMs;
    clearTimeout(idle); await awaitUntil(start(l, env), { deadline, signal }); clearTimeout(idle);
    if (signal?.aborted) fail('CANCELLED');
    if (pending.size + tombstones.size >= 4) fail('CONCURRENCY_LIMIT');
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const retireWith = code => retire(id, p, code, l, timeoutMs);
      const p = { resolve, reject, timer: setTimeout(() => retireWith('TIMEOUT'), Math.max(1, deadline - Date.now())) };
      const cancelled = () => retireWith('CANCELLED');
      const settle = fn => value => { signal?.removeEventListener('abort', cancelled); fn(value); };
      p.resolve = settle(resolve); p.reject = settle(reject);
      signal?.addEventListener('abort', cancelled, { once: true });
      pending.set(id, p);
      child.stdin.write(JSON.stringify({ id, state: request.state, questions: request.questions, inputFit: l.inputFit ?? 'lossless' }) + '\n');
    });
  }
  async function prepare(settings, { timeoutMs, signal, env = process.env, resident: keepResident = true } = {}) {
    const l = settings.laya;
    if (!l) fail('LAYA_NOT_CONFIGURED');
    if (signal?.aborted) fail('CANCELLED');
    const limit = timeoutMs ?? l.startupTimeoutMs;
    if (!Number.isInteger(limit) || limit < 100 || limit > 180000 || typeof keepResident !== 'boolean') fail('INVALID_PREPARE');
    const result = await awaitUntil(start(l, env), { deadline: Date.now() + limit, signal });
    resident = keepResident; armIdle(l); return result;
  }
  return Object.freeze({ infer, prepare, close: stop,
    status: () => ({ running: Boolean(child), ready: Boolean(readyIdentity), resident, inFlight: pending.size + tombstones.size, generation, identity: readyIdentity ?? null }) });
}

/** Path of the resident-server Unix socket for a given POINTSMAN_HOME (L3 resident Laya server). */
export function layaSocketPath(home) { return path.join(home, 'run', 'laya.sock'); }
/** True only when the path exists and is actually a socket (never a symlink, regular file or directory). */
export function layaSocketExists(home) {
  try { return fs.lstatSync(layaSocketPath(home)).isSocket(); } catch { return false; }
}
/** Thin client for the resident `laya serve` process: one JSON line request, one JSON line response, then the
 * connection closes. No retry, no persistent connection; the server owns the worker's lifecycle. */
export function createLayaSocketClient({ socketPath: sock, connectImpl = net.createConnection }) {
  function request(op, extra, { timeoutMs, signal }) {
    return new Promise((resolve, reject) => {
      let settled = false, socket;
      const finish = (fn, value) => {
        if (settled) return; settled = true;
        clearTimeout(connectTimer); clearTimeout(replyTimer); signal?.removeEventListener('abort', onAbort);
        try { socket?.destroy(); } catch { /* already closed */ }
        fn(value);
      };
      const onAbort = () => finish(reject, new ControlError('CANCELLED'));
      let buffer = '';
      try { socket = connectImpl({ path: sock }); }
      catch { finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')); return; }
      const connectTimer = setTimeout(() => finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')), 50);
      const replyTimer = setTimeout(() => finish(reject, new ControlError('TIMEOUT')), Math.max(51, timeoutMs));
      socket.on('error', () => finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')));
      socket.on('connect', () => { clearTimeout(connectTimer); try { socket.write(JSON.stringify({ op, ...extra }) + '\n'); } catch { finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')); } });
      socket.on('close', () => finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')));
      socket.on('data', chunk => {
        buffer += chunk.toString('utf8');
        const end = buffer.indexOf('\n'); if (end === -1) return;
        let msg; try { msg = JSON.parse(buffer.slice(0, end)); } catch { finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')); return; }
        if (!isObject(msg)) { finish(reject, new ControlError('LAYA_SERVER_UNAVAILABLE')); return; }
        if (msg.error) finish(reject, new ControlError(/^[A-Z_]{1,64}$/.test(msg.error) ? msg.error : 'LAYA_SERVER_ERROR'));
        else finish(resolve, msg);
      });
      if (signal?.aborted) { onAbort(); return; }
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }
  return Object.freeze({
    async infer(payload, settings, { timeoutMs = 30000, signal, wait = false } = {}) {
      if (!Number.isInteger(timeoutMs) || timeoutMs < 1) fail('INVALID_TIMEOUT');
      const capped = Math.min(timeoutMs, 60000);
      const res = await request('infer', { request: payload, timeoutMs: capped, wait: Boolean(wait) }, { timeoutMs: capped + 200, signal });
      if (!res.ok) fail('LAYA_SERVER_ERROR');
      return res.raw;
    },
    status: ({ timeoutMs = 2000, signal } = {}) => request('status', {}, { timeoutMs, signal }),
  });
}
