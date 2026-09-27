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

export function loadProviderConfig(home) {
  const source = readText(path.join(home, 'providers.json'), { optional: true, privateFile: true, maxBytes: 8192 });
  return validateProviderConfig(source === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(source));
}
/** The single providers.json contract; the Laya lifecycle validates what it writes with this same function. */
export function validateProviderConfig(c) {
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
      only(l.qualification, ['checkpoint', 'calibrationVersion', 'purposes', 'minConfidence', 'minChoiceProbability', 'noulCertainty', 'precision', 'routeGate'],
        ['checkpoint', 'calibrationVersion', 'purposes', 'minConfidence', 'minChoiceProbability', 'noulCertainty']);
      if (l.qualification.checkpoint !== l.checkpoint || !Array.isArray(l.qualification.purposes) ||
          l.qualification.purposes.some(x => !['route', 'select', 'retry', 'review', 'judge', 'escalate'].includes(x))) fail('INVALID_PROVIDER_CONFIG');
      if (l.qualification.precision != null && l.qualification.precision !== l.precision) fail('INVALID_PROVIDER_CONFIG');
      text(l.qualification.calibrationVersion, 80);
      for (const k of ['minConfidence', 'minChoiceProbability', 'noulCertainty']) { fraction(l.qualification[k]); if (l.qualification[k] < .5) fail('INVALID_PROVIDER_CONFIG'); }
      if (l.qualification.routeGate != null) {
        const g = l.qualification.routeGate;
        only(g, ['method', 'tierCoverage', 'maxHostProbability'], ['method', 'tierCoverage', 'maxHostProbability']);
        if (g.method !== 'decision-v1' || !Number.isFinite(g.tierCoverage) || g.tierCoverage < 0.5 || g.tierCoverage > 0.999 ||
            !Number.isFinite(g.maxHostProbability) || g.maxHostProbability < 0 || g.maxHostProbability > 0.5) fail('INVALID_PROVIDER_CONFIG');
      }
    }
  }
  if (c.provider === 'laya' && !c.laya) fail('LAYA_NOT_CONFIGURED');
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
  // Canonical shape validation is shared; probability meaning and acceptance are provider-specific.
  const n = normalizeResponse({ ...raw, model: identity.model }, request, { ...config,
    minConfidence: policy?.minConfidence ?? 1, minChoiceProbability: policy?.minChoiceProbability ?? 1, noulCertainty: policy?.noulCertainty ?? 1 });
  // The training decision record's provenance (src/training/schema.mjs validateProvenance) only
  // allows the listed keys with STRING values -- an `input_fit` object there would make every
  // captured laya decision fail INVALID_TRAINING_SCHEMA. The configured MODE ('lossless' vs
  // 'task-head', from settings.laya.inputFit) is instead expressed through preprocessing_version,
  // which is exactly what that field means; the per-REQUEST outcome (was this call truncated, by
  // how much) is carried outside provenance on the normalized result as `inputFit` below.
  const preprocessingVersion = (l.inputFit ?? 'lossless') === 'task-head'
    ? 'official-laya-0.3.4-task-head-v1' : 'official-laya-0.3.4-lossless-v1';
  return { ...n, eligible: Boolean(policy?.purposes.includes(request.purpose)) && n.eligible,
    qualified: Boolean(policy?.purposes.includes(request.purpose)), inputFit: normalizeInputFit(raw.input_fit),
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
