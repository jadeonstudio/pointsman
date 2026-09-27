import { createHash, randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { DEFAULTS, VERSION, MODES, ControlError, errorCode, fail, isObject } from './constants.mjs';
import { resolveHome, loadConfig, getCredential, appendEvent } from './storage.mjs';
import { validateRequest, containsSensitiveData, wireRequest } from './contracts.mjs';
import { callTypeSafe } from './provider.mjs';
import { loadProviderConfig, layaReady, createLayaClient, normalizeInference, layaSocketExists, layaSocketPath, createLayaSocketClient } from './inference.mjs';
import { createTrainingStore } from './training/store.mjs';
import { recordHost } from './training/host.mjs';
import { CAPTURE_VERSION, digest, validateTrace } from './training/schema.mjs';

/** Shared control; the selected inference adapter does not own permissions or training labels. */
export function createDecisionEngine({ home = resolveHome(), env = process.env, provider, now = Date.now,
  layaClient = createLayaClient(), training = createTrainingStore({ home }), layaSpawn = true, layaWait = true } = {}) {
  let inFlight = 0, calls = [], monitor, layaSocket;
  const circuits = new Map(), pendingFeedback = new Map();
  function getLayaSocket() { layaSocket ??= createLayaSocketClient({ socketPath: layaSocketPath(home) }); return layaSocket; }
  // L3 (2026-09-23): a live server socket always wins; otherwise `layaSpawn` decides whether this
  // process may spawn its own worker (true, the historical behavior) or must fail immediately (hooks).
  async function inferLaya(payload, settings, opts) {
    if (layaSocketExists(home)) return getLayaSocket().infer(payload, settings, { ...opts, wait: layaWait });
    if (!layaSpawn) fail('LAYA_SERVER_UNAVAILABLE');
    return layaClient.infer(payload, settings, opts);
  }
  const revision = (config, settings) => createHash('sha256').update(JSON.stringify({ config, settings })).digest('hex');
  function remember(id, normalized) {
    for (const [key, value] of pendingFeedback) if (now() - value.at > 300000) pendingFeedback.delete(key);
    while (pendingFeedback.size >= 128) pendingFeedback.delete(pendingFeedback.keys().next().value);
    pendingFeedback.set(id, { at: now(), answers: normalized.answers });
  }
  function status() {
    let config, credential = 'missing', configError = null, settings = { version: 1, provider: 'jev', laya: null };
    try { config = loadConfig(home, env); settings = loadProviderConfig(home); }
    catch (e) { config ??= { ...DEFAULTS }; configError = errorCode(e); }
    let ready = false;
    if (settings.provider === 'jev') {
      try { credential = getCredential(home, env).source; } catch { credential = 'invalid'; }
      ready = !['missing', 'invalid'].includes(credential);
    } else { credential = 'not-required'; ready = layaReady(settings.laya); }
    const circuit = circuits.get(settings.provider);
    return { version: VERSION, mode: config.mode, killSwitch: env.POINTSMAN_DISABLE === '1', credential, ready: !configError && ready,
      configError, provider: settings.provider, model: settings.provider === 'jev' ? config.model : settings.laya?.model,
      checkpoint: settings.provider === 'laya' ? settings.laya?.checkpoint : null,
      home, telemetry: config.telemetry, inFlight, circuitOpen: now() < (circuit?.until ?? 0),
      limits: { timeoutMs: config.timeoutMs, maxCallsPerMinute: config.maxCallsPerMinute, maxInFlight: config.maxInFlight, scope: 'per-process' },
      requestLimits: { maxInputBytes: config.maxInputBytes, maxQuestions: config.maxQuestions },
      policyRevision: revision(config, settings), trainingCapture: training.status().trainingCapture,
      localWorker: layaClient.status(), authorizesExecution: false };
  }
  function close() { clearInterval(monitor); monitor = null; layaClient.close(); }
  function monitorLaya() {
    if (!monitor) { monitor = setInterval(() => { try { if (loadConfig(home, env).mode === 'off') close(); } catch { close(); } }, 1000); monitor.unref(); }
  }
  async function prepare({ signal, timeoutMs, resident = true } = {}) {
    const config = loadConfig(home, env);
    if (env.POINTSMAN_DISABLE === '1' || config.mode === 'off') fail('OFF');
    const settings = loadProviderConfig(home);
    if (settings.provider !== 'laya' || !settings.laya) fail('LAYA_NOT_SELECTED');
    monitorLaya(); return layaClient.prepare(settings, { signal, timeoutMs, resident, env });
  }
  async function decide(input, { signal, modeLimit = 'on', onEvaluated, modelOverride, providerOverride, trace: suppliedTrace } = {}) {
    const start = performance.now();
    const result = { version: 1, id: randomUUID(), mode: 'off', apply: false, source: 'host', reason: 'OFF', answers: {},
      usage: { inputTokens: null, outputTokens: null }, networkCalls: 0, inferenceCalls: 0, elapsedMs: 0, authorizesExecution: false };
    const captureTicket = training.ticket();
    let config, settings, request, normalized, trace = {}, key = '', inputBytes = 0, eligible = false, reserved = false, circuit;
    try {
      config = loadConfig(home, env);
      if (!MODES.includes(modeLimit)) fail('INVALID_MODE_LIMIT');
      result.mode = config.mode === 'off' || modeLimit === 'off' ? 'off' : (config.mode === 'shadow' || modeLimit === 'shadow' ? 'shadow' : 'on');
      if (result.mode === 'off') { close(); return result; }
      if (signal?.aborted) fail('CANCELLED');
      settings = loadProviderConfig(home);
      const selected = providerOverride ?? settings.provider;
      if (!['jev', 'laya'].includes(selected)) fail('INVALID_PROVIDER_CONFIG');
      result.provider = selected;
      trace = validateTrace(suppliedTrace ?? (isObject(input) ? input.trace : undefined) ?? {});
      const rawInput = isObject(input) ? Object.fromEntries(Object.entries(input).filter(([name]) => name !== 'trace')) : input;
      request = validateRequest(rawInput, config);
      if (request.risk === 'sensitive') fail('SENSITIVE_SCOPE');
      if (selected === 'jev') {
        if (env.NODE_TLS_REJECT_UNAUTHORIZED === '0') fail('INSECURE_TLS_REFUSED');
        key = getCredential(home, env).key; if (!key) fail('NO_API_KEY');
        if (modelOverride !== undefined && (typeof modelOverride !== 'string' || !/^jev-\d+\.\d+\.\d+$/.test(modelOverride))) fail('INVALID_MODEL_OVERRIDE');
      } else {
        if (!settings.laya) fail('LAYA_NOT_CONFIGURED');
        if (modelOverride !== undefined && modelOverride !== settings.laya.model) fail('PROVIDER_MODEL_OVERRIDE_REFUSED');
      }
      if (containsSensitiveData(request, key)) fail('SENSITIVE_INPUT');
      const payload = wireRequest(request, selected === 'jev' ? modelOverride ?? config.model : settings.laya.model);
      inputBytes = Buffer.byteLength(JSON.stringify(payload));
      if (inputBytes > config.maxInputBytes) fail('INPUT_TOO_LARGE');
      if (!circuits.has(selected)) circuits.set(selected, { failures: 0, until: 0 });
      circuit = circuits.get(selected);
      if (now() < circuit.until) fail('CIRCUIT_OPEN');
      if (inFlight >= config.maxInFlight) fail('CONCURRENCY_LIMIT');
      calls = calls.filter(t => now() - t < 60000);
      if (calls.length >= config.maxCallsPerMinute) fail('LOCAL_RATE_LIMIT');
      calls.push(now()); inFlight++; reserved = true; result.inferenceCalls = 1; result.networkCalls = selected === 'jev' ? 1 : 0;
      if (selected === 'laya') monitorLaya();
      const raw = provider ? await provider(payload, key, { timeoutMs: config.timeoutMs, signal }) : selected === 'jev' ?
        await callTypeSafe(payload, key, { timeoutMs: config.timeoutMs, signal }) :
        await inferLaya(payload, settings, { timeoutMs: config.timeoutMs, signal, env });
      const n = normalizeInference(selected, raw, request, config, settings);
      circuit.failures = 0; circuit.until = 0;
      eligible = n.eligible; result.usage = n.usage; result.model = n.model; result.provenance = n.provenance;
      if (selected === 'jev' && modelOverride && n.model !== modelOverride) fail('MODEL_VERSION_MISMATCH');
      const current = loadConfig(home, env), currentSettings = loadProviderConfig(home);
      if (signal?.aborted) fail('CANCELLED');
      if (current.mode !== config.mode) fail('MODE_CHANGED');
      if (JSON.stringify(current) !== JSON.stringify(config)) fail('POLICY_CHANGED');
      if (JSON.stringify(currentSettings) !== JSON.stringify(settings)) fail('PROVIDER_CHANGED');
      remember(result.id, n); normalized = n;
      onEvaluated?.(structuredClone(n));
      if (result.mode === 'shadow') result.reason = 'SHADOW';
      else if (!eligible) result.reason = n.qualified === false ? 'UNQUALIFIED_PROVIDER' : 'LOW_CONFIDENCE';
      else { result.apply = true; result.source = selected; result.reason = 'ACCEPTED'; result.answers = n.answers; }
    } catch (e) {
      result.reason = errorCode(e);
      if (reserved && circuit && !['CANCELLED', 'MODE_CHANGED', 'POLICY_CHANGED', 'PROVIDER_CHANGED'].includes(result.reason)) {
        if (++circuit.failures >= config.circuitFailureThreshold) circuit.until = now() + config.circuitCooldownMs;
      }
    } finally {
      if (reserved) inFlight--;
      result.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
      if (normalized && ['ACCEPTED', 'LOW_CONFIDENCE', 'UNQUALIFIED_PROVIDER', 'SHADOW'].includes(result.reason)) {
        const captured = training.decision({ decision_id: result.id, trace, arm: result.mode === 'shadow' ? 'shadow' : 'active',
          request, request_hash: digest(request), provenance: normalized.provenance, answers: normalized.answers,
          mode: result.mode, apply: result.apply, latency_ms: result.elapsedMs, usage: result.usage,
          inference_calls: result.inferenceCalls, network_calls: result.networkCalls, capture_policy_version: CAPTURE_VERSION }, { secret: key, expectedTicket: captureTicket });
        result.trainingCapture = { stored: captured.stored, reason: captured.reason ?? null };
      }
      if (config?.telemetry && result.mode !== 'off') {
        result.telemetryStored = appendEvent(home, { kind: 'decision', at: new Date().toISOString(), id: result.id, mode: result.mode,
          captureStored: result.trainingCapture?.stored ?? null, captureReason: result.trainingCapture?.reason ?? null,
          provider: result.provider ?? null, model: result.model ?? null, purpose: request?.purpose ?? 'unknown',
          reason: result.reason, apply: result.apply, eligible, inputBytes, questionCount: request ? Object.keys(request.questions).length : 0,
          networkCalls: result.networkCalls, inferenceCalls: result.inferenceCalls, elapsedMs: result.elapsedMs, usage: result.usage,
          inputFitTruncated: Boolean(normalized?.inputFit?.truncated) });
      }
    }
    return result;
  }
  function feedback(input) {
    if (!isObject(input) || Object.keys(input).some(k => !['id', 'baseline', 'baselineUsage', 'baselineElapsedMs', 'taskSucceeded'].includes(k))) fail('INVALID_FEEDBACK');
    if (typeof input.id !== 'string' || !isObject(input.baseline)) fail('INVALID_FEEDBACK');
    const saved = pendingFeedback.get(input.id);
    if (!saved || now() - saved.at > 300000) fail('FEEDBACK_EXPIRED_OR_UNKNOWN');
    const names = Object.keys(saved.answers);
    if (Object.keys(input.baseline).length !== names.length) fail('INVALID_FEEDBACK');
    let matched = 0;
    for (const name of names) {
      const expected = saved.answers[name].value, actual = input.baseline[name];
      if (typeof actual !== typeof expected || (typeof actual === 'number' && !Number.isFinite(actual))) fail('INVALID_FEEDBACK');
      if (typeof expected === 'number' ? Math.abs(expected - actual) <= .05 : expected === actual) matched++;
    }
    const usage = { inputTokens: null, outputTokens: null };
    if (input.baselineUsage !== undefined) {
      if (!isObject(input.baselineUsage) || Object.keys(input.baselineUsage).some(k => !Object.hasOwn(usage, k))) fail('INVALID_FEEDBACK');
      for (const k of Object.keys(usage)) { const v = input.baselineUsage[k]; if (v != null && (!Number.isSafeInteger(v) || v < 0)) fail('INVALID_FEEDBACK'); usage[k] = v ?? null; }
    }
    if (input.baselineElapsedMs !== undefined && (!Number.isFinite(input.baselineElapsedMs) || input.baselineElapsedMs < 0 || input.baselineElapsedMs > 86400000)) fail('INVALID_FEEDBACK');
    if (input.taskSucceeded !== undefined && typeof input.taskSucceeded !== 'boolean') fail('INVALID_FEEDBACK');
    const config = loadConfig(home, env);
    const event = { kind: 'feedback', at: new Date().toISOString(), id: input.id, matched, total: names.length, baselineUsage: usage,
      baselineElapsedMs: input.baselineElapsedMs ?? null, taskSucceeded: input.taskSucceeded ?? null };
    const telemetryStored = config.telemetry && config.mode !== 'off' ? appendEvent(home, event) : false;
    pendingFeedback.delete(input.id); return { matched, total: names.length, agreementOnly: true, telemetryStored };
  }
  async function compare(input, { remoteConsent = false, signal } = {}) {
    const initial = status();
    if (initial.mode === 'off') return { active: await decide(input, { signal }), observers: [], agreementIsAccuracy: false };
    if (!remoteConsent) fail('EXPLICIT_REMOTE_COMPARISON_CONSENT_REQUIRED');
    const comparison_id = randomUUID();
    const trace = { ...validateTrace(input?.trace ?? {}), comparison_id };
    const snapshot = structuredClone(input);
    const active = await decide(snapshot, { signal, trace });
    const observer = initial.provider === 'jev' ? 'laya' : 'jev';
    const shadow = await decide(snapshot, { signal, trace, modeLimit: 'shadow', providerOverride: observer });
    if (status().policyRevision !== initial.policyRevision || signal?.aborted) {
      active.apply = false; active.source = 'host'; active.answers = {}; active.reason = 'COMPARISON_CONTEXT_CHANGED';
    }
    return { active, comparison_id, observers: [{ provider: observer, decision_id: shadow.id, reason: shadow.reason,
      elapsedMs: shadow.elapsedMs, applied: false }], agreementIsAccuracy: false };
  }
  return Object.freeze({ decide, status, feedback, close, compare, prepare,
    recordOutcome: input => training.outcome(input, { trust: 'host' }),
    recordHost: input => recordHost(training, input) });
}
export async function decideOrDelegate(engine, request, { use, delegate, signal } = {}) {
  if (typeof use !== 'function' || typeof delegate !== 'function') throw new ControlError('HANDLERS_REQUIRED');
  const decision = await engine.decide(request, { signal });
  return decision.apply ? use(decision.answers, decision) : delegate(decision);
}
