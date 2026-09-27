import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { errorCode, fail, isObject } from './constants.mjs';
import { resolveHome, appendEvent } from './storage.mjs';
import { containsSensitiveData } from './contracts.mjs';
import { createDecisionEngine } from './engine.mjs';
import { loadFeaturePolicy, policyFingerprint, effectiveMode, TIERS } from './feature-policy.mjs';
import { validateRouteInput, routeGuard, routeRequest, chooseRoute, chooseRouteByDecision } from './routing.mjs';
import { loadProviderConfig } from './inference.mjs';
import { validateFilterInput, filterRequest, filterChoices, packFilterBatches } from './filtering.mjs';
import { discoverHostRoles } from './host-roles.mjs';

/** Advisory-only: a missing profile or a configured role the host no longer has an agent definition for. Never blocks status. */
function routerWarnings(policy, env) {
  const warnings = [];
  for (const host of ['codex', 'claude']) {
    const profile = policy.router.profiles[host] ?? {};
    const targets = [...TIERS.map(tier => profile[tier]).filter(Boolean), ...Object.values(profile.intents ?? {})];
    if (!targets.length) { warnings.push(`ROUTER_PROFILE_EMPTY:${host}`); continue; }
    let discovered;
    try { discovered = discoverHostRoles(host, { env, userHome: env.HOME || os.homedir() }); } catch { discovered = []; }
    for (const target of targets) if (!discovered.includes(target.role)) warnings.push(`ROUTER_ROLE_MISSING:${host}:${target.role}`);
  }
  return warnings;
}

/** Common routing/filtering policies. Provider choice is local configuration, never another model decision. */
export function createControlLayer({ home = resolveHome(), env = process.env, engine = createDecisionEngine({ home, env }), now = Date.now } = {}) {
  const pending = new Map();
  function remember(id, value) {
    for (const [key, item] of pending) if (now() - item.at > 300000) pending.delete(key);
    while (pending.size >= 128) pending.delete(pending.keys().next().value);
    pending.set(id, { ...value, at: now() });
  }
  const expected = (initial, policy) => initial.provider === 'laya' ? initial.model : policy.expectedModel;
  function status() {
    const base = engine.status();
    try {
      const p = loadFeaturePolicy(home);
      return { ...base, features: { router: { mode: effectiveMode(base.mode, p.router.mode), configuredMode: p.router.mode,
        expectedModel: expected(base, p.router), configuredTargets: Object.fromEntries(Object.entries(p.router.profiles).map(([host, targets]) => [host, Object.keys(targets)])),
        warnings: routerWarnings(p, env) },
      bulk: { mode: effectiveMode(base.mode, p.bulk.mode), configuredMode: p.bulk.mode, expectedModel: expected(base, p.bulk) } }, featurePolicyError: null };
    } catch (error) { return { ...base, features: { router: { mode: 'off', warnings: [] }, bulk: { mode: 'off' } }, featurePolicyError: errorCode(error) }; }
  }
  function current(policy, feature, mode, revision) {
    const base = engine.status();
    if (base.configError) fail('INVALID_CONFIG');
    if (base.policyRevision !== revision) fail('GLOBAL_POLICY_CHANGED');
    if (policyFingerprint(loadFeaturePolicy(home)) !== policyFingerprint(policy)) fail('FEATURE_POLICY_CHANGED');
    if (effectiveMode(base.mode, policy[feature].mode) !== mode) fail('MODE_CHANGED');
  }
  function log(kind, result, policy, extra = {}) {
    if (result.mode === 'off' || !engine.status().telemetry) return;
    appendEvent(home, { kind, at: new Date().toISOString(), id: result.id, mode: result.mode, apply: result.apply,
      reason: result.reason, networkCalls: result.networkCalls, inferenceCalls: result.inferenceCalls, elapsedMs: result.elapsedMs,
      model: result.model ?? null, policy: policy ? policyFingerprint(policy) : null, ...extra });
  }
  async function route(input, { signal, trace } = {}) {
    const start = performance.now();
    const result = { version: 1, id: randomUUID(), mode: 'off', apply: false, source: 'host', reason: 'OFF', route: null,
      networkCalls: 0, inferenceCalls: 0, usage: { inputTokens: null, outputTokens: null }, authorizesExecution: false, changesHostModel: false };
    let policy, proposal, revision, routeGate;
    try {
      policy = loadFeaturePolicy(home);
      // Read the provider config once, BEFORE decide(), so the routeGate below always describes
      // the exact snapshot the engine inferred with -- decide() itself fails PROVIDER_CHANGED (and
      // never calls onEvaluated) if providers.json changed underneath it, so `normalized` is only
      // ever set here when this snapshot is still current.
      const providerConfig = loadProviderConfig(home);
      const initial = engine.status(); revision = initial.policyRevision;
      result.mode = effectiveMode(initial.mode, policy.router.mode);
      if (result.mode === 'off') return result;
      if (signal?.aborted) fail('CANCELLED');
      const request = validateRouteInput(input);
      const guard = routeGuard(request, policy.router);
      if (guard) { result.reason = guard; return result; }
      if (containsSensitiveData(request)) fail('SENSITIVE_INPUT');
      const model = expected(initial, policy.router);
      let normalized;
      const d = await engine.decide(routeRequest(request), { signal, trace, modeLimit: policy.router.mode, modelOverride: model, onEvaluated: n => { normalized = n; } });
      Object.assign(result, { id: d.id, networkCalls: d.networkCalls, inferenceCalls: d.inferenceCalls ?? d.networkCalls,
        usage: d.usage, model: d.model ?? null, reason: d.reason, provenance: d.provenance });
      current(policy, 'router', result.mode, revision);
      if (signal?.aborted) fail('CANCELLED');
      routeGate = providerConfig.provider === 'laya' && providerConfig.laya?.qualification?.purposes.includes('route')
        ? providerConfig.laya.qualification.routeGate : undefined;
      if (routeGate) {
        // Decision-level gate: per-answer eligibility (normalized.eligible, driven by the legacy
        // minConfidence/minChoiceProbability thresholds) does not apply here -- the routeGate was
        // fitted directly against the tier decision, so a per-answer threshold failure must not
        // block it. Only require that the provider actually qualifies this checkpoint for 'route'
        // and that the answers came from the exact model this call expected.
        if (!normalized || normalized.qualified === false) return result;
        if (normalized.model !== model) fail('MODEL_VERSION_MISMATCH');
        // The gate was fitted for one checkpoint; never apply it to answers from another.
        if (normalized.provenance?.checkpoint !== providerConfig.laya.checkpoint) fail('PROVIDER_CHANGED');
        proposal = chooseRouteByDecision(normalized.answers, request, policy.router, routeGate);
        if (!proposal.route) { result.reason = proposal.reason; return result; }
        if (result.mode === 'shadow') { remember(result.id, { kind: 'route', proposal: proposal.route.tier }); result.reason = 'SHADOW'; }
        else { result.apply = true; result.source = 'policy'; result.route = proposal.route; result.features = proposal.features; result.reason = proposal.reason; }
      } else {
        if (!normalized || !normalized.eligible) return result;
        if (normalized.model !== model) fail('MODEL_VERSION_MISMATCH');
        proposal = chooseRoute(normalized.answers, request, policy.router);
        if (!proposal.route) { result.reason = proposal.reason; return result; }
        if (result.mode === 'shadow') { remember(result.id, { kind: 'route', proposal: proposal.route.tier }); result.reason = 'SHADOW'; }
        else if (d.apply) { result.apply = true; result.source = 'policy'; result.route = proposal.route; result.features = proposal.features; result.reason = proposal.reason; }
      }
    } catch (error) { result.reason = errorCode(error); result.apply = false; result.route = null; delete result.features; }
    finally {
      result.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
      log('route', result, policy, { candidateAvailable: Boolean(proposal?.route),
        ...(routeGate ? { gate: 'decision-v1', tierProbabilities: proposal?.features?.tierProbabilities ?? null } : {}) });
    }
    return result;
  }
  async function filter(input, { signal, trace } = {}) {
    const start = performance.now();
    const result = { version: 1, id: randomUUID(), mode: 'off', apply: false, reason: 'OFF', keepIds: [], rejectIds: [], reviewIds: [],
      networkCalls: 0, inferenceCalls: 0, decisionIds: [], valid: false, authorizesExecution: false, mutatesSource: false };
    let policy, request, revision;
    const rejected = new Set(), review = new Set(); let resolved = 0;
    try {
      policy = loadFeaturePolicy(home); request = validateFilterInput(input, policy.bulk.maxItems);
      result.valid = true; result.keepIds = request.items.map(i => i.id);
      const initial = engine.status(); revision = initial.policyRevision;
      const model = expected(initial, policy.bulk);
      result.mode = effectiveMode(initial.mode, policy.bulk.mode);
      if (result.mode === 'off') return result;
      if (request.coverage === 'exhaustive') { result.reason = 'EXHAUSTIVE_KEEP_ALL'; return result; }
      if (request.risk === 'sensitive') { result.reason = 'SENSITIVE_SCOPE'; return result; }
      if (containsSensitiveData(request)) fail('SENSITIVE_INPUT');
      if (signal?.aborted) fail('CANCELLED');
      const candidates = request.items.filter(i => !i.required);
      if (!candidates.length) { result.reason = 'NOTHING_TO_FILTER'; return result; }
      // Laya evaluates only one snippet per state; a token-admission failure keeps that snippet.
      const batchSize = initial.provider === 'laya' ? 1 : policy.bulk.batchSize;
      const packed = packFilterBatches(request.query, candidates, { ...initial.requestLimits, batchSize, model });
      for (const id of packed.deferred) review.add(id);
      const deadline = AbortSignal.timeout(policy.bulk.maxTotalMs), combined = signal ? AbortSignal.any([signal, deadline]) : deadline;
      for (let index = 0; index < packed.batches.length; index++) {
        current(policy, 'bulk', result.mode, revision);
        if (signal?.aborted) fail('CANCELLED');
        if (result.inferenceCalls >= policy.bulk.maxRequests || deadline.aborted) {
          for (const item of packed.batches.slice(index).flat()) review.add(item.id); break;
        }
        const batch = packed.batches[index]; let normalized;
        const d = await engine.decide(filterRequest(request.query, batch), { signal: combined, trace, modeLimit: policy.bulk.mode,
          modelOverride: model, onEvaluated: n => { normalized = n; } });
        result.networkCalls += d.networkCalls; result.inferenceCalls += d.inferenceCalls ?? d.networkCalls; result.decisionIds.push(d.id);
        if (d.model) result.model = d.model;
        current(policy, 'bulk', result.mode, revision);
        if (signal?.aborted) fail('CANCELLED');
        if (['MODEL_VERSION_MISMATCH', 'LAYA_IDENTITY_MISMATCH'].includes(d.reason)) fail('MODEL_VERSION_MISMATCH');
        if (!normalized?.eligible || (!d.apply && d.reason !== 'SHADOW') || normalized.model !== model) {
          for (const item of batch) review.add(item.id);
          if (!['LOW_CONFIDENCE', 'SHADOW', 'ACCEPTED'].includes(d.reason) || (normalized && normalized.model !== model)) {
            for (const item of packed.batches.slice(index + 1).flat()) review.add(item.id); break;
          }
          continue;
        }
        resolved += batch.length;
        for (const item of filterChoices(normalized.answers, batch, policy.bulk)) { if (item.reject) rejected.add(item.id); if (item.review) review.add(item.id); }
      }
      current(policy, 'bulk', result.mode, revision);
      if (signal?.aborted) fail('CANCELLED');
      if (result.mode === 'shadow') {
        if (resolved) remember(result.id, { kind: 'filter', rejected: [...rejected], ids: request.items.map(i => i.id) }); result.reason = 'SHADOW';
      } else {
        result.apply = resolved > 0; result.reason = resolved ? (review.size ? 'PARTIAL_KEEP' : 'FILTERED') : 'KEEP_ALL_FALLBACK';
        result.rejectIds = request.items.filter(i => rejected.has(i.id)).map(i => i.id);
        result.keepIds = request.items.filter(i => !rejected.has(i.id)).map(i => i.id);
        result.reviewIds = request.items.filter(i => review.has(i.id)).map(i => i.id);
      }
    } catch (error) { result.reason = errorCode(error); result.apply = false; result.rejectIds = []; result.reviewIds = []; result.keepIds = request?.items.map(i => i.id) ?? []; }
    finally {
      result.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
      result.counts = { input: request?.items.length ?? 0, kept: result.keepIds.length, rejected: result.rejectIds.length, review: result.reviewIds.length };
      log('filter', result, policy, { inputCount: result.counts.input, candidateRejectCount: rejected.size, rejectedCount: result.rejectIds.length });
    }
    return result;
  }
  function observe(input) {
    if (!isObject(input) || Object.keys(input).some(k => !['id', 'tier', 'relevantIds'].includes(k)) || typeof input.id !== 'string') fail('INVALID_OBSERVATION');
    const item = pending.get(input.id);
    if (!item || now() - item.at > 300000) fail('FEEDBACK_EXPIRED_OR_UNKNOWN');
    let metrics;
    if (item.kind === 'route') {
      if (!TIERS.includes(input.tier) || input.relevantIds !== undefined) fail('INVALID_OBSERVATION');
      metrics = { matched: item.proposal === input.tier, agreementOnly: true, taskQualityMeasured: false };
    } else {
      if (input.tier !== undefined || !Array.isArray(input.relevantIds) || new Set(input.relevantIds).size !== input.relevantIds.length || input.relevantIds.some(id => !item.ids.includes(id))) fail('INVALID_OBSERVATION');
      const misses = input.relevantIds.filter(id => item.rejected.includes(id)).length;
      metrics = { relevant: input.relevantIds.length, missed: misses, recall: input.relevantIds.length ? 1 - misses / input.relevantIds.length : null, suppliedLabelsOnly: true };
    }
    pending.delete(input.id); const base = engine.status();
    if (base.mode !== 'off' && base.telemetry) appendEvent(home, { kind: 'observation', at: new Date().toISOString(), id: input.id, feature: item.kind, ...metrics });
    return metrics;
  }
  return Object.freeze({ route, filter, observe, status });
}
export const observeSchema = { type: 'object', additionalProperties: false, required: ['id'], properties: {
  id: { type: 'string' }, tier: { type: 'string', enum: TIERS }, relevantIds: { type: 'array', maxItems: 128, uniqueItems: true, items: { type: 'string' } },
} };
