import { randomUUID, randomInt } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { errorCode, fail, isObject, RESERVED } from './constants.mjs';
import { resolveHome, appendEvent } from './storage.mjs';
import { containsSensitiveData } from './contracts.mjs';
import { createDecisionEngine } from './engine.mjs';
import { loadFeaturePolicy, policyFingerprint, effectiveMode, TIERS } from './feature-policy.mjs';
import { validateRouteInput, routeGuard, routeRequest, chooseRoute, chooseRouteByDecision, effortRecommendation, ROUTE_QUESTIONS } from './routing.mjs';
import { FIXED_ROUTE_CONTEXT } from './training/laya-distill.mjs';
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
        abControlShare: p.router.abControlShare, warnings: routerWarnings(p, env) },
      bulk: { mode: effectiveMode(base.mode, p.bulk.mode), configuredMode: p.bulk.mode, expectedModel: expected(base, p.bulk) },
      evidence: { mode: effectiveMode(base.mode, p.evidence.mode), configuredMode: p.evidence.mode, automatic: p.evidence.automatic },
      effort: { mode: effectiveMode(base.mode, p.effort.mode), configuredMode: p.effort.mode, input: p.effort.input, abControlShare: p.effort.abControlShare },
      workflow: { ...p.workflow, configuredMode: p.workflow.mode, mode: effectiveMode(base.mode, p.workflow.mode),
        nativeMode: effectiveMode(effectiveMode(base.mode, p.workflow.mode), p.workflow.nativeMode) } },
      featurePolicyError: null };
    } catch (error) { return { ...base, features: { router: { mode: 'off', warnings: [] }, bulk: { mode: 'off' }, evidence: { mode: 'off' }, effort: { mode: 'off' }, workflow: { mode: 'off', nativeMode: 'off' } }, featurePolicyError: errorCode(error) }; }
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
  function effortKeys(value, allowed) {
    if (!isObject(value) || Object.keys(value).some(k => !allowed.includes(k) || RESERVED.has(k))) fail('INVALID_EFFORT_REQUEST');
  }
  function validateEffortInput(input) {
    effortKeys(input, ['text', 'context', 'loop', 'sinceLastMainMs']);
    if (typeof input.text !== 'string' || !input.text.trim() || Buffer.byteLength(input.text) > 32768) fail('INVALID_EFFORT_REQUEST');
    if (input.context !== undefined && input.context !== null &&
        (typeof input.context !== 'string' || Buffer.byteLength(input.context) > 4096)) fail('INVALID_EFFORT_REQUEST');
    if (input.loop !== undefined && !['main', 'subagent'].includes(input.loop)) fail('INVALID_EFFORT_REQUEST');
    if (input.sinceLastMainMs !== undefined && input.sinceLastMainMs !== null &&
        (!Number.isFinite(input.sinceLastMainMs) || input.sinceLastMainMs < 0)) fail('INVALID_EFFORT_REQUEST');
    return { text: input.text, context: typeof input.context === 'string' && input.context ? input.context : null,
      loop: input.loop ?? 'main', sinceLastMainMs: input.sinceLastMainMs ?? null };
  }
  // Measured 2026-09-27: an effort change rewrites the whole cached MESSAGES block, so the main loop
  // may only take one when the previous main-loop request of the session is already this stale (or
  // there was none yet); a fresh context (a subagent's first step) never pays that rewrite cost.
  function effortCold(sinceLastMainMs, coldAfterSeconds) {
    return sinceLastMainMs === null || sinceLastMainMs === undefined || sinceLastMainMs > coldAfterSeconds * 1000;
  }
  function effortTask(text, context, variant, contextChars) {
    if (variant !== 'prompt+context' || !context) return text;
    const tail = context.length > contextChars ? context.slice(context.length - contextChars) : context;
    return `${text}\n\nPrevious assistant message (tail):\n${tail}`;
  }
  const round4 = v => v === null || v === undefined ? null : Math.round(v * 10000) / 10000;
  /**
   * Phase 1 (shadow) mod: attaches the existing route judgment (intent/difficulty/risk heads, same
   * checkpoint/questions as `route()`) to a single main-loop turn or subagent run to recommend
   * raising or lowering reasoning effort. Never resolves a role/model. A main-loop change may only
   * apply on an already-cold turn (`effort.mainLoop`); a subagent change may only apply when opted
   * in (`effort.subagents`), always from that subagent's own first step. Mirrors route()'s mode,
   * validation, sensitive-input and provider-qualification guarantees; SHADOW never applies.
   */
  async function effort(input, { signal, trace } = {}) {
    const start = performance.now();
    const result = { version: 1, id: randomUUID(), mode: 'off', apply: false, reason: 'OFF', direction: null, level: null, loop: null, cacheCold: null,
      pLow: null, pHigh: null, pRiskHigh: null, arm: 'none', networkCalls: 0, inferenceCalls: 0, authorizesExecution: false, changesHostModel: false };
    let policy, revision, request, model, primary; const variants = [];
    try {
      policy = loadFeaturePolicy(home);
      const initial = engine.status(); revision = initial.policyRevision;
      result.mode = effectiveMode(initial.mode, policy.effort.mode);
      if (result.mode === 'off') return result;
      if (signal?.aborted) fail('CANCELLED');
      request = validateEffortInput(input);
      if (containsSensitiveData(request)) fail('SENSITIVE_INPUT');
      result.loop = request.loop;
      result.cacheCold = request.loop === 'main' ? effortCold(request.sinceLastMainMs, policy.effort.coldAfterSeconds) : null;
      const providerConfig = loadProviderConfig(home);
      if (!(providerConfig.provider === 'laya' && providerConfig.laya?.qualification?.purposes.includes('route'))) {
        result.reason = 'UNQUALIFIED_PROVIDER'; return result;
      }
      model = initial.model;
      const runVariant = async variant => {
        let normalized;
        const taskText = effortTask(request.text, request.context, variant, policy.effort.contextChars);
        const d = await engine.decide({ purpose: 'route', risk: 'routine', state: { task: taskText, context: FIXED_ROUTE_CONTEXT },
          questions: structuredClone(ROUTE_QUESTIONS) }, { signal, trace, modeLimit: policy.effort.mode, modelOverride: model, onEvaluated: n => { normalized = n; } });
        result.networkCalls += d.networkCalls; result.inferenceCalls += d.inferenceCalls ?? d.networkCalls;
        current(policy, 'effort', result.mode, revision);
        if (signal?.aborted) fail('CANCELLED');
        if (['MODEL_VERSION_MISMATCH', 'LAYA_IDENTITY_MISMATCH'].includes(d.reason)) fail('MODEL_VERSION_MISMATCH');
        if (!normalized || !normalized.qualified) return { variant, reason: 'UNQUALIFIED_PROVIDER', rec: null };
        if (normalized.model !== model) fail('MODEL_VERSION_MISMATCH');
        if (normalized.provenance?.checkpoint !== providerConfig.laya.checkpoint) fail('PROVIDER_CHANGED');
        if (!normalized.answers?.difficulty || !normalized.answers?.risk) return { variant, reason: 'MISSING_DIMENSIONS', rec: null };
        // effortRecommendation reads the raw per-answer probabilities directly, not the legacy
        // minConfidence/minChoiceProbability eligibility gate route() applies -- that gate is tuned
        // for a routing DECISION, not for the softer raise/lower thresholds this mod uses.
        return { variant, rec: effortRecommendation(normalized.answers, policy.effort) };
      };
      if (result.mode === 'shadow' && request.context) {
        // SHADOW records BOTH input variants (when context is available) so phase 2 can compare
        // 'prompt' vs 'prompt+context' offline; SHADOW itself never applies either way.
        const both = await Promise.all([runVariant('prompt'), runVariant('prompt+context')]);
        variants.push(...both);
        primary = both.find(v => v.variant === policy.effort.input) ?? both[0];
        result.reason = 'SHADOW';
      } else {
        primary = await runVariant(policy.effort.input);
        variants.push(primary);
        if (!primary.rec) { result.reason = primary.reason; return result; }
        if (result.mode === 'shadow') { result.reason = 'SHADOW'; }
        else {
          result.reason = primary.rec.reason;
          if (primary.rec.direction) {
            // Cache-cost gate (measured 2026-09-27): a main-loop change may only apply on an
            // already-cold turn; a subagent change may only apply once the operator opted in, since
            // it is always applied from that subagent's own first step (no warm turn to protect).
            const allowed = request.loop === 'subagent' ? policy.effort.subagents === 'on'
              : policy.effort.mainLoop === 'cold-only' && result.cacheCold === true;
            if (!allowed) {
              result.reason = request.loop === 'subagent' ? 'SUBAGENTS_DISABLED'
                : (policy.effort.mainLoop === 'off' ? 'MAIN_LOOP_DISABLED' : 'CACHE_WARM');
            } else {
              const share = Number.isFinite(policy.effort.abControlShare) ? policy.effort.abControlShare : 0;
              result.arm = share > 0 && randomInt(1_000_000) < Math.round(share * 1_000_000) ? 'control' : 'treatment';
              if (result.arm === 'treatment') { result.apply = true; result.level = primary.rec.level; }
            }
          }
        }
      }
      if (primary?.rec) {
        result.direction = primary.rec.direction;
        result.pLow = round4(primary.rec.pLow); result.pHigh = round4(primary.rec.pHigh); result.pRiskHigh = round4(primary.rec.pRiskHigh);
      }
    } catch (error) { result.reason = errorCode(error); result.apply = false; result.level = null; }
    finally {
      result.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000;
      log('effort', result, policy, { loop: result.loop, cacheCold: result.cacheCold, direction: result.direction, level: result.level,
        pLow: result.pLow, pHigh: result.pHigh, pRiskHigh: result.pRiskHigh,
        arm: result.arm, variants: variants.map(v => ({ variant: v.variant, reason: v.rec?.reason ?? v.reason,
          direction: v.rec?.direction ?? null, level: v.rec?.level ?? null,
          pLow: round4(v.rec?.pLow), pHigh: round4(v.rec?.pHigh), pRiskHigh: round4(v.rec?.pRiskHigh) })) });
    }
    return result;
  }
  async function filter(input, { signal, trace, feature = 'bulk', maxRequests } = {}) {
    const start = performance.now();
    const result = { version: 1, id: randomUUID(), mode: 'off', apply: false, reason: 'OFF', keepIds: [], rejectIds: [], reviewIds: [],
      networkCalls: 0, inferenceCalls: 0, decisionIds: [], valid: false, authorizesExecution: false, mutatesSource: false };
    let policy, request, revision;
    const rejected = new Set(), review = new Set(); let resolved = 0;
    try {
      if (!['bulk', 'evidence'].includes(feature) || (maxRequests !== undefined && (!Number.isSafeInteger(maxRequests) || maxRequests < 0))) fail('INVALID_FILTER_OPTIONS');
      policy = loadFeaturePolicy(home); request = validateFilterInput(input, policy.bulk.maxItems);
      result.valid = true; result.keepIds = request.items.map(i => i.id);
      const initial = engine.status(); revision = initial.policyRevision;
      const model = expected(initial, policy.bulk);
      result.mode = effectiveMode(initial.mode, policy[feature].mode);
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
        current(policy, feature, result.mode, revision);
        if (signal?.aborted) fail('CANCELLED');
        if (index >= Math.min(policy.bulk.maxRequests, maxRequests ?? policy.bulk.maxRequests) || deadline.aborted) {
          for (const item of packed.batches.slice(index).flat()) review.add(item.id); break;
        }
        const batch = packed.batches[index]; let normalized;
        const d = await engine.decide(filterRequest(request.query, batch), { signal: combined, trace, modeLimit: policy[feature].mode,
          modelOverride: model, onEvaluated: n => { normalized = n; } });
        result.networkCalls += d.networkCalls; result.inferenceCalls += d.inferenceCalls ?? d.networkCalls; result.decisionIds.push(d.id);
        if (d.model) result.model = d.model;
        current(policy, feature, result.mode, revision);
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
      current(policy, feature, result.mode, revision);
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
  return Object.freeze({ route, filter, effort, observe, status });
}
export const observeSchema = { type: 'object', additionalProperties: false, required: ['id'], properties: {
  id: { type: 'string' }, tier: { type: 'string', enum: TIERS }, relevantIds: { type: 'array', maxItems: 128, uniqueItems: true, items: { type: 'string' } },
} };
