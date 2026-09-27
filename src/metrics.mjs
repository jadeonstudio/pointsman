import fs from 'node:fs';
import path from 'node:path';
import { readText, noSymlinks } from './storage.mjs';
import { fail, errorCode } from './constants.mjs';
import { createTrainingStore } from './training/store.mjs';
const percentile = (array, p) => array.length ? [...array].sort((a, b) => a - b)[Math.ceil(array.length * p) - 1] : null;
const maxOf = array => array.length ? Math.max(...array) : null;

// P4 SHADOW measurement: everything below is content-free (counts, reason codes, role names, latencies).
// Hook events (`kind:'hook'`, src/hooks.mjs) and route/filter/decision events (src/control-layer.mjs,
// src/engine.mjs) are joined only by `decision_id`/`id` — never by prompt or task content.
function emptyRecommendationBucket() { return { total: 0, byRole: Object.create(null), mismatch: 0, applied: 0 }; }
function summarizeHooks(hookEvents, routeEvents, decisionEvents) {
  const byHostEvent = Object.create(null), reasons = Object.create(null), elapsedByEvent = Object.create(null);
  const recommendations = { shadow: emptyRecommendationBucket(), on: emptyRecommendationBucket() };
  const hookDecisionIds = new Set();
  for (const h of hookEvents) {
    const key = `${h.host}:${h.event}`;
    byHostEvent[key] = (byHostEvent[key] || 0) + 1;
    if (typeof h.reason === 'string') reasons[h.reason] = (reasons[h.reason] || 0) + 1;
    if (!elapsedByEvent[h.event]) elapsedByEvent[h.event] = [];
    if (Number.isFinite(h.elapsedMs)) elapsedByEvent[h.event].push(h.elapsedMs);
    if (typeof h.decision_id === 'string' && h.decision_id) hookDecisionIds.add(h.decision_id);
    if (h.event === 'pre-spawn' && typeof h.recommended_role === 'string' && h.recommended_role && ['shadow', 'on'].includes(h.mode)) {
      const bucket = recommendations[h.mode];
      bucket.total++;
      bucket.byRole[h.recommended_role] = (bucket.byRole[h.recommended_role] || 0) + 1;
      if (h.recommended_role !== h.original_role) bucket.mismatch++;
      if (h.applied) bucket.applied++;
    }
  }
  const hookLatencyMs = Object.fromEntries(Object.entries(elapsedByEvent).map(([event, values]) =>
    [event, { n: values.length, p50: percentile(values, 0.5), p95: percentile(values, 0.95), max: maxOf(values) }]));
  const routeById = new Map(routeEvents.map(e => [e.id, e]));
  const decisionById = new Map(decisionEvents.map(e => [e.id, e]));
  const linkedRoutes = [...hookDecisionIds].map(id => routeById.get(id)).filter(Boolean);
  const linkedDecisions = [...hookDecisionIds].map(id => decisionById.get(id)).filter(Boolean);
  const linkedRouteLatencies = linkedRoutes.map(e => e.elapsedMs).filter(Number.isFinite);
  const tok = (rows, key) => rows.reduce((sum, e) => sum + (Number.isSafeInteger(e.usage?.[key]) ? e.usage[key] : 0), 0);
  return {
    total: hookEvents.length, byHostEvent, reasons, recommendations, hookLatencyMs,
    routeDecisionLatencyMs: { n: linkedRouteLatencies.length, p50: percentile(linkedRouteLatencies, 0.5), p95: percentile(linkedRouteLatencies, 0.95) },
    networkCallsFromHooks: { total: linkedRoutes.reduce((s, e) => s + (e.networkCalls || 0), 0),
      decisionsReferenced: hookDecisionIds.size, decisionsLinkedToRoute: linkedRoutes.length,
      limitation: 'Only pre-spawn decision_ids (and the Codex subagent-start heuristic link) can be joined; post-spawn/subagent-stop hook events carry no decision_id. A decision logged outside the queried day-window cannot be joined either.' },
    typeSafeUsage: { inputTokens: tok(linkedDecisions, 'inputTokens'), outputTokens: tok(linkedDecisions, 'outputTokens'), decisionsMatched: linkedDecisions.length,
      note: 'Sourced from decision events sharing the same decision_id as a hook event; route events do not carry usage.' },
    costUsd: null, costReason: 'No USD rate is embedded anywhere in this tool; only raw token counts are reported.',
  };
}
function emptyOutcomeBucket() { return { pass: 0, fail: 0, uncertain: 0, runnerPassed: 0, runnerFailed: 0, n: 0, insufficientSample: true }; }
/** Observed counts only, joined by decision_id; not a causal claim about whether following the recommendation helped. */
function summarizeFollowedOutcomes(hookEvents, outcomeRows) {
  const byDecision = new Map();
  for (const o of outcomeRows) { if (!byDecision.has(o.decision_id)) byDecision.set(o.decision_id, []); byDecision.get(o.decision_id).push(o); }
  const buckets = { applied: emptyOutcomeBucket(), notApplied: emptyOutcomeBucket() };
  const seen = { applied: new Set(), notApplied: new Set() };
  for (const h of hookEvents) {
    if (typeof h.decision_id !== 'string' || !h.decision_id) continue;
    const group = h.applied ? 'applied' : 'notApplied';
    if (seen[group].has(h.decision_id)) continue;
    const outs = byDecision.get(h.decision_id);
    if (!outs || !outs.length) continue;
    seen[group].add(h.decision_id);
    for (const o of outs) {
      if (o.source === 'host_review' && ['pass', 'fail', 'uncertain'].includes(o.host_review)) buckets[group][o.host_review]++;
      else if (o.executed === true && typeof o.metrics?.task_succeeded === 'boolean') buckets[group][o.metrics.task_succeeded ? 'runnerPassed' : 'runnerFailed']++;
    }
  }
  for (const group of ['applied', 'notApplied']) { buckets[group].n = seen[group].size; buckets[group].insufficientSample = seen[group].size < 30; }
  return { ...buckets, note: 'Counts observed outcomes for decisions the hook already made; this is not a controlled comparison.' };
}
/** Read-only: never calls store.decision/outcome/writeDerived/evaluateStore. A store error is reported as labels.error only. */
function readTrainingMetrics(home, hookEvents) {
  try {
    const store = createTrainingStore({ home });
    const minStrongLabelsPerPurpose = store.config().minStrongLabelsPerPurpose;
    const snapshot = store.scan();
    const outcomeEvents = snapshot.events.filter(e => e.kind === 'outcomes');
    const evaluationEvents = snapshot.events.filter(e => e.kind === 'evaluations');
    const bySource = Object.create(null);
    for (const e of outcomeEvents) if (typeof e.data.source === 'string') bySource[e.data.source] = (bySource[e.data.source] || 0) + 1;
    const strongLabelsByPurpose = Object.create(null);
    for (const e of evaluationEvents) {
      const purpose = e.data.derived?.purpose;
      const labels = Array.isArray(e.data.derived?.labels) ? e.data.derived.labels : [];
      const strong = labels.filter(l => ['objective', 'human'].includes(l?.source)).length;
      if (typeof purpose === 'string' && strong) strongLabelsByPurpose[purpose] = (strongLabelsByPurpose[purpose] || 0) + strong;
    }
    const trainingCandidateReady = Object.fromEntries(Object.keys(strongLabelsByPurpose).map(p => [p, strongLabelsByPurpose[p] >= minStrongLabelsPerPurpose]));
    return { labels: { bySource, minStrongLabelsPerPurpose, strongLabelsByPurpose, trainingCandidateReady, evaluationsAvailable: evaluationEvents.length > 0 },
      followedOutcomes: summarizeFollowedOutcomes(hookEvents, outcomeEvents.map(e => e.data)) };
  } catch (error) {
    return { labels: { error: errorCode(error) }, followedOutcomes: summarizeFollowedOutcomes(hookEvents, []) };
  }
}
export function readMetrics(home, days = 7) {
  if (!Number.isInteger(days) || days < 1 || days > 30) fail('INVALID_DAYS');
  const dir = path.join(home, 'logs'); noSymlinks(dir);
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter(n => /^events-\d{4}-\d\d-\d\d\.jsonl$/.test(n)) : [];
  const cutoff = new Date(Date.now() - (days - 1) * 86400000).toISOString().slice(0, 10);
  const events = []; let skipped = 0, cappedFiles = 0;
  for (const name of files.filter(n => n.slice(7, 17) >= cutoff)) {
    const text = readText(path.join(dir, name), { privateFile: true, maxBytes: 11 * 1024 * 1024 });
    if (Buffer.byteLength(text) >= 10 * 1024 * 1024) cappedFiles++;
    for (const line of text.split('\n').filter(Boolean)) {
      try { const row = JSON.parse(line); if (!['decision', 'feedback', 'route', 'filter', 'observation', 'hook'].includes(row.kind)) throw new Error(); events.push(row); }
      catch { skipped++; }
    }
  }
  const decisions = events.filter(e => e.kind === 'decision'), feedback = events.filter(e => e.kind === 'feedback');
  const latency = decisions.filter(e => e.networkCalls === 1).map(e => e.elapsedMs).filter(Number.isFinite);
  const tokens = (rows, field, key) => rows.reduce((sum, e) => sum + (Number.isSafeInteger(e[field]?.[key]) ? e[field][key] : 0), 0);
  const featureActivity = Object.fromEntries(['route', 'filter'].map(kind => {
    const rows = events.filter(e => e.kind === kind);
    return [kind, { requests: rows.length, applied: rows.filter(e => e.apply).length,
      reasons: rows.reduce((o, r) => { o[r.reason] = (o[r.reason] || 0) + 1; return o; }, Object.create(null)),
      elapsedMs: { p50: percentile(rows.map(e => e.elapsedMs).filter(Number.isFinite), 0.5), p95: percentile(rows.map(e => e.elapsedMs).filter(Number.isFinite), 0.95) } }];
  }));
  const jevDecisions = decisions.filter(e => (e.provider ?? 'jev') === 'jev');
  const providerActivity = Object.fromEntries(['jev', 'laya'].map(provider => {
    const rows = decisions.filter(e => (e.provider ?? 'jev') === provider);
    const inference = rows.filter(e => (e.inferenceCalls ?? e.networkCalls ?? 0) > 0);
    const normal = ['ACCEPTED','LOW_CONFIDENCE','UNQUALIFIED_PROVIDER','SHADOW','MODE_CHANGED','POLICY_CHANGED','PROVIDER_CHANGED','CANCELLED'];
    return [provider, { requests:rows.length, inferenceCalls:inference.reduce((s,e)=>s+(e.inferenceCalls ?? e.networkCalls),0), networkCalls:rows.reduce((s,e)=>s+(e.networkCalls||0),0),
      accepted:rows.filter(e=>e.apply).length, hostFallbacks:rows.filter(e=>e.mode==='on' && !e.apply).length, errors:inference.filter(e=>!normal.includes(e.reason)).length,
      inputTokens:tokens(rows,'usage','inputTokens'), outputTokens:tokens(rows,'usage','outputTokens'), missingUsageCalls:inference.filter(e=>e.usage?.inputTokens==null||e.usage?.outputTokens==null).length,
      // `laya.inputFit:'task-head'` opt-in only (src/inference.mjs, workers/laya_worker.py fit_task_head); always 0 for jev/'lossless'.
      routeInputFitTruncated: rows.filter(e => e.purpose === 'route' && e.inputFitTruncated).length,
      latencyMs:{p50:percentile(inference.map(e=>e.elapsedMs).filter(Number.isFinite),.5),p95:percentile(inference.map(e=>e.elapsedMs).filter(Number.isFinite),.95)} }];
  }));
  const observations = events.filter(e => e.kind === 'observation');
  const hookEvents = events.filter(e => e.kind === 'hook');
  const routeEvents = events.filter(e => e.kind === 'route');
  const hooks = summarizeHooks(hookEvents, routeEvents, decisions);
  const { labels, followedOutcomes } = readTrainingMetrics(home, hookEvents);
  return { days, decisions: decisions.length, networkCalls: decisions.reduce((s, e) => s + (e.networkCalls || 0), 0),
    inferenceCalls: decisions.reduce((s,e)=>s+(e.inferenceCalls ?? e.networkCalls ?? 0),0), providerActivity,
    accepted: decisions.filter(e => e.apply).length,
    byMode: Object.fromEntries(['off', 'shadow', 'on'].map(mode => [mode, decisions.filter(e => e.mode === mode).length])),
    reasons: decisions.reduce((out, row) => { out[row.reason] = (out[row.reason] || 0) + 1; return out; }, Object.create(null)),
    networkDecisionLatencyMs: { p50: percentile(latency, 0.5), p95: percentile(latency, 0.95) },
    jevReportedTokens: { input: tokens(jevDecisions, 'usage', 'inputTokens'), output: tokens(jevDecisions, 'usage', 'outputTokens'),
      missingUsageCalls: jevDecisions.filter(e => e.networkCalls && (e.usage?.inputTokens == null || e.usage?.outputTokens == null)).length },
    baselineReportedTokens: { input: tokens(feedback, 'baselineUsage', 'inputTokens'), output: tokens(feedback, 'baselineUsage', 'outputTokens') },
    baselineAgreement: { matched: feedback.reduce((s, e) => s + (e.matched || 0), 0), total: feedback.reduce((s, e) => s + (e.total || 0), 0), isAccuracy: false },
    featureActivity,
    routeAgreement: { observed: observations.filter(e => e.feature === 'route').length,
      matched: observations.filter(e => e.feature === 'route' && e.matched === true).length, isTaskAccuracy: false },
    filterLabels: { relevant: observations.filter(e => e.feature === 'filter').reduce((n, e) => n + (e.relevant || 0), 0),
      missed: observations.filter(e => e.feature === 'filter').reduce((n, e) => n + (e.missed || 0), 0), suppliedLabelsOnly: true },
    hooks, labels, followedOutcomes,
    tokenSavings: null, costSavings: null, skippedLines: skipped, cappedFiles,
    note: 'OFF requests are not logged. API counts use decision events only, not wrapper events. Accepted primitives are not executed model routes. Missing usage is unknown, not zero. Host tokens and Jev tokens are not directly interchangeable. Whole-task savings require matched A/B runs. Logs are local best-effort metadata only. Hook events are only logged when router mode is shadow/on; hooks/labels/followedOutcomes are observational joins by decision_id, not causal measurements, and a training store read error surfaces only as labels.error.' };
}
