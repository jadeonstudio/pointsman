import { randomUUID } from 'node:crypto';
import { POLICY_VERSION, encode, digest, validateTarget } from './schema.mjs';

export const EVALUATION_POLICY = Object.freeze({
  version: POLICY_VERSION, implementation_revision: 'evidence-guards-v1',
  minLabelConfidence: .9, labelSources: ['objective', 'human'],
  agreementIsAccuracy: false, hostIsOracle: false,
  qualityRule: 'runner/human report: task success AND required task-scoped checks AND no explicit failed quality metric, rollback, regression, runtime error or timeout',
  labelRule: 'explicit question-specific assertion or human correction; never derive labels from provider votes or generic success',
});
const PURPOSE_SIGNALS = Object.freeze({
  route: ['task_succeeded', 'retry_count', 'escalated', 'latency_ms', 'token_usage', 'cost_usd', 'regression'],
  select: ['task_succeeded', 'artifact_created', 'runtime_error', 'retry_count'],
  retry: ['retry_needed', 'retry_count', 'task_succeeded', 'timeout'],
  review: ['serious_review_issue', 'regression', 'human_override', 'task_succeeded'],
  judge: ['tests_passed', 'build_passed', 'lint_passed', 'typecheck_passed', 'runtime_error', 'artifact_created'],
  escalate: ['escalated', 'human_override', 'additional_agent_calls', 'task_succeeded', 'retry_count'],
});
export function indexEvents(snapshot) {
  const decisions = new Map(), excluded = new Set(), outcomes = new Map();
  let duplicateDecisions = 0, orphanOutcomes = 0, orphanEvaluations = 0;
  for (const e of snapshot.events.filter(e => e.kind === 'decisions')) {
    const key = e.data.decision_id;
    if (decisions.has(key)) { excluded.add(key); duplicateDecisions++; }
    else decisions.set(key, e);
  }
  for (const e of snapshot.events.filter(e => e.kind === 'outcomes')) {
    const key = e.data.decision_id;
    if (!decisions.has(key) || excluded.has(key)) { orphanOutcomes++; continue; }
    if (!outcomes.has(key)) outcomes.set(key, []);
    outcomes.get(key).push(e);
  }
  const byId = new Map(snapshot.events.map(e => [e.event_id, e]));
  for (const e of snapshot.events.filter(e => e.kind === 'evaluations')) {
    if (!decisions.has(e.data.decision_id) || excluded.has(e.data.decision_id) || e.data.outcome_ids.some(k => {
      const o = byId.get(k); return !o || o.kind !== 'outcomes' || o.data.decision_id !== e.data.decision_id;
    })) orphanEvaluations++;
  }
  for (const key of excluded) decisions.delete(key);
  return { decisions, outcomes, report: { invalidEvents: snapshot.invalid.length, duplicateDecisions, orphanOutcomes, orphanEvaluations } };
}
export function evaluateDecision(decisionEvent, outcomeEvents = []) {
  const d = decisionEvent.data;
  const base = {
    decision_id: d.decision_id, policy_version: POLICY_VERSION,
    implementation_revision: EVALUATION_POLICY.implementation_revision,
    outcome_ids: outcomeEvents.map(e => e.event_id).sort(), purpose: d.request.purpose,
    quality_observed: null, quality_is_global_correctness: false, labels: [], signals: {},
    utility: null, issues: [], evidence_sources: [], metrics: null, executed: false,
    execution_matches_prediction: false, baseline_skip_supported: false,
  };
  // Enforce the join even for direct library callers, not just store-indexed calls.
  if (outcomeEvents.some(e => e.data.decision_id !== d.decision_id)) return { ...base, state: 'INVALID_OUTCOME_LINK' };
  const distinct = new Map(outcomeEvents.filter(e => e.data.final).map(e => [digest(e.data), e]));
  const final = [...distinct.values()];
  const executions = final.filter(e => e.data.executed);
  if (executions.length > 1) return { ...base, state: 'AMBIGUOUS_OUTCOME' };
  if (!final.length) return { ...base, state: 'AWAITING_OUTCOME' };
  const primary = executions[0] ?? final[0];
  const o = primary.data, m = o.metrics;
  base.metrics = m;
  base.evidence_sources = [...new Set(final.flatMap(e => [e.data.source, ...e.data.labels.map(a => a.source)]))].sort();
  // A shadow prediction cannot inherit an active arm's downstream execution. Independent
  // non-executed label annotations ARE permitted, for calibration and supervised export.
  if (d.arm === 'shadow' && o.executed) return { ...base, state: 'SHADOW_HAS_NO_COUNTERFACTUAL_OUTCOME' };
  for (const [name, value] of Object.entries(o.executed_answers)) {
    const q = d.request.questions[name];
    try {
      if (!q) return { ...base, state: 'INVALID_EXECUTION_LINK' };
      if (q.type === 'score') {
        if (!Number.isFinite(value) || value < 0 || value > q.criteria.length - 1) return { ...base, state: 'INVALID_EXECUTION_LINK' };
      } else validateTarget(q, value);
    } catch { return { ...base, state: 'INVALID_EXECUTION_LINK' }; }
  }
  base.executed = o.executed;
  base.execution_matches_prediction = o.executed && Object.keys(o.executed_answers).length === Object.keys(d.answers).length &&
    Object.entries(d.answers).every(([name, a]) => encode(a.value) === encode(o.executed_answers[name]));
  base.baseline_skip_supported = base.execution_matches_prediction && d.apply && d.arm === 'active' &&
    o.source === 'runner' && m.baseline_call_skipped === true && m.human_override !== true;
  const required = o.checks.filter(c => c.required && c.scope === 'task' && c.kind !== 'label');
  // Explicit failure wins over a broad success claim. Missing/null checks are unknown,
  // not failed, and independent label evidence is evaluated separately below.
  const failedQuality = ['tests_passed', 'build_passed', 'lint_passed', 'typecheck_passed', 'artifact_created'].some(k => m[k] === false);
  const negative = failedQuality || ['runtime_error', 'rollback', 'timeout', 'regression', 'serious_review_issue'].some(k => m[k] === true) || required.some(c => !c.passed);
  if (negative && m.task_succeeded === true) base.issues.push('CONTRADICTORY_OUTCOME_EVIDENCE');
  if (o.executed && o.source !== 'host_review') {
    base.quality_observed = negative || m.task_succeeded === false ? false :
      (m.task_succeeded === true && required.length && required.every(c => c.passed) ? true : null);
  }
  for (const key of PURPOSE_SIGNALS[d.request.purpose]) base.signals[key] = m[key] ?? null;
  if (d.request.purpose === 'route') base.utility = {
    quality: base.quality_observed, latency_ms: m.latency_ms ?? null, token_usage: m.token_usage ?? null,
    cost_usd: m.cost_usd ?? null, retry_count: m.retry_count ?? null, escalated: m.escalated ?? null,
    note: 'Observed dimensions; successful execution alone does not label the optimal route.',
  };
  if (d.request.purpose === 'retry' && m.retry_needed === true) base.issues.push('RETRY_WAS_NEEDED');
  if (d.request.purpose === 'review' && m.serious_review_issue === true) base.issues.push('SERIOUS_ISSUE_FOUND');
  if (d.request.purpose === 'escalate' && m.escalated === true) base.issues.push('ESCALATION_OCCURRED');
  if (d.request.purpose === 'judge' && negative) base.issues.push('OBJECTIVE_FAILURE_OBSERVED');
  if (d.request.purpose === 'select' && m.artifact_created === false) base.issues.push('REQUESTED_ARTIFACT_MISSING');
  if (m.human_override === true) base.issues.push('HUMAN_OVERRIDE_RECORDED');
  const candidates = new Map();
  for (const annotation of final) for (const a of annotation.data.labels) {
    const report = annotation.data;
    if (!EVALUATION_POLICY.labelSources.includes(a.source) || a.label_confidence < EVALUATION_POLICY.minLabelConfidence) continue;
    const q = d.request.questions[a.question_id];
    try { if (!q) continue; validateTarget(q, a.value); } catch { base.issues.push('INVALID_LABEL'); continue; }
    if (a.source === 'objective') {
      if (report.source !== 'runner') continue;
      const assertions = report.checks.filter(c => c.kind === 'label' && c.required && c.scope === 'task' &&
        c.question_id === a.question_id && c.evidence_ref === a.evidence_ref);
      if (!assertions.length || assertions.some(c => !c.passed)) continue;
    }
    if (a.source === 'human' && report.source !== 'human') continue;
    if (!candidates.has(a.question_id)) candidates.set(a.question_id, []);
    candidates.get(a.question_id).push({ ...a, outcome_id: annotation.event_id });
  }
  for (const values of candidates.values()) {
    if (new Set(values.map(a => encode(a.value))).size !== 1) { base.issues.push('CONFLICTING_LABELS'); continue; }
    const a = [...values].sort((a, b) => a.label_confidence - b.label_confidence)[0];
    base.labels.push({ ...a, reliability_is_calibrated_probability: false });
  }
  return { ...base, state: base.labels.length ? 'LABEL_CANDIDATE' : o.executed ? 'EVIDENCE_ONLY' : 'NOT_EXECUTED' };
}
export function evaluateStore(store) {
  if (!store.config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
  return store.lock(() => {
    const snapshot = store.scanUnlocked(), index = indexEvents(snapshot);
    const existing = new Set(snapshot.events.filter(e => e.kind === 'evaluations').map(e => digest(e.data)));
    let written = 0;
    for (const [key, d] of index.decisions) {
      const derived = evaluateDecision(d, index.outcomes.get(key));
      const data = { decision_id: key, outcome_ids: derived.outcome_ids, policy_version: POLICY_VERSION, derived };
      if (!existing.has(digest(data))) { const r = store.appendUnlocked('evaluations', data, randomUUID()); if (r.stored) written++; }
    }
    return { stored: true, evaluationsWritten: written, policyVersion: POLICY_VERSION, ...index.report };
  });
}
export function pairedPreferences(rows) {
  const pairs = [];
  for (let i = 0; i < rows.length; i++) for (let j = i + 1; j < rows.length; j++) {
    const a = rows[i], b = rows[j], x = a.decision.data, y = b.decision.data;
    if (x.request.purpose !== 'route' || x.request_hash !== y.request_hash ||
        !x.trace.task_id || x.trace.task_id !== y.trace.task_id || !x.trace.snapshot_id || x.trace.snapshot_id !== y.trace.snapshot_id ||
        a.evaluation.quality_observed !== true || b.evaluation.quality_observed !== true ||
        !a.evaluation.execution_matches_prediction || !b.evaluation.execution_matches_prediction ||
        !a.outcome || !b.outcome || a.outcome.execution_id === b.outcome.execution_id ||
        encode(a.outcome.executed_answers) === encode(b.outcome.executed_answers)) continue;
    const am = a.evaluation.metrics, bm = b.evaluation.metrics;
    if (![am.latency_ms, bm.latency_ms, am.token_usage, bm.token_usage, am.retry_count, bm.retry_count].every(Number.isFinite) ||
        [am.escalated, bm.escalated].some(v => typeof v !== 'boolean')) continue;
    const dominates = (u, v) => u.latency_ms <= v.latency_ms && u.token_usage <= v.token_usage && u.retry_count <= v.retry_count &&
      (u.latency_ms < v.latency_ms || u.token_usage < v.token_usage || u.retry_count < v.retry_count) && !u.escalated;
    const chosen = dominates(am, bm) ? a : dominates(bm, am) ? b : null;
    if (chosen) pairs.push({ chosen_decision_id: chosen.decision.data.decision_id,
      rejected_decision_id: (chosen === a ? b : a).decision.data.decision_id,
      relation: 'observed_success_pareto_dominance', is_ground_truth: false, policy_version: POLICY_VERSION });
  }
  return pairs.sort((a, b) => encode(a).localeCompare(encode(b)));
}
export function summarizeComparisons(snapshot) {
  const index = indexEvents(snapshot), groups = new Map(), execution = new Map(), calibration = new Map();
  for (const [key, event] of index.decisions) {
    const d = event.data, e = evaluateDecision(event, index.outcomes.get(key));
    const identity = encode({ ...d.provenance, purpose: d.request.purpose });
    if (!execution.has(identity)) execution.set(identity, { provider: d.provenance.provider, model: d.provenance.model,
      checkpoint: d.provenance.checkpoint, purpose: d.request.purpose, decisions: 0, applied: 0, executed: 0,
      followedPredictions: 0, measuredSuccesses: 0, measuredFailures: 0, unknownQuality: 0,
      baselineCallsReportedSkipped: 0, inferenceLatencyMs: [] });
    const stats = execution.get(identity); stats.decisions++; if (d.apply) stats.applied++; stats.inferenceLatencyMs.push(d.latency_ms);
    if (e.executed && d.arm !== 'shadow') {
      stats.executed++;
      if (e.execution_matches_prediction) {
        stats.followedPredictions++;
        if (e.quality_observed === true) stats.measuredSuccesses++;
        else if (e.quality_observed === false) stats.measuredFailures++; else stats.unknownQuality++;
      }
      if (e.baseline_skip_supported) stats.baselineCallsReportedSkipped++;
    }
    if (d.trace.comparison_id && d.trace.task_id && d.trace.snapshot_id) {
      const group = `${d.trace.comparison_id}:${d.trace.task_id}:${d.trace.snapshot_id}:${d.request_hash}`;
      if (!groups.has(group)) groups.set(group, []); groups.get(group).push(d);
    }
    for (const label of e.labels) {
      const q = d.request.questions[label.question_id], p = d.answers[label.question_id];
      const probs = q.type === 'noul' && p.probabilityTrue != null ? { false: 1 - p.probabilityTrue, true: p.probabilityTrue } : p.probabilities;
      if (!probs) continue;
      const calKey = `${identity}:${q.type}:${Object.keys(probs).length}:${label.source}`;
      if (!calibration.has(calKey)) calibration.set(calKey, { provider: d.provenance.provider, model: d.provenance.model,
        checkpoint: d.provenance.checkpoint, purpose: d.request.purpose, primitive: q.type,
        optionCount: Object.keys(probs).length, labelSource: label.source, samples: [] });
      const selected = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
      calibration.get(calKey).samples.push({ confidence: selected[1], correct: selected[0] === String(label.value),
        brier: Object.entries(probs).reduce((s, [k, v]) => s + (v - (k === String(label.value) ? 1 : 0)) ** 2, 0) });
    }
  }
  let matched = 0, compared = 0;
  for (const values of groups.values()) for (let i = 0; i < values.length; i++) for (let j = i + 1; j < values.length; j++) {
    if (encode(values[i].provenance) === encode(values[j].provenance)) continue;
    for (const name of Object.keys(values[i].answers)) { compared++; if (encode(values[i].answers[name].value) === encode(values[j].answers[name]?.value)) matched++; }
  }
  const percentile = (arr, p) => arr.length ? [...arr].sort((a, b) => a - b)[Math.ceil(arr.length * p) - 1] : null;
  return { ...index.report, agreement: { matched, compared, isAccuracy: false },
    arms: [...execution.values()].map(({ inferenceLatencyMs, ...s }) => ({ ...s,
      latencyMs: { p50: percentile(inferenceLatencyMs, .5), p95: percentile(inferenceLatencyMs, .95) } })),
    calibration: [...calibration.values()].map(({ samples, ...s }) => {
      let ece = 0;
      for (let i = 0; i < 10; i++) { const bin = samples.filter(x => Math.min(9, Math.floor(x.confidence * 10)) === i);
        if (bin.length) ece += Math.abs(bin.reduce((v, x) => v + Number(x.correct) - x.confidence, 0)) / samples.length; }
      return { ...s, count: samples.length, brier: samples.reduce((v, x) => v + x.brier, 0) / samples.length, ece,
        confidenceBasis: 'top-class probability, NOT entropy confidence or task success probability' };
    }), note: 'Evidence is locally reported, not cryptographically attested. Host review is weak evidence. Shadow agreement supplies no counterfactual task success.' };
}
