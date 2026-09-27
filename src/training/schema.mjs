import { createHash } from 'node:crypto';
import { DEFAULTS, PURPOSES, fail, isObject } from '../constants.mjs';
import { validateRequest, containsSensitiveData } from '../contracts.mjs';

export const SCHEMA_VERSION = 1;
export const POLICY_VERSION = 'downstream-evidence-v2';
export const CAPTURE_VERSION = 'minimal-state-v1';
export const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const HASH = /^[0-9a-f]{64}$/;
export const EVIDENCE = /^sha256:[0-9a-f]{64}$/;
export const KINDS = ['decisions', 'outcomes', 'evaluations'];
export const LABEL_SOURCES = ['objective', 'human', 'host_review', 'provider_agreement'];
// Single cap for every DERIVED training artifact (canonical dataset + preferences, dataset manifest
// reads, laya export files, frozen holdouts, distill run files/inputs). Raised from 64 MiB to 256 MiB
// (dataset growth, 2026-09-26): the canonical dataset for ~24,000 samples already exceeded 64 MiB.
// This does NOT cover raw captured-event limits (single event 48 KiB, 100,000 events per kind, the
// raw snapshot scan cap in store.mjs), which are a separate, intentionally small, per-event budget.
export const MAX_DERIVED_BYTES = 256 * 1024 * 1024;

// Preserve state, option and question order: all are part of the model's actual input.
export function stable(value, parent = '') {
  if (parent === 'state') return structuredClone(value);
  if (Array.isArray(value)) return value.map(v => stable(v));
  if (!isObject(value)) return value;
  const keys = ['criteria', 'questions'].includes(parent) ? Object.keys(value) : Object.keys(value).sort();
  return Object.fromEntries(keys.map(k => [k, stable(value[k], k)]));
}
export const encode = value => JSON.stringify(stable(value));
export const digest = value => createHash('sha256').update(typeof value === 'string' ? value : encode(value)).digest('hex');
export function only(value, allowed, required = []) {
  if (!isObject(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value)) ||
      Object.keys(value).some(k => !allowed.includes(k)) || required.some(k => !Object.hasOwn(value, k))) fail('INVALID_TRAINING_SCHEMA');
}
export function text(value, max = 160) {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\x00-\x1f]/.test(value)) fail('INVALID_TRAINING_SCHEMA');
}
export function id(value) { if (typeof value !== 'string' || !UUID.test(value)) fail('INVALID_TRAINING_ID'); }
export function fraction(value) { if (!Number.isFinite(value) || value < 0 || value > 1) fail('INVALID_TRAINING_SCHEMA'); }
export function count(value) { if (!Number.isSafeInteger(value) || value < 0) fail('INVALID_TRAINING_SCHEMA'); }
export function safeContent(value, key = '') {
  const json = JSON.stringify(value);
  if (json === undefined || Buffer.byteLength(json) > 49152 || containsSensitiveData(value, key)) fail('TRAINING_SENSITIVE_OR_OVERSIZED');
  const forbidden = /^(?:messages|transcript|conversation|repository|source_code|file_contents|environment|env|customer|customer_data|personal_data|session_cookie|secret|secrets)$/i;
  let nodes = 0;
  function visit(v, depth = 0) {
    if (++nodes > 4000 || depth > 14) fail('TRAINING_SENSITIVE_OR_OVERSIZED');
    if (typeof v === 'number' && !Number.isFinite(v)) fail('INVALID_TRAINING_SCHEMA');
    if (typeof v === 'string') {
      const trimmed = v.trim();
      if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
        let parsed; try { parsed = JSON.parse(trimmed); } catch { /* ordinary text */ }
        // A JSON-encoded value is screened by its parsed content only; pattern-matching its escaped text
        // produced false positives (a newline before "@team" became "\n@team", read as an email).
        if (parsed && typeof parsed === 'object') { if (containsSensitiveData(parsed, key)) fail('TRAINING_SENSITIVE_OR_OVERSIZED'); visit(parsed, depth + 1); return; }
      }
      if (/[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}/.test(v) || /\b(?:010|011)[ -]?\d{3,4}[ -]?\d{4}\b/.test(v) ||
          /\b\d{6}[- ]?[1-4]\d{6}\b/.test(v) || /\b(?:sk_live_|sk_test_|xox[baprs]-|AIza)[A-Za-z0-9_-]{8,}/.test(v)) fail('TRAINING_SENSITIVE_OR_OVERSIZED');
    }
    if (v && typeof v === 'object') {
      if (!Array.isArray(v) && ![Object.prototype, null].includes(Object.getPrototypeOf(v))) fail('INVALID_TRAINING_SCHEMA');
      for (const [k, child] of Object.entries(v)) {
        if (['__proto__', 'prototype', 'constructor'].includes(k) || forbidden.test(k)) fail('TRAINING_SENSITIVE_OR_OVERSIZED');
        visit(child, depth + 1);
      }
    }
  }
  visit(value);
}
export function validateTrace(value = {}) {
  only(value, ['task_id', 'snapshot_id', 'comparison_id']);
  if (value.task_id !== undefined) id(value.task_id);
  if (value.comparison_id !== undefined) id(value.comparison_id);
  if (value.snapshot_id !== undefined && !HASH.test(value.snapshot_id)) fail('INVALID_TRAINING_SCHEMA');
  return structuredClone(value);
}
export const traceSchema = { type: 'object', additionalProperties: false, properties: {
  task_id: { type: 'string', format: 'uuid' }, snapshot_id: { type: 'string', pattern: '^[0-9a-f]{64}$' },
  comparison_id: { type: 'string', format: 'uuid' },
} };
export function validateProvenance(p) {
  only(p, ['provider', 'model', 'model_version', 'checkpoint', 'runtime_version', 'preprocessing_version', 'confidence_semantics', 'device', 'precision'],
    ['provider', 'model', 'model_version', 'checkpoint', 'runtime_version', 'preprocessing_version', 'confidence_semantics']);
  if (!['jev', 'laya', 'host'].includes(p.provider)) fail('INVALID_TRAINING_SCHEMA');
  for (const key of Object.keys(p)) text(p[key], 200);
  if (p.provider === 'laya' && !HASH.test(p.checkpoint)) fail('INVALID_TRAINING_SCHEMA');
}
export function validateTarget(q, value) {
  if (q.type === 'choice' && (typeof value !== 'string' || !Object.hasOwn(q.criteria, value))) fail('INVALID_TRAINING_TARGET');
  if (q.type === 'noul' && typeof value !== 'boolean') fail('INVALID_TRAINING_TARGET');
  // Fractional Score targets require independently measured distributions; do not invent one from an average.
  if (q.type === 'score' && (!Number.isInteger(value) || value < 0 || value >= q.criteria.length)) fail('INVALID_TRAINING_TARGET');
}
export function targetDistribution(q, value) {
  validateTarget(q, value);
  const keys = q.type === 'choice' ? Object.keys(q.criteria) : q.type === 'noul' ? ['false', 'true'] : q.criteria.map((_, i) => String(i));
  return Object.fromEntries(keys.map(k => [k, k === String(value) ? 1 : 0]));
}
export function validatePrediction(request, answers, host = false) {
  if (!isObject(answers) || Object.keys(answers).length !== Object.keys(request.questions).length) fail('INVALID_TRAINING_SCHEMA');
  for (const [name, q] of Object.entries(request.questions)) {
    const a = answers[name];
    only(a, ['type', 'value', 'confidence', 'probabilities', 'selectedProbability', 'probabilityTrue'], ['type', 'value', 'confidence']);
    if (a.type !== q.type) fail('INVALID_TRAINING_SCHEMA');
    if (a.confidence !== null) fraction(a.confidence);
    if (q.type === 'score') { if (!Number.isFinite(a.value) || a.value < 0 || a.value > q.criteria.length - 1) fail('INVALID_TRAINING_SCHEMA'); }
    else validateTarget(q, a.value);
    if (host && a.probabilities == null && a.probabilityTrue == null) continue;
    if (q.type === 'noul') {
      fraction(a.probabilityTrue);
      if (a.value !== (a.probabilityTrue >= .5)) fail('INVALID_TRAINING_SCHEMA');
    } else {
      const keys = q.type === 'choice' ? Object.keys(q.criteria) : q.criteria.map((_, i) => String(i));
      only(a.probabilities, keys, keys);
      Object.values(a.probabilities).forEach(fraction);
      if (Math.abs(Object.values(a.probabilities).reduce((s, p) => s + p, 0) - 1) > .02) fail('INVALID_TRAINING_SCHEMA');
      if (q.type === 'score' && Math.abs(keys.reduce((s, k) => s + Number(k) * a.probabilities[k], 0) - a.value) > .05) fail('INVALID_TRAINING_SCHEMA');
      if (q.type === 'choice' && a.probabilities[a.value] + 1e-9 < Math.max(...Object.values(a.probabilities))) fail('INVALID_TRAINING_SCHEMA');
      if (a.selectedProbability !== undefined && (q.type !== 'choice' || a.selectedProbability !== a.probabilities[a.value])) fail('INVALID_TRAINING_SCHEMA');
    }
  }
}
export function validateDecision(d) {
  only(d, ['decision_id', 'trace', 'arm', 'request', 'request_hash', 'provenance', 'answers', 'mode', 'apply', 'latency_ms', 'usage', 'inference_calls', 'network_calls', 'capture_policy_version'],
    ['decision_id', 'trace', 'arm', 'request', 'request_hash', 'provenance', 'answers', 'mode', 'apply', 'latency_ms', 'usage', 'inference_calls', 'network_calls', 'capture_policy_version']);
  id(d.decision_id); validateTrace(d.trace); validateProvenance(d.provenance);
  if (!['active', 'shadow', 'host'].includes(d.arm) || !['shadow', 'on'].includes(d.mode) || typeof d.apply !== 'boolean') fail('INVALID_TRAINING_SCHEMA');
  if ((d.arm === 'shadow' || d.mode === 'shadow') && d.apply) fail('INVALID_TRAINING_SCHEMA');
  if ((d.arm === 'host') !== (d.provenance.provider === 'host')) fail('INVALID_TRAINING_SCHEMA');
  const req = validateRequest(d.request, DEFAULTS);
  if (req.risk !== 'routine' || Buffer.byteLength(JSON.stringify(req.state)) > 4096 || d.request_hash !== digest(req)) fail('INVALID_TRAINING_SCHEMA');
  validatePrediction(req, d.answers, d.arm === 'host');
  if (!Number.isFinite(d.latency_ms) || d.latency_ms < 0) fail('INVALID_TRAINING_SCHEMA');
  only(d.usage, ['inputTokens', 'outputTokens'], ['inputTokens', 'outputTokens']);
  for (const n of Object.values(d.usage)) if (n !== null) count(n);
  count(d.inference_calls); count(d.network_calls);
  if (d.capture_policy_version !== CAPTURE_VERSION) fail('INVALID_TRAINING_SCHEMA');
  safeContent(d); return d;
}
export const BOOL_METRICS = ['task_succeeded', 'tests_passed', 'build_passed', 'lint_passed', 'typecheck_passed', 'runtime_error', 'artifact_created', 'escalated', 'human_override', 'rollback', 'timeout', 'regression', 'retry_needed', 'serious_review_issue', 'baseline_call_skipped'];
export const COUNT_METRICS = ['retry_count', 'additional_agent_calls', 'token_usage', 'input_tokens', 'output_tokens'];
export function validateOutcome(o) {
  only(o, ['decision_id', 'execution_id', 'executed', 'final', 'source', 'executed_answers', 'metrics', 'checks', 'labels', 'host_review'],
    ['decision_id', 'execution_id', 'executed', 'final', 'source', 'executed_answers', 'metrics', 'checks', 'labels']);
  id(o.decision_id); id(o.execution_id);
  if (typeof o.executed !== 'boolean' || typeof o.final !== 'boolean' || !['runner', 'human', 'host_review'].includes(o.source)) fail('INVALID_TRAINING_SCHEMA');
  if (!isObject(o.executed_answers) || Object.keys(o.executed_answers).length > 8) fail('INVALID_TRAINING_SCHEMA');
  for (const [k, v] of Object.entries(o.executed_answers)) { text(k, 64); if (!['string', 'boolean', 'number'].includes(typeof v)) fail('INVALID_TRAINING_SCHEMA'); }
  only(o.metrics, [...BOOL_METRICS, ...COUNT_METRICS, 'latency_ms', 'cost_usd']);
  for (const k of BOOL_METRICS) if (o.metrics[k] != null && typeof o.metrics[k] !== 'boolean') fail('INVALID_TRAINING_SCHEMA');
  for (const k of COUNT_METRICS) if (o.metrics[k] != null) count(o.metrics[k]);
  for (const k of ['latency_ms', 'cost_usd']) if (o.metrics[k] != null && (!Number.isFinite(o.metrics[k]) || o.metrics[k] < 0)) fail('INVALID_TRAINING_SCHEMA');
  if (!Array.isArray(o.checks) || o.checks.length > 32 || !Array.isArray(o.labels) || o.labels.length > 8) fail('INVALID_TRAINING_SCHEMA');
  for (const c of o.checks) {
    only(c, ['kind', 'passed', 'required', 'scope', 'evidence_ref', 'question_id', 'exit_status'], ['kind', 'passed', 'required', 'scope', 'evidence_ref']);
    if (!['tests', 'build', 'lint', 'typecheck', 'runtime', 'artifact', 'command', 'label'].includes(c.kind) || typeof c.passed !== 'boolean' || typeof c.required !== 'boolean' || !['task', 'partial'].includes(c.scope) || !EVIDENCE.test(c.evidence_ref)) fail('INVALID_TRAINING_SCHEMA');
    if (c.question_id !== undefined) text(c.question_id, 64);
    if (c.exit_status !== undefined && (!Number.isInteger(c.exit_status) || c.passed !== (c.exit_status === 0))) fail('INVALID_TRAINING_SCHEMA');
  }
  for (const a of o.labels) {
    only(a, ['question_id', 'value', 'source', 'label_confidence', 'evidence_ref'], ['question_id', 'value', 'source', 'label_confidence', 'evidence_ref']);
    text(a.question_id, 64); fraction(a.label_confidence);
    if (!LABEL_SOURCES.includes(a.source) || !EVIDENCE.test(a.evidence_ref) || !['string', 'number', 'boolean'].includes(typeof a.value)) fail('INVALID_TRAINING_SCHEMA');
    if (o.source === 'host_review' && !['host_review', 'provider_agreement'].includes(a.source)) fail('UNTRUSTED_LABEL_SOURCE');
    if (a.source === 'human' && o.source !== 'human') fail('UNTRUSTED_LABEL_SOURCE');
  }
  if (o.host_review !== undefined && !['pass', 'fail', 'uncertain'].includes(o.host_review)) fail('INVALID_TRAINING_SCHEMA');
  safeContent(o); return o;
}
export function validateEvent(e) {
  only(e, ['schema_version', 'event_id', 'kind', 'created_at', 'data', 'checksum'], ['schema_version', 'event_id', 'kind', 'created_at', 'data', 'checksum']);
  id(e.event_id);
  if (e.schema_version !== SCHEMA_VERSION || !KINDS.includes(e.kind) || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(e.created_at) || !Number.isFinite(Date.parse(e.created_at))) fail('INVALID_TRAINING_SCHEMA');
  const { checksum, ...body } = e;
  if (checksum !== digest(body)) fail('TRAINING_CHECKSUM_MISMATCH');
  if (e.kind === 'decisions') validateDecision(e.data);
  else if (e.kind === 'outcomes') validateOutcome(e.data);
  else {
    only(e.data, ['decision_id', 'outcome_ids', 'policy_version', 'derived'], ['decision_id', 'outcome_ids', 'policy_version', 'derived']);
    id(e.data.decision_id); text(e.data.policy_version, 80);
    if (!Array.isArray(e.data.outcome_ids) || e.data.outcome_ids.length > 128) fail('INVALID_TRAINING_SCHEMA');
    e.data.outcome_ids.forEach(id);
    if (!isObject(e.data.derived)) fail('INVALID_TRAINING_SCHEMA');
  }
  safeContent(e); return e;
}
