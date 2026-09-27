import { randomUUID } from 'node:crypto';
import { DEFAULTS, fail } from '../constants.mjs';
import { validateRequest } from '../contracts.mjs';
import { only, CAPTURE_VERSION, digest, validateTrace } from './schema.mjs';

/** A reported host baseline is a comparison arm, never a teacher label. */
export function recordHost(store, input) {
  if (!store.config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
  only(input, ['request', 'trace', 'model', 'model_version', 'values', 'latency_ms', 'usage', 'decision_id'],
    ['request', 'trace', 'model', 'model_version', 'values', 'latency_ms']);
  const request = validateRequest(input.request, DEFAULTS), answers = {};
  only(input.values, Object.keys(request.questions), Object.keys(request.questions));
  for (const [name, q] of Object.entries(request.questions)) answers[name] = { type: q.type, value: input.values[name], confidence: null };
  const decision_id = input.decision_id ?? randomUUID();
  const result = store.decision({ decision_id, trace: validateTrace(input.trace), arm: 'host', request, request_hash: digest(request),
    provenance: { provider: 'host', model: input.model, model_version: input.model_version, checkpoint: input.model_version,
      runtime_version: 'host-reported', preprocessing_version: 'host-state-v1', confidence_semantics: 'none' },
    answers, mode: 'on', apply: true, latency_ms: input.latency_ms,
    usage: input.usage ?? { inputTokens: null, outputTokens: null }, inference_calls: 0, network_calls: 0,
    capture_policy_version: CAPTURE_VERSION });
  return { ...result, decision_id, isGroundTruth: false };
}
