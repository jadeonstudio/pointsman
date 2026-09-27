import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { CAPTURE_VERSION, digest } from '../src/training/schema.mjs';
export const KEY = 'offline-training-test-key-not-a-real-credential';
export const REF = 'sha256:' + 'a'.repeat(64);
export const request = () => ({ purpose: 'route', risk: 'routine', state: { task: 'A bounded documentation change' }, questions: {
  worker: { type: 'choice', instructions: 'Choose the worker for this bounded task.', criteria: { light: 'Narrow known scope', strong: 'Unclear or broad scope' } },
} });
export const trace = () => ({ task_id: randomUUID(), snapshot_id: digest('synthetic snapshot') });
export const response = () => ({ model: 'jev-1.13.0', answers: {
  worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } },
}, usage: { input_tokens: 100, output_tokens: 0 } });
export function decision(provider = 'jev', changes = {}) {
  const r = request();
  return { decision_id: randomUUID(), trace: trace(), arm: 'active', request: r, request_hash: digest(r),
    provenance: { provider, model: provider === 'laya' ? 'laya/base' : 'jev-1.13.0', model_version: provider === 'laya' ? 'a'.repeat(64) : 'jev-1.13.0',
      checkpoint: provider === 'laya' ? 'a'.repeat(64) : 'jev-1.13.0', runtime_version: provider === 'laya' ? '0.3.4' : 'systemone-v1',
      preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' },
    answers: { worker: { type: 'choice', value: 'light', confidence: .97, selectedProbability: .99, probabilities: { light: .99, strong: .01 } } },
    mode: 'on', apply: true, latency_ms: 600, usage: { inputTokens: 100, outputTokens: 0 },
    inference_calls: 1, network_calls: provider === 'laya' ? 0 : 1, capture_policy_version: CAPTURE_VERSION, ...changes };
}
export function outcome(d, changes = {}) {
  return { decision_id: d.decision_id, execution_id: randomUUID(), executed: true, final: true, source: 'runner',
    executed_answers: Object.fromEntries(Object.entries(d.answers).map(([k, a]) => [k, a.value])),
    metrics: { task_succeeded: true, tests_passed: true, retry_count: 0, escalated: false, latency_ms: 1200, token_usage: 320, baseline_call_skipped: true },
    checks: [{ kind: 'tests', required: true, passed: true, scope: 'task', evidence_ref: REF },
      { kind: 'label', required: true, passed: true, scope: 'task', evidence_ref: REF, question_id: 'worker' }],
    labels: [{ question_id: 'worker', value: 'light', source: 'objective', label_confidence: .95, evidence_ref: REF }], ...changes };
}
export function fixture(t, capture = true) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-training-')));
  const store = createTrainingStore({ home });
  // Existing tests build datasets from a handful of samples; the production default (100) is
  // exercised by its own dedicated tests instead of every dataset test needing --allow-small.
  if (capture) { store.setCapture(true); store.setMinLabels(1); }
  setMode(home, 'on', {}); const env = { TYPESAFE_API_KEY: KEY };
  const engine = createDecisionEngine({ home, env, provider: async () => response() });
  t.after(() => { engine.close(); fs.rmSync(home, { recursive: true, force: true }); });
  const save = d => { const result = store.decision(d); if (!result.stored) throw new Error(result.reason); return d; };
  return { home, store, engine, env, save, writeProviders: p => atomicWrite(path.join(home, 'providers.json'), JSON.stringify(p)) };
}
// Reproduces dataset.mjs groupedSplits() for an isolated single-sample group (no shared
// task/request/state with any other sample): its group representative is always 'input:'+request_hash.
export const splitFor = requestHash => { const n = parseInt(digest('input:' + requestHash).slice(0, 8), 16) % 100; return n < 80 ? 'train' : n < 90 ? 'calibration' : 'test'; };
export function stateForSplit(buildRequest, target, start = 0) {
  for (let i = start; i < start + 20000; i++) if (splitFor(digest(buildRequest(i))) === target) return i;
  throw new Error('NO_DETERMINISTIC_CASE_FOUND_FOR_SPLIT_' + target);
}
export const fillerRequest = i => { const r = request(); r.state = { task: `filler case ${i}` }; return r; };
export function layaConfig(home, patch = {}) {
  return { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'a'.repeat(64),
    runtimeVersion: '0.3.4', device: 'cpu', startupTimeoutMs: 1000, idleTimeoutMs: 1000, ...patch } };
}
