import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createDecisionEngine } from '../src/engine.mjs';
import { createControlLayer } from '../src/control-layer.mjs';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { validateFeaturePolicy } from '../src/feature-policy.mjs';

export const ROUTE_INPUT = { task: 'Fix the typo in the README heading at the supplied location.', host: 'codex', risk: 'routine',
  context: { complete: true, scope: 'local', previousFailures: 0, highImpact: false, modelLocked: false, exhaustive: false },
  availableRoles: ['fixture-economy-role', 'fixture-standard-role', 'fixture-strong-role'],
  availableModels: ['fixture-economy', 'fixture-standard', 'fixture-strong'], availableSkills: ['fixture-search'] };
export const FILTER_INPUT = { query: 'Evidence about database transactions', risk: 'routine', coverage: 'selective', items: [
  { id: 'keep_1', text: 'Database transaction isolation evidence' }, { id: 'drop_1', text: 'drop: an unrelated weather summary' },
  { id: 'uncertain_1', text: 'uncertain: partial context' }, { id: 'required_1', text: 'drop: but explicitly required', required: true },
] };
function choice(q, label, confidence = 0.99, custom) {
  const labels = Object.keys(q.criteria);
  return { type: 'choice', choice: label, confidence, probabilities: custom ?? Object.fromEntries(labels.map(k => [k, k === label ? 1 : 0])) };
}
export function response(payload, overrides = {}) {
  const answers = {};
  if (payload.questions.intent) {
    answers.intent = choice(payload.questions.intent, overrides.intent ?? 'edit', overrides.intentConfidence ?? 0.99, overrides.intentProbabilities);
    answers.risk = choice(payload.questions.risk, overrides.risk ?? 'safe', 0.99, overrides.riskProbabilities);
    const probabilities = Object.fromEntries((overrides.difficulty ?? [1, 0, 0, 0, 0]).map((p, i) => [String(i), p]));
    answers.difficulty = { type: 'score', confidence: overrides.difficultyConfidence ?? 0.99, probabilities,
      score: Object.entries(probabilities).reduce((n, [k, p]) => n + Number(k) * p, 0) };
  } else {
    for (const [name, q] of Object.entries(payload.questions)) {
      const text = payload.state.candidates[name];
      answers[name] = choice(q, text.startsWith('drop:') ? 'exclude' : text.startsWith('uncertain:') ? 'uncertain' : 'include');
    }
  }
  return { model: 'jev-1.13.0', usage: { input_tokens: 123, output_tokens: 0 }, answers };
}
export function setup({ mode = 'on', featureMode = 'on', provider, now } = {}) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-features-')));
  fs.chmodSync(home, 0o700);
  const env = { HOME: home, POINTSMAN_HOME: home, TYPESAFE_API_KEY: 'offline-feature-fixture-not-real' };
  setMode(home, mode, env);
  const policy = validateFeaturePolicy({ version: 2, router: { mode: featureMode, profiles: {
    codex: { economy: { role: 'fixture-economy-role', model: 'fixture-economy', reasoning: 'low', skills: { explain: ['fixture-search'] } },
      standard: { role: 'fixture-standard-role', model: 'fixture-standard', reasoning: 'medium' },
      strong: { role: 'fixture-strong-role', model: 'fixture-strong', reasoning: 'high' } },
    claude: { economy: { role: 'fixture-economy-role', model: 'haiku' }, standard: { role: 'fixture-standard-role', model: 'sonnet' },
      strong: { role: 'fixture-strong-role', model: 'opus' } },
  } }, bulk: { mode: featureMode } });
  const save = () => atomicWrite(path.join(home, 'features.json'), JSON.stringify(policy)); save();
  const calls = [];
  const engine = createDecisionEngine({ home, env, provider: async (payload, key, options) => {
    calls.push(structuredClone(payload)); return provider ? provider(payload, key, options, calls.length) : response(payload);
  }, ...(now ? { now } : {}) });
  const layer = createControlLayer({ home, env, engine, ...(now ? { now } : {}) });
  return { home, env, policy, save, calls, engine, layer, cleanup: () => fs.rmSync(home, { recursive: true, force: true }) };
}
