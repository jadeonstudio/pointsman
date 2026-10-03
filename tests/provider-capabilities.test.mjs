import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULTS, MAX_FRAME_BYTES, MAX_RESPONSE_BYTES } from '../src/constants.mjs';
import { validateConfig, atomicWrite } from '../src/storage.mjs';
import { validateRequest, normalizeResponse, decisionSchema, providerRequestLimits } from '../src/contracts.mjs';
import { fixture, deferred } from './helpers.mjs';
import path from 'node:path';

const expanded = { version: 2, capabilityProfile: 'jev-expanded-v1', maxChoiceOptions: 255 };
const choice = (count = 16) => ({ purpose: 'select', risk: 'routine', state: 'Frozen candidate evidence', questions: {
  candidate: { type: 'choice', instructions: 'Select one candidate.', criteria: Object.fromEntries(Array.from({ length: count }, (_, i) => [`c${i}`, null])) },
} });
function answer(payload, confidence = .99) {
  return { model: 'jev-1.13.0', answers: Object.fromEntries(Object.entries(payload.questions).map(([name, q]) => [name,
    { type: 'choice', choice: Object.keys(q.criteria)[0], confidence,
      probabilities: Object.fromEntries(Object.keys(q.criteria).map((key, i) => [key, i === 0 ? 1 : 0])) }])) };
}

test('portable config remains v1/8 questions/24k bytes/16 options; expansion requires v2 and explicit profile', () => {
  assert.equal(validateConfig({}).version, 1);
  assert.equal(DEFAULTS.maxQuestions, 8); assert.equal(DEFAULTS.maxInputBytes, 24000); assert.equal(DEFAULTS.maxChoiceOptions, 16);
  assert.equal(validateConfig(expanded).maxChoiceOptions, 255);
  for (const patch of [{ maxChoiceOptions: 17 }, { ...expanded, version: 1 },
    { version: 2, maxChoiceOptions: 255 }, { ...expanded, maxChoiceOptions: 256 },
    { ...expanded, maxQuestions: 9 }, { ...expanded, maxInputBytes: 48001 }, { capabilityProfile: '__proto__' }]) {
    assert.throws(() => validateConfig(patch), /INVALID_CONFIG/);
  }
  assert.ok(48000 < MAX_FRAME_BYTES); assert.equal(MAX_RESPONSE_BYTES, 131072);
});
test('Score 10 levels is zero based; legacy 11 levels is rejected without migration', () => {
  const req = { purpose: 'judge', risk: 'routine', state: 'evidence', questions: {
    severity: { type: 'score', instructions: 'Rate severity.', criteria: Array.from({ length: 10 }, (_, i) => `level ${i}`) },
  } };
  validateRequest(req, DEFAULTS);
  const probabilities = Object.fromEntries(req.questions.severity.criteria.map((_, i) => [i, i === 9 ? 1 : 0]));
  assert.equal(normalizeResponse({ model: 'jev-1.13.0', answers: { severity: { type: 'score', score: 9, confidence: .99, probabilities } } }, req, DEFAULTS).answers.severity.value, 9);
  req.questions.severity.criteria.push('legacy level');
  assert.throws(() => validateRequest(req, DEFAULTS), /INVALID_REQUEST/);
  assert.throws(() => normalizeResponse({ model: 'jev-1.13.0', answers: { severity: { type: 'score' } } }, req, DEFAULTS), /MALFORMED_RESPONSE/);
  const criteria = decisionSchema.properties.questions.additionalProperties.properties.criteria;
  assert.equal(criteria.anyOf.find(x => x.type === 'array').maxItems, 10);
  assert.match(criteria.description, /v2 jev-expanded-v1/);
});
test('Jev opt-in admits 255 options, 256/default17/local17 reject before inference', async t => {
  let calls = 0;
  const f = fixture(t, { provider: async payload => { calls++; return answer(payload); } });
  t.after(() => f.engine.close());
  assert.equal((await f.engine.decide(choice(17))).reason, 'INVALID_REQUEST');
  f.config(expanded);
  assert.equal(f.engine.status().requestLimits.maxChoiceOptions, 255);
  assert.equal((await f.engine.decide(choice(255))).apply, true);
  assert.equal((await f.engine.decide(choice(256))).reason, 'INVALID_REQUEST');
  assert.equal((await f.engine.decide(choice(17), { providerOverride: 'laya' })).reason, 'INVALID_REQUEST');
  assert.equal(calls, 1);
  assert.equal(providerRequestLimits(validateConfig(expanded), 'laya').maxChoiceOptions, 16);
});
test('expanded profiles retain OFF/SHADOW, response bounds and atomic batch confidence', async t => {
  let calls = 0, low = false, oversize = false;
  const f = fixture(t, { provider: async payload => { calls++; const raw = answer(payload);
    if (low) raw.answers.second.confidence = .1;
    if (oversize) raw.padding = 'x'.repeat(MAX_RESPONSE_BYTES);
    return raw; } });
  t.after(() => f.engine.close());
  f.config({ ...expanded, mode: 'off' });
  assert.equal((await f.engine.decide(choice(255))).reason, 'OFF'); assert.equal(calls, 0);
  f.config({ mode: 'shadow' });
  const shadow = await f.engine.decide(choice(255)); assert.equal(shadow.apply, false); assert.deepEqual(shadow.answers, {});
  f.config({ mode: 'on' });
  const req = choice(255); req.questions.second = structuredClone(req.questions.candidate); low = true;
  const atomic = await f.engine.decide(req); assert.equal(atomic.reason, 'LOW_CONFIDENCE'); assert.deepEqual(atomic.answers, {});
  low = false; oversize = true;
  assert.equal((await f.engine.decide(choice(255))).reason, 'RESPONSE_TOO_LARGE');
});
test('profile policy and provider rereads prevent late expanded result application', async t => {
  const pending = deferred(), entered = deferred();
  const f = fixture(t, { provider: async payload => { entered.resolve(); await pending.promise; return answer(payload); } });
  t.after(() => f.engine.close()); f.config(expanded);
  const decision = f.engine.decide(choice(255)); await entered.promise;
  f.config({ capabilityProfile: 'portable', maxChoiceOptions: 16 }); pending.resolve();
  const result = await decision; assert.equal(result.reason, 'POLICY_CHANGED'); assert.equal(result.apply, false); assert.deepEqual(result.answers, {});
  const pending2 = deferred(), entered2 = deferred();
  const g = fixture(t, { provider: async payload => { entered2.resolve(); await pending2.promise; return answer(payload); } });
  t.after(() => g.engine.close()); g.config(expanded);
  const result2 = g.engine.decide(choice(255)); await entered2.promise;
  atomicWrite(path.join(g.home, 'providers.json'), JSON.stringify({ version: 1, provider: 'jev', laya: {
    python: '/usr/bin/python3', modelPath: g.home, model: 'laya/base', checkpoint: 'b'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu',
  } })); pending2.resolve();
  assert.equal((await result2).reason, 'PROVIDER_CHANGED');
});
