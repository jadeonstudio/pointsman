import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { DEFAULTS, ControlError } from '../src/constants.mjs';
import { createDecisionEngine, decideOrDelegate } from '../src/engine.mjs';
import { containsSensitiveData, validateRequest, normalizeResponse, wireRequest } from '../src/contracts.mjs';
import { setMode } from '../src/storage.mjs';
import { readMetrics } from '../src/metrics.mjs';
import { fixture, request, response, KEY, deferred } from './helpers.mjs';

test('ON normalizes all three primitives, including fractional score and null Noul confidence', async t => {
  const f = fixture(t); const r = await f.engine.decide(request());
  assert.equal(r.apply, true); assert.equal(r.answers.category.value, 'documentation');
  assert.equal(r.answers.passed.confidence, null); assert.equal(r.answers.passed.probabilityTrue, .99);
  assert.equal(r.answers.complexity.value, .02); assert.equal(r.authorizesExecution, false);
  assert.deepEqual(r.usage, { inputTokens: 200, outputTokens: 0 }); assert.equal(f.calls(), 1);
});
test('OFF is zero-network, skips malformed input and keys, and writes no telemetry', async t => {
  const f = fixture(t); setMode(f.home, 'off', {}); f.env.TYPESAFE_API_KEY = 'bad key';
  const r = await f.engine.decide(null); assert.equal(r.reason, 'OFF'); assert.equal(f.calls(), 0);
  assert.equal(fs.existsSync(path.join(f.home, 'logs')), false);
});
test('environment kill switch overrides persistent ON', async t => {
  const f = fixture(t); f.env.POINTSMAN_DISABLE = '1'; assert.equal((await f.engine.decide(request())).reason, 'OFF'); assert.equal(f.calls(), 0);
});
test('package-root prepare is explicit Laya residency, without inference', async t => {
  const f=fixture(t),providers=path.join(f.home,'providers.json');const settings={version:1,provider:'laya',laya:{python:'/bin/python3',modelPath:'/tmp/model',model:'laya/base',checkpoint:'a'.repeat(64),runtimeVersion:'0.3.4',device:'cpu'}};fs.writeFileSync(providers,JSON.stringify(settings),{mode:0o600});
  let seen;const layaClient={status:()=>({running:false}),close:()=>{},prepare:async(...args)=>{seen=args;return{model:'laya/base'};}};
  const engine=createDecisionEngine({home:f.home,env:f.env,layaClient});t.after(()=>engine.close());
  assert.deepEqual(await engine.prepare({timeoutMs:500,resident:false}),{model:'laya/base'});assert.equal(seen[0].provider,'laya');assert.equal(seen[0].laya.idleTimeoutMs,60000);assert.deepEqual(seen[1],{signal:undefined,timeoutMs:500,resident:false,env:f.env});
  settings.provider='jev';fs.writeFileSync(providers,JSON.stringify(settings));await assert.rejects(engine.prepare(),/LAYA_NOT_SELECTED/);
  settings.provider='laya';fs.writeFileSync(providers,JSON.stringify(settings));setMode(f.home,'off',{});await assert.rejects(engine.prepare(),/OFF/);
});
test('SHADOW is blind and feedback records agreement, not decisions or accuracy', async t => {
  const f = fixture(t); setMode(f.home, 'shadow', {}); const r = await f.engine.decide(request());
  assert.equal(r.reason, 'SHADOW'); assert.equal(r.apply, false); assert.deepEqual(r.answers, {});
  const feedback = f.engine.feedback({ id: r.id, baseline: { category: 'documentation', passed: true, complexity: .02 }, baselineUsage: { inputTokens: 900, outputTokens: 20 }, baselineElapsedMs: 400, taskSucceeded: true });
  assert.equal(feedback.matched, 3); assert.equal(feedback.agreementOnly, true);
  assert.throws(() => f.engine.feedback({ id: r.id, baseline: {} }), /FEEDBACK_EXPIRED/);
  const report = readMetrics(f.home); assert.equal(report.baselineAgreement.isAccuracy, false);
  assert.equal(report.tokenSavings, null); assert.equal(report.baselineReportedTokens.input, 900);
  const logs = fs.readFileSync(path.join(f.home, 'logs', fs.readdirSync(path.join(f.home, 'logs'))[0]), 'utf8');
  for (const hidden of [KEY, 'Documentation typo', 'documentation', 'criteria', 'baseline"']) assert.equal(logs.includes(hidden), false, hidden);
});
test('missing usage stays unknown', () => { const r = response(); delete r.usage; assert.deepEqual(normalizeResponse(r, request(), DEFAULTS).usage, { inputTokens: null, outputTokens: null }); });
test('choice confidence and selected probability are independent thresholds', () => {
  for (const change of [r => { r.answers.category.confidence = .7; }, r => { r.answers.category.probabilities = { documentation: .7, implementation: .3 }; }]) {
    const raw = response(); change(raw); assert.equal(normalizeResponse(raw, request(), DEFAULTS).eligible, false);
  }
});
test('uncertain Noul and score disable the entire batch', () => {
  for (const change of [r => { r.answers.passed.noul = .7; }, r => { r.answers.complexity.confidence = .4; }]) {
    const raw = response(); change(raw); assert.equal(normalizeResponse(raw, request(), DEFAULTS).eligible, false);
  }
});
test('Noul can confidently choose false without inventing confidence', () => { const raw = response(); raw.answers.passed.noul = .01; const r = normalizeResponse(raw, request(), DEFAULTS); assert.equal(r.eligible, true); assert.equal(r.answers.passed.value, false); });
test('LOW_CONFIDENCE exposes no usable answers', async t => {
  const f = fixture(t, { provider: async () => { const raw = response(); raw.answers.category.confidence = .1; return raw; } });
  const r = await f.engine.decide(request()); assert.equal(r.reason, 'LOW_CONFIDENCE'); assert.deepEqual(r.answers, {});
});
test('malformed distributions, types, expected scores and usage reject safely', () => {
  const mutations = [
    r => { r.answers.category.choice = 'invented'; }, r => { r.answers.category.probabilities.documentation = .1; },
    r => { r.answers.category.confidence = NaN; }, r => { r.answers.category.choice = 'implementation'; },
    r => { r.answers.complexity.score = 1; }, r => { r.answers.passed.type = 'choice'; },
    r => { delete r.answers.passed; }, r => { r.usage.input_tokens = -1; }, r => { r.model = 'bad model'; },
    r => { r.answers.category.probabilities.extra = 0; }, r => { r.answers.passed.noul = 2; },
  ];
  for (const mutate of mutations) { const raw = response(); mutate(raw); assert.throws(() => normalizeResponse(raw, request(), DEFAULTS), /MALFORMED_RESPONSE/); }
});
test('request validation rejects prototype keys, unknown fields, invalid criteria and budgets', () => {
  const mutations = [q => { q.hack = true; }, q => { q.state = JSON.parse('{"__proto__":{}}'); }, q => { q.questions = {}; },
    q => { q.questions.category.criteria = { only: null }; }, q => { q.questions.category.instructions = ''; },
    q => { q.questions.complexity.criteria = ['one']; }, q => { q.questions.passed.criteria = { yes: 'invalid' }; },
    q => { q.state = 'x'.repeat(25000); }, q => { q.risk = 'safe'; }];
  for (const mutate of mutations) { const q = request(); mutate(q); assert.throws(() => validateRequest(q, DEFAULTS)); }
  const input = request(); const detached = validateRequest(input, DEFAULTS); input.state.change = 'changed'; assert.notEqual(detached.state.change, input.state.change);
});
test('wire payload uses model/state/questions and JSON instructions, never key', () => {
  const raw = wireRequest(request(), 'jev-latest'); assert.deepEqual(Object.keys(raw), ['model', 'state', 'questions']);
  assert.ok(Array.isArray(raw.questions.category.instructions)); assert.equal(JSON.stringify(raw).includes(KEY), false);
});
test('sensitive-data screen covers state, descriptions, credentials and question instructions', async t => {
  const f = fixture(t);
  const strings = [KEY, 'Bearer abc123456789', '-----BEGIN PRIVATE KEY-----', 'password=verysecret', 'https://alice:secret@example.com', 'ghp_abcdefghijklmno'];
  for (const text of strings) {
    const q = request(); q.questions.category.instructions = text;
    assert.equal(containsSensitiveData(q, KEY), true);
    assert.equal((await f.engine.decide(q)).reason, 'SENSITIVE_INPUT');
  }
  const q = request(); q.state = { authorization: 'opaque' }; assert.equal((await f.engine.decide(q)).reason, 'SENSITIVE_INPUT'); assert.equal(f.calls(), 0);
});
test('sensitive scope, absent credentials and disabled TLS stay local', async t => {
  const f = fixture(t); const q = request(); q.risk = 'sensitive'; assert.equal((await f.engine.decide(q)).reason, 'SENSITIVE_SCOPE');
  delete f.env.TYPESAFE_API_KEY; assert.equal((await f.engine.decide(request())).reason, 'NO_API_KEY');
  f.env.TYPESAFE_API_KEY = KEY; f.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'; assert.equal((await f.engine.decide(request())).reason, 'INSECURE_TLS_REFUSED'); assert.equal(f.calls(), 0);
});
test('provider exceptions never leak raw text; circuit opens and recovers', async t => {
  let clock = 0, broken = true, calls = 0;
  const f = fixture(t, { now: () => clock, provider: async () => { calls++; if (broken) throw new Error(`oops ${KEY}`); return response(); } });
  for (let i = 0; i < 3; i++) { const r = await f.engine.decide(request()); assert.equal(r.reason, 'INTERNAL_ERROR'); assert.equal(JSON.stringify(r).includes(KEY), false); }
  assert.equal((await f.engine.decide(request())).reason, 'CIRCUIT_OPEN'); assert.equal(calls, 3);
  clock = 31000; broken = false; assert.equal((await f.engine.decide(request())).apply, true);
});
test('upstream 429 fallback has no retry', async t => { let count = 0; const f = fixture(t, { provider: async () => { count++; throw new ControlError('RATE_LIMITED'); } }); assert.equal((await f.engine.decide(request())).reason, 'RATE_LIMITED'); assert.equal(count, 1); });
test('per-process rate limit prevents another request and expires', async t => {
  let clock = 0; const f = fixture(t, { now: () => clock }); f.config({ maxCallsPerMinute: 1 });
  assert.equal((await f.engine.decide(request())).apply, true); assert.equal((await f.engine.decide(request())).reason, 'LOCAL_RATE_LIMIT'); clock = 60001; assert.equal((await f.engine.decide(request())).apply, true);
});
test('in-flight cap and switch-off while awaiting response', async t => {
  const d = deferred(); const f = fixture(t, { provider: async () => { await d.promise; return response(); } }); f.config({ maxInFlight: 1 });
  const first = f.engine.decide(request()); assert.equal((await f.engine.decide(request())).reason, 'CONCURRENCY_LIMIT');
  setMode(f.home, 'off', {}); d.resolve(); const r = await first; assert.equal(r.reason, 'MODE_CHANGED'); assert.equal(r.apply, false); assert.equal(f.engine.status().inFlight, 0);
});
test('policy changes and cancellation prevent applying in-flight responses', async t => {
  const d = deferred(); const f = fixture(t, { provider: async () => { await d.promise; return response(); } });
  const first = f.engine.decide(request()); f.config({ minConfidence: .99 }); d.resolve(); assert.equal((await first).reason, 'POLICY_CHANGED');
  const controller = new AbortController(); controller.abort(); assert.equal((await f.engine.decide(request(), { signal: controller.signal })).reason, 'CANCELLED');
});
test('feedback expiry and strict baseline validation', async t => {
  let clock = 0; const f = fixture(t, { now: () => clock }); const r = await f.engine.decide(request());
  assert.throws(() => f.engine.feedback({ id: r.id, baseline: { category: true, passed: true, complexity: .02 } }), /INVALID_FEEDBACK/);
  clock = 300001; assert.throws(() => f.engine.feedback({ id: r.id, baseline: {} }), /FEEDBACK_EXPIRED/);
});
test('owned fast path replaces the delegate call, with OFF preserving it', async t => {
  const f = fixture(t); let hostCalls = 0, useCalls = 0;
  const handlers = { use: () => ++useCalls, delegate: () => ++hostCalls };
  await decideOrDelegate(f.engine, request(), handlers); setMode(f.home, 'off', {}); await decideOrDelegate(f.engine, request(), handlers);
  assert.equal(useCalls, 1); assert.equal(hostCalls, 1);
});
