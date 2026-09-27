import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { setup, response, ROUTE_INPUT, FILTER_INPUT } from './features-helpers.mjs';
import { validateFeaturePolicy, setFeatureMode, loadFeaturePolicy, effectiveMode, CLAUDE_MODELS, setAbControlShare } from '../src/feature-policy.mjs';
import { routeOrDelegate, ROUTE_QUESTIONS } from '../src/routing.mjs';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { evaluatePairedRuns } from '../src/evaluation.mjs';
import { ControlError } from '../src/constants.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createControlLayer } from '../src/control-layer.mjs';
const copy = x => structuredClone(x);
const run = (name, fn) => test(name, async t => { const s = setup(); t.after(s.cleanup); await fn(s, t); });

for (const global of ['off', 'shadow', 'on']) for (const feature of ['off', 'shadow', 'on']) {
  test(`mode cap ${global}/${feature}`, async t => {
    const s = setup({ mode: global, featureMode: feature }); t.after(s.cleanup);
    const r = await s.layer.route(copy(ROUTE_INPUT));
    const mode = effectiveMode(global, feature);
    assert.equal(r.mode, mode); assert.equal(s.calls.length, mode === 'off' ? 0 : 1);
    assert.equal(r.apply, mode === 'on'); if (mode !== 'on') assert.equal(r.route, null);
  });
}
run('route makes ONE pinned TypeSafe call with three independent questions; no role/model/skill names on wire', async s => {
  const r = await s.layer.route(copy(ROUTE_INPUT));
  assert.equal(r.apply, true); assert.equal(r.route.tier, 'economy'); assert.equal(r.route.role, 'fixture-economy-role');
  assert.equal(r.changesHostModel, false); assert.equal(r.authorizesExecution, false);
  assert.equal(s.calls.length, 1); assert.equal(s.calls[0].model, 'jev-1.13.0');
  assert.deepEqual(Object.keys(s.calls[0].questions), ['intent', 'difficulty', 'risk']);
  const wire = JSON.stringify(s.calls);
  for (const secret of ['fixture-economy', 'fixture-economy-role', 'fixture-standard-role', 'fixture-strong-role', 'fixture-search']) assert.ok(!wire.includes(secret));
});
for (const [name, context] of [['locked', { modelLocked: true }], ['exhaustive', { exhaustive: true }], ['whole-repo', { scope: 'repository' }],
  ['cross-module', { scope: 'cross-module' }], ['unknown-scope', { scope: 'unknown' }], ['incomplete', { complete: false }], ['retry', { previousFailures: 1 }], ['high-impact', { highImpact: true }]]) {
  run(`route guard ${name} makes no API call`, async s => {
    const req = copy(ROUTE_INPUT); Object.assign(req.context, context);
    assert.equal((await s.layer.route(req)).apply, false); assert.equal(s.calls.length, 0);
  });
}
run('sensitive route remains local', async s => {
  assert.equal((await s.layer.route({ ...copy(ROUTE_INPUT), risk: 'sensitive' })).reason, 'SENSITIVE_SCOPE'); assert.equal(s.calls.length, 0);
});
run('empty/unknown profile and single distinct model skip classification', async s => {
  s.policy.router.profiles.codex = {}; s.save();
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).reason, 'INSUFFICIENT_TARGETS'); assert.equal(s.calls.length, 0);
});
for (const intent of ['other', 'operate', 'architecture']) test(`intent ${intent} cannot take cheap path`, async t => {
  const s = setup({ provider: p => response(p, { intent }) }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).apply, false);
});
test('expected difficulty and hard tail prevent misleading easy average', async t => {
  const s = setup({ provider: p => response(p, { difficulty: [0.8, 0, 0, 0, 0.2] }) }); t.after(s.cleanup);
  const r = await s.layer.route(copy(ROUTE_INPUT)); assert.equal(r.features.difficulty, 1.8); assert.equal(r.route.tier, 'strong');
});
test('moderate task chooses configured standard tier', async t => {
  const s = setup({ provider: p => response(p, { difficulty: [0, 0, 1, 0, 0] }) }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).route.tier, 'standard');
});
test('debugging retains strong tier even when difficulty is low', async t => {
  const s = setup({ provider: p => response(p, { intent: 'debug' }) }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).route.tier, 'strong');
});
test('safe argmax cannot override dangerous tail probability', async t => {
  const s = setup({ provider: p => response(p, { riskProbabilities: { safe: 0.96, caution: 0, high: 0.04, unknown: 0 } }) }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).reason, 'RISK_REQUIRES_HOST');
});
test('weak dimension delegates', async t => {
  const s = setup({ provider: p => response(p, { intentConfidence: 0.2 }) }); t.after(s.cleanup);
  const r = await s.layer.route(copy(ROUTE_INPUT)); assert.equal(r.apply, false); assert.equal(r.route, null);
});
test('served version drift cannot authorize a route', async t => {
  const s = setup({ provider: p => ({ ...response(p), model: 'jev-1.14.0' }) }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).reason, 'MODEL_VERSION_MISMATCH');
});
run('unavailable selected target delegates instead of inventing an ID', async s => {
  const req = copy(ROUTE_INPUT); req.availableModels = ['fixture-standard', 'fixture-strong'];
  assert.equal((await s.layer.route(req)).reason, 'TARGET_UNAVAILABLE');
});
run('a role the host does not actually have delegates even when the model is available', async s => {
  const req = copy(ROUTE_INPUT); req.availableRoles = ['fixture-standard-role', 'fixture-strong-role'];
  const r = await s.layer.route(req); assert.equal(r.reason, 'TARGET_UNAVAILABLE'); assert.equal(r.route, null);
});
run('fewer than two distinct roles actually available delegates without a network call', async s => {
  const req = copy(ROUTE_INPUT); req.availableRoles = ['fixture-economy-role'];
  assert.equal((await s.layer.route(req)).reason, 'INSUFFICIENT_TARGETS'); assert.equal(s.calls.length, 0);
});
test('an economy/standard intent override replaces the tier target and is flagged intentOverride', async t => {
  const s = setup({ provider: p => response(p, { intent: 'explain' }) }); t.after(s.cleanup);
  s.policy.router.profiles.codex.intents = { explain: { role: 'fixture-explain-role', model: 'fixture-explain', reasoning: 'low' } }; s.save();
  const req = copy(ROUTE_INPUT); req.availableRoles = [...req.availableRoles, 'fixture-explain-role']; req.availableModels = [...req.availableModels, 'fixture-explain'];
  const r = await s.layer.route(req);
  assert.equal(r.apply, true); assert.equal(r.route.tier, 'economy'); assert.equal(r.route.role, 'fixture-explain-role'); assert.equal(r.route.intentOverride, true);
});
test('an intent override target unavailable to this host reports TARGET_UNAVAILABLE, not the tier default', async t => {
  const s = setup({ provider: p => response(p, { intent: 'explain' }) }); t.after(s.cleanup);
  s.policy.router.profiles.codex.intents = { explain: { role: 'fixture-explain-role' } }; s.save();
  // fixture-explain-role is never added to availableRoles here, even though the economy tier target is.
  const r = await s.layer.route(copy(ROUTE_INPUT));
  assert.equal(r.reason, 'TARGET_UNAVAILABLE'); assert.equal(r.route, null);
});
test('an intent override never applies to STRONG_DEFAULT (debug always keeps the strong tier role)', async t => {
  const s = setup({ provider: p => response(p, { intent: 'debug' }) }); t.after(s.cleanup);
  s.policy.router.profiles.codex.intents = { debug: { role: 'fixture-explain-role' } }; s.save();
  const req = copy(ROUTE_INPUT); req.availableRoles = [...req.availableRoles, 'fixture-explain-role'];
  const r = await s.layer.route(req);
  assert.equal(r.route.tier, 'strong'); assert.equal(r.route.role, 'fixture-strong-role'); assert.equal(r.route.intentOverride, undefined);
});
run('router status warns about an empty host profile and a role the host no longer has', async s => {
  s.policy.router.profiles.claude = {}; s.save();
  const status = s.layer.status();
  assert.ok(status.features.router.warnings.includes('ROUTER_PROFILE_EMPTY:claude'));
  assert.ok(status.features.router.warnings.some(w => w.startsWith('ROUTER_ROLE_MISSING:codex:')));
});
test('claude model "fable" cannot be configured as a profile target', () => {
  assert.ok(!CLAUDE_MODELS.includes('fable'));
  assert.throws(() => validateFeaturePolicy({ version: 2, router: { profiles: { claude: { economy: { role: 'implementer', model: 'fable' } }, codex: {} } } }));
});
test('unknown skills are not returned', async t => {
  const s = setup({ provider: p => response(p, { intent: 'explain' }) }); t.after(s.cleanup);
  assert.deepEqual((await s.layer.route({ ...copy(ROUTE_INPUT), availableSkills: [] })).route.skills, []);
  assert.deepEqual((await s.layer.route(copy(ROUTE_INPUT))).route.skills, ['fixture-search']);
});
run('input with secret-looking data is not transmitted', async s => {
  const req = copy(ROUTE_INPUT); req.task = 'Bearer abcdefghijklmnopqrstuvwxyz';
  assert.equal((await s.layer.route(req)).reason, 'SENSITIVE_INPUT'); assert.equal(s.calls.length, 0);
});
run('malformed route and missing context do not execute', async s => {
  const req = copy(ROUTE_INPUT); delete req.context.complete;
  assert.equal((await s.layer.route(req)).reason, 'INVALID_ROUTE_REQUEST'); assert.equal(s.calls.length, 0);
});
run('library fast path skips delegate exactly once', async s => {
  let uses = 0, delegates = 0;
  const handlers = { use: () => { uses++; }, delegate: () => { delegates++; } };
  await routeOrDelegate(s.layer, copy(ROUTE_INPUT), handlers);
  setFeatureMode(s.home, 'router', 'off'); await routeOrDelegate(s.layer, copy(ROUTE_INPUT), handlers);
  assert.equal(uses, 1); assert.equal(delegates, 1);
});
run('shadow hides proposal then records only agreement, not task quality', async s => {
  setFeatureMode(s.home, 'router', 'shadow');
  const r = await s.layer.route(copy(ROUTE_INPUT)); assert.equal(r.route, null); assert.equal(r.features, undefined);
  const f = s.layer.observe({ id: r.id, tier: 'economy' }); assert.equal(f.matched, true); assert.equal(f.taskQualityMeasured, false);
  assert.throws(() => s.layer.observe({ id: r.id, tier: 'economy' }));
});
test('global OFF during inference rejects late response', async t => {
  let s; s = setup({ provider: p => { setMode(s.home, 'off', s.env); return response(p); } }); t.after(s.cleanup);
  const r = await s.layer.route(copy(ROUTE_INPUT)); assert.equal(r.apply, false); assert.equal(r.route, null);
});
test('feature policy change during inference rejects late response', async t => {
  let s; s = setup({ provider: p => { s.policy.router.economyMaxDifficulty = 1.5; s.save(); return response(p); } }); t.after(s.cleanup);
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).reason, 'FEATURE_POLICY_CHANGED');
});
run('filter drops only clear exclusions, keeps unknown and mandatory; preserves order', async s => {
  const r = await s.layer.filter(copy(FILTER_INPUT));
  assert.equal(r.apply, true); assert.deepEqual(r.rejectIds, ['drop_1']);
  assert.deepEqual(r.keepIds, ['keep_1', 'uncertain_1', 'required_1']); assert.deepEqual(r.reviewIds, ['uncertain_1']);
  assert.ok(!JSON.stringify(r).includes('transaction')); assert.ok(!JSON.stringify(s.calls).includes('but explicitly required'));
});
for (const [name, changes] of [['exhaustive', { coverage: 'exhaustive' }], ['sensitive', { risk: 'sensitive' }]]) {
  run(`filter ${name} keeps every item without inference`, async s => {
    const r = await s.layer.filter({ ...copy(FILTER_INPUT), ...changes });
    assert.equal(r.apply, false); assert.equal(r.keepIds.length, 4); assert.equal(s.calls.length, 0);
  });
}
run('bulk OFF keeps all even with global ON', async s => {
  setFeatureMode(s.home, 'bulk', 'off'); const r = await s.layer.filter(copy(FILTER_INPUT));
  assert.equal(r.keepIds.length, 4); assert.equal(r.rejectIds.length, 0); assert.equal(s.calls.length, 0);
});
run('bulk SHADOW never hides actual input and computes recall against supplied labels', async s => {
  setFeatureMode(s.home, 'bulk', 'shadow'); const r = await s.layer.filter(copy(FILTER_INPUT));
  assert.equal(r.keepIds.length, 4); assert.equal(r.rejectIds.length, 0);
  const f = s.layer.observe({ id: r.id, relevantIds: ['keep_1', 'drop_1'] }); assert.equal(f.missed, 1); assert.equal(f.recall, 0.5);
});
test('one uncertain probability retains the entire batch', async t => {
  const s = setup({ provider: p => { const r = response(p); r.answers.item_0.confidence = 0.1; return r; } }); t.after(s.cleanup);
  const r = await s.layer.filter(copy(FILTER_INPUT)); assert.equal(r.apply, false); assert.equal(r.keepIds.length, 4);
});
run('batching uses bounded questions and request budget keeps unprocessed input', async s => {
  s.policy.bulk.batchSize = 2; s.policy.bulk.maxRequests = 1; s.save();
  const req = copy(FILTER_INPUT); req.items.push({ id: 'drop_2', text: 'drop: another item' });
  const r = await s.layer.filter(req); assert.equal(s.calls.length, 1); assert.equal(Object.keys(s.calls[0].questions).length, 2);
  assert.ok(r.keepIds.includes('drop_2')); assert.ok(r.reviewIds.includes('drop_2'));
});
test('API outage does not produce a retry storm or drop input', async t => {
  const s = setup({ provider: () => { throw new ControlError('RATE_LIMITED'); } }); t.after(s.cleanup);
  const r = await s.layer.filter(copy(FILTER_INPUT)); assert.equal(s.calls.length, 1); assert.equal(r.keepIds.length, 4);
});
test('version changes mid-batch restore all items', async t => {
  const s = setup({ provider: (p, key, opts, count) => ({ ...response(p), model: count > 1 ? 'jev-1.14.0' : 'jev-1.13.0' }) }); t.after(s.cleanup);
  s.policy.bulk.batchSize = 1; s.save();
  const r = await s.layer.filter(copy(FILTER_INPUT)); assert.equal(r.apply, false); assert.equal(r.keepIds.length, 4); assert.equal(r.rejectIds.length, 0);
});
run('duplicate IDs invalidate filter rather than lose or reorder source', async s => {
  const req = copy(FILTER_INPUT); req.items[1].id = req.items[0].id;
  const r = await s.layer.filter(req); assert.equal(r.valid, false); assert.equal(r.apply, false); assert.equal(s.calls.length, 0);
});
run('secret detected in later snippet blocks all network calls', async s => {
  const req = copy(FILTER_INPUT); req.items[2].text = 'password=secret123456789';
  const r = await s.layer.filter(req); assert.equal(r.reason, 'SENSITIVE_INPUT'); assert.equal(s.calls.length, 0); assert.equal(r.keepIds.length, 4);
});
run('required-only and empty input need no classifier', async s => {
  assert.equal((await s.layer.filter({ ...copy(FILTER_INPUT), items: [] })).keepIds.length, 0);
  assert.equal((await s.layer.filter({ ...copy(FILTER_INPUT), items: [FILTER_INPUT.items[3]] })).keepIds.length, 1);
  assert.equal(s.calls.length, 0);
});
run('local telemetry does not contain tasks, snippets, keys or configured model IDs', async s => {
  await s.layer.route(copy(ROUTE_INPUT)); await s.layer.filter(copy(FILTER_INPUT));
  const dir = path.join(s.home, 'logs'); const text = fs.readdirSync(dir).map(f => fs.readFileSync(path.join(dir, f), 'utf8')).join('');
  for (const secret of [ROUTE_INPUT.task, FILTER_INPUT.items[0].text, s.env.TYPESAFE_API_KEY, 'fixture-economy']) assert.ok(!text.includes(secret));
});
run('feature policy denies malformed keys, provider URLs, aliases and permissive drop thresholds', async s => {
  for (const raw of [{ endpoint: 'https://other.example' }, { bulk: { minRejectProbability: 0.5 } }, { router: { expectedModel: 'jev-latest' } },
    JSON.parse('{"router":{"profiles":{"__proto__":{}}}}')]) assert.throws(() => validateFeaturePolicy(raw));
  atomicWrite(path.join(s.home, 'features.json'), '{bad'); assert.equal(s.layer.status().features.router.mode, 'off');
  setFeatureMode(s.home, 'router', 'off'); assert.equal(loadFeaturePolicy(s.home).router.mode, 'off');
});
run('policy symlink cannot be followed', async s => {
  fs.renameSync(path.join(s.home, 'features.json'), path.join(s.home, 'original.json'));
  fs.symlinkSync(path.join(s.home, 'original.json'), path.join(s.home, 'features.json'));
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).apply, false); assert.equal(s.calls.length, 0);
});
run('CLI mode/status/filter are runnable without original host executables', async s => {
  const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
  const exec = args => spawnSync(process.execPath, [bin, ...args, '--home', s.home], { encoding: 'utf8', env: { PATH: process.env.PATH, HOME: s.home } });
  assert.equal(exec(['router', 'off']).status, 0);
  assert.equal(JSON.parse(exec(['status']).stdout).features.router.mode, 'off');
  assert.equal(exec(['policy', 'check']).status, 0);
});
run('abControlShare defaults to 0, is bounded to [0, 0.5], and `router ab` writes only that field', async s => {
  assert.equal(loadFeaturePolicy(s.home).router.abControlShare, 0);
  for (const bad of [-0.1, 0.51, 1, NaN, 'x']) assert.throws(() => validateFeaturePolicy({ router: { abControlShare: bad } }));
  const before = loadFeaturePolicy(s.home);
  const updated = setAbControlShare(s.home, '0.3');
  assert.equal(updated.router.abControlShare, 0.3);
  assert.deepEqual({ ...updated.router, abControlShare: before.router.abControlShare }, before.router); // every other router field preserved
  assert.equal(setAbControlShare(s.home, 'off').router.abControlShare, 0);
  assert.throws(() => setAbControlShare(s.home, '0.9'));
});
run('CLI `router ab` sets/clears abControlShare and is visible in status', async s => {
  const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
  const exec = args => spawnSync(process.execPath, [bin, ...args, '--home', s.home], { encoding: 'utf8', env: { PATH: process.env.PATH, HOME: s.home } });
  const set = exec(['router', 'ab', '0.25']);
  assert.equal(set.status, 0);
  assert.equal(JSON.parse(set.stdout).features.router.abControlShare, 0.25);
  assert.equal(JSON.parse(exec(['status']).stdout).features.router.abControlShare, 0.25);
  assert.equal(JSON.parse(exec(['router', 'ab', 'off']).stdout).features.router.abControlShare, 0);
  assert.notEqual(exec(['router', 'ab', '0.9']).status, 0);
});
test('paired evaluator rejects SHADOW predictions without execution evidence', () => {
  assert.throws(() => evaluatePairedRuns({ cases: [{ downgraded: true, baseline: { executed: true }, routed: { executed: false } }] }));
});
test('paired evaluator separates errors, savings and confidence bound', () => {
  const pass = { executed: true, passed: true, elapsedMs: 100, outputTokens: 100 };
  const r = evaluatePairedRuns({ cases: Array.from({ length: 100 }, () => ({ downgraded: true, baseline: pass, routed: { ...pass, elapsedMs: 80, outputTokens: 70 } })) });
  assert.equal(r.wrongDowngrades, 0); assert.ok(r.zeroErrorUpper95 > 0.029); assert.ok(r.zeroErrorUpper95 < 0.03);
  assert.ok(Math.abs(r.elapsedReduction - 0.2) < 1e-10); assert.equal(r.monetarySavings, null); assert.equal(r.automaticApproval, false);
});

// --- decision-level route gate (laya provider only; providers.json laya.qualification.routeGate) ---
function layaProvidersConfig(home, gate = { method: 'decision-v1', tierCoverage: .9, maxHostProbability: .05 }) {
  return { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'a'.repeat(64),
    runtimeVersion: '0.3.4', device: 'cpu', qualification: { checkpoint: 'a'.repeat(64), calibrationVersion: 'v1', purposes: ['route'],
      minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9, ...(gate ? { routeGate: gate } : {}) } } };
}
// One-hot laya-shaped answers: intent/risk/difficulty each fully concentrated on one value, so
// tierDistribution collapses to a single bucket regardless of the (independently low) confidence.
function layaOneHotResponse(laya, { intent = 'edit', difficultyIndex = 0, risk = 'safe', confidence = .99 } = {}) {
  const oneHot = (keys, value) => Object.fromEntries(keys.map(k => [k, k === value ? 1 : 0]));
  return {
    identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device, precision: 'torch.float32' },
    answers: {
      intent: { type: 'choice', choice: intent, confidence, probabilities: oneHot(Object.keys(ROUTE_QUESTIONS.intent.criteria), intent) },
      difficulty: { type: 'score', score: difficultyIndex, confidence, probabilities: oneHot(['0', '1', '2', '3', '4'], String(difficultyIndex)) },
      risk: { type: 'choice', choice: risk, confidence, probabilities: oneHot(Object.keys(ROUTE_QUESTIONS.risk.criteria), risk) },
    },
    usage: { input_tokens: 5, output_tokens: 0 },
  };
}
run('laya decision-gate route applies even when per-answer confidence is below qualification.minConfidence', async s => {
  const providers = layaProvidersConfig(s.home);
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify(providers));
  const engine = createDecisionEngine({ home: s.home, env: s.env, provider: async () => layaOneHotResponse(providers.laya, { confidence: .3 }) });
  const layer = createControlLayer({ home: s.home, env: s.env, engine });
  const r = await layer.route(copy(ROUTE_INPUT));
  assert.equal(r.apply, true);
  assert.equal(r.reason, 'DECISION_ECONOMY');
  assert.equal(r.route.tier, 'economy');
  assert.equal(r.route.role, 'fixture-economy-role');
  assert.equal(r.features.tierProbabilities.economy, 1);
  engine.close();
});
run('laya without a routeGate keeps the legacy per-answer path: the same low-confidence answers do not apply', async s => {
  const providers = layaProvidersConfig(s.home, null);
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify(providers));
  const engine = createDecisionEngine({ home: s.home, env: s.env, provider: async () => layaOneHotResponse(providers.laya, { confidence: .3 }) });
  const layer = createControlLayer({ home: s.home, env: s.env, engine });
  const r = await layer.route(copy(ROUTE_INPUT));
  assert.equal(r.apply, false);
  assert.equal(r.route, null);
  engine.close();
});
test('laya decision-gate SHADOW mode records the proposal without applying it', async t => {
  const s = setup({ featureMode: 'shadow' }); t.after(s.cleanup);
  const providers = layaProvidersConfig(s.home);
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify(providers));
  const engine = createDecisionEngine({ home: s.home, env: s.env, provider: async () => layaOneHotResponse(providers.laya) });
  const layer = createControlLayer({ home: s.home, env: s.env, engine });
  const r = await layer.route(copy(ROUTE_INPUT));
  assert.equal(r.mode, 'shadow'); assert.equal(r.apply, false); assert.equal(r.reason, 'SHADOW');
  const metrics = layer.observe({ id: r.id, tier: 'economy' });
  assert.equal(metrics.matched, true);
  engine.close();
});
