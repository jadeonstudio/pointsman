import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { setup } from './features-helpers.mjs';
import { validateFeaturePolicy, loadFeaturePolicy, setFeatureMode, setAbControlShare, FEATURE_DEFAULTS } from '../src/feature-policy.mjs';
import { effortRecommendation, ROUTE_QUESTIONS } from '../src/routing.mjs';
import { atomicWrite } from '../src/storage.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createControlLayer } from '../src/control-layer.mjs';
import { processHookEvent, HOOK_EVENTS } from '../src/hooks.mjs';

const copy = x => structuredClone(x);
const run = (name, fn) => test(name, async t => { const s = setup(); t.after(s.cleanup); await fn(s, t); });

function readEvents(home) {
  const dir = path.join(home, 'logs');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).flatMap(name => fs.readFileSync(path.join(dir, name), 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l)));
}
function layaProvidersConfig(home) {
  return { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'b'.repeat(64),
    runtimeVersion: '0.3.4', device: 'cpu', qualification: { checkpoint: 'b'.repeat(64), calibrationVersion: 'v1', purposes: ['route'],
      minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 } } };
}
// Custom (non-one-hot) probability maps, so effortRecommendation's thresholds can be exercised
// end-to-end, not only as a pure function.
function layaResponse(laya, { intentProbs, difficultyProbs, riskProbs, confidence = .99 } = {}) {
  const pick = probs => Object.entries(probs).reduce((best, [k, v]) => (v > best.v ? { k, v } : best), { k: null, v: -1 }).k;
  return {
    identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device, precision: 'torch.float32' },
    answers: {
      intent: { type: 'choice', choice: pick(intentProbs), confidence, probabilities: intentProbs },
      difficulty: { type: 'score', score: Object.entries(difficultyProbs).reduce((n, [k, p]) => n + Number(k) * p, 0), confidence, probabilities: difficultyProbs },
      risk: { type: 'choice', choice: pick(riskProbs), confidence, probabilities: riskProbs },
    },
    usage: { input_tokens: 5, output_tokens: 0 },
  };
}
const EASY_LOW_RISK = { intentProbs: { explain: .8, edit: .2, debug: 0, operate: 0, research: 0, architecture: 0, other: 0 },
  difficultyProbs: { 0: .8, 1: .1, 2: .1, 3: 0, 4: 0 }, riskProbs: { safe: .95, caution: .05, high: 0, unknown: 0 } };
const HARD_LOW_RISK = { intentProbs: { explain: 0, edit: .2, debug: .8, operate: 0, research: 0, architecture: 0, other: 0 },
  difficultyProbs: { 0: 0, 1: 0, 2: .1, 3: .3, 4: .6 }, riskProbs: { safe: .95, caution: .05, high: 0, unknown: 0 } };
const EASY_HIGH_RISK = { intentProbs: { explain: .8, edit: .2, debug: 0, operate: 0, research: 0, architecture: 0, other: 0 },
  difficultyProbs: { 0: .8, 1: .1, 2: .1, 3: 0, 4: 0 }, riskProbs: { safe: .1, caution: .1, high: .8, unknown: 0 } };
function setupLaya(s, response) {
  const providers = layaProvidersConfig(s.home);
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify(providers));
  const engine = createDecisionEngine({ home: s.home, env: s.env, provider: async () => layaResponse(providers.laya, response) });
  const layer = createControlLayer({ home: s.home, env: s.env, engine });
  return { providers, engine, layer };
}
function withEffort(s, overrides) {
  const policy = loadFeaturePolicy(s.home);
  Object.assign(policy.effort, overrides);
  atomicWrite(path.join(s.home, 'features.json'), JSON.stringify(validateFeaturePolicy(policy)));
}

// --- routing.effortRecommendation (pure) ---------------------------------------------------------
test('effortRecommendation: lower when pLow high and risk low', () => {
  const answers = { difficulty: { probabilities: { 0: .5, 1: .3, 2: .1, 3: .1, 4: 0 } }, risk: { probabilities: { high: .05 } } };
  const r = effortRecommendation(answers, FEATURE_DEFAULTS.effort);
  assert.equal(r.direction, 'lower'); assert.equal(r.level, FEATURE_DEFAULTS.effort.lowerTo); assert.equal(r.reason, 'LOWER');
  assert.equal(Math.round(r.pLow * 100) / 100, 0.8);
});
test('effortRecommendation: raise wins over lower when both thresholds are met', () => {
  // pLow (0+1) = 0.75 >= 0.7, but pHigh (3+4) = 0.6 >= 0.5 too -> raise must win.
  const answers = { difficulty: { probabilities: { 0: .4, 1: .35, 2: .05, 3: .3, 4: .3 } }, risk: { probabilities: { high: .05 } } };
  const r = effortRecommendation(answers, FEATURE_DEFAULTS.effort);
  assert.equal(r.direction, 'raise'); assert.equal(r.level, FEATURE_DEFAULTS.effort.raiseTo); assert.equal(r.reason, 'RAISE');
});
test('effortRecommendation: high risk alone raises even when difficulty looks easy', () => {
  const answers = { difficulty: { probabilities: { 0: .8, 1: .1, 2: .1, 3: 0, 4: 0 } }, risk: { probabilities: { high: .8 } } };
  const r = effortRecommendation(answers, FEATURE_DEFAULTS.effort);
  assert.equal(r.direction, 'raise');
});
test('effortRecommendation: high risk blocks a lower even when difficulty is easy', () => {
  const answers = { difficulty: { probabilities: { 0: .8, 1: .1, 2: .1, 3: 0, 4: 0 } }, risk: { probabilities: { high: .25 } } };
  const r = effortRecommendation(answers, FEATURE_DEFAULTS.effort);
  assert.equal(r.direction, null); assert.equal(r.reason, 'NO_CHANGE');
});
test('effortRecommendation: neither threshold met returns null direction', () => {
  const answers = { difficulty: { probabilities: { 0: .2, 1: .2, 2: .4, 3: .1, 4: .1 } }, risk: { probabilities: { high: .05 } } };
  const r = effortRecommendation(answers, FEATURE_DEFAULTS.effort);
  assert.equal(r.direction, null); assert.equal(r.reason, 'NO_CHANGE');
});
test('effortRecommendation: missing dimensions', () => {
  assert.equal(effortRecommendation({ difficulty: null, risk: { probabilities: {} } }, FEATURE_DEFAULTS.effort).reason, 'MISSING_DIMENSIONS');
});

// --- feature-policy: effort defaults, validation, migration --------------------------------------
test('effort defaults are OFF and an old features.json without `effort` still loads', () => {
  const policy = validateFeaturePolicy({ version: 2, router: { mode: 'on' } }); // no `effort` key at all
  assert.deepEqual(policy.effort, FEATURE_DEFAULTS.effort);
  assert.equal(policy.effort.mode, 'off');
});
test('effort policy rejects out-of-range fields', () => {
  for (const bad of [{ mode: 'nope' }, { input: 'nope' }, { contextChars: -1 }, { contextChars: 2001 },
    { lowerTo: 'xhigh' }, { raiseTo: 'low' }, { minLowerProbability: 1.5 }, { abControlShare: 0.6 },
    { mainLoop: 'always' }, { coldAfterSeconds: 100 }, { coldAfterSeconds: 90000 }, { subagents: 'maybe' }]) {
    assert.throws(() => validateFeaturePolicy({ version: 2, effort: bad }), /INVALID_FEATURE_POLICY/);
  }
});
test('`pointsman effort` off|shadow|on and `effort ab` write only the effort section', () => {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-effort-cli-')));
  fs.chmodSync(home, 0o700);
  setFeatureMode(home, 'effort', 'shadow');
  let policy = loadFeaturePolicy(home);
  assert.equal(policy.effort.mode, 'shadow'); assert.equal(policy.router.mode, 'off'); // router untouched
  setAbControlShare(home, '0.25', 'effort');
  policy = loadFeaturePolicy(home);
  assert.equal(policy.effort.abControlShare, 0.25); assert.equal(policy.router.abControlShare, 0);
  fs.rmSync(home, { recursive: true, force: true });
});

// --- control-layer.effort(): mode, sensitivity, provider qualification --------------------------
run('effort OFF returns at once, no inference, no log', async (s, t) => {
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.' });
  assert.equal(r.mode, 'off'); assert.equal(r.apply, false);
  assert.deepEqual(readEvents(s.home).filter(e => e.kind === 'effort'), []);
});
run('effort SENSITIVE_INPUT never calls the provider', async (s, t) => {
  withEffort(s, { mode: 'on' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'api_key: sk-abcdefghijklmnop123456' });
  assert.equal(r.reason, 'SENSITIVE_INPUT'); assert.equal(r.apply, false); assert.equal(r.inferenceCalls, 0);
});
run('effort on a non-laya provider reports UNQUALIFIED_PROVIDER without inference', async (s, t) => {
  withEffort(s, { mode: 'on' }); // no providers.json written -> defaults to jev
  const r = await s.layer.effort({ text: 'Explain what this function does.' });
  assert.equal(r.reason, 'UNQUALIFIED_PROVIDER'); assert.equal(r.apply, false); assert.equal(r.inferenceCalls, 0);
});

// --- control-layer.effort(): SHADOW records both variants, never applies -------------------------
run('effort SHADOW with context runs BOTH input variants and never applies', async (s, t) => {
  withEffort(s, { mode: 'shadow' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable across the file.', context: 'earlier turn tail text' });
  assert.equal(r.mode, 'shadow'); assert.equal(r.apply, false); assert.equal(r.reason, 'SHADOW');
  const events = readEvents(s.home).filter(e => e.kind === 'effort');
  assert.equal(events.length, 1);
  assert.equal(events[0].variants.length, 2);
  assert.deepEqual(new Set(events[0].variants.map(v => v.variant)), new Set(['prompt', 'prompt+context']));
  assert.equal(JSON.stringify(events[0]).includes('Rename this local variable'), false); // never logs text
});
run('effort SHADOW without context runs a single variant', async (s, t) => {
  withEffort(s, { mode: 'shadow' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.' });
  assert.equal(r.reason, 'SHADOW');
  const events = readEvents(s.home).filter(e => e.kind === 'effort');
  assert.equal(events[0].variants.length, 1);
});

// --- control-layer.effort(): main-loop cache-cost gate -------------------------------------------
run('main loop ON with mainLoop=off never applies even on a cold turn', async (s, t) => {
  withEffort(s, { mode: 'on', mainLoop: 'off' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.', loop: 'main', sinceLastMainMs: null });
  assert.equal(r.direction, 'lower'); assert.equal(r.apply, false); assert.equal(r.reason, 'MAIN_LOOP_DISABLED');
});
run('main loop ON with mainLoop=cold-only applies when there was no earlier main-loop request', async (s, t) => {
  withEffort(s, { mode: 'on', mainLoop: 'cold-only' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.', loop: 'main', sinceLastMainMs: null });
  assert.equal(r.apply, true); assert.equal(r.level, FEATURE_DEFAULTS.effort.lowerTo); assert.equal(r.arm, 'treatment');
});
run('main loop ON with mainLoop=cold-only refuses on a warm turn (CACHE_WARM)', async (s, t) => {
  withEffort(s, { mode: 'on', mainLoop: 'cold-only', coldAfterSeconds: 3600 });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.', loop: 'main', sinceLastMainMs: 5000 });
  assert.equal(r.direction, 'lower'); assert.equal(r.apply, false); assert.equal(r.reason, 'CACHE_WARM');
});
run('main loop ON with mainLoop=cold-only applies once the previous request is old enough', async (s, t) => {
  withEffort(s, { mode: 'on', mainLoop: 'cold-only', coldAfterSeconds: 300 });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.', loop: 'main', sinceLastMainMs: 301000 });
  assert.equal(r.apply, true);
});
run('SHADOW still records the main-loop recommendation and cache-cold flag regardless of mainLoop policy', async (s, t) => {
  withEffort(s, { mode: 'shadow', mainLoop: 'off' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Rename this local variable.', loop: 'main', sinceLastMainMs: 10 });
  assert.equal(r.reason, 'SHADOW'); assert.equal(r.direction, 'lower'); assert.equal(r.cacheCold, false);
  const events = readEvents(s.home).filter(e => e.kind === 'effort');
  assert.equal(events[0].cacheCold, false); assert.equal(events[0].loop, 'main');
});

// --- control-layer.effort(): subagent opt-in, always from a fresh context ------------------------
run('subagent loop is refused by default (subagents=off)', async (s, t) => {
  withEffort(s, { mode: 'on' });
  const { layer, engine } = setupLaya(s, HARD_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Investigate why the integration test intermittently fails.', loop: 'subagent' });
  assert.equal(r.direction, 'raise'); assert.equal(r.apply, false); assert.equal(r.reason, 'SUBAGENTS_DISABLED');
});
run('subagent loop applies once opted in, independent of any cache-cold state', async (s, t) => {
  withEffort(s, { mode: 'on', subagents: 'on' });
  const { layer, engine } = setupLaya(s, HARD_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Investigate why the integration test intermittently fails.', loop: 'subagent' });
  assert.equal(r.apply, true); assert.equal(r.level, FEATURE_DEFAULTS.effort.raiseTo);
});
run('effort ab control share holds back an otherwise-applied ON-mode change but still logs it', async (s, t) => {
  withEffort(s, { mode: 'on', subagents: 'on', abControlShare: 0.5 });
  const { layer, engine } = setupLaya(s, HARD_LOW_RISK); t.after(() => engine.close());
  const r = await layer.effort({ text: 'Investigate why the integration test intermittently fails.', loop: 'subagent' });
  assert.equal(r.direction, 'raise');
  assert.ok(['control', 'treatment'].includes(r.arm));
  assert.equal(r.apply, r.arm === 'treatment');
});

// --- hooks.mjs: turn-effort / turn-outcome --------------------------------------------------------
test('HOOK_EVENTS includes the two Claude-only effort-mod events', () => {
  assert.ok(HOOK_EVENTS.includes('turn-effort')); assert.ok(HOOK_EVENTS.includes('turn-outcome'));
});
run('turn-effort is skipped (BARE_COMMAND) before any inference for a bare slash command', async (s, t) => {
  withEffort(s, { mode: 'on' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await processHookEvent({ host: 'claude', event: 'turn-effort', input: { text: '/compact' }, home: s.home, env: s.env, layer });
  assert.equal(r.output.reason, 'BARE_COMMAND'); assert.equal(r.output.apply, false); assert.equal(r.output.decision_id, null);
});
run('turn-effort produces no output when effort mode is OFF, even if router is ON', async (s, t) => {
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close()); // effort stays default OFF
  setFeatureMode(s.home, 'router', 'on');
  const r = await processHookEvent({ host: 'claude', event: 'turn-effort', input: { text: 'Rename this local variable.' }, home: s.home, env: s.env, layer });
  assert.equal(r.output, null);
  assert.deepEqual(readEvents(s.home), []);
});
run('router OFF does not disable turn-effort/turn-outcome events', async (s, t) => {
  withEffort(s, { mode: 'shadow' });
  setFeatureMode(s.home, 'router', 'off'); // router explicitly off
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await processHookEvent({ host: 'claude', event: 'turn-effort', input: { text: 'Rename this local variable.' }, home: s.home, env: s.env, layer });
  assert.equal(r.output.mode, 'shadow'); assert.equal(r.output.apply, false);
});
run('turn-effort ON produces a decision that turn-outcome reads back (arm + level) and logs content-free numbers', async (s, t) => {
  withEffort(s, { mode: 'on', mainLoop: 'cold-only' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const effortResult = await processHookEvent({ host: 'claude', event: 'turn-effort',
    input: { text: 'Rename this local variable.', loop: 'main', since_last_main_ms: null }, home: s.home, env: s.env, layer });
  assert.equal(effortResult.output.apply, true);
  const decisionId = effortResult.output.decision_id;
  const outcomeResult = await processHookEvent({ host: 'claude', event: 'turn-outcome',
    input: { decision_id: decisionId, turn_id: 't1', loop: 'main', steps: 3, duration_ms: 12345, effort_used: effortResult.output.level,
      stop_reason: 'end_turn', tool_uses: 2, usage: { input: 1000, output: 200, cache_creation: 50, cache_read: 900 } },
    home: s.home, env: s.env, layer });
  assert.equal(outcomeResult.output, null);
  const outcomeEvents = readEvents(s.home).filter(e => e.kind === 'hook' && e.reason === 'TURN_OUTCOME');
  assert.equal(outcomeEvents.length, 1);
  assert.equal(outcomeEvents[0].decision_id, decisionId);
  assert.equal(outcomeEvents[0].arm, effortResult.output.arm);
  assert.equal(outcomeEvents[0].loop, 'main');
  assert.equal(outcomeEvents[0].usage.input, 1000);
  assert.equal(JSON.stringify(outcomeEvents[0]).includes('Rename this local variable'), false);
});
run('turn-outcome with an unknown/absent decision_id logs arm "none" rather than failing', async (s, t) => {
  withEffort(s, { mode: 'shadow' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await processHookEvent({ host: 'claude', event: 'turn-outcome',
    input: { turn_id: 't2', steps: 1, duration_ms: 10, tool_uses: 0, usage: { input: 1, output: 1, cache_creation: 0, cache_read: 0 } },
    home: s.home, env: s.env, layer });
  assert.equal(r.output, null);
  const events = readEvents(s.home).filter(e => e.reason === 'TURN_OUTCOME');
  assert.equal(events[0].arm, 'none'); assert.equal(events[0].decision_id, null);
});
run('turn-outcome logs nothing when effort mode is OFF', async (s, t) => {
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await processHookEvent({ host: 'claude', event: 'turn-outcome',
    input: { turn_id: 't3', steps: 1, duration_ms: 10, tool_uses: 0, usage: { input: 1, output: 1, cache_creation: 0, cache_read: 0 } },
    home: s.home, env: s.env, layer });
  assert.equal(r.output, null);
  assert.deepEqual(readEvents(s.home), []);
});
run('turn-outcome rejects malformed usage/steps rather than logging garbage', async (s, t) => {
  withEffort(s, { mode: 'shadow' });
  const { layer, engine } = setupLaya(s, EASY_LOW_RISK); t.after(() => engine.close());
  const r = await processHookEvent({ host: 'claude', event: 'turn-outcome',
    input: { turn_id: 't4', steps: -1, duration_ms: 10, tool_uses: 0, usage: { input: 1, output: 1, cache_creation: 0, cache_read: 0 } },
    home: s.home, env: s.env, layer });
  assert.equal(r.output, null); assert.equal(r.telemetry.reason, 'INVALID_HOOK_INPUT');
  assert.deepEqual(readEvents(s.home).filter(e => e.reason === 'TURN_OUTCOME'), []);
});
