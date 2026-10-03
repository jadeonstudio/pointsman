import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawnSync, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { atomicWrite, readText } from '../src/storage.mjs';
import { ControlError } from '../src/constants.mjs';
import { digest } from '../src/training/schema.mjs';
import { buildDataset } from '../src/training/dataset.mjs';
import { fixture as trainingFixture, trace, outcome, REF } from './training-helpers.mjs';
import { ROUTE_QUESTIONS } from '../src/routing.mjs';
import { registerCheckpoint, activateCandidate, freezeHoldout, listHoldouts, qualifyCandidate, compareCandidate,
  promoteCandidate, rollbackLaya, layaStatus, loadQualification } from '../src/training/laya-lifecycle.mjs';

// --- synthetic checkpoint fixture -------------------------------------------------
function makeCheckpointDir(root, tag = 'weights') {
  const dir = fs.mkdtempSync(path.join(root, 'ckpt-'));
  // `tag` varies the content (never just the tmp path, which is not part of the fingerprint input)
  // so callers that need two DISTINCT registered checkpoints in the same home get distinct hashes.
  fs.writeFileSync(path.join(dir, 'model.safetensors'), tag);
  fs.writeFileSync(path.join(dir, 'rl_agent_config.json'), '{}');
  fs.mkdirSync(path.join(dir, 'encoder'));
  fs.writeFileSync(path.join(dir, 'encoder', 'config.json'), '{}');
  fs.mkdirSync(path.join(dir, 'tokenizer'));
  fs.writeFileSync(path.join(dir, 'tokenizer', 'tokenizer_config.json'), '{}');
  return dir;
}
// Deterministic stand-in for the offline `python -I workers/laya_worker.py --fingerprint`
// hashing step: hash the sorted file list, same shape contract (64-hex sha256), no python needed.
function fakeFingerprint(_python, dir) {
  const records = [];
  const walk = d => { for (const name of fs.readdirSync(d).sort()) { const p = path.join(d, name); const st = fs.statSync(p); if (st.isDirectory()) walk(p); else records.push([path.relative(dir, p), fs.readFileSync(p, 'utf8')]); } };
  walk(dir);
  return digest(records);
}

// --- synthetic dataset fixture ----------------------------------------------------
// Splits are computed from digest('input:'+request_hash) for an isolated single-sample group,
// mirroring dataset.mjs groupedSplits(); reproduces the training-helpers.mjs splitFor() rule.
const splitFor = requestHash => { const n = parseInt(digest('input:' + requestHash).slice(0, 8), 16) % 100; return n < 80 ? 'train' : n < 90 ? 'calibration' : 'test'; };
function buildRequest(i, want, conf) {
  return { purpose: 'route', risk: 'routine', state: { task: `synthetic laya case ${i}`, want, conf },
    questions: { worker: { type: 'choice', instructions: 'Choose the worker for this bounded task.', criteria: { light: 'Narrow known scope', strong: 'Unclear or broad scope' } } } };
}
function findIndex(target, want, conf, cursor) {
  for (let i = cursor.n; i < cursor.n + 20000; i++) if (splitFor(digest(buildRequest(i, want, conf))) === target) { cursor.n = i + 1; return i; }
  throw new Error('NO_INDEX_FOUND');
}
// Builds a dataset with `count` (want, conf, truth) samples placed in `split`.
function addSamples(f, cursor, split, rows) {
  for (const { want, conf, truth } of rows) {
    const i = findIndex(split, want, conf, cursor);
    const r = buildRequest(i, want, conf);
    const d = { decision_id: randomUUID(), trace: trace(), arm: 'active', request: r, request_hash: digest(r),
      provenance: { provider: 'jev', model: 'jev-1.13.0', model_version: 'jev-1.13.0', checkpoint: 'jev-1.13.0',
        runtime_version: 'systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' },
      answers: { worker: { type: 'choice', value: want, confidence: .9, selectedProbability: .9, probabilities: { light: want === 'light' ? .9 : .1, strong: want === 'strong' ? .9 : .1 } } },
      mode: 'on', apply: true, latency_ms: 10, usage: { inputTokens: 10, outputTokens: 0 },
      inference_calls: 1, network_calls: 1, capture_policy_version: 'minimal-state-v1' };
    f.save(d);
    f.store.outcome(outcome(d, { labels: [{ question_id: 'worker', value: truth, source: 'objective', label_confidence: .95, evidence_ref: REF }] }));
  }
}
// 5 correct@.95 + 5 incorrect@.55 in calibration: t=0.50 gives accuracy .5 (fails 0.75 target),
// t=0.56 excludes the .55 batch giving accuracy 1.0 (passes) -- exercises real grid search, not the t=0.50 edge.
function calibrationRows() {
  const rows = [];
  for (let k = 0; k < 5; k++) rows.push({ want: 'light', conf: .95, truth: 'light' });
  for (let k = 0; k < 5; k++) rows.push({ want: 'light', conf: .55, truth: 'strong' });
  return rows;
}
// 6 confident-correct + 2 low-confidence (excluded from coverage either way) per split.
function evalRows(strongAccuracy = true) {
  const rows = [];
  for (let k = 0; k < 6; k++) rows.push({ want: 'light', conf: .95, truth: strongAccuracy ? 'light' : 'strong' });
  for (let k = 0; k < 2; k++) rows.push({ want: 'light', conf: .3, truth: 'strong' });
  return rows;
}
function buildRouteDataset(f) {
  const cursor = { n: 0 };
  addSamples(f, cursor, 'calibration', calibrationRows());
  addSamples(f, cursor, 'test', evalRows(true));
  addSamples(f, cursor, 'test', [{ want: 'strong', conf: .99, truth: 'strong' }]); // filler for other purposes/type mix, ignored
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.built, undefined, JSON.stringify(built));
  return built.dataset_version;
}

// --- full three-question route dataset (decision-level gate fixtures) --------------
// Unlike buildRouteDataset() above (a single synthetic 'worker' choice question used to exercise
// the legacy per-answer threshold), these fixtures record real ROUTE_QUESTIONS (intent/difficulty/risk)
// decisions so the decision-level route gate (labelTier/tierDistribution/chooseRouteByDecision) has
// something to group and grade. `pred` travels in state purely so the fake worker below can look up
// what to answer for each of the three per-question requests this one decision produces.
function decisionRequest(i, pred, tag) {
  return { purpose: 'route', risk: 'routine', state: { task: `decision route case ${tag} ${i}`, pred }, questions: structuredClone(ROUTE_QUESTIONS) };
}
function findDecisionIndex(target, pred, tag, cursor) {
  for (let i = cursor.n; i < cursor.n + 20000; i++) if (splitFor(digest(decisionRequest(i, pred, tag))) === target) { cursor.n = i + 1; return i; }
  throw new Error('NO_INDEX_FOUND');
}
function oneHotChoice(criteria, value) { return Object.fromEntries(Object.keys(criteria).map(k => [k, k === value ? 1 : 0])); }
function oneHotScore(index) { return Object.fromEntries([0, 1, 2, 3, 4].map(i => [String(i), i === index ? 1 : 0])); }
// Adds `count` decisions whose TARGET labels are (targetIntent, targetDifficulty, targetRisk) and whose
// worker PREDICTION (used only at qualify time by fakeDecisionLayaClient) is `pred`. The recorded
// decision.answers themselves are a one-hot match of the target -- only the outcome's objective labels
// actually determine each dataset sample's target (see evaluate.mjs), so this is just satisfying
// validateDecision/validatePrediction, not part of what qualify grades.
function addDecisionTasks(f, cursor, split, count, { pred, targetIntent, targetDifficulty, targetRisk, tag }) {
  for (let k = 0; k < count; k++) {
    const i = findDecisionIndex(split, pred, tag, cursor);
    const r = decisionRequest(i, pred, tag);
    const answers = {
      intent: { type: 'choice', value: targetIntent, confidence: .9, selectedProbability: 1, probabilities: oneHotChoice(ROUTE_QUESTIONS.intent.criteria, targetIntent) },
      difficulty: { type: 'score', value: targetDifficulty, confidence: .9, probabilities: oneHotScore(targetDifficulty) },
      risk: { type: 'choice', value: targetRisk, confidence: .9, selectedProbability: 1, probabilities: oneHotChoice(ROUTE_QUESTIONS.risk.criteria, targetRisk) },
    };
    const d = { decision_id: randomUUID(), trace: trace(), arm: 'active', request: r, request_hash: digest(r),
      provenance: { provider: 'jev', model: 'jev-1.13.0', model_version: 'jev-1.13.0', checkpoint: 'jev-1.13.0',
        runtime_version: 'systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' },
      answers, mode: 'on', apply: true, latency_ms: 10, usage: { inputTokens: 10, outputTokens: 0 },
      inference_calls: 1, network_calls: 1, capture_policy_version: 'minimal-state-v1' };
    f.save(d);
    f.store.outcome(outcome(d, {
      checks: [{ kind: 'tests', required: true, passed: true, scope: 'task', evidence_ref: REF },
        ...['intent', 'difficulty', 'risk'].map(question_id => ({ kind: 'label', required: true, passed: true, scope: 'task', evidence_ref: REF, question_id }))],
      labels: [{ question_id: 'intent', value: targetIntent, source: 'objective', label_confidence: .95, evidence_ref: REF },
        { question_id: 'difficulty', value: targetDifficulty, source: 'objective', label_confidence: .95, evidence_ref: REF },
        { question_id: 'risk', value: targetRisk, source: 'objective', label_confidence: .95, evidence_ref: REF }],
    }));
  }
}
// A one-hot 'economy' prediction (edit / difficulty 0 / safe): labelTier('edit', 0, 'safe', <default
// router policy>) is 'economy' (displayed difficulty 1 <= economyMaxDifficulty 2).
const ECONOMY_PRED = { intent: { choice: 'edit', confidence: .99, probabilities: oneHotChoice(ROUTE_QUESTIONS.intent.criteria, 'edit') },
  difficulty: { score: 0, confidence: .99, probabilities: oneHotScore(0) },
  risk: { choice: 'safe', confidence: .99, probabilities: oneHotChoice(ROUTE_QUESTIONS.risk.criteria, 'safe') } };
function fakeDecisionLayaClient() {
  return { calls: () => 0, status: () => ({ running: false }), close: () => {}, prepare: async () => ({}),
    async infer(payload, settings) {
      const laya = settings.laya;
      const [qid] = Object.keys(payload.questions);
      const p = payload.state.pred[qid];
      const answer = qid === 'difficulty' ? { type: 'score', score: p.score, confidence: p.confidence, probabilities: p.probabilities }
        : { type: 'choice', choice: p.choice, confidence: p.confidence, probabilities: p.probabilities };
      return { identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device,
        precision: laya.precision === 'fp16' ? 'torch.float16' : 'torch.float32' }, answers: { [qid]: answer }, usage: { input_tokens: 5, output_tokens: 0 } };
    } };
}
// Builds a dataset with N calibration tasks and N test-split tasks, all with true label 'economy'
// (target edit/0/safe) and a confidently and correctly ECONOMY_PRED worker prediction. freezeHoldout
// (below) freezes whatever lands in the 'test' split as the holdout, so both splits end up identical.
// n=30 keeps the Wilson upper bound on 0 observed unsafe decisions under the default routeMaxError
// (0.15): with a small n, even zero errors carries a wide upper bound (e.g. n=8 gives ~0.32).
function buildDecisionRouteDataset(f, n = 30) {
  const cursor = { n: 0 };
  addDecisionTasks(f, cursor, 'calibration', n, { pred: ECONOMY_PRED, targetIntent: 'edit', targetDifficulty: 0, targetRisk: 'safe', tag: 'good-calib' });
  addDecisionTasks(f, cursor, 'test', n, { pred: ECONOMY_PRED, targetIntent: 'edit', targetDifficulty: 0, targetRisk: 'safe', tag: 'good-test' });
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.built, undefined, JSON.stringify(built));
  return built.dataset_version;
}

// --- fake Laya worker (deterministic; no python/network/training) -----------------
function fakeLayaClient() {
  let calls = 0;
  return {
    calls: () => calls,
    status: () => ({ running: false }), close: () => {}, prepare: async () => ({}),
    async infer(payload, settings) {
      calls++;
      const laya = settings.laya;
      const [qid, q] = Object.entries(payload.questions)[0];
      const want = payload.state.want, conf = payload.state.conf;
      const other = Object.keys(q.criteria).find(k => k !== want);
      // `confidence` and the choice distribution are independent fields in the real contract
      // (contracts.mjs normalizeResponse): pin the distribution's max share on the picked choice
      // (a valid distribution) and drive the calibration grid purely through `confidence`.
      return {
        identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device,
          precision: laya.precision === 'fp16' ? 'torch.float16' : 'torch.float32' },
        answers: { [qid]: { type: 'choice', choice: want, confidence: conf, probabilities: { [want]: .99, [other]: .01 } } },
        usage: { input_tokens: 5, output_tokens: 0 },
      };
    },
  };
}

function holdoutFixture(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
function writeProviders(home, config) { atomicWrite(path.join(home, 'providers.json'), JSON.stringify(config, null, 2) + '\n'); }

// ============================== register ==============================

test('register copies a checkpoint, fingerprints it, and never touches providers.json', t => {
  const root = holdoutFixture(t);
  const ckpt = makeCheckpointDir(root);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const before = fs.existsSync(path.join(home, 'providers.json')) ? readText(path.join(home, 'providers.json'), { optional: true }) : null;
  const r = registerCheckpoint(home, { checkpointDir: ckpt, model: 'laya/test', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.match(r.checkpoint, /^[0-9a-f]{64}$/);
  assert.equal(r.reused, false);
  assert.ok(fs.existsSync(r.modelPath));
  assert.ok(fs.existsSync(path.join(r.modelPath, 'model.safetensors')));
  const candidate = JSON.parse(fs.readFileSync(r.candidate, 'utf8'));
  assert.equal(candidate.checkpoint, r.checkpoint);
  assert.equal(candidate.model, 'laya/test');
  assert.equal(candidate.device, 'cpu');
  assert.equal(candidate.runtimeVersion, '0.3.4');
  const after = fs.existsSync(path.join(home, 'providers.json')) ? readText(path.join(home, 'providers.json'), { optional: true }) : null;
  assert.equal(after, before);
});

test('register reuses an existing checkpoint copy after verifying the fingerprint matches', t => {
  const root = holdoutFixture(t);
  const ckpt = makeCheckpointDir(root);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const first = registerCheckpoint(home, { checkpointDir: ckpt, model: 'laya/one', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const second = registerCheckpoint(home, { checkpointDir: ckpt, model: 'laya/one', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.equal(second.checkpoint, first.checkpoint);
  assert.equal(second.reused, true);
  assert.equal(second.modelPath, first.modelPath);
});

test('register refuses a symlinked checkpoint directory and requires configured python/device', t => {
  const root = holdoutFixture(t);
  const ckpt = makeCheckpointDir(root);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const link = path.join(root, 'ckpt-link');
  fs.symlinkSync(ckpt, link);
  assert.throws(() => registerCheckpoint(home, { checkpointDir: link, model: 'x', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint }), /UNSAFE_SYMLINK|LAYA_MODEL_SYMLINK_REFUSED/);
  assert.throws(() => registerCheckpoint(home, { checkpointDir: ckpt, model: 'x', device: 'cpu', fingerprintImpl: fakeFingerprint }), /LAYA_PYTHON_NOT_CONFIGURED/);
  assert.throws(() => registerCheckpoint(home, { checkpointDir: ckpt, model: 'x', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint }), /LAYA_DEVICE_REQUIRED/);
});

test('register falls back to the active providers.json python/device when not given explicitly', t => {
  const root = holdoutFixture(t);
  const ckpt = makeCheckpointDir(root);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  writeProviders(home, { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'mps' } });
  const r = registerCheckpoint(home, { checkpointDir: ckpt, model: 'laya/new', fingerprintImpl: fakeFingerprint });
  const candidate = JSON.parse(fs.readFileSync(r.candidate, 'utf8'));
  assert.equal(candidate.python, '/usr/bin/python3');
  assert.equal(candidate.device, 'mps');
});

test('register defaults precision to fp32, accepts an explicit --precision, and falls back to the active config otherwise', t => {
  const root = holdoutFixture(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const noConfig = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/p1', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(noConfig.candidate, 'utf8')).precision, 'fp32');
  const explicit = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/p2', device: 'cpu', python: '/usr/bin/python3', precision: 'fp16', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(explicit.candidate, 'utf8')).precision, 'fp16');
  writeProviders(home, { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu', precision: 'fp16' } });
  const fromActive = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/p3', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(fromActive.candidate, 'utf8')).precision, 'fp16');
  assert.throws(() => registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/p4', device: 'cpu', python: '/usr/bin/python3', precision: 'int8', fingerprintImpl: fakeFingerprint }), /INVALID_PROVIDER_CONFIG/);
});

// ============================== holdout ==============================

test('holdout freeze is immutable and content-hashed; freezing twice under the same name is refused', t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const first = freezeHoldout(f.home, { datasetVersion: version, name: 'shared' });
  assert.equal(first.sample_count, 9); // 8 evalRows() + 1 filler, all placed in the test split
  assert.match(first.sha256, /^[0-9a-f]{64}$/);
  assert.throws(() => freezeHoldout(f.home, { datasetVersion: version, name: 'shared' }), /HOLDOUT_ALREADY_EXISTS/);
  const list = listHoldouts(f.home);
  assert.equal(list.length, 1);
  assert.equal(list[0].name, 'shared');
  assert.equal(list[0].sample_count, first.sample_count);
  assert.equal(Object.keys(list[0]).includes('state') || Object.keys(list[0]).includes('samples'), false);
});

// ============================== qualify ==============================

test('qualify passes a purpose whose calibration threshold clears test+holdout, with no raw state in evidence', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const home = f.home;
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/q', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const layaClient = fakeLayaClient();
  const result = await qualifyCandidate(home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient,
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(result.checkpoint, reg.checkpoint);
  assert.equal(result.qualified, true);
  assert.deepEqual(result.purposes, ['route']);
  assert.equal(result.minConfidence, .56);
  assert.equal(result.minChoiceProbability, .56);
  assert.equal(result.noulCertainty, .56);
  assert.equal(result.evidence.route.test.n, 7); // 6 confident correct + the confident filler, both >= 0.56
  assert.equal(result.evidence.route.test.accuracy, 1);
  assert.equal(result.evidence.route.holdout.n, 7);
  const raw = JSON.stringify(result);
  assert.ok(!raw.includes('synthetic laya case'));
  const stored = loadQualification(home, reg.checkpoint);
  assert.deepEqual(stored, result);
});

test('qualify hands the worker client normalized candidate settings (startup/idle timeouts filled in)', async t => {
  // 2026-09-23 real run: a registered candidate file carries no startupTimeoutMs, and the validator only
  // normalized a clone, so the real worker client got setTimeout(undefined) and failed at once with
  // LAYA_STARTUP_TIMEOUT. The fake client never looked at the timeouts, so no test caught it.
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/qt', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const inner = fakeLayaClient();
  const seen = [];
  const layaClient = { ...inner, async infer(payload, settings, opts) { seen.push(settings.laya); return inner.infer(payload, settings, opts); } };
  await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient,
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.ok(seen.length > 0);
  for (const l of seen) {
    assert.equal(l.startupTimeoutMs, 120000);
    assert.equal(l.idleTimeoutMs, 60000);
    assert.equal(l.inputFit, 'lossless');
  }
});

test('qualify fails a purpose when calibration sample count is below the minimum', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/q2', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const result = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 999, minTest: 5, minLowerBound: .5 });
  assert.equal(result.qualified, false);
  assert.deepEqual(result.purposes, []);
  assert.equal(result.evidence.route.calibration.threshold, null);
});

test('qualify fails when the worker predicts wrong: accuracy never clears the target', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/bad', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const alwaysWrong = { ...fakeLayaClient(), async infer(payload, settings) {
    const laya = settings.laya; const [qid, q] = Object.entries(payload.questions)[0];
    const wrong = Object.keys(q.criteria).find(k => k !== payload.state.want);
    return { identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device, precision: 'torch.float32' },
      answers: { [qid]: { type: 'choice', choice: wrong, confidence: .95, probabilities: { [wrong]: .95, [payload.state.want]: .05 } } }, usage: { input_tokens: 1, output_tokens: 0 } };
  } };
  const result = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: alwaysWrong,
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(result.qualified, false);
});

// ============================== score-question scoring (real 2026-09-23 bug) ==============================
// A trained candidate's normalized `score` answer is the continuous EXPECTED level (a probability-weighted
// average, e.g. 2.0257), never an integer. The target is a 0-based integer index. Comparing them with
// `===` means a score question can never be counted correct. Difficulty agreement must instead compare
// argmax(answer.probabilities) to the target index.
const SCORE_LEVELS = 3;
function scoreProbabilities(want, conf) {
  const other = (1 - conf) / (SCORE_LEVELS - 1);
  return Object.fromEntries(Array.from({ length: SCORE_LEVELS }, (_, i) => [String(i), i === want ? conf : other]));
}
function buildScoreRequest(i, want, conf) {
  return { purpose: 'judge', risk: 'routine', state: { task: `synthetic score case ${i}`, want, conf },
    questions: { severity: { type: 'score', instructions: 'Rate the severity of this issue.', criteria: ['low', 'medium', 'high'] } } };
}
function findScoreIndex(target, want, conf, cursor) {
  for (let i = cursor.n; i < cursor.n + 20000; i++) if (splitFor(digest(buildScoreRequest(i, want, conf))) === target) { cursor.n = i + 1; return i; }
  throw new Error('NO_INDEX_FOUND');
}
function addScoreSamples(f, cursor, split, rows) {
  for (const { want, conf, truth } of rows) {
    const i = findScoreIndex(split, want, conf, cursor);
    const r = buildScoreRequest(i, want, conf);
    const oneHot = Object.fromEntries(Array.from({ length: SCORE_LEVELS }, (_, k) => [String(k), k === want ? 1 : 0]));
    const d = { decision_id: randomUUID(), trace: trace(), arm: 'active', request: r, request_hash: digest(r),
      provenance: { provider: 'jev', model: 'jev-1.13.0', model_version: 'jev-1.13.0', checkpoint: 'jev-1.13.0',
        runtime_version: 'systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' },
      answers: { severity: { type: 'score', value: want, confidence: .9, probabilities: oneHot } },
      mode: 'on', apply: true, latency_ms: 10, usage: { inputTokens: 10, outputTokens: 0 },
      inference_calls: 1, network_calls: 1, capture_policy_version: 'minimal-state-v1' };
    f.save(d);
    f.store.outcome(outcome(d, {
      checks: [{ kind: 'tests', required: true, passed: true, scope: 'task', evidence_ref: REF },
        { kind: 'label', required: true, passed: true, scope: 'task', evidence_ref: REF, question_id: 'severity' }],
      labels: [{ question_id: 'severity', value: truth, source: 'objective', label_confidence: .95, evidence_ref: REF }] }));
  }
}
// 5 correct-high-confidence (want===truth) + 5 incorrect-low-confidence, exercising the real grid search.
function scoreCalibrationRows() {
  const rows = [];
  for (let k = 0; k < 5; k++) rows.push({ want: 2, conf: .95, truth: 2 });
  for (let k = 0; k < 5; k++) rows.push({ want: 2, conf: .55, truth: 1 });
  return rows;
}
function scoreEvalRows() {
  const rows = [];
  for (let k = 0; k < 6; k++) rows.push({ want: 2, conf: .95, truth: 2 });
  for (let k = 0; k < 2; k++) rows.push({ want: 2, conf: .3, truth: 1 });
  return rows;
}
function buildScoreDataset(f) {
  const cursor = { n: 0 };
  addScoreSamples(f, cursor, 'calibration', scoreCalibrationRows());
  addScoreSamples(f, cursor, 'test', scoreEvalRows());
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.built, undefined, JSON.stringify(built));
  return built.dataset_version;
}
function fakeScoreLayaClient() {
  return {
    calls: () => 0, status: () => ({ running: false }), close: () => {}, prepare: async () => ({}),
    async infer(payload, settings) {
      const laya = settings.laya;
      const [qid, q] = Object.entries(payload.questions)[0];
      const want = payload.state.want, conf = payload.state.conf;
      const probabilities = scoreProbabilities(want, conf);
      const labels = q.criteria.map((_, idx) => idx);
      // The real worker's `score` is the probability-weighted expected value -- continuous, and never
      // exactly equal to the target index unless probabilities happen to be exactly one-hot.
      const expected = labels.reduce((s, k) => s + k * probabilities[String(k)], 0);
      return {
        identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device,
          precision: laya.precision === 'fp16' ? 'torch.float16' : 'torch.float32' },
        answers: { [qid]: { type: 'score', score: expected, confidence: conf, probabilities } },
        usage: { input_tokens: 5, output_tokens: 0 },
      };
    },
  };
}

test('qualify credits score questions by argmax(probabilities) vs the target index, not raw continuous value equality', async t => {
  const f = trainingFixture(t);
  const version = buildScoreDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'hs' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/score', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const result = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeScoreLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(result.qualified, true, JSON.stringify(result));
  assert.deepEqual(result.purposes, ['judge']);
  assert.equal(result.evidence.judge.test.accuracy, 1);
  assert.equal(result.evidence.judge.holdout.accuracy, 1);
  // Content-free per-question raw stats (bug 4): n, raw agreement, refused — reported per split, never
  // pooled: calibration also picks the epoch/threshold (optimistic) and the holdout freezes the test split,
  // so a pooled figure would double-count test and mix in tuned-on data.
  assert.ok(result.by_question, 'qualify result must report per-question raw stats');
  assert.deepEqual(Object.keys(result.by_question), ['calibration', 'test', 'holdout']);
  assert.equal(result.by_question.calibration.severity.n, 10);
  assert.equal(result.by_question.test.severity.n, 8);
  assert.equal(result.by_question.holdout.severity.n, 8);
  assert.equal(result.by_question.test.severity.refused, 0);
  assert.ok(result.by_question.test.severity.raw_agreement > 0 && result.by_question.test.severity.raw_agreement <= 1);
});

// Owner decision (2026-09-27): an operator may keep the acceptance threshold at or above a floor while
// lowering targetAccuracy, so a lower target never silently lowers the confidence bar the model earned.
test('qualify --min-threshold floors the calibrated threshold and is recorded in params', async t => {
  const f = trainingFixture(t);
  const version = buildScoreDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'hmin' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/score', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const base = { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeScoreLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 };
  const plain = await qualifyCandidate(f.home, base);
  const floored = await qualifyCandidate(f.home, { ...base, layaClient: fakeScoreLayaClient(), minThreshold: 0.9 });
  assert.equal(plain.params.minThreshold, 0.5);
  assert.equal(floored.params.minThreshold, 0.9);
  const t0 = plain.evidence.judge.calibration.threshold, t1 = floored.evidence.judge.calibration.threshold;
  assert.ok(t0 !== null && t0 < 0.9, `fixture should calibrate below the floor, got ${t0}`);
  assert.ok(t1 === null || t1 >= 0.9, `floored threshold ${t1} must be >= 0.9`);
  await assert.rejects(qualifyCandidate(f.home, { ...base, minThreshold: 0.4 }), /INVALID_QUALIFY_PARAMS/);
  await assert.rejects(qualifyCandidate(f.home, { ...base, minThreshold: 1 }), /INVALID_QUALIFY_PARAMS/);
});

// ============================== decision-level route gate ==============================

test('qualify fits a decision-level route gate from synthetic records and reports it in evidence.route.decision', async t => {
  const f = trainingFixture(t);
  const version = buildDecisionRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'dh1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/decision', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const result = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeDecisionLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  const decision = result.evidence.route.decision;
  assert.ok(decision, 'evidence.route.decision must be present for purpose route');
  assert.notEqual(decision.gate, null, JSON.stringify(decision));
  // Every prediction is a confident, correct one-hot economy answer, so every combination in the grid
  // gives 0 unsafe decisions; the deterministic tie-break (largest tierCoverage, then smallest
  // maxHostProbability) must pick the first grid point, 0.99 / 0.01.
  assert.equal(decision.gate.tierCoverage, 0.99);
  assert.equal(decision.gate.maxHostProbability, 0.01);
  assert.equal(decision.calibration.tasks, 30);
  assert.equal(decision.calibration.applied, 30);
  assert.equal(decision.calibration.economy, 30);
  assert.equal(decision.calibration.unsafe, 0);
  assert.equal(decision.test.tasks, 30);
  assert.equal(decision.test.applied, 30);
  assert.equal(decision.holdout.tasks, 30);
  assert.equal(decision.holdout.applied, 30);
  assert.equal(decision.passes, true);
  assert.deepEqual(result.routeGate, { method: 'decision-v1', tierCoverage: 0.99, maxHostProbability: 0.01 });
  assert.ok(result.purposes.includes('route'));
  assert.equal(result.qualified, true);
  const raw = JSON.stringify(result);
  assert.ok(!raw.includes('decision route case'));
});

test('qualify: a failing test split leaves routeGate unset even when calibration alone would fit one', async t => {
  const f = trainingFixture(t);
  const cursor = { n: 0 };
  addDecisionTasks(f, cursor, 'calibration', 30, { pred: ECONOMY_PRED, targetIntent: 'edit', targetDifficulty: 0, targetRisk: 'safe', tag: 'good-calib' });
  // Test split: worker confidently predicts economy (edit/0/safe) but the true label is host
  // (intent 'operate'), so every decided task on this split is unsafe.
  addDecisionTasks(f, cursor, 'test', 8, { pred: ECONOMY_PRED, targetIntent: 'operate', targetDifficulty: 0, targetRisk: 'safe', tag: 'bad-test' });
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.built, undefined, JSON.stringify(built));
  const version = built.dataset_version;
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'dh2' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/decision-fail', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const result = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeDecisionLayaClient(),
    targetAccuracy: .5, minCoverage: .01, minCalibration: 1, minTest: 1, minLowerBound: 0 });
  const decision = result.evidence.route.decision;
  assert.notEqual(decision.gate, null, JSON.stringify(decision)); // calibration alone still fits (it is all-correct)
  assert.equal(decision.test.applied, 8);
  assert.equal(decision.test.unsafe, 8);
  assert.equal(decision.passes, false);
  assert.equal(result.routeGate, undefined);
});

test('qualify route-gate params validate: fractions in (0,1], routeMaxErrorUpper >= routeMaxError, routeMinApplied a positive integer', async t => {
  const f = trainingFixture(t);
  const version = buildDecisionRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'dh3' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/decision-params', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const base = { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeDecisionLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 };
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMaxError: 0 }), /INVALID_QUALIFY_PARAMS/);
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMaxError: 1.5 }), /INVALID_QUALIFY_PARAMS/);
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMaxErrorUpper: 0 }), /INVALID_QUALIFY_PARAMS/);
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMaxError: 0.2, routeMaxErrorUpper: 0.1 }), /INVALID_QUALIFY_PARAMS/); // upper below maxError
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMinApplied: 0 }), /INVALID_QUALIFY_PARAMS/);
  await assert.rejects(qualifyCandidate(f.home, { ...base, routeMinApplied: 1.5 }), /INVALID_QUALIFY_PARAMS/);
  const ok = await qualifyCandidate(f.home, { ...base, routeMaxError: 0.2, routeMaxErrorUpper: 0.2, routeMinApplied: 1 });
  assert.deepEqual(ok.params.routeMaxError, 0.2);
  assert.deepEqual(ok.params.routeMaxErrorUpper, 0.2);
  assert.deepEqual(ok.params.routeMinApplied, 1);
});

// Recovery cases use already-issued legacy evidence. New purpose-only artifacts
// cannot promote; decision-family.test exercises the new operational guard.
function issuedLegacyQualification(home, candidateHash, qualification) {
  const legacy = structuredClone(qualification);
  delete legacy.decisionIdentity; delete legacy.familyQualification; delete legacy.operationalEligible;
  atomicWrite(path.join(home, 'laya', 'qualifications', `${candidateHash}.json`), JSON.stringify(legacy));
}
test('promote copies routeGate into the promoted providers.json qualification when present', async t => {
  const f = trainingFixture(t);
  const version = buildDecisionRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'dh4' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/decision-promote', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qual = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeDecisionLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.ok(qual.routeGate, JSON.stringify(qual));
  issuedLegacyQualification(f.home, reg.checkpoint, qual);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeDecisionLayaClient(), noActiveBaseline: true });
  const promoted = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  assert.equal(promoted.promoted, true, JSON.stringify(promoted));
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.deepEqual(providers.laya.qualification.routeGate, qual.routeGate);
});

// ============================== compare + promote ==============================

async function qualifiedCandidate(f, version, holdoutName, model = 'laya/promote-me') {
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model, device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qual = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName, layaClient: fakeLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(qual.qualified, true, JSON.stringify(qual));
  issuedLegacyQualification(f.home, reg.checkpoint, qual);
  return { reg, qual };
}

test('qualify records the candidate precision, and promote carries it into providers.json laya + qualification', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))),
    model: 'laya/fp16', device: 'cpu', python: '/usr/bin/python3', precision: 'fp16', fingerprintImpl: fakeFingerprint });
  const qual = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(qual.qualified, true, JSON.stringify(qual));
  assert.equal(qual.precision, 'fp16');
  issuedLegacyQualification(f.home, reg.checkpoint, qual);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  const promoted = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  assert.equal(promoted.promoted, true, JSON.stringify(promoted));
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.equal(providers.laya.precision, 'fp16');
  assert.equal(providers.laya.qualification.precision, 'fp16');
});

test('compare + promote (no active baseline) swaps providers.json laya, keeps provider selection, and logs history', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg, qual } = await qualifiedCandidate(f, version, holdout.name);
  const cmp = await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  assert.equal(cmp.active, null);
  assert.equal(cmp.candidate, reg.checkpoint);
  assert.equal(cmp.purposes.route.candidate.selective.accuracy, 1);
  assert.ok(cmp.purposes.route.candidate.raw.accuracy < 1); // low-confidence answers are included in raw
  const promoted = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  assert.equal(promoted.promoted, true);
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.equal(providers.provider, 'jev'); // provider selection is never changed by promote
  assert.equal(providers.laya.checkpoint, reg.checkpoint);
  assert.equal(providers.laya.qualification.checkpoint, reg.checkpoint);
  assert.deepEqual(providers.laya.qualification.purposes, qual.purposes);
  const history = fs.readFileSync(path.join(f.home, 'laya', 'history.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(history.length, 1);
  assert.equal(history[0].action, 'promote');
  assert.equal(history[0].before, null);
});

// (2026-09-23 real bug) `--no-active-baseline` must skip inferring the active checkpoint entirely when
// one is configured, not just skip the ACTIVE_BASELINE_REQUIRED guard.
test('compare --no-active-baseline skips inferring the active checkpoint even when one is configured', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const activeReg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-'))), 'active-weights'), model: 'laya/active', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  writeProviders(f.home, { version: 1, provider: 'laya', laya: JSON.parse(fs.readFileSync(activeReg.candidate, 'utf8')) });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  const seenCheckpoints = [];
  const inner = fakeLayaClient();
  const layaClient = { ...inner, async infer(payload, settings, opts) { seenCheckpoints.push(settings.laya.checkpoint); return inner.infer(payload, settings, opts); } };
  const cmp = await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient, noActiveBaseline: true });
  assert.equal(cmp.active, null);
  assert.ok(!('active' in cmp.purposes.route), 'no active side must be recorded when --no-active-baseline is set');
  assert.ok(!seenCheckpoints.includes(activeReg.checkpoint), 'the active checkpoint must never be inferred when --no-active-baseline is set');
});

// (2026-09-23 real bug) the active baseline refusing one sample (e.g. INPUT_TRUNCATED from the official
// tokenizer admission check) must not abort the whole compare; it must be recorded as `refused` and the
// rest of the holdout still evaluated. Also covers bug 4: content-free per-question raw stats.
test('compare records a per-sample INPUT_TRUNCATED refusal instead of aborting, and reports refused counts', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const activeReg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-'))), 'active-weights'), model: 'laya/active', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  writeProviders(f.home, { version: 1, provider: 'laya', laya: JSON.parse(fs.readFileSync(activeReg.candidate, 'utf8')) });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  const inner = fakeLayaClient();
  const counters = new Map();
  const flakyClient = { ...inner, async infer(payload, settings, opts) {
    const k = settings.laya.checkpoint;
    const n = (counters.get(k) ?? 0) + 1; counters.set(k, n);
    if (k === activeReg.checkpoint && n % 3 === 0) throw new ControlError('INPUT_TRUNCATED');
    return inner.infer(payload, settings, opts);
  } };
  const cmp = await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: flakyClient });
  assert.equal(cmp.active, activeReg.checkpoint);
  assert.ok(cmp.by_question, 'compare result must report per-question raw stats');
  assert.ok(cmp.by_question.active, 'compare result must report per-question raw stats for the active side');
  const activeRefused = Object.values(cmp.by_question.active).reduce((s, q) => s + q.refused, 0);
  assert.ok(activeRefused > 0, 'refused samples must be counted');
  const candidateRefused = Object.values(cmp.by_question.candidate).reduce((s, q) => s + q.refused, 0);
  assert.equal(candidateRefused, 0); // only the active side was made flaky
  // A refused sample must never count as correct, and must never be covered by any threshold (raw uses threshold 0).
  assert.ok(cmp.purposes.route.active.raw.refused > 0);
  assert.ok(cmp.purposes.route.active.raw.n < 9); // fewer than the full 9-sample holdout are covered
});

test('promote is refused without qualification, without a matching comparison report, or on checkpoint mismatch', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/unqualified', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const noQual = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  assert.equal(noQual.promoted, false);
  assert.equal(noQual.reason, 'PROMOTION_REFUSED');
  assert.ok(noQual.violations.includes('QUALIFICATION_NOT_FOUND'));

  const { reg: reg2 } = await qualifiedCandidate(f, version, holdout.name);
  const noComparison = promoteCandidate(f.home, { candidateHash: reg2.checkpoint, holdoutName: holdout.name });
  assert.equal(noComparison.promoted, false);
  assert.ok(noComparison.violations.includes('COMPARISON_NOT_FOUND'));
});

test('promote is refused when the candidate regresses holdout accuracy past maxRegression vs. the active checkpoint', async t => {
  // Unit-level: hand-construct the qualification/comparison artifacts promote() reads, so the
  // regression gate is exercised directly instead of through a second full inference scenario.
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const activeReg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/active', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const activeQualification = { checkpoint: activeReg.checkpoint, calibrationVersion: 'v1', purposes: ['route'], minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 };
  writeProviders(f.home, { version: 1, provider: 'laya', laya: { ...JSON.parse(fs.readFileSync(activeReg.candidate, 'utf8')), qualification: activeQualification } });
  const reg2 = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/regressive', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qualPath = path.join(f.home, 'laya', 'qualifications', `${reg2.checkpoint}.json`);
  atomicWrite(qualPath, JSON.stringify({ checkpoint: reg2.checkpoint, qualified: true, purposes: ['route'], holdout: holdout.name,
    calibrationVersion: 'v1', minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 }, null, 2) + '\n');
  const comparisonPath = path.join(f.home, 'laya', 'comparisons', `${activeReg.checkpoint}__${reg2.checkpoint}__${holdout.name}.json`);
  atomicWrite(comparisonPath, JSON.stringify({ active: activeReg.checkpoint, candidate: reg2.checkpoint, holdout: holdout.name,
    purposes: { route: { active: { qualified: true, raw: { n: 10, coverage: 1, accuracy: .95 }, selective: { n: 10, coverage: 1, accuracy: .95 } },
      candidate: { qualified: true, raw: { n: 10, coverage: 1, accuracy: .5 }, selective: { n: 10, coverage: 1, accuracy: .5 } } } } }, null, 2) + '\n');
  const promoted = promoteCandidate(f.home, { candidateHash: reg2.checkpoint, holdoutName: holdout.name, maxRegression: .02 });
  assert.equal(promoted.promoted, false);
  assert.equal(promoted.reason, 'PROMOTION_REFUSED');
  assert.ok(promoted.violations.includes('route:ACCURACY_REGRESSION'));
  // providers.json must be unchanged after a refusal.
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.equal(providers.laya.checkpoint, activeReg.checkpoint);
});

// ============================== rollback ==============================

test('rollback restores the laya block active before the last promote/rollback; refuses with no history', async t => {
  const f = trainingFixture(t);
  assert.throws(() => rollbackLaya(f.home), /NO_HISTORY_TO_ROLLBACK/);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  const rolled = rollbackLaya(f.home);
  assert.equal(rolled.rolledBack, true);
  assert.equal(rolled.restored, null);
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.equal(providers.laya, null);
  assert.equal(providers.provider, 'jev');
});

// ============================== status ==============================

test('status reports active checkpoint, candidate qualification state, and holdout metadata without sample content', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  const status = layaStatus(f.home);
  assert.equal(status.active_checkpoint, null);
  assert.equal(status.candidates.length, 1);
  assert.equal(status.candidates[0].checkpoint, reg.checkpoint);
  assert.equal(status.candidates[0].qualified, true);
  assert.equal(status.holdouts.length, 1);
  assert.equal(status.holdouts[0].name, 'h1');
});

test('rollback walks back promotes like a stack and can never re-apply a rolled-back checkpoint', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg: a } = await qualifiedCandidate(f, version, holdout.name, 'laya/a');
  await compareCandidate(f.home, { candidateHash: a.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  assert.equal(promoteCandidate(f.home, { candidateHash: a.checkpoint, holdoutName: holdout.name }).promoted, true);
  const { reg: b } = await qualifiedCandidate(f, version, holdout.name, 'laya/b');
  await compareCandidate(f.home, { candidateHash: b.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient() });
  assert.equal(promoteCandidate(f.home, { candidateHash: b.checkpoint, holdoutName: holdout.name }).promoted, true);
  const active = () => JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8')).laya?.checkpoint ?? null;
  assert.equal(active(), b.checkpoint);
  rollbackLaya(f.home); assert.equal(active(), a.checkpoint);
  rollbackLaya(f.home); assert.equal(active(), null);
  // A third rollback must not "undo the rollback" and silently re-promote a checkpoint without its gates.
  assert.throws(() => rollbackLaya(f.home), /NO_HISTORY_TO_ROLLBACK/);
  assert.equal(active(), null);
});
test('rollback refuses when providers.json no longer holds the checkpoint the last promote installed', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  assert.throws(() => rollbackLaya(f.home), /ROLLBACK_STATE_MISMATCH/);
});
test('promote requires the qualification and the comparison to use the same fixed holdout', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const h1 = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const h2 = freezeHoldout(f.home, { datasetVersion: version, name: 'h2' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg } = await qualifiedCandidate(f, version, h1.name);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: h2.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  const r = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: h2.name });
  assert.equal(r.promoted, false);
  assert.ok(r.violations.includes('QUALIFICATION_HOLDOUT_MISMATCH'));
});
test('an unqualified active checkpoint is compared on raw holdout accuracy, not on an incomparable coverage', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const baseReg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/base', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  // Active base checkpoint without any qualification block (the usual first-fine-tune situation).
  writeProviders(f.home, { version: 1, provider: 'jev', laya: JSON.parse(fs.readFileSync(baseReg.candidate, 'utf8')) });
  const { reg } = await qualifiedCandidate(f, version, holdout.name, 'laya/first-finetune');
  const cmp = await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient() });
  assert.equal(cmp.purposes.route.active.qualified, false);
  assert.ok(cmp.purposes.route.active.raw.n > 0);
  const r = promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name });
  assert.equal(r.promoted, true, JSON.stringify(r));
});

test('a promoted providers.json is loadable by the runtime provider loader', async t => {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  writeProviders(f.home, { version: 1, provider: 'jev' });
  const { reg } = await qualifiedCandidate(f, version, holdout.name);
  await compareCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name, layaClient: fakeLayaClient(), noActiveBaseline: true });
  assert.equal(promoteCandidate(f.home, { candidateHash: reg.checkpoint, holdoutName: holdout.name }).promoted, true);
  const { loadProviderConfig } = await import('../src/inference.mjs');
  const c = loadProviderConfig(f.home);
  assert.equal(c.laya.checkpoint, reg.checkpoint);
  assert.ok(c.laya.qualification.calibrationVersion.length <= 80);
});

test('CLI laya register accepts --python so the first checkpoint can be registered before providers.json has a laya block', t => {
  const f = trainingFixture(t);
  const ckpt = makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-'))));
  const python = execFileSync('python3', ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' }).trim();
  const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
  const r = spawnSync(process.execPath, [bin, 'laya', 'register', '--checkpoint', ckpt, '--python', python, '--device', 'cpu', '--precision', 'fp16', '--home', f.home],
    { encoding: 'utf8', env: { ...process.env, HOME: f.home, POINTSMAN_HOME: f.home } });
  assert.equal(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout);
  const candidate = JSON.parse(fs.readFileSync(out.candidate, 'utf8'));
  assert.equal(candidate.python, python);
  assert.equal(candidate.precision, 'fp16');
  assert.equal(fs.existsSync(path.join(f.home, 'providers.json')), false); // register never activates
  const bad = spawnSync(process.execPath, [bin, 'laya', 'register', '--checkpoint', ckpt, '--python', 'relative/python', '--device', 'cpu', '--home', f.home],
    { encoding: 'utf8', env: { ...process.env, HOME: f.home, POINTSMAN_HOME: f.home } });
  assert.notEqual(bad.status, 0);
});

test('laya activate installs an unqualified candidate for shadow/data collection only and is undone by rollback', t => {
  const f = trainingFixture(t);
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-')))), model: 'laya/base', device: 'cpu', python: '/usr/bin/python3', precision: 'fp16', fingerprintImpl: fakeFingerprint });
  const r = activateCandidate(f.home, { candidateHash: reg.checkpoint });
  assert.equal(r.activated, true); assert.equal(r.qualified, false);
  const providers = JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8'));
  assert.equal(providers.provider, 'jev'); // provider selection is a separate explicit command
  assert.equal(providers.laya.checkpoint, reg.checkpoint);
  assert.equal(providers.laya.precision, 'fp16');
  assert.equal(providers.laya.qualification, undefined); // never applied in ON without a qualification
  // An active qualified checkpoint is never silently replaced by an unqualified one.
  writeProviders(f.home, { ...providers, laya: { ...providers.laya, qualification: { checkpoint: reg.checkpoint, calibrationVersion: 'v1', purposes: ['route'], minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 } } });
  assert.throws(() => activateCandidate(f.home, { candidateHash: reg.checkpoint }), /ACTIVE_CHECKPOINT_QUALIFIED/);
  writeProviders(f.home, providers);
  rollbackLaya(f.home);
  assert.equal(JSON.parse(fs.readFileSync(path.join(f.home, 'providers.json'), 'utf8')).laya, null);
});
