// Explicit Laya checkpoint lifecycle: register -> holdout freeze -> qualify -> compare -> promote -> rollback.
// Nothing here starts training or promotes automatically; every step is one operator-invoked CLI command.
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { noSymlinks, ensureDir, readText, atomicWrite } from '../storage.mjs';
import { fail, PURPOSES, DEFAULTS, ControlError } from '../constants.mjs';
import { HASH, MAX_DERIVED_BYTES, digest, encode, only } from './schema.mjs';
import { readDataset } from './dataset.mjs';
import { createTrainingStore } from './store.mjs';
import { loadProviderConfig, validateProviderConfig, normalizeInference, createLayaClient } from '../inference.mjs';
import { validateRequest, wireRequest } from '../contracts.mjs';
import { loadFeaturePolicy } from '../feature-policy.mjs';
import { labelTier, tierDistribution, decideTier } from '../routing.mjs';

export const LAYA_RUNTIME_VERSION = '0.3.4';
const DEVICES = ['cpu', 'mps', 'cuda'];
export const MODEL_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$/;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$/;

export const layaRoot = home => path.join(home, 'laya');
export const candidatesDir = home => path.join(layaRoot(home), 'candidates');
export const checkpointsDir = home => path.join(layaRoot(home), 'checkpoints');
const holdoutsDir = home => path.join(layaRoot(home), 'holdouts');
const qualificationsDir = home => path.join(layaRoot(home), 'qualifications');
const comparisonsDir = home => path.join(layaRoot(home), 'comparisons');
export const publishedDir = home => path.join(layaRoot(home), 'published');
const historyFile = home => path.join(layaRoot(home), 'history.jsonl');
const providersFile = home => path.join(home, 'providers.json');

// Validate against the exact providers.json contract the runtime loader enforces (never a divergent copy).
// Returns the normalized copy (defaults such as startupTimeoutMs/idleTimeoutMs/inputFit filled in). Callers that hand
// settings to a worker must use the return value: the raw candidate file omits those defaults.
function validateLayaSettings(l) { return validateProviderConfig({ version: 1, provider: 'jev', laya: structuredClone(l) }).laya; }
function assertNoSymlinksDeep(root) {
  noSymlinks(root);
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    for (const name of fs.readdirSync(dir)) {
      const p = path.join(dir, name);
      const st = fs.lstatSync(p);
      if (st.isSymbolicLink()) fail('LAYA_MODEL_SYMLINK_REFUSED');
      if (st.isDirectory()) stack.push(p);
    }
  }
}
/** Offline, file-hashing only: `python -I workers/laya_worker.py --fingerprint <dir>`. No inference, no download. */
export function defaultFingerprint(python, dir) {
  if (typeof python !== 'string' || !path.isAbsolute(python)) fail('LAYA_PYTHON_NOT_CONFIGURED');
  const script = fileURLToPath(new URL('../../workers/laya_worker.py', import.meta.url));
  const r = spawnSync(python, ['-I', script, '--fingerprint', dir], { shell: false, encoding: 'utf8', timeout: 60000 });
  if (r.error || r.status !== 0) fail('LAYA_FINGERPRINT_FAILED');
  let parsed; try { parsed = JSON.parse(String(r.stdout).trim().split('\n').pop()); } catch { fail('LAYA_FINGERPRINT_FAILED'); }
  if (!parsed || !HASH.test(parsed.checkpoint)) fail('LAYA_FINGERPRINT_FAILED');
  return parsed.checkpoint;
}

// --- register ---------------------------------------------------------
export function registerCheckpoint(home, { checkpointDir, model, device, python, precision, inputFit, fingerprintImpl = defaultFingerprint } = {}) {
  if (typeof checkpointDir !== 'string' || !path.isAbsolute(checkpointDir)) fail('LAYA_CHECKPOINT_PATH_REQUIRED');
  assertNoSymlinksDeep(checkpointDir);
  let stat; try { stat = fs.statSync(checkpointDir); } catch { fail('LAYA_CHECKPOINT_NOT_DIRECTORY'); }
  if (!stat.isDirectory()) fail('LAYA_CHECKPOINT_NOT_DIRECTORY');
  let active = { version: 1, provider: 'jev', laya: null };
  try { active = loadProviderConfig(home); } catch { /* providers.json absent/invalid: fall back to explicit flags only */ }
  const resolvedPython = python ?? active.laya?.python;
  if (typeof resolvedPython !== 'string' || !path.isAbsolute(resolvedPython)) fail('LAYA_PYTHON_NOT_CONFIGURED');
  const resolvedDevice = device ?? active.laya?.device;
  if (!DEVICES.includes(resolvedDevice)) fail('LAYA_DEVICE_REQUIRED');
  // fp16 changes the answer distribution slightly (measured max delta 0.0074), so it is fixed at
  // register time and travels with the candidate through qualify/promote rather than being an
  // independent runtime knob.
  const resolvedPrecision = precision ?? active.laya?.precision ?? 'fp32';
  if (!['fp32', 'fp16'].includes(resolvedPrecision)) fail('INVALID_PROVIDER_CONFIG');
  // Like precision, inputFit is fixed at register time and travels with the candidate through
  // activate/promote (see providers.json `laya.inputFit`, default OFF/'lossless').
  const resolvedInputFit = inputFit ?? active.laya?.inputFit ?? 'lossless';
  if (!['lossless', 'task-head'].includes(resolvedInputFit)) fail('INVALID_PROVIDER_CONFIG');
  const checkpoint = fingerprintImpl(resolvedPython, checkpointDir);
  const resolvedModel = model ?? `laya/${checkpoint.slice(0, 12)}`;
  if (!MODEL_RE.test(resolvedModel)) fail('LAYA_MODEL_INVALID');
  ensureDir(layaRoot(home), true); ensureDir(checkpointsDir(home), true); ensureDir(candidatesDir(home), true);
  const dest = path.join(checkpointsDir(home), checkpoint);
  const reused = fs.existsSync(dest);
  if (reused) {
    noSymlinks(dest);
    if (fingerprintImpl(resolvedPython, dest) !== checkpoint) fail('LAYA_CHECKPOINT_STORE_CORRUPTED');
  } else {
    const tmp = path.join(checkpointsDir(home), `.tmp-${randomUUID()}`);
    fs.cpSync(checkpointDir, tmp, { recursive: true, dereference: false });
    try {
      assertNoSymlinksDeep(tmp);
      if (fingerprintImpl(resolvedPython, tmp) !== checkpoint) fail('LAYA_CHECKPOINT_COPY_MISMATCH');
      fs.chmodSync(tmp, 0o700);
      fs.renameSync(tmp, dest);
    } catch (e) { fs.rmSync(tmp, { recursive: true, force: true }); throw e; }
  }
  const candidate = { python: resolvedPython, modelPath: dest, model: resolvedModel, checkpoint, runtimeVersion: LAYA_RUNTIME_VERSION, device: resolvedDevice, precision: resolvedPrecision, inputFit: resolvedInputFit };
  validateLayaSettings(candidate);
  const file = path.join(candidatesDir(home), `${checkpoint}.json`);
  atomicWrite(file, JSON.stringify(candidate, null, 2) + '\n');
  return { checkpoint, candidate: file, modelPath: dest, reused };
}

// --- holdout ------------------------------------------------------------
export function freezeHoldout(home, { datasetVersion, name = datasetVersion, store } = {}) {
  if (!NAME_RE.test(String(name))) fail('INVALID_HOLDOUT_NAME');
  const s = store ?? createTrainingStore({ home });
  const { samples } = readDataset(s, datasetVersion);
  const testSamples = samples.filter(x => x.split === 'test');
  if (!testSamples.length) fail('EMPTY_HOLDOUT');
  ensureDir(layaRoot(home), true); ensureDir(holdoutsDir(home), true);
  const dataFile = path.join(holdoutsDir(home), `${name}.jsonl`), manifestFile = path.join(holdoutsDir(home), `${name}.json`);
  if (fs.existsSync(dataFile) || fs.existsSync(manifestFile)) fail('HOLDOUT_ALREADY_EXISTS');
  const contents = testSamples.map(encode).join('\n') + '\n';
  const manifest = { name: String(name), dataset_version: datasetVersion, sample_count: testSamples.length, sha256: digest(contents), created_at: new Date().toISOString() };
  atomicWrite(dataFile, contents, { expected: null });
  atomicWrite(manifestFile, JSON.stringify(manifest, null, 2) + '\n', { expected: null });
  return manifest;
}
export function listHoldouts(home) {
  const dir = holdoutsDir(home);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort()
    .map(f => JSON.parse(readText(path.join(dir, f), { privateFile: true, maxBytes: 65536 })));
}
function readHoldout(home, name) {
  if (!NAME_RE.test(String(name))) fail('INVALID_HOLDOUT_NAME');
  const manifest = JSON.parse(readText(path.join(holdoutsDir(home), `${name}.json`), { privateFile: true, maxBytes: 65536 }));
  const contents = readText(path.join(holdoutsDir(home), `${name}.jsonl`), { privateFile: true, maxBytes: MAX_DERIVED_BYTES });
  if (manifest.sha256 !== digest(contents)) fail('HOLDOUT_CORRUPTED');
  const samples = contents.trim() ? contents.trim().split('\n').map(JSON.parse) : [];
  if (samples.length !== manifest.sample_count) fail('HOLDOUT_CORRUPTED');
  return { manifest, samples };
}

// --- inference over dataset/holdout samples ------------------------------
// Per-sample input-admission refusals from the official worker (e.g. the tokenizer budget rejecting
// the request outright) are not a model quality signal and must not abort the whole qualify/compare
// run. Any other error code still aborts (a real bug should not be silently swallowed as "refused").
const INPUT_REFUSAL_CODES = new Set(['INPUT_TRUNCATED', 'INPUT_REWRITE_REFUSED']);
function argmaxKey(probabilities) {
  let bestKey = null, bestValue = -Infinity;
  for (const [key, value] of Object.entries(probabilities)) if (value > bestValue) { bestValue = value; bestKey = key; }
  return bestKey;
}
// Exported so scripts/laya-benchmark.mjs (a separate reporting tool, not the qualify/compare
// lifecycle) reuses the exact same per-sample scoring semantics instead of duplicating them:
// argmax-of-probabilities correctness, INPUT_TRUNCATED/INPUT_REWRITE_REFUSED counted as refused
// (never correct), and the same threshold-comparable `metric` per question type.
export async function inferOne(layaClient, laya, sample, { timeoutMs = 30000, env = process.env, signal } = {}) {
  // Groups this sample's answer with its siblings from the same route request (see buildRouteTasks
  // below): prefer a future distill-run task id if present, falling back to the dataset's own task_id.
  const taskKey = sample.raw_refs?.distill?.task_id ?? sample.task_id ?? null;
  const request = validateRequest({ purpose: sample.purpose, risk: 'routine', state: sample.state, questions: { [sample.question_id]: sample.question } }, DEFAULTS);
  const payload = wireRequest(request, laya.model);
  let raw;
  try {
    raw = await layaClient.infer(payload, { laya }, { timeoutMs, env, signal });
  } catch (e) {
    if (e instanceof ControlError && INPUT_REFUSAL_CODES.has(e.code)) {
      // Never covered by any qualify/compare threshold (metric -Infinity), and never counted correct.
      return { purpose: sample.purpose, question_id: sample.question_id, correct: false, metric: -Infinity, predicted: null,
        label_source: sample.label_source, refused: true, task_key: taskKey, target: sample.target.value, probabilities: null };
    }
    throw e;
  }
  const relaxed = { ...DEFAULTS, minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0 };
  const n = normalizeInference('laya', raw, request, relaxed, { laya });
  const answer = n.answers[sample.question_id];
  // Score questions report a continuous probability-weighted expected value in `answer.value` (e.g.
  // 2.0257), never the integer target index -- agreement must compare the argmax label instead.
  // `predicted` is the same argmax label used for `correct` (score: numeric index; choice: key),
  // exposed so a caller (e.g. scripts/laya-benchmark.mjs) can build confusion/within-N stats against
  // `sample.target.value` without re-deriving the argmax itself.
  const predicted = sample.question.type === 'score' ? Number(argmaxKey(answer.probabilities)) : answer.value;
  const correct = sample.question.type === 'score'
    ? argmaxKey(answer.probabilities) === String(sample.target.value)
    : answer.value === sample.target.value;
  // Threshold mapping: choice questions gate on BOTH selected probability and confidence (the
  // minChoiceProbability/minConfidence pair); noul gates on max(p,1-p) i.e. certainty away from 0.5;
  // score gates on confidence alone. A single grid value t is compared against whichever metric applies.
  const metric = sample.question.type === 'noul' ? Math.max(answer.probabilityTrue, 1 - answer.probabilityTrue)
    : sample.question.type === 'choice' ? Math.min(answer.confidence, answer.selectedProbability)
    : answer.confidence;
  return { purpose: sample.purpose, question_id: sample.question_id, correct, metric, predicted, label_source: sample.label_source, refused: false,
    task_key: taskKey, target: sample.target.value, probabilities: answer.probabilities };
}
// Content-free per-question raw stats: n, raw agreement (argmax-correct / n), refused.
function byQuestionStats(records) {
  const buckets = {};
  for (const r of records) {
    const b = buckets[r.question_id] ??= { n: 0, correct: 0, refused: 0 };
    b.n++;
    if (r.refused) b.refused++; else if (r.correct) b.correct++;
  }
  return Object.fromEntries(Object.entries(buckets).map(([qid, b]) => [qid, { n: b.n, raw_agreement: b.n ? b.correct / b.n : 0, refused: b.refused }]));
}
// When any evaluated sample carries an 'ai_reference' label (owner decision 2026-09-23), the metric
// this checkpoint is scored against is agreement with that reference model, never "accuracy".
// Exported so scripts/laya-benchmark.mjs reports the same metric_semantics gate qualify/compare
// already enforce, instead of a second copy that could drift.
export function labelSourceSummary(records) {
  const counts = {};
  for (const r of records) counts[r.label_source] = (counts[r.label_source] ?? 0) + 1;
  const metricSemantics = Object.hasOwn(counts, 'ai_reference') ? 'agreement-with-ai-reference' : 'accuracy';
  return { evaluation_label_sources: counts, metric_semantics: metricSemantics };
}
async function inferAll(layaClient, laya, samples, opts) {
  const out = [];
  for (const sample of samples) out.push(await inferOne(layaClient, laya, sample, opts));
  return out;
}
function groupByPurpose(records) {
  const out = Object.create(null);
  for (const r of records) (out[r.purpose] ??= []).push(r);
  return out;
}
function wilsonLowerBound(successes, n) {
  if (n === 0) return 0;
  const z = 1.959963985;
  const p = successes / n, denom = 1 + z * z / n;
  const center = p + z * z / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n);
  return Math.max(0, (center - margin) / denom);
}
function wilsonUpperBound(successes, n) {
  if (n === 0) return 0;
  const z = 1.959963985;
  const p = successes / n, denom = 1 + z * z / n;
  const center = p + z * z / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n);
  return Math.min(1, (center + margin) / denom);
}

// --- decision-level route gate (purpose 'route' only) ----------------------------------
// Groups this split's inferOne records (one per (task, question) with question_id in
// {intent, difficulty, risk}) into complete route tasks; anything refused or missing a dimension
// is skipped (never silently dropped, always counted).
function buildRouteTasks(records) {
  const byTask = new Map();
  let skipped = 0;
  for (const r of records) {
    if (r.purpose !== 'route') continue;
    if (r.refused || !r.task_key) { skipped++; continue; }
    const entry = byTask.get(r.task_key) ?? {};
    entry[r.question_id] = r;
    byTask.set(r.task_key, entry);
  }
  const tasks = [];
  for (const entry of byTask.values()) {
    if (!entry.intent || !entry.difficulty || !entry.risk) { skipped++; continue; }
    tasks.push(entry);
  }
  return { tasks, skipped };
}
const TIER_RANK = { economy: 0, standard: 1, strong: 2 };
// Applies one (tierCoverage, maxHostProbability) gate to a set of built route tasks. Content-free:
// only counts, never task text or predicted/target labels themselves.
function evalRouteGate(tasks, routerPolicy, gate) {
  let applied = 0, economy = 0, standard = 0, strong = 0, exact = 0, over = 0, hostStrong = 0, unsafe = 0;
  for (const t of tasks) {
    const ref = labelTier(t.intent.target, t.difficulty.target, t.risk.target, routerPolicy);
    const predicted = tierDistribution({ intent: t.intent, difficulty: t.difficulty, risk: t.risk }, routerPolicy);
    const d = decideTier(predicted, gate);
    if (!d.applied) continue;
    applied++;
    if (d.tier === 'economy') economy++; else if (d.tier === 'standard') standard++; else strong++;
    if (ref === 'host') { if (d.tier === 'strong') hostStrong++; else unsafe++; }
    else if (d.tier === ref) exact++;
    else if (TIER_RANK[d.tier] > TIER_RANK[ref]) over++;
    else unsafe++;
  }
  return { tasks: tasks.length, applied, rate: tasks.length ? applied / tasks.length : 0, economy, standard, strong,
    exact, over, host_strong: hostStrong, unsafe, unsafe_rate: applied ? unsafe / applied : 0, unsafe_upper: wilsonUpperBound(unsafe, applied) };
}
const ROUTE_GATE_TAU_GRID = [0.99, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7];
const ROUTE_GATE_EPS_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3];
// Grid search over calibration only: maximise decided economy+standard subject to applied >=
// routeMinApplied and Wilson-upper(unsafe/applied) <= routeMaxError; ties broken by larger applied,
// then larger tierCoverage, then smaller maxHostProbability (deterministic).
function selectRouteGate(calibTasks, routerPolicy, { routeMinApplied, routeMaxError }) {
  let best = null;
  // Traversal order (tau descending, eps ascending within a tau) already puts the largest-tau,
  // smallest-eps candidate first for any tied (objective, applied) pair, so "keep first on ties,
  // replace only on strict improvement" implements the full deterministic tie-break by construction.
  for (const tau of ROUTE_GATE_TAU_GRID) {
    for (const eps of ROUTE_GATE_EPS_GRID) {
      const stats = evalRouteGate(calibTasks, routerPolicy, { tierCoverage: tau, maxHostProbability: eps });
      if (stats.applied < routeMinApplied || stats.unsafe_upper > routeMaxError) continue;
      const objective = stats.economy + stats.standard;
      if (!best || objective > best.objective || (objective === best.objective && stats.applied > best.stats.applied)) {
        best = { tau, eps, stats, objective };
      }
    }
  }
  return best;
}
// Computes the full decision-gate evidence block for one qualify run: the selected (tau, eps) from
// calibration (or null if nothing satisfies the constraints) plus the same evalRouteGate() stats on
// test and holdout, and whether both independently pass the (looser) verification thresholds.
function routeDecisionEvidence(calibRecords, testRecords, holdoutRecords, routerPolicy, params) {
  const { routeMinApplied, routeMaxError, routeMaxErrorUpper } = params;
  const calib = buildRouteTasks(calibRecords), test = buildRouteTasks(testRecords), holdout = buildRouteTasks(holdoutRecords);
  const best = selectRouteGate(calib.tasks, routerPolicy, { routeMinApplied, routeMaxError });
  const empty = { tasks: 0, applied: 0, rate: 0, economy: 0, standard: 0, strong: 0, exact: 0, over: 0, host_strong: 0, unsafe: 0, unsafe_rate: 0, unsafe_upper: 0 };
  if (!best) {
    return { gate: null, calibration: { ...empty, tasks: calib.tasks.length, skipped: calib.skipped },
      test: { ...empty, tasks: test.tasks.length, skipped: test.skipped }, holdout: { ...empty, tasks: holdout.tasks.length, skipped: holdout.skipped }, passes: false };
  }
  const gate = { tierCoverage: best.tau, maxHostProbability: best.eps };
  const testStats = evalRouteGate(test.tasks, routerPolicy, gate), holdoutStats = evalRouteGate(holdout.tasks, routerPolicy, gate);
  const verifies = s => s.applied >= routeMinApplied && s.unsafe_rate <= routeMaxError && s.unsafe_upper <= routeMaxErrorUpper;
  return { gate, calibration: { ...best.stats, skipped: calib.skipped }, test: { ...testStats, skipped: test.skipped },
    holdout: { ...holdoutStats, skipped: holdout.skipped }, passes: verifies(testStats) && verifies(holdoutStats) };
}
function evalSplit(records, threshold) {
  const covered = records.filter(r => r.metric >= threshold);
  const successes = covered.filter(r => r.correct).length;
  const refused = records.filter(r => r.refused).length;
  return { n: covered.length, coverage: records.length ? covered.length / records.length : 0,
    accuracy: covered.length ? successes / covered.length : 0, lowerBound: wilsonLowerBound(successes, covered.length), refused };
}
// Grid search minThreshold..0.99 step 0.01 (default 0.50): lowest threshold meeting selective accuracy and
// coverage floors. minThreshold lets an operator lower targetAccuracy without also lowering the confidence
// bar the checkpoint earned (owner decision 2026-09-27).
function selectThreshold(records, { targetAccuracy, minCoverage, minCalibration, minThreshold = 0.5 }) {
  if (records.length < minCalibration) return null;
  for (let step = Math.round(minThreshold * 100); step <= 99; step++) {
    const threshold = step / 100;
    const ev = evalSplit(records, threshold);
    if (ev.n > 0 && ev.coverage >= minCoverage && ev.accuracy >= targetAccuracy) return threshold;
  }
  return null;
}

// --- qualify --------------------------------------------------------------
export function loadCandidate(home, candidateHash) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  const file = path.join(candidatesDir(home), `${candidateHash}.json`);
  let laya; try { laya = JSON.parse(readText(file, { privateFile: true, maxBytes: 8192 })); } catch { fail('LAYA_CANDIDATE_NOT_FOUND'); }
  const normalized = validateLayaSettings(laya);
  if (normalized.checkpoint !== candidateHash) fail('LAYA_CANDIDATE_MISMATCH');
  return normalized;
}
export async function qualifyCandidate(home, { candidateHash, datasetVersion, holdoutName, layaClient = createLayaClient(), store,
  timeoutMs = 30000, env = process.env, signal,
  targetAccuracy = 0.9, minCoverage = 0.2, minCalibration = 30, minTest = 30, minLowerBound = 0.8, minThreshold = 0.5,
  routeMaxError = 0.15, routeMaxErrorUpper = 0.25, routeMinApplied = 30 } = {}) {
  if (!Number.isFinite(minThreshold) || minThreshold < 0.5 || minThreshold > 0.99) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isFinite(targetAccuracy) || targetAccuracy < 0.5 || targetAccuracy > 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isFinite(minCoverage) || minCoverage <= 0 || minCoverage > 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isInteger(minCalibration) || minCalibration < 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isInteger(minTest) || minTest < 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isFinite(minLowerBound) || minLowerBound < 0 || minLowerBound > 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isFinite(routeMaxError) || routeMaxError <= 0 || routeMaxError > 1) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isFinite(routeMaxErrorUpper) || routeMaxErrorUpper <= 0 || routeMaxErrorUpper > 1 || routeMaxErrorUpper < routeMaxError) fail('INVALID_QUALIFY_PARAMS');
  if (!Number.isInteger(routeMinApplied) || routeMinApplied < 1) fail('INVALID_QUALIFY_PARAMS');
  const laya = loadCandidate(home, candidateHash);
  const s = store ?? createTrainingStore({ home });
  const { samples } = readDataset(s, datasetVersion);
  const { manifest: holdoutManifest, samples: holdoutSamples } = readHoldout(home, holdoutName);
  const opts = { timeoutMs, env, signal };
  const calibrationRecords = await inferAll(layaClient, laya, samples.filter(x => x.split === 'calibration'), opts);
  const testRecords = await inferAll(layaClient, laya, samples.filter(x => x.split === 'test'), opts);
  const holdoutRecords = await inferAll(layaClient, laya, holdoutSamples, opts);
  const calibrationByPurpose = groupByPurpose(calibrationRecords);
  const testByPurpose = groupByPurpose(testRecords);
  const holdoutByPurpose = groupByPurpose(holdoutRecords);
  const evidence = {}, qualifiedPurposes = [];
  let globalThreshold = null;
  const routerPolicy = loadFeaturePolicy(home).router;
  for (const purpose of PURPOSES) {
    const calib = calibrationByPurpose[purpose] || [];
    const threshold = selectThreshold(calib, { targetAccuracy, minCoverage, minCalibration, minThreshold });
    const entry = { calibration: { n: calib.length, threshold } };
    if (threshold != null) {
      entry.test = evalSplit(testByPurpose[purpose] || [], threshold);
      entry.holdout = evalSplit(holdoutByPurpose[purpose] || [], threshold);
      entry.passes = entry.test.n >= minTest && entry.holdout.n >= minTest &&
        entry.test.accuracy >= targetAccuracy && entry.holdout.accuracy >= targetAccuracy &&
        entry.test.lowerBound >= minLowerBound && entry.holdout.lowerBound >= minLowerBound;
      if (entry.passes) { qualifiedPurposes.push(purpose); globalThreshold = globalThreshold == null ? threshold : Math.max(globalThreshold, threshold); }
    }
    // The decision-level gate is evaluated independently of the per-answer threshold above: a
    // checkpoint may qualify 'route' through this gate alone, through the per-answer path alone,
    // through both, or through neither.
    if (purpose === 'route') {
      entry.decision = routeDecisionEvidence(calibrationRecords, testRecords, holdoutRecords, routerPolicy,
        { routeMinApplied, routeMaxError, routeMaxErrorUpper });
    }
    evidence[purpose] = entry;
  }
  let routeGate = null;
  if (evidence.route?.decision?.passes) {
    routeGate = { method: 'decision-v1', tierCoverage: evidence.route.decision.gate.tierCoverage, maxHostProbability: evidence.route.decision.gate.maxHostProbability };
    if (!qualifiedPurposes.includes('route')) qualifiedPurposes.push('route');
  }
  const params = { targetAccuracy, minCoverage, minCalibration, minTest, minLowerBound, minThreshold, routeMaxError, routeMaxErrorUpper, routeMinApplied };
  // Must fit the provider contract's 80-character calibrationVersion limit.
  const calibrationVersion = `${datasetVersion.slice(0, 32)}:q-${digest(params).slice(0, 32)}`;
  const allRecords = [...calibrationRecords, ...testRecords, ...holdoutRecords];
  const result = { checkpoint: candidateHash, qualified: qualifiedPurposes.length > 0, purposes: qualifiedPurposes,
    // A purpose qualified ONLY through the decision gate has no per-answer threshold of its own; if
    // no purpose has one either, globalThreshold stays null and this 1 fallback applies (per-answer
    // eligibility then always fails for everyone, which is harmless for a decision-gated route since
    // the runtime path above never reads it).
    minConfidence: globalThreshold ?? 1, minChoiceProbability: globalThreshold ?? 1, noulCertainty: globalThreshold ?? 1,
    calibrationVersion, dataset_version: datasetVersion, holdout: holdoutManifest.name, holdout_sha256: holdoutManifest.sha256,
    precision: laya.precision ?? 'fp32', params, evidence, ...(routeGate ? { routeGate } : {}),
    // Per split, never pooled: calibration also tunes epoch/threshold and the holdout freezes the test split.
    by_question: { calibration: byQuestionStats(calibrationRecords), test: byQuestionStats(testRecords), holdout: byQuestionStats(holdoutRecords) },
    generated_at: new Date().toISOString(),
    ...labelSourceSummary(allRecords) };
  ensureDir(layaRoot(home), true); ensureDir(qualificationsDir(home), true);
  atomicWrite(path.join(qualificationsDir(home), `${candidateHash}.json`), JSON.stringify(result, null, 2) + '\n');
  return result;
}
export function loadQualification(home, candidateHash) {
  const file = path.join(qualificationsDir(home), `${candidateHash}.json`);
  const text = readText(file, { optional: true, privateFile: true, maxBytes: 1048576 });
  return text === null ? null : JSON.parse(text);
}

// --- compare ----------------------------------------------------------
// An invalid providers.json must fail loudly here, not look like "no active checkpoint".
function activeLayaOf(home) { return loadProviderConfig(home).laya ?? null; }
export async function compareCandidate(home, { candidateHash, holdoutName, layaClient = createLayaClient(),
  noActiveBaseline = false, timeoutMs = 30000, env = process.env, signal } = {}) {
  const candidate = loadCandidate(home, candidateHash);
  const active = activeLayaOf(home);
  if (!active && !noActiveBaseline) fail('ACTIVE_BASELINE_REQUIRED');
  const { manifest: holdoutManifest, samples: holdoutSamples } = readHoldout(home, holdoutName);
  const opts = { timeoutMs, env, signal };
  const candidateQual = loadQualification(home, candidateHash);
  // raw = every holdout answer (threshold 0): comparable across checkpoints and the basis of the forgetting check.
  // selective = the checkpoint's own qualified threshold; only comparable when BOTH sides are qualified for the purpose.
  const side = (records, qual, purpose) => {
    const qualified = Boolean(qual?.purposes?.includes(purpose));
    return { qualified, raw: evalSplit(records, 0), selective: qualified ? evalSplit(records, qual.minConfidence) : null };
  };
  const candidateRecords = await inferAll(layaClient, candidate, holdoutSamples, opts);
  const candidateByPurpose = groupByPurpose(candidateRecords);
  const purposes = {};
  for (const purpose of PURPOSES) purposes[purpose] = { candidate: side(candidateByPurpose[purpose] || [], candidateQual, purpose) };
  // `--no-active-baseline` must skip inferring the active checkpoint entirely, not just skip the
  // ACTIVE_BASELINE_REQUIRED guard above -- otherwise the flag would still spend a full inference pass
  // on a checkpoint the caller explicitly asked to exclude from this comparison.
  const activeUsed = noActiveBaseline ? null : active;
  let activeRecords = null;
  if (activeUsed) {
    activeRecords = await inferAll(layaClient, activeUsed, holdoutSamples, opts);
    const activeByPurpose = groupByPurpose(activeRecords);
    for (const purpose of PURPOSES) purposes[purpose].active = side(activeByPurpose[purpose] || [], activeUsed.qualification, purpose);
  }
  const report = { active: activeUsed?.checkpoint ?? null, candidate: candidateHash, holdout: holdoutManifest.name,
    holdout_sha256: holdoutManifest.sha256, purposes,
    by_question: { candidate: byQuestionStats(candidateRecords), active: activeRecords ? byQuestionStats(activeRecords) : null },
    generated_at: new Date().toISOString(), ...labelSourceSummary(holdoutSamples) };
  ensureDir(layaRoot(home), true); ensureDir(comparisonsDir(home), true);
  const file = path.join(comparisonsDir(home), `${activeUsed?.checkpoint ?? 'none'}__${candidateHash}__${holdoutManifest.name}.json`);
  atomicWrite(file, JSON.stringify(report, null, 2) + '\n');
  return report;
}
function loadComparison(home, activeHash, candidateHash, holdoutName) {
  const file = path.join(comparisonsDir(home), `${activeHash ?? 'none'}__${candidateHash}__${holdoutName}.json`);
  const text = readText(file, { optional: true, privateFile: true, maxBytes: 1048576 });
  return text === null ? null : JSON.parse(text);
}

// --- history (append-only; each entry can invert the previous providers.json laya block) --
function appendHistory(home, entry) {
  ensureDir(layaRoot(home), true);
  const file = historyFile(home);
  noSymlinks(file);
  const fd = fs.openSync(file, fs.constants.O_CREAT | fs.constants.O_APPEND | fs.constants.O_WRONLY | (fs.constants.O_NOFOLLOW || 0), 0o600);
  try { fs.writeFileSync(fd, JSON.stringify({ at: new Date().toISOString(), ...entry }) + '\n'); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
}
function readHistory(home) {
  const text = readText(historyFile(home), { optional: true, privateFile: true, maxBytes: 10 * 1024 * 1024 });
  if (text === null) return [];
  return text.trim() ? text.trim().split('\n').map(JSON.parse) : [];
}

// --- promote / rollback ------------------------------------------------
// Operator runtime knobs survive a checkpoint change; only the checkpoint identity and qualification move.
function runtimeSettings(laya) {
  return { startupTimeoutMs: laya?.startupTimeoutMs ?? 120000, idleTimeoutMs: laya?.idleTimeoutMs ?? 60000,
    ...(laya?.serverIdleUnloadMs !== undefined ? { serverIdleUnloadMs: laya.serverIdleUnloadMs } : {}) };
}
/** Install an UNQUALIFIED candidate for shadow/data collection. Without a qualification block its answers are
 * never applied in ON, and a currently qualified checkpoint is never silently replaced. Undone by rollback. */
export function activateCandidate(home, { candidateHash } = {}) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  const candidate = loadCandidate(home, candidateHash);
  const file = providersFile(home);
  const old = readText(file, { optional: true, privateFile: true });
  const currentConfig = old === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(old);
  if (currentConfig.laya?.qualification) fail('ACTIVE_CHECKPOINT_QUALIFIED');
  const nextLaya = { ...candidate, ...runtimeSettings(currentConfig.laya) };
  const nextConfig = { ...currentConfig, laya: nextLaya }; // provider selection is a separate explicit command
  validateProviderConfig(structuredClone(nextConfig));
  atomicWrite(file, JSON.stringify(nextConfig, null, 2) + '\n', { expected: old });
  appendHistory(home, { action: 'activate', candidate: candidateHash, before: currentConfig.laya ?? null, after: nextLaya });
  return { activated: true, qualified: false, checkpoint: candidateHash, previous: currentConfig.laya ?? null };
}
export function promoteCandidate(home, { candidateHash, holdoutName, maxRegression = 0.02 } = {}) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  if (!Number.isFinite(maxRegression) || maxRegression < 0 || maxRegression > 1) fail('INVALID_PROMOTE_PARAMS');
  const qualification = loadQualification(home, candidateHash);
  if (!qualification) return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['QUALIFICATION_NOT_FOUND'] };
  if (qualification.checkpoint !== candidateHash) return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['QUALIFICATION_CHECKPOINT_MISMATCH'] };
  if (!qualification.qualified || !qualification.purposes.length) return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['NOT_QUALIFIED'] };
  const currentActive = activeLayaOf(home);
  const comparison = loadComparison(home, currentActive?.checkpoint ?? null, candidateHash, holdoutName);
  if (!comparison) return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['COMPARISON_NOT_FOUND'] };
  if (comparison.candidate !== candidateHash || comparison.active !== (currentActive?.checkpoint ?? null) || comparison.holdout !== holdoutName) {
    return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['COMPARISON_MISMATCH'] };
  }
  if (qualification.holdout !== holdoutName) return { promoted: false, reason: 'PROMOTION_REFUSED', violations: ['QUALIFICATION_HOLDOUT_MISMATCH'] };
  const violations = [];
  for (const purpose of qualification.purposes) {
    const row = comparison.purposes[purpose];
    if (!row?.candidate?.raw || !row.candidate.raw.n) { violations.push(`${purpose}:MISSING_CANDIDATE_RESULT`); continue; }
    if (row.active) {
      // Forgetting check on the same answers for both checkpoints.
      if (row.candidate.raw.accuracy < row.active.raw.accuracy - maxRegression) violations.push(`${purpose}:ACCURACY_REGRESSION`);
      if (row.active.qualified && row.active.selective && row.candidate.selective) {
        if (row.candidate.selective.accuracy < row.active.selective.accuracy - maxRegression) violations.push(`${purpose}:SELECTIVE_ACCURACY_REGRESSION`);
        if (row.candidate.selective.coverage < row.active.selective.coverage - maxRegression) violations.push(`${purpose}:COVERAGE_REGRESSION`);
      }
    }
  }
  if (violations.length) return { promoted: false, reason: 'PROMOTION_REFUSED', violations };
  const candidate = loadCandidate(home, candidateHash);
  const file = providersFile(home);
  const old = readText(file, { optional: true, privateFile: true });
  const currentConfig = old === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(old);
  const nextLaya = { ...candidate, ...runtimeSettings(currentConfig.laya),
    qualification: { checkpoint: qualification.checkpoint, calibrationVersion: qualification.calibrationVersion, purposes: qualification.purposes,
      minConfidence: qualification.minConfidence, minChoiceProbability: qualification.minChoiceProbability, noulCertainty: qualification.noulCertainty,
      precision: qualification.precision ?? candidate.precision ?? 'fp32', ...(qualification.routeGate ? { routeGate: qualification.routeGate } : {}) } };
  const nextConfig = { ...currentConfig, laya: nextLaya }; // provider selection (jev/laya) is never changed here
  validateProviderConfig(structuredClone(nextConfig));
  atomicWrite(file, JSON.stringify(nextConfig, null, 2) + '\n', { expected: old });
  appendHistory(home, { action: 'promote', candidate: candidateHash, before: currentConfig.laya ?? null, after: nextLaya });
  return { promoted: true, checkpoint: candidateHash, previous: currentConfig.laya ?? null };
}
// --- adopt (consumer-side activation of a pulled candidate) -------------
/** Published qualification for a pulled candidate: {repo, revision, qualification, evaluation}, written by
 * laya pull. A consumer has no local holdout to run qualify/compare against, so `adopt` trusts this
 * publisher-provided record explicitly instead (a separate operator action from promote). */
export function loadPublished(home, candidateHash) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  const text = readText(path.join(publishedDir(home), `${candidateHash}.json`), { optional: true, privateFile: true, maxBytes: 1048576 });
  return text === null ? null : JSON.parse(text);
}
export function adoptCandidate(home, { candidateHash } = {}) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  const candidate = loadCandidate(home, candidateHash);
  const published = loadPublished(home, candidateHash);
  if (!published || !published.qualification) fail('PUBLISHED_RECORD_NOT_FOUND');
  const q = published.qualification;
  if (q.checkpoint !== candidateHash) fail('PUBLISHED_QUALIFICATION_MISMATCH');
  const candidatePrecision = candidate.precision ?? 'fp32';
  const qualificationPrecision = q.precision ?? candidatePrecision;
  if (qualificationPrecision !== candidatePrecision) fail('PUBLISHED_QUALIFICATION_MISMATCH');
  const file = providersFile(home);
  const old = readText(file, { optional: true, privateFile: true });
  const currentConfig = old === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(old);
  const nextLaya = { ...candidate, ...runtimeSettings(currentConfig.laya),
    qualification: { checkpoint: q.checkpoint, calibrationVersion: q.calibrationVersion, purposes: q.purposes,
      minConfidence: q.minConfidence, minChoiceProbability: q.minChoiceProbability, noulCertainty: q.noulCertainty,
      precision: qualificationPrecision, ...(q.routeGate ? { routeGate: q.routeGate } : {}) } };
  const nextConfig = { ...currentConfig, laya: nextLaya }; // provider selection (jev/laya) is never changed here
  validateProviderConfig(structuredClone(nextConfig));
  atomicWrite(file, JSON.stringify(nextConfig, null, 2) + '\n', { expected: old });
  appendHistory(home, { action: 'adopt', candidate: candidateHash, before: currentConfig.laya ?? null, after: nextLaya,
    source: { repo: published.repo, revision: published.revision } });
  return { adopted: true, checkpoint: candidateHash, previous: currentConfig.laya ?? null, source: { repo: published.repo, revision: published.revision } };
}
export function rollbackLaya(home) {
  // Promotes form a stack; each rollback pops one. A rollback never re-applies a checkpoint it removed,
  // so it cannot become a gate-free re-promotion.
  const stack = [];
  for (const entry of readHistory(home)) {
    if (entry.action === 'promote' || entry.action === 'activate' || entry.action === 'adopt') stack.push(entry);
    else if (entry.action === 'rollback') stack.pop();
  }
  if (!stack.length) fail('NO_HISTORY_TO_ROLLBACK');
  const last = stack[stack.length - 1];
  const restored = last.before ?? null;
  const file = providersFile(home);
  const old = readText(file, { optional: true, privateFile: true });
  const currentConfig = old === null ? { version: 1, provider: 'jev', laya: null } : JSON.parse(old);
  if ((currentConfig.laya?.checkpoint ?? null) !== (last.after?.checkpoint ?? null)) fail('ROLLBACK_STATE_MISMATCH');
  const nextConfig = { ...currentConfig, laya: restored };
  atomicWrite(file, JSON.stringify(nextConfig, null, 2) + '\n', { expected: old });
  appendHistory(home, { action: 'rollback', candidate: null, before: currentConfig.laya ?? null, after: restored });
  return { rolledBack: true, restored };
}

// --- status ------------------------------------------------------------
export function layaStatus(home) {
  const active = activeLayaOf(home);
  const dir = candidatesDir(home);
  const candidates = fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort().map(f => {
    const c = JSON.parse(readText(path.join(dir, f), { privateFile: true, maxBytes: 8192 }));
    const q = loadQualification(home, c.checkpoint);
    const published = loadPublished(home, c.checkpoint);
    return { checkpoint: c.checkpoint, model: c.model, device: c.device, qualified: q?.qualified ?? null, purposes: q?.purposes ?? [],
      ...(published ? { published: { repo: published.repo, revision: published.revision } } : {}) };
  }) : [];
  return { active_checkpoint: active?.checkpoint ?? null, candidates, holdouts: listHoldouts(home) };
}
