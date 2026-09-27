import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { validateProviderConfig, normalizeInference, createLayaClient } from '../src/inference.mjs';
import { registerCheckpoint } from '../src/training/laya-lifecycle.mjs';
import { atomicWrite } from '../src/storage.mjs';
import { readMetrics } from '../src/metrics.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { setMode } from '../src/storage.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { layaConfig, request as buildRequest, trace as buildTrace, KEY } from './training-helpers.mjs';

function tempHome(t) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-input-fit-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  return home;
}
function makeCheckpointDir(root) {
  const dir = fs.mkdtempSync(path.join(root, 'ckpt-'));
  for (const [name, content] of Object.entries({ 'model.safetensors': 'fixture', 'rl_agent_config.json': '{}',
    'encoder/config.json': '{}', 'tokenizer/tokenizer_config.json': '{}' })) {
    const p = path.join(dir, name); fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, content);
  }
  return dir;
}
const fakeFingerprint = () => 'b'.repeat(64);
function writeProviders(home, config) { atomicWrite(path.join(home, 'providers.json'), JSON.stringify(config, null, 2) + '\n'); }

// --- providers.json contract ------------------------------------------------
test('validateProviderConfig defaults laya.inputFit to lossless, accepts task-head, rejects unknown values', () => {
  const base = { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: '/tmp', model: 'laya/base',
    checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu' } };
  const defaulted = validateProviderConfig(structuredClone(base));
  assert.equal(defaulted.laya.inputFit, 'lossless');
  const explicit = validateProviderConfig({ ...structuredClone(base), laya: { ...base.laya, inputFit: 'task-head' } });
  assert.equal(explicit.laya.inputFit, 'task-head');
  assert.throws(() => validateProviderConfig({ ...structuredClone(base), laya: { ...base.laya, inputFit: 'aggressive' } }), /INVALID_PROVIDER_CONFIG/);
});

// --- laya register --input-fit ----------------------------------------------
test('registerCheckpoint defaults inputFit to lossless, accepts an explicit value, inherits from the active config, and rejects invalid values', t => {
  const home = tempHome(t);
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-ckpt-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const noConfig = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/i1', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(noConfig.candidate, 'utf8')).inputFit, 'lossless');
  const explicit = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/i2', device: 'cpu', python: '/usr/bin/python3', inputFit: 'task-head', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(explicit.candidate, 'utf8')).inputFit, 'task-head');
  writeProviders(home, { version: 1, provider: 'laya', laya: { python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu', inputFit: 'task-head' } });
  const fromActive = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/i3', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.equal(JSON.parse(fs.readFileSync(fromActive.candidate, 'utf8')).inputFit, 'task-head');
  assert.throws(() => registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/i4', device: 'cpu', python: '/usr/bin/python3', inputFit: 'aggressive', fingerprintImpl: fakeFingerprint }), /INVALID_PROVIDER_CONFIG/);
});

// --- worker wire payload -----------------------------------------------------
function fakeSpawn() {
  const messages = [];
  const spawnImpl = () => {
    const c = new EventEmitter();
    c.stdin = new PassThrough(); c.stdout = new PassThrough(); c.stderr = new PassThrough();
    c.kill = () => true;
    let init;
    c.stdin.on('data', b => {
      let m; try { m = JSON.parse(b.toString()); } catch { return; }
      if (m.init) { init = m.init; queueMicrotask(() => c.stdout.write(JSON.stringify({ ready: true, identity: { model: init.model, checkpoint: init.checkpoint, runtime_version: init.runtimeVersion, device: init.device, precision: 'torch.float32' } }) + '\n')); return; }
      messages.push(m);
      queueMicrotask(() => c.stdout.write(JSON.stringify({ id: m.id, result: { identity: { model: init.model, checkpoint: init.checkpoint, runtime_version: init.runtimeVersion, device: init.device, precision: 'torch.float32' },
        answers: { worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } } }, model: init.model, usage: { input_tokens: 10, output_tokens: 0 },
        input_fit: { truncated: true, original_chars: 2500, kept_chars: 1440 } } }) + '\n'));
    });
    return c;
  };
  return { spawnImpl, messages };
}
test('createLayaClient.infer sends laya.inputFit through to the worker frame', async t => {
  const home = tempHome(t);
  const fake = fakeSpawn();
  const client = createLayaClient({ spawnImpl: fake.spawnImpl });
  t.after(() => client.close());
  fs.writeFileSync(path.join(home, 'model.safetensors'), 'fake');
  const laya = layaConfig(home, { inputFit: 'task-head' }).laya;
  const raw = await client.infer({ state: { task: 'x' }, questions: {} }, { laya }, { timeoutMs: 2000 });
  assert.equal(fake.messages.length, 1);
  assert.equal(fake.messages[0].inputFit, 'task-head');
  assert.deepEqual(raw.input_fit, { truncated: true, original_chars: 2500, kept_chars: 1440 });
});
test('createLayaClient.infer defaults to lossless when providers.json has no inputFit', async t => {
  const home = tempHome(t);
  const fake = fakeSpawn();
  const client = createLayaClient({ spawnImpl: fake.spawnImpl });
  t.after(() => client.close());
  fs.writeFileSync(path.join(home, 'model.safetensors'), 'fake');
  const laya = layaConfig(home).laya;
  await client.infer({ state: { task: 'x' }, questions: {} }, { laya }, { timeoutMs: 2000 });
  assert.equal(fake.messages[0].inputFit, 'lossless');
});

// --- provenance / inputFit ----------------------------------------------------
// input_fit is a per-request OUTCOME, not provenance: src/training/schema.mjs validateProvenance
// only allows the listed keys with string values, so it must never appear inside n.provenance
// (that is exactly the regression this test guards against). The MODE (lossless vs task-head) is
// instead expressed through provenance.preprocessing_version, which stays a plain string.
test('normalizeInference keeps provenance string-only (mode via preprocessing_version) and puts the per-request outcome on n.inputFit', () => {
  const home = '/tmp';
  const l = layaConfig(home, { inputFit: 'task-head' }).laya;
  const req = buildRequest();
  const rawTruncated = { identity: { model: l.model, checkpoint: l.checkpoint, runtime_version: l.runtimeVersion, device: l.device, precision: 'torch.float32' },
    answers: { worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } } }, model: l.model, usage: { input_tokens: 5, output_tokens: 0 },
    input_fit: { truncated: true, original_chars: 900, kept_chars: 400 } };
  const n1 = normalizeInference('laya', rawTruncated, req, { minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0, questionOrder: undefined }, { laya: l });
  assert.equal(n1.provenance.input_fit, undefined);
  assert.equal(typeof n1.provenance.preprocessing_version, 'string');
  assert.equal(n1.provenance.preprocessing_version, 'official-laya-0.3.4-task-head-v1');
  assert.deepEqual(n1.inputFit, { truncated: true, original_chars: 900, kept_chars: 400 });

  const rawUntouched = { ...rawTruncated, input_fit: undefined };
  const n2 = normalizeInference('laya', rawUntouched, req, { minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0 }, { laya: l });
  assert.deepEqual(n2.inputFit, { truncated: false });

  const lossless = layaConfig(home).laya; // default inputFit: 'lossless'
  const n3 = normalizeInference('laya', { ...rawTruncated, identity: { ...rawTruncated.identity, checkpoint: lossless.checkpoint }, input_fit: undefined },
    req, { minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0 }, { laya: lossless });
  assert.equal(n3.provenance.preprocessing_version, 'official-laya-0.3.4-lossless-v1');

  // A malformed worker input_fit (bad types, kept > original) is treated as untruncated, never propagated.
  for (const bad of [{ truncated: true }, { truncated: true, original_chars: 'x', kept_chars: 1 }, { truncated: true, original_chars: 5, kept_chars: 10 }, 'not-an-object', 42]) {
    const n = normalizeInference('laya', { ...rawTruncated, input_fit: bad }, req, { minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0 }, { laya: l });
    assert.deepEqual(n.inputFit, { truncated: false }, JSON.stringify(bad));
  }
});

// --- metrics aggregation -------------------------------------------------------
test('metrics surfaces a routeInputFitTruncated count per provider from decision telemetry', async t => {
  const home = tempHome(t);
  setMode(home, 'on', {});
  const env = { TYPESAFE_API_KEY: KEY };
  const l = layaConfig(home).laya;
  writeProviders(home, { version: 1, provider: 'laya', laya: l });
  fs.writeFileSync(path.join(home, 'model.safetensors'), 'fake');
  let call = 0;
  const provider = async () => {
    call++;
    return { identity: { model: l.model, checkpoint: l.checkpoint, runtime_version: l.runtimeVersion, device: l.device, precision: 'torch.float32' },
      answers: { worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } } }, model: l.model, usage: { input_tokens: 5, output_tokens: 0 },
      input_fit: call === 1 ? { truncated: true, original_chars: 900, kept_chars: 400 } : { truncated: false } };
  };
  const engine = createDecisionEngine({ home, env, provider });
  t.after(() => engine.close());
  await engine.decide({ ...buildRequest(), trace: buildTrace() });
  await engine.decide({ ...buildRequest(), trace: buildTrace() });
  const metrics = readMetrics(home, 1);
  assert.equal(metrics.providerActivity.laya.routeInputFitTruncated, 1);
});

// --- training capture regression: input_fit must never land in provenance --------------------
// src/training/schema.mjs validateProvenance() only allows the listed keys with STRING values
// (text(p[key], 200) for every key). Putting the input_fit object directly into provenance makes
// EVERY laya decision fail capture once inputFit is used, silently, since capture failures are
// swallowed into trainingCapture.stored=false rather than thrown to the caller.
test('training capture stores a laya decision (both truncated and untruncated input_fit) without INVALID_TRAINING_SCHEMA', async t => {
  const home = tempHome(t);
  const store = createTrainingStore({ home });
  store.setCapture(true); store.setMinLabels(1);
  setMode(home, 'on', {});
  const p = layaConfig(home, { inputFit: 'task-head' });
  writeProviders(home, p);
  fs.writeFileSync(path.join(home, 'model.safetensors'), 'fake');
  const l = p.laya;
  const identity = { model: l.model, checkpoint: l.checkpoint, runtime_version: l.runtimeVersion, device: l.device, precision: 'torch.float32' };
  const answers = { worker: { type: 'choice', choice: 'light', confidence: .97, probabilities: { light: .99, strong: .01 } } };
  for (const inputFit of [{ truncated: true, original_chars: 900, kept_chars: 400 }, { truncated: false }]) {
    const raw = { identity, answers, model: l.model, usage: { input_tokens: 5, output_tokens: 0 }, input_fit: inputFit };
    const engine = createDecisionEngine({ home, env: {}, provider: async () => raw, training: store });
    t.after(() => engine.close());
    const result = await engine.decide({ ...buildRequest(), trace: buildTrace() });
    assert.equal(result.trainingCapture.stored, true, `inputFit=${JSON.stringify(inputFit)} reason=${result.trainingCapture.reason}`);
  }
});
