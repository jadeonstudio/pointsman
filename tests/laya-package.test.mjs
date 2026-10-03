import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import http from 'node:http';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { randomUUID, createHash } from 'node:crypto';
import { atomicWrite, readText } from '../src/storage.mjs';
import { digest } from '../src/training/schema.mjs';
import { buildDataset } from '../src/training/dataset.mjs';
import { fixture as trainingFixture, trace, outcome, REF } from './training-helpers.mjs';
import { registerCheckpoint, freezeHoldout, qualifyCandidate, loadCandidate, loadQualification,
  loadPublished, adoptCandidate, layaStatus, rollbackLaya, publishedDir, LAYA_RUNTIME_VERSION } from '../src/training/laya-lifecycle.mjs';
import { packageCandidate, pullCandidate } from '../src/training/laya-package.mjs';
import { TRAINING_HELP } from '../src/training/cli.mjs';
import { decisionScope } from '../src/inference.mjs';

test('CLI help text lists laya package/pull/adopt', () => {
  assert.match(TRAINING_HELP, /laya package --candidate HASH --out DIR/);
  assert.match(TRAINING_HELP, /laya pull --repo OWNER\/NAME --revision SHA40/);
  assert.match(TRAINING_HELP, /laya adopt --candidate HASH/);
});

// --- fixtures (mirrors tests/laya-lifecycle.test.mjs's synthetic-checkpoint pattern) --------------
function makeCheckpointDir(root, tag = 'weights') {
  const dir = fs.mkdtempSync(path.join(root, 'ckpt-'));
  fs.writeFileSync(path.join(dir, 'model.safetensors'), tag);
  fs.writeFileSync(path.join(dir, 'rl_agent_config.json'), '{}');
  fs.mkdirSync(path.join(dir, 'encoder'));
  fs.writeFileSync(path.join(dir, 'encoder', 'config.json'), '{}');
  fs.mkdirSync(path.join(dir, 'tokenizer'));
  fs.writeFileSync(path.join(dir, 'tokenizer', 'tokenizer.json'), '{}');
  fs.writeFileSync(path.join(dir, 'tokenizer', 'tokenizer_config.json'), '{}');
  return dir;
}
function fakeFingerprint(_python, dir) {
  const records = [];
  const walk = d => { for (const name of fs.readdirSync(d).sort()) { const p = path.join(d, name); const st = fs.statSync(p); if (st.isDirectory()) walk(p); else records.push([path.relative(dir, p), fs.readFileSync(p, 'utf8')]); } };
  walk(dir);
  return digest(records);
}
const splitFor = requestHash => { const n = parseInt(digest('input:' + requestHash).slice(0, 8), 16) % 100; return n < 80 ? 'train' : n < 90 ? 'calibration' : 'test'; };
function buildRequest(i, want, conf) {
  return { purpose: 'route', risk: 'routine', state: { task: `synthetic pull case ${i}`, want, conf },
    questions: { worker: { type: 'choice', instructions: 'Choose the worker for this bounded task.', criteria: { light: 'Narrow known scope', strong: 'Unclear or broad scope' } } } };
}
function findIndex(target, want, conf, cursor) {
  for (let i = cursor.n; i < cursor.n + 20000; i++) if (splitFor(digest(buildRequest(i, want, conf))) === target) { cursor.n = i + 1; return i; }
  throw new Error('NO_INDEX_FOUND');
}
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
function calibrationRows() {
  const rows = [];
  for (let k = 0; k < 5; k++) rows.push({ want: 'light', conf: .95, truth: 'light' });
  for (let k = 0; k < 5; k++) rows.push({ want: 'light', conf: .55, truth: 'strong' });
  return rows;
}
function evalRows() {
  const rows = [];
  for (let k = 0; k < 6; k++) rows.push({ want: 'light', conf: .95, truth: 'light' });
  for (let k = 0; k < 2; k++) rows.push({ want: 'light', conf: .3, truth: 'strong' });
  return rows;
}
function buildRouteDataset(f) {
  const cursor = { n: 0 };
  addSamples(f, cursor, 'calibration', calibrationRows());
  addSamples(f, cursor, 'test', evalRows());
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.built, undefined, JSON.stringify(built));
  return built.dataset_version;
}
function fakeLayaClient() {
  return { status: () => ({ running: false }), close: () => {}, prepare: async () => ({}),
    async infer(payload, settings) {
      const laya = settings.laya;
      const [qid, q] = Object.entries(payload.questions)[0];
      const want = payload.state.want, conf = payload.state.conf;
      const other = Object.keys(q.criteria).find(k => k !== want);
      return { identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device,
        precision: laya.precision === 'fp16' ? 'torch.float16' : 'torch.float32' },
        answers: { [qid]: { type: 'choice', choice: want, confidence: conf, probabilities: { [want]: .99, [other]: .01 } } },
        usage: { input_tokens: 5, output_tokens: 0 } };
    } };
}
function tmpRoot(t) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-pkg-')));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}
/** Registers + qualifies a real candidate (fake fingerprint/laya client, no python/network) so package/pull/adopt have something real to work with. */
async function buildQualifiedCandidate(t) {
  const f = trainingFixture(t);
  const version = buildRouteDataset(f);
  const holdout = freezeHoldout(f.home, { datasetVersion: version, name: 'h1' });
  const reg = registerCheckpoint(f.home, { checkpointDir: makeCheckpointDir(tmpRoot(t)), model: 'laya/pkg-test', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qualification = await qualifyCandidate(f.home, { candidateHash: reg.checkpoint, datasetVersion: version, holdoutName: holdout.name, layaClient: fakeLayaClient(),
    targetAccuracy: .75, minCoverage: .3, minCalibration: 5, minTest: 5, minLowerBound: .5 });
  assert.equal(qualification.qualified, true);
  // Existing transfer cases replay an already-issued legacy artifact. New
  // generic purpose-only qualifications are refused by the separate test below.
  delete qualification.decisionIdentity; delete qualification.familyQualification; delete qualification.operationalEligible;
  atomicWrite(path.join(f.home,'laya','qualifications',`${reg.checkpoint}.json`),JSON.stringify(qualification));
  return { home: f.home, checkpoint: reg.checkpoint, qualification };
}

test('package/pull/adopt retain the bounded decision proof and operational gate', async t => {
  const pub=await buildQualifiedCandidate(t), candidate=loadCandidate(pub.home,pub.checkpoint), actual=decisionScope(buildRequest(1,'light',.95),'worker');
  const qualification={...pub.qualification, operationalEligible:true, familyQualification:{status:'BOUNDED_ENVELOPE_QUALIFIED',families:{'fixture-bounded:choice':{passed:true}}},
    decisionIdentity:{version:1,checkpoint:pub.checkpoint,runtimeVersion:candidate.runtimeVersion,precision:candidate.precision,inputFit:'task-head',calibrationVersion:pub.qualification.calibrationVersion,
      scopes:[{...actual,familyId:'fixture-bounded',familyRevision:'fixture-v1',stateBuilderRevision:`payload-state-schema-v1:${actual.stateSchemaHash}`,threshold:.9}]}};
  candidate.inputFit='task-head'; atomicWrite(path.join(pub.home,'laya','candidates',`${pub.checkpoint}.json`),JSON.stringify(candidate));
  atomicWrite(path.join(pub.home,'laya','qualifications',`${pub.checkpoint}.json`),JSON.stringify(qualification));
  const out=path.join(tmpRoot(t),'out'); packageCandidate(pub.home,{candidateHash:pub.checkpoint,outDir:out});
  const manifest=JSON.parse(fs.readFileSync(path.join(out,'pointsman.json'))), home=tmpRoot(t);
  assert.deepEqual(manifest.qualification.decisionIdentity,qualification.decisionIdentity);
  const fetchImpl=async url=>{const name=new URL(url).pathname.split('/').slice(5).join('/');return new Response(fs.readFileSync(path.join(out,name)),{status:200});};
  await pullCandidate(home,{repo:'fixture/model',revision:'b'.repeat(40),python:'/usr/bin/python3',device:'cpu',fetchImpl,
    registerImpl:(h,args)=>registerCheckpoint(h,{...args,fingerprintImpl:fakeFingerprint})});
  const published=loadPublished(home,pub.checkpoint); assert.equal(published.qualification.operationalEligible,true);
  assert.deepEqual(published.qualification.familyQualification,qualification.familyQualification);
  adoptCandidate(home,{candidateHash:pub.checkpoint});
  const active=JSON.parse(fs.readFileSync(path.join(home,'providers.json'))).laya;
  assert.deepEqual(active.qualification.decisionIdentity,qualification.decisionIdentity);assert.equal(active.inputFit,'task-head');
  rollbackLaya(home); assert.equal(JSON.parse(fs.readFileSync(path.join(home,'providers.json'))).laya,null);
});
test('package refuses a new all-unqualified envelope before creating output', async t => {
  const pub=await buildQualifiedCandidate(t),candidate=loadCandidate(pub.home,pub.checkpoint),q={...pub.qualification,operationalEligible:false,
    decisionIdentity:{version:1,checkpoint:pub.checkpoint,runtimeVersion:candidate.runtimeVersion,precision:candidate.precision,inputFit:candidate.inputFit,calibrationVersion:pub.qualification.calibrationVersion,scopes:[]}};
  atomicWrite(path.join(pub.home,'laya','qualifications',`${pub.checkpoint}.json`),JSON.stringify(q));
  const out=path.join(tmpRoot(t),'out'); assert.throws(()=>packageCandidate(pub.home,{candidateHash:pub.checkpoint,outDir:out}),/NO_QUALIFIED_DECISION_FAMILY/);
  assert.equal(fs.existsSync(out),false);
});

// ============================== package ==============================

test('package writes files + a manifest with correct sha256/size and no absolute paths', async t => {
  const { home, checkpoint } = await buildQualifiedCandidate(t);
  const outDir = path.join(tmpRoot(t), 'out');
  const result = packageCandidate(home, { candidateHash: checkpoint, outDir, repo: 'owner/name' });
  assert.equal(result.candidate, checkpoint);
  assert.equal(result.repo, 'owner/name');
  assert.ok(result.files >= 1);
  const manifest = JSON.parse(fs.readFileSync(path.join(outDir, 'pointsman.json'), 'utf8'));
  assert.equal(manifest.format, 'pointsman-model-v1');
  assert.equal(manifest.checkpoint, checkpoint);
  assert.equal(manifest.runtimeVersion, LAYA_RUNTIME_VERSION);
  assert.equal(manifest.qualification.checkpoint, checkpoint);
  assert.deepEqual(Object.keys(manifest.qualification).sort(),
    ['calibrationVersion', 'checkpoint', 'minChoiceProbability', 'minConfidence', 'noulCertainty', 'precision', 'purposes'].sort());
  for (const f of manifest.files) {
    const bytes = fs.readFileSync(path.join(outDir, f.path));
    assert.equal(bytes.length, f.size);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), f.sha256);
  }
  const raw = JSON.stringify(manifest);
  assert.ok(!raw.includes(home));
  assert.ok(!raw.includes(os.tmpdir()));
  assert.equal(fs.existsSync(path.join(outDir, 'README.md')), false);
});

test('package refuses an unqualified candidate', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/unq', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.throws(() => packageCandidate(home, { candidateHash: reg.checkpoint, outDir: path.join(root, 'out') }), /NOT_QUALIFIED/);
});

test('package refuses a non-empty out directory', async t => {
  const { home, checkpoint } = await buildQualifiedCandidate(t);
  const outDir = path.join(tmpRoot(t), 'out');
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'existing.txt'), 'x');
  assert.throws(() => packageCandidate(home, { candidateHash: checkpoint, outDir }), /LAYA_PACKAGE_OUT_NOT_EMPTY/);
});

// ============================== pull ==============================

function startServer(routes) {
  const server = http.createServer((req, res) => {
    const key = req.url.split('?')[0];
    const handler = routes.get(key);
    if (!handler) { res.writeHead(404); res.end(); return; }
    handler(req, res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port, routes })));
}
function localFetch(getPort) {
  return (url, init) => {
    const u = new URL(url);
    return fetch(`http://127.0.0.1:${getPort()}${u.pathname}${u.search}`, init);
  };
}
function jsonRoute(body) { return (req, res) => { const b = Buffer.from(JSON.stringify(body)); res.writeHead(200, { 'content-type': 'application/json', 'content-length': b.length }); res.end(b); }; }
function bytesRoute(buf) { return (req, res) => { res.writeHead(200, { 'content-length': buf.length }); res.end(buf); }; }
function redirectRoute(location) { return (req, res) => { res.writeHead(302, { location }); res.end(); }; }

async function preparePullFixture(t) {
  const pub = await buildQualifiedCandidate(t);
  const outDir = path.join(tmpRoot(t), 'publish');
  const packaged = packageCandidate(pub.home, { candidateHash: pub.checkpoint, outDir, repo: 'owner/model' });
  const manifest = JSON.parse(fs.readFileSync(path.join(outDir, 'pointsman.json'), 'utf8'));
  const revision = 'a'.repeat(40);
  const routes = new Map();
  routes.set(`/owner/model/resolve/${revision}/pointsman.json`, jsonRoute(manifest));
  for (const f of manifest.files) {
    routes.set(`/owner/model/resolve/${revision}/${f.path}`, bytesRoute(fs.readFileSync(path.join(outDir, f.path))));
  }
  const { server, port } = await startServer(routes);
  t.after(() => server.close());
  const consumerHome = fs.mkdtempSync(path.join(tmpRoot(t), 'consumer-'));
  return { consumerHome, revision, manifest, routes, getPort: () => port };
}
function fakeRegister(expectedCheckpoint) {
  return (home, opts) => ({ checkpoint: expectedCheckpoint, candidate: '/fake/candidate.json', modelPath: opts.checkpointDir, reused: false });
}

test('pull happy path: downloads, verifies sha256, registers and publishes qualification', async t => {
  const { consumerHome, revision, manifest, getPort } = await preparePullFixture(t);
  const result = await pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: {}, registerImpl: fakeRegister(manifest.checkpoint) });
  assert.deepEqual(result, { candidate: manifest.checkpoint, repo: 'owner/model', revision, files: manifest.files.length,
    bytes: manifest.files.reduce((s, f) => s + f.size, 0), registered: true });
  const published = loadPublished(consumerHome, manifest.checkpoint);
  assert.equal(published.repo, 'owner/model');
  assert.equal(published.revision, revision);
  assert.deepEqual(published.qualification, manifest.qualification);
  for (const f of manifest.files) {
    const dest = path.join(consumerHome, 'laya', 'downloads', revision, f.path);
    assert.equal(fs.statSync(dest).mode & 0o777, 0o600);
  }
});

test('pull rejects a non-40-hex revision without any network call', async t => {
  const { consumerHome } = await preparePullFixture(t);
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision: 'main',
    fetchImpl: () => { throw new Error('must not fetch'); } }), /PINNED_REVISION_REQUIRED/);
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision: 'a'.repeat(39),
    fetchImpl: () => { throw new Error('must not fetch'); } }), /PINNED_REVISION_REQUIRED/);
});

test('pull rejects a malformed repo', async t => {
  const { consumerHome } = await preparePullFixture(t);
  await assert.rejects(pullCandidate(consumerHome, { repo: '../evil', revision: 'a'.repeat(40),
    fetchImpl: () => { throw new Error('must not fetch'); } }), /INVALID_REPO/);
});

test('pull rejects a redirect to an untrusted host (manifest fetch)', async t => {
  const { consumerHome, revision, routes, getPort } = await preparePullFixture(t);
  routes.set(`/owner/model/resolve/${revision}/pointsman.json`, redirectRoute('http://evil.example.com/pointsman.json'));
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, fetchImpl: localFetch(getPort), env: {} }),
    /UNTRUSTED_DOWNLOAD_HOST/);
});

test('pull rejects a redirect that downgrades to plain http on an allowed host', async t => {
  const { consumerHome, revision, routes, getPort } = await preparePullFixture(t);
  routes.set(`/owner/model/resolve/${revision}/pointsman.json`, redirectRoute('http://cdn-lfs.hf.co/pointsman.json'));
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, fetchImpl: localFetch(getPort), env: {} }),
    /UNTRUSTED_DOWNLOAD_HOST/);
});

test('pull drops the HF_TOKEN Authorization header on a cross-host redirect but sends it to huggingface.co', async t => {
  const { consumerHome, revision, manifest, routes, getPort } = await preparePullFixture(t);
  const seenAuth = {};
  const firstFile = manifest.files[0];
  const realFilePath = `/owner/model/resolve/${revision}/${firstFile.path}`;
  const mirrorPath = `/lfs-mirror/${firstFile.path}`;
  const originalFileHandler = routes.get(realFilePath); // the plain bytesRoute, captured before any wrapping
  // Wrap every existing route to record the Authorization header it received, then make the first
  // file's real route redirect once through a distinct path pretending to be an LFS host.
  for (const [key, handler] of routes) {
    routes.set(key, (req, res) => { seenAuth[key] = req.headers.authorization; handler(req, res); });
  }
  routes.set(mirrorPath, (req, res) => { seenAuth[mirrorPath] = req.headers.authorization; originalFileHandler(req, res); });
  routes.set(realFilePath, (req, res) => { seenAuth[realFilePath] = req.headers.authorization; res.writeHead(302, { location: `https://cdn-lfs.huggingface.co${mirrorPath}` }); res.end(); });
  const result = await pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: { HF_TOKEN: 'super-secret-token' }, registerImpl: fakeRegister(manifest.checkpoint) });
  assert.equal(result.registered, true);
  assert.equal(seenAuth[`/owner/model/resolve/${revision}/pointsman.json`], 'Bearer super-secret-token');
  assert.equal(seenAuth[realFilePath], 'Bearer super-secret-token'); // still the original huggingface.co host
  assert.equal(seenAuth[mirrorPath], undefined); // cross-origin hop: token must not be forwarded
});

test('pull never echoes HF_TOKEN in its returned JSON', async t => {
  const { consumerHome, revision, manifest, getPort } = await preparePullFixture(t);
  const result = await pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: { HF_TOKEN: 'super-secret-token' }, registerImpl: fakeRegister(manifest.checkpoint) });
  assert.ok(!JSON.stringify(result).includes('super-secret-token'));
});

test('pull rejects a traversal path inside the manifest', async t => {
  const { consumerHome, revision, manifest, routes, getPort } = await preparePullFixture(t);
  const tampered = { ...manifest, files: [{ path: '../../etc/passwd', sha256: 'a'.repeat(64), size: 1 }] };
  routes.set(`/owner/model/resolve/${revision}/pointsman.json`, jsonRoute(tampered));
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, fetchImpl: localFetch(getPort), env: {} }),
    /LAYA_MANIFEST_PATH_REJECTED/);
});

test('pull rejects a sha256 mismatch and registers nothing', async t => {
  const { consumerHome, revision, manifest, routes, getPort } = await preparePullFixture(t);
  const firstPath = `/owner/model/resolve/${revision}/${manifest.files[0].path}`;
  routes.set(firstPath, bytesRoute(Buffer.from('corrupted-bytes-not-matching-declared-sha')));
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: {}, registerImpl: fakeRegister(manifest.checkpoint) }), /LAYA_PULL_(HASH|SIZE)_MISMATCH/);
  assert.equal(loadPublished(consumerHome, manifest.checkpoint), null);
});

test('pull rejects a size mismatch', async t => {
  const { consumerHome, revision, manifest, routes, getPort } = await preparePullFixture(t);
  const firstPath = `/owner/model/resolve/${revision}/${manifest.files[0].path}`;
  routes.set(firstPath, bytesRoute(Buffer.alloc(manifest.files[0].size + 5, 1)));
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: {}, registerImpl: fakeRegister(manifest.checkpoint) }), /LAYA_PULL_SIZE_MISMATCH/);
});

test('pull dry-run only validates the manifest and downloads nothing', async t => {
  const { consumerHome, revision, manifest, routes, getPort } = await preparePullFixture(t);
  // Any file route would throw if actually requested during a dry run.
  for (const f of manifest.files) routes.set(`/owner/model/resolve/${revision}/${f.path}`, () => { throw new Error('dry-run must not download files'); });
  const result = await pullCandidate(consumerHome, { repo: 'owner/model', revision, dryRun: true, fetchImpl: localFetch(getPort), env: {} });
  assert.deepEqual(result, { candidate: manifest.checkpoint, repo: 'owner/model', revision, files: manifest.files.length,
    bytes: manifest.files.reduce((s, f) => s + f.size, 0), dryRun: true });
  assert.equal(fs.existsSync(path.join(consumerHome, 'laya', 'downloads')), false);
});

test('pull fails with CHECKPOINT_MISMATCH when the registered fingerprint differs, and registers nothing', async t => {
  const { consumerHome, revision, manifest, getPort } = await preparePullFixture(t);
  const wrongHash = 'b'.repeat(64);
  await assert.rejects(pullCandidate(consumerHome, { repo: 'owner/model', revision, python: '/usr/bin/python3', device: 'cpu',
    fetchImpl: localFetch(getPort), env: {}, registerImpl: fakeRegister(wrongHash) }), /CHECKPOINT_MISMATCH/);
  assert.equal(loadPublished(consumerHome, manifest.checkpoint), null);
  assert.equal(loadPublished(consumerHome, wrongHash), null);
});

// ============================== adopt ==============================

function publishRecord(home, checkpoint, qualification, repo = 'owner/model', revision = 'a'.repeat(40)) {
  fs.mkdirSync(publishedDir(home), { recursive: true, mode: 0o700 });
  atomicWrite(path.join(publishedDir(home), `${checkpoint}.json`), JSON.stringify({ repo, revision, qualification, evaluation: null }, null, 2) + '\n');
}

test('adopt writes providers.json from the published qualification (incl. routeGate), records history, and rollback undoes it', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/adopt', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qualification = { checkpoint: reg.checkpoint, calibrationVersion: 'v1:q-' + 'a'.repeat(32), purposes: ['route'],
    minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9, precision: 'fp32', routeGate: { method: 'decision-v1', tierCoverage: .9, maxHostProbability: .1 } };
  publishRecord(home, reg.checkpoint, qualification);
  const result = adoptCandidate(home, { candidateHash: reg.checkpoint });
  assert.equal(result.adopted, true);
  assert.equal(result.checkpoint, reg.checkpoint);
  assert.deepEqual(result.source, { repo: 'owner/model', revision: 'a'.repeat(40) });
  const providers = JSON.parse(readText(path.join(home, 'providers.json'), { optional: true }));
  assert.equal(providers.laya.checkpoint, reg.checkpoint);
  assert.deepEqual(providers.laya.qualification, qualification);
  assert.equal(providers.provider, 'jev'); // provider selection is never changed by adopt
  const history = readText(path.join(home, 'laya', 'history.jsonl')).trim().split('\n').map(JSON.parse);
  assert.equal(history.at(-1).action, 'adopt');
  assert.deepEqual(history.at(-1).source, { repo: 'owner/model', revision: 'a'.repeat(40) });
  const rolledBack = rollbackLaya(home);
  assert.equal(rolledBack.restored, null);
  const after = JSON.parse(readText(path.join(home, 'providers.json'), { optional: true }));
  assert.equal(after.laya, null);
});

test('adopt refuses without a published record', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/noadopt', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  assert.throws(() => adoptCandidate(home, { candidateHash: reg.checkpoint }), /PUBLISHED_RECORD_NOT_FOUND/);
});

test('adopt refuses a precision mismatch between the candidate and the published qualification', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/precmismatch', device: 'cpu', python: '/usr/bin/python3', precision: 'fp32', fingerprintImpl: fakeFingerprint });
  const qualification = { checkpoint: reg.checkpoint, calibrationVersion: 'v1:q-' + 'a'.repeat(32), purposes: ['route'],
    minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9, precision: 'fp16' };
  publishRecord(home, reg.checkpoint, qualification);
  assert.throws(() => adoptCandidate(home, { candidateHash: reg.checkpoint }), /PUBLISHED_QUALIFICATION_MISMATCH/);
});

// ============================== CLI wiring ==============================

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const cli = (home, args) => spawnSync(process.execPath, [path.join(ROOT, 'bin/pointsman.mjs'), ...args], {
  encoding: 'utf8', timeout: 8000, env: { ...process.env, HOME: home, POINTSMAN_HOME: home, TYPESAFE_API_KEY: '', POINTSMAN_DISABLE: '0' },
});

test('CLI wires laya package/pull/adopt flags through to the library functions', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  assert.equal(cli(home, ['laya', 'package', '--out', path.join(root, 'out')]).status, 2); // missing --candidate
  assert.equal(cli(home, ['laya', 'package', '--candidate', 'a'.repeat(64)]).status, 2); // missing --out
  assert.equal(cli(home, ['laya', 'pull', '--repo', 'owner/model']).status, 2); // missing --revision
  assert.equal(cli(home, ['laya', 'adopt']).status, 2); // missing --candidate
  const bad = cli(home, ['laya', 'package', '--candidate', 'a'.repeat(64), '--out', path.join(root, 'out2')]);
  assert.equal(bad.status, 2);
  assert.match(bad.stderr, /LAYA_CANDIDATE_NOT_FOUND/);
});

test('laya status reports published repo/revision for candidates that have them', t => {
  const root = tmpRoot(t);
  const home = fs.mkdtempSync(path.join(root, 'home-'));
  const reg = registerCheckpoint(home, { checkpointDir: makeCheckpointDir(root), model: 'laya/statuspub', device: 'cpu', python: '/usr/bin/python3', fingerprintImpl: fakeFingerprint });
  const qualification = { checkpoint: reg.checkpoint, calibrationVersion: 'v1:q-' + 'a'.repeat(32), purposes: ['route'],
    minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9, precision: 'fp32' };
  publishRecord(home, reg.checkpoint, qualification, 'owner/pub-model', 'c'.repeat(40));
  const status = layaStatus(home);
  const row = status.candidates.find(c => c.checkpoint === reg.checkpoint);
  assert.deepEqual(row.published, { repo: 'owner/pub-model', revision: 'c'.repeat(40) });
});
