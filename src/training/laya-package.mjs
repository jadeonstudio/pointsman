// Publisher/consumer transfer of a qualified Laya checkpoint through a Hugging Face repo.
// `package` only ever writes a local folder (never uploads); `pull` only ever reads a pinned,
// already-published revision (never a branch/tag, never an unlisted host). Neither command starts
// training, changes providers.json's active checkpoint by itself (see `adopt` in laya-lifecycle.mjs),
// or transmits private evaluation data/task text -- the manifest carries only content-free numbers.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { randomUUID, createHash } from 'node:crypto';
import { ensureDir, atomicWrite, noSymlinks } from '../storage.mjs';
import { fail } from '../constants.mjs';
import { HASH, only } from './schema.mjs';
import { loadProviderConfig, validateProviderConfig } from '../inference.mjs';
import { LAYA_RUNTIME_VERSION, MODEL_RE, layaRoot, checkpointsDir, candidatesDir, publishedDir,
  loadCandidate, loadQualification, operationalFamiliesPassed, registerCheckpoint as defaultRegisterCheckpoint } from './laya-lifecycle.mjs';

const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const REVISION_RE = /^[0-9a-f]{40}$/i;
// huggingface.co (the manifest and default resolve host) plus its LFS/CDN subdomains and the hf.co
// short domain; never anything else, so a redirect cannot be used to exfiltrate HF_TOKEN or pull an
// arbitrary payload from an attacker-controlled host.
const ALLOWED_HOST_RE = /^(?:[a-z0-9-]+\.)*(?:huggingface\.co|hf\.co)$/i;
const MANIFEST_PATH_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}$/;
const MAX_FILE_SIZE = 4 * 1024 ** 3;
const MAX_MANIFEST_FILES = 20;
// Exactly what a registered checkpoint directory (see registerCheckpoint) may contain; package copies
// only these, exactly what exists, and pull only ever writes these into a fresh download directory.
const PACKAGE_FILES = ['model.safetensors', 'encoder/config.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'rl_agent_config.json'];

const sha256Hex = buf => createHash('sha256').update(buf).digest('hex');
function sha256File(file) { return sha256Hex(fs.readFileSync(file)); }

// --- package (publisher side; never uploads) -----------------------------
export function packageCandidate(home, { candidateHash, outDir, repo } = {}) {
  if (!HASH.test(candidateHash)) fail('INVALID_CANDIDATE_HASH');
  if (typeof outDir !== 'string' || !path.isAbsolute(outDir)) fail('LAYA_PACKAGE_OUT_REQUIRED');
  if (repo !== undefined && repo !== null && !REPO_RE.test(repo)) fail('INVALID_REPO');
  const candidate = loadCandidate(home, candidateHash);
  const qualification = loadQualification(home, candidateHash);
  if (!qualification || !qualification.qualified || !qualification.purposes.length) fail('NOT_QUALIFIED');
  if (!operationalFamiliesPassed(qualification)) fail('NO_QUALIFIED_DECISION_FAMILY');
  validateProviderConfig({ version: 1, provider: 'laya', laya: { ...candidate, qualification: {
    checkpoint: qualification.checkpoint, calibrationVersion: qualification.calibrationVersion, purposes: qualification.purposes,
    minConfidence: qualification.minConfidence, minChoiceProbability: qualification.minChoiceProbability, noulCertainty: qualification.noulCertainty,
    ...(qualification.decisionIdentity ? { decisionIdentity: qualification.decisionIdentity } : {}) } } });
  let outStat = null;
  try { outStat = fs.lstatSync(outDir); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  if (outStat) {
    if (outStat.isSymbolicLink()) fail('UNSAFE_SYMLINK');
    if (!outStat.isDirectory()) fail('LAYA_PACKAGE_OUT_NOT_DIRECTORY');
    if (fs.readdirSync(outDir).length) fail('LAYA_PACKAGE_OUT_NOT_EMPTY');
  } else {
    fs.mkdirSync(outDir, { recursive: true, mode: 0o700 });
  }
  const srcRoot = candidate.modelPath;
  noSymlinks(srcRoot);
  const files = [];
  for (const rel of PACKAGE_FILES) {
    const src = path.join(srcRoot, rel);
    let st; try { st = fs.lstatSync(src); } catch { continue; } // copy exactly what exists
    if (st.isSymbolicLink()) fail('LAYA_MODEL_SYMLINK_REFUSED');
    if (!st.isFile()) continue;
    const buf = fs.readFileSync(src);
    const dest = path.join(outDir, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true, mode: 0o700 });
    fs.writeFileSync(dest, buf, { mode: 0o600 });
    files.push({ path: rel, sha256: sha256Hex(buf), size: buf.length });
  }
  if (!files.length) fail('LAYA_PACKAGE_EMPTY');
  const precision = qualification.precision ?? candidate.precision ?? 'fp32';
  const manifestQualification = { checkpoint: qualification.checkpoint, calibrationVersion: qualification.calibrationVersion,
    purposes: qualification.purposes, minConfidence: qualification.minConfidence, minChoiceProbability: qualification.minChoiceProbability,
    noulCertainty: qualification.noulCertainty, precision, ...(qualification.routeGate ? { routeGate: qualification.routeGate } : {}),
    ...(qualification.decisionIdentity ? { decisionIdentity: qualification.decisionIdentity } : {}) };
  const evaluation = { dataset_version: qualification.dataset_version, holdout: { name: qualification.holdout, sha256: qualification.holdout_sha256 },
    route_decision: qualification.evidence?.route?.decision ?? null, by_question_test: qualification.by_question?.test ?? {},
    ...(qualification.decisionIdentity ? { operationalEligible: qualification.operationalEligible, familyQualification: qualification.familyQualification } : {}) };
  const manifest = { format: 'pointsman-model-v1', model: candidate.model, checkpoint: candidateHash, runtimeVersion: candidate.runtimeVersion,
    precision, inputFit: candidate.inputFit ?? 'lossless', files, qualification: manifestQualification, evaluation, generated_at: new Date().toISOString() };
  fs.writeFileSync(path.join(outDir, 'pointsman.json'), JSON.stringify(manifest, null, 2) + '\n', { mode: 0o600 });
  return { candidate: candidateHash, outDir, repo: repo ?? null, files: files.length, bytes: files.reduce((s, f) => s + f.size, 0) };
}

// --- pull (consumer side) -------------------------------------------------
function validateManifest(m) {
  only(m, ['format', 'model', 'checkpoint', 'runtimeVersion', 'precision', 'inputFit', 'files', 'qualification', 'evaluation', 'generated_at'],
    ['format', 'model', 'checkpoint', 'runtimeVersion', 'precision', 'inputFit', 'files', 'qualification']);
  if (m.format !== 'pointsman-model-v1') fail('LAYA_MANIFEST_INVALID');
  if (!HASH.test(m.checkpoint)) fail('LAYA_MANIFEST_INVALID');
  if (m.runtimeVersion !== LAYA_RUNTIME_VERSION) fail('LAYA_RUNTIME_VERSION_UNSUPPORTED');
  if (!['fp32', 'fp16'].includes(m.precision)) fail('LAYA_MANIFEST_INVALID');
  if (!['lossless', 'task-head'].includes(m.inputFit)) fail('LAYA_MANIFEST_INVALID');
  if (typeof m.model !== 'string' || !MODEL_RE.test(m.model)) fail('LAYA_MANIFEST_INVALID');
  if (!Array.isArray(m.files) || m.files.length < 1 || m.files.length > MAX_MANIFEST_FILES) fail('LAYA_MANIFEST_INVALID');
  const seen = new Set();
  for (const f of m.files) {
    only(f, ['path', 'sha256', 'size'], ['path', 'sha256', 'size']);
    if (typeof f.path !== 'string' || path.isAbsolute(f.path) || f.path.includes('\\') || f.path.split('/').includes('..') || !MANIFEST_PATH_RE.test(f.path)) fail('LAYA_MANIFEST_PATH_REJECTED');
    if (seen.has(f.path)) fail('LAYA_MANIFEST_INVALID');
    seen.add(f.path);
    if (!HASH.test(f.sha256)) fail('LAYA_MANIFEST_INVALID');
    if (!Number.isInteger(f.size) || f.size < 0 || f.size > MAX_FILE_SIZE) fail('LAYA_MANIFEST_INVALID');
  }
  if (!m.qualification || m.qualification.checkpoint !== m.checkpoint) fail('LAYA_MANIFEST_INVALID');
  const dummyAbs = path.join(os.tmpdir(), 'pointsman-manifest-check', randomUUID());
  try {
    validateProviderConfig({ version: 1, provider: 'jev', laya: { python: dummyAbs, modelPath: dummyAbs, model: m.model,
      checkpoint: m.checkpoint, runtimeVersion: m.runtimeVersion, device: 'cpu', precision: m.precision, inputFit: m.inputFit, qualification: m.qualification } });
    if (m.qualification.decisionIdentity && !operationalFamiliesPassed({ ...m.qualification,
      operationalEligible: m.evaluation?.operationalEligible, familyQualification: m.evaluation?.familyQualification })) fail('LAYA_MANIFEST_INVALID');
  } catch { fail('LAYA_MANIFEST_INVALID'); }
}
// Manual redirect handling (never `redirect: 'follow'`): the allowed-host check and the Authorization
// header decision are both re-evaluated at every hop from the URL we are about to request, so the
// token is attached only when the CURRENT hop's host is exactly huggingface.co and is dropped the
// instant a redirect (e.g. to an LFS/cdn-lfs host) points anywhere else -- it is never something we
// received and blindly forwarded.
async function fetchFollowingRedirects(url, { fetchImpl, timeoutMs, env }) {
  let current = new URL(url);
  for (let hop = 0; hop <= 5; hop++) {
    if (current.protocol !== 'https:' || !ALLOWED_HOST_RE.test(current.hostname)) fail('UNTRUSTED_DOWNLOAD_HOST');
    const headers = {};
    if (current.hostname.toLowerCase() === 'huggingface.co' && env.HF_TOKEN) headers.Authorization = `Bearer ${env.HF_TOKEN}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let res;
    try { res = await fetchImpl(current.toString(), { headers, redirect: 'manual', signal: controller.signal }); }
    catch { fail('LAYA_PULL_FETCH_FAILED'); }
    finally { clearTimeout(timer); }
    if ([301, 302, 303, 307, 308].includes(res.status)) {
      const location = res.headers.get('location');
      if (!location) fail('LAYA_PULL_FETCH_FAILED');
      current = new URL(location, current);
      continue;
    }
    return res;
  }
  fail('LAYA_PULL_TOO_MANY_REDIRECTS');
}
async function downloadToFile(res, tempPath, expectedSize) {
  const hash = createHash('sha256');
  let total = 0;
  const ws = fs.createWriteStream(tempPath, { mode: 0o600, flags: 'wx' });
  try {
    for await (const chunk of res.body) {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      total += buf.length;
      if (total > expectedSize) fail('LAYA_PULL_SIZE_MISMATCH');
      hash.update(buf);
      if (!ws.write(buf)) await once(ws, 'drain');
    }
    await new Promise((resolve, reject) => ws.end(err => (err ? reject(err) : resolve())));
  } catch (e) {
    await new Promise(resolve => ws.destroy(undefined, () => resolve()));
    throw e;
  }
  return { sha256: hash.digest('hex'), size: total };
}
export async function pullCandidate(home, { repo, revision, python, device, dryRun = false, fetchImpl = fetch,
  env = process.env, timeoutMs = 20000, maxManifestBytes = 262144, registerImpl = defaultRegisterCheckpoint } = {}) {
  if (typeof repo !== 'string' || !REPO_RE.test(repo)) fail('INVALID_REPO');
  if (typeof revision !== 'string' || !REVISION_RE.test(revision)) fail('PINNED_REVISION_REQUIRED');
  const manifestUrl = `https://huggingface.co/${repo}/resolve/${revision}/pointsman.json`;
  const res = await fetchFollowingRedirects(manifestUrl, { fetchImpl, timeoutMs, env });
  if (res.status !== 200) fail('LAYA_PULL_MANIFEST_NOT_FOUND');
  const declared = Number(res.headers.get('content-length') || 0);
  if (declared > maxManifestBytes) fail('LAYA_MANIFEST_TOO_LARGE');
  const ab = await res.arrayBuffer();
  if (ab.byteLength > maxManifestBytes) fail('LAYA_MANIFEST_TOO_LARGE');
  let manifest;
  try { manifest = JSON.parse(Buffer.from(ab).toString('utf8')); } catch { fail('LAYA_MANIFEST_INVALID'); }
  validateManifest(manifest);
  const totalBytes = manifest.files.reduce((s, f) => s + f.size, 0);
  if (dryRun) return { candidate: manifest.checkpoint, repo, revision, files: manifest.files.length, bytes: totalBytes, dryRun: true };
  const downloadDir = path.join(layaRoot(home), 'downloads', revision);
  ensureDir(layaRoot(home), true);
  ensureDir(path.join(layaRoot(home), 'downloads'), true);
  ensureDir(downloadDir, true);
  for (const f of manifest.files) {
    const dest = path.join(downloadDir, f.path);
    let reuse = false;
    try {
      const st = fs.lstatSync(dest);
      if (st.isFile() && st.size === f.size && sha256File(dest) === f.sha256) reuse = true;
      else fs.rmSync(dest, { force: true });
    } catch { /* not present yet */ }
    if (reuse) continue;
    fs.mkdirSync(path.dirname(dest), { recursive: true, mode: 0o700 });
    const fileUrl = `https://huggingface.co/${repo}/resolve/${revision}/${f.path}`;
    const fres = await fetchFollowingRedirects(fileUrl, { fetchImpl, timeoutMs, env });
    if (fres.status !== 200) fail('LAYA_PULL_FILE_NOT_FOUND');
    const temp = `${dest}.${randomUUID()}.tmp`;
    let result;
    try { result = await downloadToFile(fres, temp, f.size); }
    catch (e) { fs.rmSync(temp, { force: true }); throw e; }
    if (result.size !== f.size) { fs.rmSync(temp, { force: true }); fail('LAYA_PULL_SIZE_MISMATCH'); }
    if (result.sha256 !== f.sha256) { fs.rmSync(temp, { force: true }); fail('LAYA_PULL_HASH_MISMATCH'); }
    fs.renameSync(temp, dest);
  }
  let active = { laya: null };
  try { active = loadProviderConfig(home); } catch { /* providers.json absent/invalid: fall back to explicit flags only */ }
  const resolvedPython = python ?? active.laya?.python;
  if (typeof resolvedPython !== 'string' || !path.isAbsolute(resolvedPython)) fail('LAYA_PYTHON_NOT_CONFIGURED');
  const resolvedDevice = device ?? active.laya?.device ?? 'cpu';
  const registered = await registerImpl(home, { checkpointDir: downloadDir, model: manifest.model, device: resolvedDevice,
    python: resolvedPython, precision: manifest.precision, inputFit: manifest.inputFit });
  if (registered.checkpoint !== manifest.checkpoint) {
    if (!registered.reused) {
      fs.rmSync(path.join(checkpointsDir(home), registered.checkpoint), { recursive: true, force: true });
      fs.rmSync(path.join(candidatesDir(home), `${registered.checkpoint}.json`), { force: true });
    }
    fail('CHECKPOINT_MISMATCH');
  }
  ensureDir(publishedDir(home), true);
  const published = { repo, revision, qualification: { ...manifest.qualification,
    ...(manifest.qualification.decisionIdentity ? { operationalEligible: manifest.evaluation.operationalEligible, familyQualification: manifest.evaluation.familyQualification } : {}) }, evaluation: manifest.evaluation ?? null };
  atomicWrite(path.join(publishedDir(home), `${manifest.checkpoint}.json`), JSON.stringify(published, null, 2) + '\n');
  return { candidate: manifest.checkpoint, repo, revision, files: manifest.files.length, bytes: totalBytes, registered: true };
}
