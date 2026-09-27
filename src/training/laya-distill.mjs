// Teacher (Jev/TypeSafe) distillation + human-review pipeline for Laya route training (2026-09-23 owner decision).
// Teacher labels ground ONLY the train split; calibration/test/holdout accept human review only. Only
// synthetic (owner-authored) task sentences are ever sent to the remote teacher; captured shadow task
// text is imported with egress:'forbidden' and never leaves the machine.
import fs from 'node:fs';
import path from 'node:path';
import { noSymlinks, ensureDir, readText } from '../storage.mjs';
import { ID, fail, errorCode, ControlError } from '../constants.mjs';
import { validateRequest, wireRequest, normalizeResponse, containsSensitiveData } from '../contracts.mjs';
import { callTypeSafe } from '../provider.mjs';
import { getCredential, loadConfig } from '../storage.mjs';
import { loadFeaturePolicy } from '../feature-policy.mjs';
import { ROUTE_QUESTIONS } from '../routing.mjs';
import { createTrainingStore } from './store.mjs';
import { POLICY_VERSION, MAX_DERIVED_BYTES, digest, encode, only, text, safeContent, targetDistribution, validateTarget } from './schema.mjs';
import { EVALUATION_POLICY } from './evaluate.mjs';

// route requests always use this fixed context so routeGuard would let a real spawn reach the model;
// teacher/train/inference inputs for distillation must match that exact shape.
export const FIXED_ROUTE_CONTEXT = Object.freeze({ complete: true, scope: 'local', previousFailures: 0, highImpact: false, modelLocked: false, exhaustive: false });
export const TEACHER_LABEL_CONFIDENCE = EVALUATION_POLICY.minLabelConfidence; // trust-in-source floor for unreviewed teacher output, not the model's own probability
const MIN_CALL_INTERVAL_MS = 1200; // <=50 calls/minute
// Owner decision (2026-09-23, evaluation criteria): Claude
// (or another reviewer-model) reference labels replace human TTY review for calibration/test/holdout
// (role:'eval') because owner-run TTY review of English options/sentences proved unworkable. A
// measured Jev-vs-Claude agreement gap (intent .85 / risk .68 / difficulty .38 on 300 tasks) means
// the owner's "Laya ~= Claude" goal also needs Claude labels grounding TRAIN (role:'train'): those
// REPLACE the Jev teacher sample for that task rather than adding to it. Never recorded as 'human'
// or 'runner'; always 'ai_reference' with the labeling model recorded as the source. Results against
// role:'eval' or role:'train' ai_reference labels are "Claude agreement", never "accuracy".
const MODEL_SOURCE_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const REFERENCE_ROLES = ['eval', 'train'];

export function validateRunName(run) { if (typeof run !== 'string' || !ID.test(run)) fail('INVALID_DISTILL_RUN'); }

function distillRoot(home) { return path.join(home, 'laya', 'distill'); }
export function runDir(home, run) { validateRunName(run); return path.join(distillRoot(home), run); }
function ensureRunDir(home, run) {
  validateRunName(run);
  ensureDir(home, true);
  ensureDir(path.join(home, 'laya'), true);
  ensureDir(distillRoot(home), true);
  const dir = path.join(distillRoot(home), run);
  ensureDir(dir, true);
  return dir;
}
function withRunLock(dir, fn) {
  const lockDir = path.join(dir, '.lock');
  noSymlinks(lockDir);
  try { fs.mkdirSync(lockDir, { mode: 0o700 }); }
  catch (e) { if (e.code === 'EEXIST') fail('DISTILL_LOCKED'); throw e; }
  try { return fn(); } finally { fs.rmdirSync(lockDir); }
}
// Async-safe variant: the lock is held across every `await` in `fn` (mkdirSync/rmdirSync are
// synchronous, so no other call in this same process can slip in between acquire and release),
// so two concurrent `distill label` runs on the same run never both call the teacher.
async function withRunLockAsync(dir, fn) {
  const lockDir = path.join(dir, '.lock');
  noSymlinks(lockDir);
  try { fs.mkdirSync(lockDir, { mode: 0o700 }); }
  catch (e) { if (e.code === 'EEXIST') fail('DISTILL_LOCKED'); throw e; }
  try { return await fn(); } finally { fs.rmdirSync(lockDir); }
}
function appendPrivateJsonl(file, obj) {
  noSymlinks(file);
  const fd = fs.openSync(file, fs.constants.O_CREAT | fs.constants.O_APPEND | fs.constants.O_WRONLY | (fs.constants.O_NOFOLLOW || 0), 0o600);
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || stat.nlink !== 1) fail('UNSAFE_FILE');
    if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) fail('WRONG_OWNER');
    if (stat.mode & 0o077) fail('PRIVATE_FILE_REQUIRED');
    fs.writeSync(fd, JSON.stringify(obj) + '\n');
    fs.fsyncSync(fd);
  } finally { fs.closeSync(fd); }
}
function readPrivateJsonl(file, maxBytes = MAX_DERIVED_BYTES) {
  const t = readText(file, { optional: true, privateFile: true, maxBytes });
  if (t === null) return [];
  return t.trim() ? t.trim().split('\n').map(line => JSON.parse(line)) : [];
}
const tasksFile = dir => path.join(dir, 'tasks.jsonl');
const teacherFile = dir => path.join(dir, 'teacher.jsonl');
const teacherFailuresFile = dir => path.join(dir, 'teacher-failures.jsonl');
const reviewFile = dir => path.join(dir, 'review.jsonl');
const referenceFile = dir => path.join(dir, 'reference.jsonl');

function normalizeTaskText(s) { return s.trim().replace(/\s+/g, ' '); }
// No dependency on locale/ICU: a same-user local heuristic, adequate for splitting an owner-authored ko/en corpus.
// Exported so scripts/laya-benchmark.mjs classifies a built dataset sample's ko/en language the exact
// same way distillImportShadow does, instead of a second drifting regex.
export function detectLang(s) { return /[\uAC00-\uD7A3]/.test(s) ? 'ko' : 'en'; }
function buildTaskRequest(task) {
  return { purpose: 'route', risk: 'routine', state: { task, context: structuredClone(FIXED_ROUTE_CONTEXT) }, questions: structuredClone(ROUTE_QUESTIONS) };
}
// Shared sensitive-content screen used at both import time (distillImport/distillImportShadow) and
// build time (buildDistillDataset), so the two stages never disagree about what "sensitive" means
// (2026-09-23 bug fix: `distillImport` screened with only `containsSensitiveData`, which misses
// emails/phones/RRNs that `safeContent`'s stricter regexes catch, so a task could import clean and
// then fail the WHOLE build later). Runs the exact `safeContent` check `buildDistillDataset` already
// applies to the final sample's `state`, against the same `{ task, context: FIXED_ROUTE_CONTEXT }`
// shape, and returns the thrown error (or null if the state is clean) instead of throwing itself, so
// callers can decide whether a given failure means "skip as sensitive" or "fail loudly" (a genuine
// schema bug, e.g. TRAINING_SENSITIVE_OR_OVERSIZED's oversize/forbidden-key checks misfiring, must
// never be swallowed as if the content were merely sensitive).
function screenTaskState(taskText) {
  try { safeContent({ state: { task: taskText, context: FIXED_ROUTE_CONTEXT } }); return null; }
  catch (e) { return e; }
}

// --- import ---------------------------------------------------------------
export function distillImport(home, { run, inputFile, readFileImpl = (f) => fs.readFileSync(f, 'utf8') } = {}) {
  validateRunName(run);
  if (typeof inputFile !== 'string' || !inputFile) fail('DISTILL_INPUT_REQUIRED');
  const dir = ensureRunDir(home, run);
  return withRunLock(dir, () => {
    const raw = readFileImpl(inputFile);
    if (Buffer.byteLength(raw) > MAX_DERIVED_BYTES) fail('DISTILL_INPUT_TOO_LARGE');
    const file = tasksFile(dir);
    const seen = new Set(readPrivateJsonl(file).map(t => t.task_id));
    let total = 0, added = 0, skippedSensitive = 0, skippedDuplicate = 0;
    for (const line of raw.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      total++;
      let parsed;
      try { parsed = JSON.parse(trimmed); } catch { fail('INVALID_DISTILL_TASK_LINE'); }
      only(parsed, ['lang', 'domain', 'task', 'group', 'reviewable'], ['lang', 'task']);
      if (!['ko', 'en'].includes(parsed.lang)) fail('INVALID_DISTILL_TASK_LINE');
      // UTF-8 byte limit (not a character count) matching the hook's own truncateUtf8(..., 8000)
      // cap (src/hooks.mjs preSpawn task=description+prompt), so owner-authored Korean prompts up
      // to ~2,500 chars (well over the old 2000-char cap) import.
      if (typeof parsed.task !== 'string' || !parsed.task.trim() || Buffer.byteLength(parsed.task) > 8000) fail('INVALID_DISTILL_TASK_LINE');
      if (parsed.domain !== undefined && parsed.domain !== null) text(parsed.domain, 80);
      // Optional dedup/leak-prevention key (e.g. shared by a ko/en translation pair of the same
      // underlying task) so a human-reviewed member and its teacher-labeled sibling never split
      // across train and eval (see buildDistillDataset).
      if (parsed.group !== undefined && parsed.group !== null && (typeof parsed.group !== 'string' || !ID.test(parsed.group))) fail('INVALID_DISTILL_TASK_LINE');
      // Short synthetic augmentation items can be marked reviewable:false so the human-reviewed
      // calibration/test set reflects realistic long agent prompts; they only ever ground train.
      if (parsed.reviewable !== undefined && typeof parsed.reviewable !== 'boolean') fail('INVALID_DISTILL_TASK_LINE');
      // Skip as sensitive on EITHER screen: containsSensitiveData (credential-shaped patterns) or the
      // stricter safeContent screen (also emails/phones/RRNs/key-prefixes), so an import-time skip
      // always matches what buildDistillDataset would later accept for this same task.
      if (containsSensitiveData(parsed) || screenTaskState(parsed.task)) { skippedSensitive++; continue; }
      const normalized = normalizeTaskText(parsed.task);
      const task_id = digest({ lang: parsed.lang, text: normalized });
      if (seen.has(task_id)) { skippedDuplicate++; continue; }
      seen.add(task_id);
      appendPrivateJsonl(file, { task_id, lang: parsed.lang, domain: parsed.domain ?? null, task: parsed.task,
        group: parsed.group ?? null, reviewable: parsed.reviewable ?? true,
        source: 'synthetic', egress: 'allowed', added_at: new Date().toISOString() });
      added++;
    }
    return { run, total, added, skippedSensitive, skippedDuplicate };
  });
}
export function distillImportShadow(home, { run, trainingStore } = {}) {
  validateRunName(run);
  const dir = ensureRunDir(home, run);
  const store = trainingStore ?? createTrainingStore({ home });
  return withRunLock(dir, () => {
    const snapshot = store.scan();
    const file = tasksFile(dir);
    const seen = new Set(readPrivateJsonl(file).map(t => t.task_id));
    let scannedDecisions = 0, added = 0, skippedSensitive = 0, skippedDuplicate = 0;
    for (const e of snapshot.events) {
      if (e.kind !== 'decisions' || e.data.request.purpose !== 'route') continue;
      scannedDecisions++;
      const task = e.data.request.state?.task;
      if (typeof task !== 'string' || !task.trim()) continue;
      if (containsSensitiveData({ task }) || screenTaskState(task)) { skippedSensitive++; continue; }
      const lang = detectLang(task);
      const task_id = digest({ lang, text: normalizeTaskText(task) });
      if (seen.has(task_id)) { skippedDuplicate++; continue; }
      seen.add(task_id);
      appendPrivateJsonl(file, { task_id, lang, domain: null, group: null, reviewable: true, task, source: 'shadow', egress: 'forbidden', added_at: new Date().toISOString() });
      added++;
    }
    return { run, scannedDecisions, added, skippedSensitive, skippedDuplicate };
  });
}

// --- import-reference (owner/AI-reference labels for eval or train grounding) ---
// Input labels use the human-facing 1..5 difficulty scale (like `distill review`'s typed input);
// stored/validated as the internal 0-based score value.
function normalizeReferenceLabels(labels) {
  only(labels, Object.keys(ROUTE_QUESTIONS), Object.keys(ROUTE_QUESTIONS));
  const out = {};
  for (const [qid, q] of Object.entries(ROUTE_QUESTIONS)) {
    const raw = labels[qid];
    let value = raw;
    if (q.type === 'score') {
      if (!Number.isInteger(raw) || raw < 1 || raw > q.criteria.length) fail('INVALID_DISTILL_REFERENCE_LINE');
      value = raw - 1;
    }
    try { validateTarget(q, value); } catch { fail('INVALID_DISTILL_REFERENCE_LINE'); }
    out[qid] = value;
  }
  return out;
}
export function distillImportReference(home, { run, inputFile, source, role = 'eval', readFileImpl = (f) => fs.readFileSync(f, 'utf8') } = {}) {
  validateRunName(run);
  if (typeof inputFile !== 'string' || !inputFile) fail('DISTILL_INPUT_REQUIRED');
  if (typeof source !== 'string' || !MODEL_SOURCE_RE.test(source)) fail('INVALID_DISTILL_REFERENCE_SOURCE');
  if (!REFERENCE_ROLES.includes(role)) fail('INVALID_DISTILL_REFERENCE_ROLE');
  const dir = ensureRunDir(home, run);
  return withRunLock(dir, () => {
    const raw = readFileImpl(inputFile);
    if (Buffer.byteLength(raw) > MAX_DERIVED_BYTES) fail('DISTILL_INPUT_TOO_LARGE');
    const tasksById = new Map(readPrivateJsonl(tasksFile(dir)).map(t => [t.task_id, t]));
    const file = referenceFile(dir);
    const existing = new Set(readPrivateJsonl(file).map(l => l.task_id));
    let total = 0, unknownCount = 0;
    const parsedItems = [];
    for (const line of raw.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      total++;
      let parsed;
      try { parsed = JSON.parse(trimmed); } catch { fail('INVALID_DISTILL_REFERENCE_LINE'); }
      only(parsed, ['lang', 'task', 'labels'], ['lang', 'task', 'labels']);
      if (!['ko', 'en'].includes(parsed.lang)) fail('INVALID_DISTILL_REFERENCE_LINE');
      if (typeof parsed.task !== 'string' || !parsed.task.trim()) fail('INVALID_DISTILL_REFERENCE_LINE');
      const labels = normalizeReferenceLabels(parsed.labels);
      // Same task_id rule as `import`, so a reference line resolves to the exact same imported task.
      const task_id = digest({ lang: parsed.lang, text: normalizeTaskText(parsed.task) });
      if (!tasksById.has(task_id)) { unknownCount++; continue; }
      parsedItems.push({ task_id, labels });
    }
    // Fail the whole import (nothing written) rather than silently skipping unresolved tasks, so an
    // operator notices a typo'd/stale input file instead of quietly losing reference labels.
    if (unknownCount > 0) { const e = new ControlError('DISTILL_REFERENCE_UNKNOWN_TASK'); e.unknownCount = unknownCount; throw e; }
    let added = 0, skippedDuplicate = 0;
    for (const item of parsedItems) {
      if (existing.has(item.task_id)) { skippedDuplicate++; continue; }
      existing.add(item.task_id);
      appendPrivateJsonl(file, { task_id: item.task_id, labels: item.labels, source, role, at: new Date().toISOString() });
      added++;
    }
    return { run, total, added, skippedDuplicate, unknown: unknownCount, source, role };
  });
}

// --- label (teacher) -------------------------------------------------------
const LABEL_CONSECUTIVE_FAILURE_LIMIT = 5;
export async function distillLabel(home, { run, confirmEgress, limit, provider = callTypeSafe, key, model, timeoutMs, env = process.env,
  now = Date.now, sleepImpl = ms => new Promise(resolve => setTimeout(resolve, ms)) } = {}) {
  if (!confirmEgress) fail('EXPLICIT_REMOTE_TEACHER_CONSENT_REQUIRED');
  validateRunName(run);
  const dir = ensureRunDir(home, run);
  // Held for the whole batch (across every await below), not just individual writes: two
  // concurrent `distill label` runs on the same run must never both call the teacher.
  return withRunLockAsync(dir, async () => {
    const tasks = readPrivateJsonl(tasksFile(dir));
    const labeled = new Set(readPrivateJsonl(teacherFile(dir)).map(l => l.task_id));
    const eligible = tasks.filter(t => t.egress === 'allowed' && !labeled.has(t.task_id));
    const batch = Number.isInteger(limit) ? eligible.slice(0, Math.max(0, limit)) : eligible;
    const resolvedKey = key ?? getCredential(home, env).key;
    if (!resolvedKey) fail('NO_API_KEY');
    const policy = loadFeaturePolicy(home);
    const resolvedModel = model ?? policy.router.expectedModel;
    const config = loadConfig(home, env);
    const resolvedTimeout = timeoutMs ?? config.timeoutMs;
    let lastCallAt = null, attempted = 0, succeeded = 0, failed = 0, consecutiveFailures = 0, aborted = false, abortReason = null;
    const usage = { inputTokens: 0, outputTokens: 0 };
    const failureCodes = {};
    for (const task of batch) {
      if (lastCallAt !== null) {
        const wait = MIN_CALL_INTERVAL_MS - (now() - lastCallAt);
        if (wait > 0) await sleepImpl(wait);
      }
      lastCallAt = now();
      attempted++;
      try {
        const request = validateRequest(buildTaskRequest(task.task), config);
        const payload = wireRequest(request, resolvedModel);
        const raw = await provider(payload, resolvedKey, { timeoutMs: resolvedTimeout });
        const n = normalizeResponse(raw, request, { ...config, minConfidence: 0, minChoiceProbability: 0, noulCertainty: 0 });
        const answers = {};
        for (const qid of Object.keys(ROUTE_QUESTIONS)) answers[qid] = { probabilities: n.answers[qid].probabilities };
        appendPrivateJsonl(teacherFile(dir), { task_id: task.task_id, model: n.model, answers, usage: n.usage, at: new Date().toISOString() });
        succeeded++; usage.inputTokens += n.usage.inputTokens ?? 0; usage.outputTokens += n.usage.outputTokens ?? 0;
        consecutiveFailures = 0;
      } catch (e) {
        const code = errorCode(e);
        appendPrivateJsonl(teacherFailuresFile(dir), { task_id: task.task_id, code, at: new Date().toISOString() });
        failed++; failureCodes[code] = (failureCodes[code] ?? 0) + 1;
        consecutiveFailures++;
        // Circuit breaker: 5 consecutive failures stop the batch (already-recorded results stay;
        // an unattempted task has no failure record, so the next `distill label` run retries it).
        if (consecutiveFailures >= LABEL_CONSECUTIVE_FAILURE_LIMIT) { aborted = true; abortReason = code; break; }
      }
    }
    return { run, eligible: eligible.length, attempted, succeeded, failed, failureCodes, usage, aborted, abortReason };
  });
}

// --- review (human TTY) -----------------------------------------------------
// TTY checks are a same-user local-authorship signal only, not proof of a human operator (see SECURITY.md).
export async function distillReview(home, { run, count = 200, isStdinTTY, isStdoutTTY, prompt, write = () => {} } = {}) {
  if (!isStdinTTY || !isStdoutTTY) fail('HUMAN_TTY_REQUIRED');
  validateRunName(run);
  if (!Number.isInteger(count) || count < 2) fail('INVALID_DISTILL_REVIEW_COUNT');
  const dir = ensureRunDir(home, run);
  const tasksById = new Map(readPrivateJsonl(tasksFile(dir)).map(t => [t.task_id, t]));
  const teacherById = new Map(readPrivateJsonl(teacherFile(dir)).map(l => [l.task_id, l]));
  const reviewedIds = new Set(readPrivateJsonl(reviewFile(dir)).map(l => l.task_id));
  const perLang = Math.floor(count / 2);
  const byLang = { ko: [], en: [] };
  for (const task_id of teacherById.keys()) {
    const t = tasksById.get(task_id);
    if (t && t.reviewable !== false && byLang[t.lang]) byLang[t.lang].push(task_id);
  }
  for (const lang of ['ko', 'en']) byLang[lang].sort();
  // Group-aware selection: never put two members of the same group (e.g. a ko/en translation
  // pair sharing `group`) into the review budget together, and never re-select a group that
  // already has a reviewed member, so review spend isn't wasted labeling near-duplicate content.
  const reviewedGroupKeys = new Set();
  for (const task_id of reviewedIds) { const t = tasksById.get(task_id); if (t) reviewedGroupKeys.add(t.group ?? t.task_id); }
  const selectedGroupKeys = new Set();
  const selected = [];
  for (const lang of ['ko', 'en']) {
    let picked = 0;
    for (const task_id of byLang[lang]) {
      if (picked >= perLang) break;
      const t = tasksById.get(task_id);
      const key = t.group ?? t.task_id;
      if (reviewedGroupKeys.has(key) || selectedGroupKeys.has(key)) continue;
      selectedGroupKeys.add(key);
      selected.push(task_id);
      picked++;
    }
  }
  const pending = selected.filter(id => !reviewedIds.has(id));
  let reviewed = 0, skipped = 0, quit = false;
  for (let idx = 0; idx < pending.length; idx++) {
    const task_id = pending[idx];
    const task = tasksById.get(task_id), teacher = teacherById.get(task_id);
    write(`\n[${idx + 1}/${pending.length}] lang=${task.lang}\n`);
    write(`task (${task.lang}): ${task.task}\n`);
    const labels = {};
    let taskSkipped = false, taskQuit = false;
    for (const [qid, q] of Object.entries(ROUTE_QUESTIONS)) {
      const teacherAnswer = teacher.answers[qid];
      const probs = teacherAnswer.probabilities;
      const optionKeys = q.type === 'score' ? q.criteria.map((_, i) => String(i)) : Object.keys(q.criteria);
      const topKey = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
      write(`question ${qid}: ${q.instructions}\n`);
      optionKeys.forEach((key, i) => {
        const label = q.type === 'score' ? q.criteria[i] : `${key} — ${q.criteria[key]}`;
        const marker = key === topKey ? '*' : ' ';
        const p = (probs[key] ?? 0).toFixed(2);
        write(`${marker}${i + 1}) ${label} (p=${p})\n`);
      });
      let value, resolved = false, invalidAttempts = 0;
      while (!resolved) {
        const answer = (await prompt(`value for "${qid}" (Enter=accept, number/key, s=skip task, q=save & quit): `)).trim();
        if (answer === 'q') { taskQuit = true; break; }
        if (answer === 's') { taskSkipped = true; break; }
        if (answer === '') { value = q.type === 'score' ? Number(topKey) : topKey; resolved = true; break; }
        const n = Number(answer);
        if (q.type === 'score') {
          if (Number.isInteger(n) && n >= 1 && n <= q.criteria.length) { value = n - 1; resolved = true; break; }
        } else {
          if (Number.isInteger(n) && n >= 1 && n <= optionKeys.length) { value = optionKeys[n - 1]; resolved = true; break; }
          const match = optionKeys.find(k => k.toLowerCase() === answer.toLowerCase());
          if (match) { value = match; resolved = true; break; }
        }
        invalidAttempts++;
        if (invalidAttempts >= 5) { taskSkipped = true; write(`too many invalid attempts for "${qid}"; skipping task\n`); break; }
        write(`invalid value "${answer}" for "${qid}"; try again\n`);
      }
      if (taskQuit || taskSkipped) break;
      validateTarget(q, value);
      labels[qid] = value;
    }
    if (taskQuit) { quit = true; break; }
    if (taskSkipped) { skipped++; continue; }
    appendPrivateJsonl(reviewFile(dir), { task_id, labels, reviewer: 'human-tty', at: new Date().toISOString() });
    reviewed++;
  }
  return { run, selected: selected.length, reviewed, skipped, remaining: pending.length - reviewed - skipped, quit };
}

// --- build (standard datasets/<version> layout) -----------------------------
function hashToUuid(hex) {
  const bytes = hex.slice(0, 32).padEnd(32, '0').match(/.{2}/g).map(h => parseInt(h, 16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40; bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const h = bytes.map(x => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`;
}
function argmax(probabilities) { return Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0]; }
function teacherProvenance(model) {
  return { provider: 'jev', model, model_version: model, checkpoint: model, runtime_version: 'systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' };
}
const HOST_REVIEW_PROVENANCE = Object.freeze({ provider: 'host', model: 'human-review', model_version: 'human-review', checkpoint: 'human-review', runtime_version: 'host-reviewed', preprocessing_version: 'wire-request-v1', confidence_semantics: 'none' });
// Same provenance SHAPE for a role:'eval' or role:'train' reference label; only `model`/`model_version`/`checkpoint`
// (the labeling model recorded at import time) differ between reference lines.
function aiReferenceProvenance(source) {
  return { provider: 'host', model: source, model_version: source, checkpoint: source, runtime_version: 'ai-reference', preprocessing_version: 'wire-request-v1', confidence_semantics: 'none' };
}
// Shared by buildDistillDataset and distillCompareTeacher so the calibration/test split assignment
// for a human-reviewed or role:'eval' reference task can never drift between the two call sites
// (2026-09-25). `group_id` namespacing matches buildDistillDataset's pre-existing formula exactly:
// a `group` name can never collide with a bare task_id's own digest, and an ungrouped task's
// group_id is byte-identical to the pre-group `digest({distill_task: task_id})` formula.
function groupIdForTask(task) {
  const groupKey = task.group ?? task.task_id;
  return task.group ? digest({ distill_group: groupKey }) : digest({ distill_task: task.task_id });
}
// Deterministic 50/50 calibration/test split on the GROUP key's hash (not the task's own id), so
// every eval-grounding member of one group lands in the same split. For an ungrouped task the group
// key IS the task_id, so this is byte-identical to the pre-group per-task-id formula.
function evalSplitForTask(task) {
  const group_id = groupIdForTask(task);
  const splitSeed = task.group ? group_id : task.task_id;
  return parseInt(splitSeed.slice(0, 8), 16) % 2 === 0 ? 'calibration' : 'test';
}

export function buildDistillDataset(home, { run, trainingStore } = {}) {
  validateRunName(run);
  const dir = ensureRunDir(home, run);
  const store = trainingStore ?? createTrainingStore({ home });
  const { tasks, teacherById, reviewById, referenceById } = withRunLock(dir, () => ({
    tasks: readPrivateJsonl(tasksFile(dir)),
    teacherById: new Map(readPrivateJsonl(teacherFile(dir)).map(l => [l.task_id, l])),
    reviewById: new Map(readPrivateJsonl(reviewFile(dir)).map(l => [l.task_id, l])),
    referenceById: new Map(readPrivateJsonl(referenceFile(dir)).map(l => [l.task_id, l])),
  }));
  return store.lock(() => {
    const snapshot_id = digest(`distill-run:${run}`);
    const samples = [];
    // Group leak prevention: a `group` (e.g. shared by a ko/en translation pair of the same
    // underlying task) that has ANY eval-grounding member (human review OR a role:'eval' reference
    // label) must never also ground train via a teacher/role:'train' label on another member —
    // that would leak eval content into train.
    const reviewedGroupKeys = new Set();
    for (const task of tasks) {
      if (reviewById.has(task.task_id)) reviewedGroupKeys.add(task.group ?? task.task_id);
      const ref = referenceById.get(task.task_id);
      if (ref && ref.role === 'eval') reviewedGroupKeys.add(task.group ?? task.task_id);
    }
    let excludedGroupLeak = 0, excludedUnsafe = 0;
    for (const task of [...tasks].sort((a, b) => a.task_id.localeCompare(b.task_id))) {
      // Re-run the same screen distillImport/distillImportShadow apply at import time, so a task
      // that slipped past an older/weaker import screen (pre-fix data) never fails the WHOLE build —
      // it is excluded on its own (no samples for ANY of its questions) and counted, never thrown.
      // A non-sensitivity failure (a genuine schema bug) still fails loudly.
      const screenErr = screenTaskState(task.task);
      if (screenErr) {
        if (errorCode(screenErr) !== 'TRAINING_SENSITIVE_OR_OVERSIZED') throw screenErr;
        excludedUnsafe++; continue;
      }
      const teacher = teacherById.get(task.task_id);
      const reviewed = reviewById.get(task.task_id);
      const reference = referenceById.get(task.task_id);
      const groupKey = task.group ?? task.task_id;
      const group_id = groupIdForTask(task);
      let split, provenance, labelSource, labelConfidence, valueFor, probsFor;
      // Precedence per task: human (eval) > reference eval > reference train > Jev teacher (train).
      if (reviewed) {
        split = evalSplitForTask(task);
        provenance = teacher ? teacherProvenance(teacher.model) : HOST_REVIEW_PROVENANCE;
        labelSource = 'human'; labelConfidence = 1;
        valueFor = qid => reviewed.labels[qid];
        probsFor = (q, value) => targetDistribution(q, value);
      } else if (reference && reference.role === 'eval') {
        split = evalSplitForTask(task);
        provenance = aiReferenceProvenance(reference.source);
        labelSource = 'ai_reference'; labelConfidence = 1;
        valueFor = qid => reference.labels[qid];
        probsFor = (q, value) => targetDistribution(q, value);
      } else if (reference && reference.role === 'train') {
        if (reviewedGroupKeys.has(groupKey)) { excludedGroupLeak++; continue; } // a train-role reference never grounds train when a group sibling is in eval
        split = 'train';
        provenance = aiReferenceProvenance(reference.source);
        labelSource = 'ai_reference'; labelConfidence = 1;
        valueFor = qid => reference.labels[qid];
        probsFor = (q, value) => targetDistribution(q, value);
      } else if (teacher && task.egress === 'allowed') {
        if (reviewedGroupKeys.has(groupKey)) { excludedGroupLeak++; continue; } // teacher label never grounds train when a group sibling is in eval
        split = 'train';
        provenance = teacherProvenance(teacher.model);
        labelSource = 'teacher'; labelConfidence = TEACHER_LABEL_CONFIDENCE;
        valueFor = qid => { const top = argmax(teacher.answers[qid].probabilities); return ROUTE_QUESTIONS[qid].type === 'score' ? Number(top) : top; };
        probsFor = (q, value, qid) => teacher.answers[qid].probabilities;
      } else continue;
      const request = buildTaskRequest(task.task);
      const request_hash = digest(request);
      const task_uuid = hashToUuid(task.task_id);
      for (const [qid, q] of Object.entries(ROUTE_QUESTIONS)) {
        const value = valueFor(qid);
        validateTarget(q, value);
        const target = { value, probabilities: probsFor(q, value, qid) };
        const sample = { schema_version: 1, sample_id: digest({ request_hash, question_id: qid, target }),
          task_id: task_uuid, snapshot_id, task_ids: [task_uuid], snapshot_ids: [snapshot_id], request_hash,
          purpose: 'route', state: request.state, question_id: qid, question: structuredClone(q),
          target, label_source: labelSource, label_confidence: labelConfidence, label_confidence_is_calibrated: false,
          evaluation_policy_version: POLICY_VERSION, provenance: [provenance],
          raw_refs: { distill: { run, task_id: task.task_id } }, group_id, split };
        safeContent(sample);
        samples.push(sample);
      }
    }
    samples.sort((a, b) => a.sample_id.localeCompare(b.sample_id));
    const data = samples.map(encode).join('\n') + (samples.length ? '\n' : '');
    const data_sha256 = digest(data);
    const version = digest({ kind: 'distill', run, data_sha256 });
    const counts = { split: {}, lang: {}, label_source: {}, label_source_by_split: {}, excluded_group_leak: excludedGroupLeak, excluded_unsafe: excludedUnsafe };
    for (const s of samples) {
      counts.split[s.split] = (counts.split[s.split] ?? 0) + 1;
      counts.label_source[s.label_source] = (counts.label_source[s.label_source] ?? 0) + 1;
      const bySplit = (counts.label_source_by_split[s.split] ??= {});
      bySplit[s.label_source] = (bySplit[s.label_source] ?? 0) + 1;
    }
    for (const task of tasks) if (teacherById.has(task.task_id) || reviewById.has(task.task_id) || referenceById.has(task.task_id)) counts.lang[task.lang] = (counts.lang[task.lang] ?? 0) + 1;
    const teacherModels = [...new Set([...teacherById.values()].map(l => l.model))];
    const manifest = { schema_version: 1, dataset_version: version, kind: 'distill', run, generated_at: new Date().toISOString(),
      sample_count: samples.length, data_sha256, evaluation_policy_version: POLICY_VERSION,
      teacher_model: teacherModels.length <= 1 ? (teacherModels[0] ?? null) : teacherModels, counts,
      policy: 'precedence per task: human (eval) > reference eval (ai_reference) > reference train (ai_reference, replaces teacher) > Jev teacher (train); only synthetic tasks are ever sent to the remote teacher' };
    const manifestPath = path.join(store.root, 'manifests', `${version}.json`);
    const previous = readText(manifestPath, { optional: true, privateFile: true, maxBytes: 1048576 });
    store.writeDerived(`datasets/${version}/canonical.jsonl`, data);
    if (previous === null) store.writeDerived(`manifests/${version}.json`, encode(manifest) + '\n');
    else { const old = JSON.parse(previous); if (old.dataset_version !== version || old.data_sha256 !== data_sha256) fail('DATASET_MANIFEST_MISMATCH'); }
    return { dataset_version: version, sample_count: samples.length, manifest: manifestPath, counts };
  });
}

// --- status (content-free) --------------------------------------------------
export function distillStatus(home, { run } = {}) {
  validateRunName(run);
  const dir = runDir(home, run);
  const tasks = readPrivateJsonl(tasksFile(dir));
  const teacherLines = readPrivateJsonl(teacherFile(dir));
  const failures = readPrivateJsonl(teacherFailuresFile(dir));
  const reviewLines = readPrivateJsonl(reviewFile(dir));
  const referenceLines = readPrivateJsonl(referenceFile(dir));
  const bySourceLang = {};
  for (const t of tasks) { const k = `${t.source}:${t.lang}`; bySourceLang[k] = (bySourceLang[k] ?? 0) + 1; }
  const usage = teacherLines.reduce((acc, l) => ({ inputTokens: acc.inputTokens + (l.usage?.inputTokens ?? 0), outputTokens: acc.outputTokens + (l.usage?.outputTokens ?? 0) }), { inputTokens: 0, outputTokens: 0 });
  const groups = new Set(tasks.map(t => t.group ?? t.task_id)).size;
  const reviewable = tasks.filter(t => t.reviewable !== false).length;
  const referenceByRole = {};
  for (const l of referenceLines) referenceByRole[l.role] = (referenceByRole[l.role] ?? 0) + 1;
  return { run, tasks: tasks.length, tasks_by_source_lang: bySourceLang, teacher_labels: teacherLines.length,
    teacher_label_failures: failures.length, reviewed: reviewLines.length, teacher_usage: usage, groups, reviewable,
    reference_labels: referenceLines.length, reference_labels_by_role: referenceByRole };
}

// --- compare-teacher (numbers-only Jev teacher vs reference-label report) ---
function teacherArgmax(teacher, qid, q) {
  const top = argmax(teacher.answers[qid].probabilities);
  return q.type === 'score' ? Number(top) : top;
}
function questionAgreementReport(pairs, q) {
  const n = pairs.length;
  const agreementCount = pairs.filter(p => p.teacherValue === p.referenceValue).length;
  const confusion = {};
  for (const p of pairs) {
    const t = String(p.teacherValue);
    (confusion[t] ??= {});
    confusion[t][String(p.referenceValue)] = (confusion[t][String(p.referenceValue)] ?? 0) + 1;
  }
  const byLang = {};
  for (const lang of ['ko', 'en']) {
    const subset = pairs.filter(p => p.lang === lang);
    const agree = subset.filter(p => p.teacherValue === p.referenceValue).length;
    byLang[lang] = { n: subset.length, agreement_rate: subset.length ? agree / subset.length : null };
  }
  const report = { n, agreement: { count: agreementCount, rate: n ? agreementCount / n : null }, confusion, by_lang: byLang };
  if (q.type === 'score') {
    const diffs = pairs.map(p => Math.abs(Number(p.teacherValue) - Number(p.referenceValue)));
    report.mean_absolute_difference = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null;
    const within1 = diffs.filter(d => d <= 1).length;
    report.within_one = { count: within1, rate: diffs.length ? within1 / diffs.length : null };
  }
  return report;
}
/** Numbers-only Jev-teacher-vs-reference-label agreement report (never accuracy: teacher is not
 * ground truth — see TRAINING_DATA.md §7). Reports role:'eval' and role:'train' reference lines
 * SEPARATELY, since (per the owner's 2026-09-23 measurement) they ground different splits. */
export function distillCompareTeacher(home, { run } = {}) {
  validateRunName(run);
  const dir = runDir(home, run);
  const tasksById = new Map(readPrivateJsonl(tasksFile(dir)).map(t => [t.task_id, t]));
  const teacherById = new Map(readPrivateJsonl(teacherFile(dir)).map(l => [l.task_id, l]));
  const referenceLines = readPrivateJsonl(referenceFile(dir));
  const bySection = { eval_reference: [], train_reference: [] };
  for (const r of referenceLines) {
    const teacher = teacherById.get(r.task_id), task = tasksById.get(r.task_id);
    if (!teacher || !task) continue; // only tasks with BOTH a teacher label and a reference label are comparable
    const key = r.role === 'train' ? 'train_reference' : 'eval_reference';
    bySection[key].push({ lang: task.lang, teacher, reference: r, task });
  }
  function questionsFor(items) {
    const questions = {};
    for (const [qid, q] of Object.entries(ROUTE_QUESTIONS)) {
      const pairs = items.map(it => ({ lang: it.lang, teacherValue: teacherArgmax(it.teacher, qid, q), referenceValue: it.reference.labels[qid] }));
      questions[qid] = questionAgreementReport(pairs, q);
    }
    return questions;
  }
  const report = { run };
  for (const [key, items] of Object.entries(bySection)) {
    const section = { compared: items.length, questions: questionsFor(items) };
    // by_split uses the exact same evalSplitForTask helper buildDistillDataset uses for these same
    // reference lines (human review / role:'eval'), so these counts can never drift from the build's
    // actual calibration/test assignment (2026-09-25).
    if (key === 'eval_reference') {
      const bySplit = { calibration: [], test: [] };
      for (const it of items) bySplit[evalSplitForTask(it.task)].push(it);
      section.by_split = {};
      for (const splitName of ['calibration', 'test']) {
        section.by_split[splitName] = { compared: bySplit[splitName].length, questions: questionsFor(bySplit[splitName]) };
      }
    }
    report[key] = section;
  }
  return report;
}
