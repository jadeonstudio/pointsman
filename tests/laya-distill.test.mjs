import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { ROUTE_QUESTIONS } from '../src/routing.mjs';
import { readDataset, exportDataset } from '../src/training/dataset.mjs';
import { freezeHoldout, qualifyCandidate, compareCandidate } from '../src/training/laya-lifecycle.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { digest, encode } from '../src/training/schema.mjs';
import { setMode, atomicWrite } from '../src/storage.mjs';
import {
  distillImport, distillImportReference, distillImportShadow, distillLabel, distillReview, buildDistillDataset,
  distillCompareTeacher, distillStatus, runDir,
} from '../src/training/laya-distill.mjs';

function fixture(t) {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-distill-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  return home;
}
function writeInputFile(t, lines) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-distill-input-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'tasks.jsonl');
  fs.writeFileSync(file, lines.map(l => JSON.stringify(l)).join('\n') + '\n');
  return file;
}
function choiceProbs(criteria, picked) {
  const keys = Object.keys(criteria);
  const other = keys.length > 1 ? (1 - 0.82) / (keys.length - 1) : 0;
  return Object.fromEntries(keys.map(k => [k, k === picked ? 0.82 : other]));
}
function scoreProbs(pickIndex, n) {
  const other = n > 1 ? 0.2 / (n - 1) : 0;
  const probs = {};
  for (let i = 0; i < n; i++) probs[String(i)] = i === pickIndex ? 0.8 : other;
  return probs;
}
function teacherRaw(model, picks = { intent: 'edit', difficulty: 1, risk: 'safe' }) {
  const intentProbs = choiceProbs(ROUTE_QUESTIONS.intent.criteria, picks.intent);
  const riskProbs = choiceProbs(ROUTE_QUESTIONS.risk.criteria, picks.risk);
  const diffProbs = scoreProbs(picks.difficulty, ROUTE_QUESTIONS.difficulty.criteria.length);
  const diffScore = Object.entries(diffProbs).reduce((s, [k, p]) => s + Number(k) * p, 0);
  return {
    model, usage: { input_tokens: 12, output_tokens: 3},
    answers: {
      intent: { type: 'choice', choice: picks.intent, confidence: .9, probabilities: intentProbs },
      difficulty: { type: 'score', score: diffScore, confidence: .9, probabilities: diffProbs },
      risk: { type: 'choice', choice: picks.risk, confidence: .9, probabilities: riskProbs },
    },
  };
}
function fakeProvider(picks) {
  return async () => teacherRaw('jev-1.13.0', picks);
}

// ============================== import ==============================

test('distill import validates schema, screens sensitive lines, and deduplicates by normalized text', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'en', task: 'Explain how the router chooses a tier' },
    { lang: 'en', task: 'Explain how the router chooses a tier' }, // duplicate (exact)
    { lang: 'en', task: '  Explain   how the router  chooses a tier ' }, // duplicate (normalized)
    { lang: 'ko', domain: 'billing', task: '\uACB0\uC81C \uBAA8\uB4C8\uC758 \uC2E4\uD328 \uC6D0\uC778\uC744 \uBD84\uC11D\uD574\uC918' },
    { lang: 'en', task: 'api_key=sk-abcdefghijklmnop123456 leaked in logs' }, // sensitive
  ]);
  const r = distillImport(home, { run: 'run1', inputFile: file });
  assert.equal(r.total, 5);
  assert.equal(r.added, 2);
  assert.equal(r.skippedDuplicate, 2);
  assert.equal(r.skippedSensitive, 1);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.length, 2);
  assert.ok(lines.every(l => l.source === 'synthetic' && l.egress === 'allowed'));
  assert.match(lines[0].task_id, /^[0-9a-f]{64}$/);
});

test('distill import skips a task containing an example email address, even though containsSensitiveData alone would not catch it (safeContent shares the screen with buildDistillDataset)', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'en', task: 'Escalate this to test@company.com if the router keeps failing' }, // email: containsSensitiveData misses it, safeContent does not
    { lang: 'en', task: 'Add a retry to the fetch call' },
  ]);
  const r = distillImport(home, { run: 'run1', inputFile: file });
  assert.equal(r.total, 2);
  assert.equal(r.added, 1);
  assert.equal(r.skippedSensitive, 1);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.length, 1);
  assert.ok(!lines.some(l => l.task.includes('test@company.com')));
});

test('distill import rejects malformed lines and out-of-range language/length', t => {
  const home = fixture(t);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'fr', task: 'x' }]) }), /INVALID_DISTILL_TASK_LINE/);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: '' }]) }), /INVALID_DISTILL_TASK_LINE/);
  assert.throws(() => distillImport(home, { run: 'not a valid run name', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x' }]) }), /INVALID_DISTILL_RUN/);
});

test('distill import uses a UTF-8 byte limit of 8000 (matching the hook truncateUtf8 cap), not a character count', t => {
  const home = fixture(t);
  // Long owner-authored Korean prompts (~2,500 chars) are well over the old 2000-char cap but
  // comfortably under 8000 bytes; the hook (src/hooks.mjs truncateUtf8) uses the same 8000-byte cap.
  const koreanTask = '\uC791\uC5C5 \uC9C0\uC2DC\uBB38 \uD55C\uAD6D\uC5B4 \uBB38\uC7A5\uC785\uB2C8\uB2E4. '.repeat(150).trim().slice(0, 2500);
  assert.ok(Buffer.byteLength(koreanTask) > 2000 && Buffer.byteLength(koreanTask) <= 8000);
  const r = distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'ko', task: koreanTask }]) });
  assert.equal(r.added, 1);
  const asciiUnder = 'x'.repeat(2001); // over the OLD char cap, well under the new byte cap
  const r2 = distillImport(home, { run: 'run2', inputFile: writeInputFile(t, [{ lang: 'en', task: asciiUnder }]) });
  assert.equal(r2.added, 1);
  const overByteLimit = 'x'.repeat(8001);
  assert.throws(() => distillImport(home, { run: 'run3', inputFile: writeInputFile(t, [{ lang: 'en', task: overByteLimit }]) }), /INVALID_DISTILL_TASK_LINE/);
  const koreanOverByteLimit = '\uC791\uC5C5'.repeat(3000); // multi-byte chars pushing well past 8000 bytes
  assert.throws(() => distillImport(home, { run: 'run4', inputFile: writeInputFile(t, [{ lang: 'ko', task: koreanOverByteLimit }]) }), /INVALID_DISTILL_TASK_LINE/);
});

test('distill import accepts an optional group field matching ID and stores null when absent', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uACE0\uC591\uC774\uB97C \uBD80\uD0C1\uD574', group: 'pair1' },
    { lang: 'en', task: 'Please take care of the cat', group: 'pair1' },
    { lang: 'en', task: 'ungrouped task text' },
  ]);
  const r = distillImport(home, { run: 'run1', inputFile: file });
  assert.equal(r.added, 3);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.filter(l => l.group === 'pair1').length, 2);
  assert.equal(lines.find(l => l.task === 'ungrouped task text').group, null);
});

test('distill import rejects a group value that does not match the ID pattern', t => {
  const home = fixture(t);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x', group: 'not a valid id' }]) }), /INVALID_DISTILL_TASK_LINE/);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x', group: 123 }]) }), /INVALID_DISTILL_TASK_LINE/);
});

test('distill import accepts an optional reviewable boolean, defaulting to true', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'en', task: 'short synthetic augmentation item', reviewable: false },
    { lang: 'en', task: 'a realistic long agent prompt task', reviewable: true },
    { lang: 'en', task: 'a task with no reviewable field at all' },
  ]);
  const r = distillImport(home, { run: 'run1', inputFile: file });
  assert.equal(r.added, 3);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.find(l => l.task === 'short synthetic augmentation item').reviewable, false);
  assert.equal(lines.find(l => l.task === 'a realistic long agent prompt task').reviewable, true);
  assert.equal(lines.find(l => l.task === 'a task with no reviewable field at all').reviewable, true); // default
});

test('distill import rejects a non-boolean reviewable value', t => {
  const home = fixture(t);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x', reviewable: 'yes' }]) }), /INVALID_DISTILL_TASK_LINE/);
  assert.throws(() => distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x', reviewable: 0 }]) }), /INVALID_DISTILL_TASK_LINE/);
});

test('distill import-shadow pulls captured route task text with forbidden egress and skips non-route captures', t => {
  const home = fixture(t);
  setMode(home, 'on', {});
  const store = createTrainingStore({ home });
  store.setCapture(true);
  const routeRequest = { purpose: 'route', risk: 'routine', state: { task: 'Investigate a flaky CI job', context: { complete: true, scope: 'local', previousFailures: 0, highImpact: false, modelLocked: false, exhaustive: false } }, questions: ROUTE_QUESTIONS };
  const otherRequest = { purpose: 'judge', risk: 'routine', state: { task: 'unrelated judge task' }, questions: { ok: { type: 'noul', instructions: 'ok?' } } };
  for (const [request, purpose] of [[routeRequest, 'route'], [otherRequest, 'judge']]) {
    const r = store.decision({
      decision_id: randomUUID(), trace: { task_id: randomUUID(), snapshot_id: digest('s') }, arm: 'active', request, request_hash: digest(request),
      provenance: { provider: 'jev', model: 'jev-1.13.0', model_version: 'jev-1.13.0', checkpoint: 'jev-1.13.0', runtime_version: 'systemone-v1', preprocessing_version: 'wire-request-v1', confidence_semantics: 'reported-statistic' },
      answers: purpose === 'route'
        ? { intent: { type: 'choice', value: 'debug', confidence: .9, selectedProbability: .82, probabilities: choiceProbs(ROUTE_QUESTIONS.intent.criteria, 'debug') },
            difficulty: { type: 'score', value: 2, confidence: .9, probabilities: scoreProbs(2, 5) },
            risk: { type: 'choice', value: 'safe', confidence: .9, selectedProbability: .82, probabilities: choiceProbs(ROUTE_QUESTIONS.risk.criteria, 'safe') } }
        : { ok: { type: 'noul', value: true, probabilityTrue: .9, confidence: null } },
      mode: 'on', apply: true, latency_ms: 5, usage: { inputTokens: 1, outputTokens: 1 }, inference_calls: 1, network_calls: 1, capture_policy_version: 'minimal-state-v1',
    });
    assert.ok(r.stored, JSON.stringify(r));
  }
  const result = distillImportShadow(home, { run: 'run1' });
  assert.equal(result.scannedDecisions, 1);
  assert.equal(result.added, 1);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].source, 'shadow');
  assert.equal(lines[0].egress, 'forbidden');
  assert.equal(lines[0].task, 'Investigate a flaky CI job');
  assert.equal(lines[0].group, null);
  assert.equal(lines[0].reviewable, true);
});

// ============================== label ==============================

test('distill label refuses without --confirm-egress and never calls the provider', async t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'Add a unit test for the parser' }]) });
  let called = false;
  await assert.rejects(distillLabel(home, { run: 'run1', confirmEgress: false, key: 'k', provider: async () => { called = true; return teacherRaw('jev-1.13.0'); } }), /EXPLICIT_REMOTE_TEACHER_CONSENT_REQUIRED/);
  assert.equal(called, false);
});

test('distill label calls only egress-allowed unlabeled tasks, records probabilities and usage, and is idempotent', async t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [
    { lang: 'en', task: 'Add a unit test for the parser' },
    { lang: 'ko', task: '\uD30C\uC11C\uC5D0 \uB2E8\uC704 \uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918' },
  ]) });
  distillImportShadow(home, { run: 'run1' }); // no captures yet; still exercises the egress:'forbidden' exclusion path once present
  let calls = 0;
  const r1 = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => { calls++; return teacherRaw('jev-1.13.0'); }, sleepImpl: async () => {} });
  assert.equal(r1.eligible, 2);
  assert.equal(r1.succeeded, 2);
  assert.equal(calls, 2);
  const teacherLines = fs.readFileSync(path.join(runDir(home, 'run1'), 'teacher.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(teacherLines.length, 2);
  assert.ok(teacherLines.every(l => l.answers.intent.probabilities && l.answers.difficulty.probabilities && l.answers.risk.probabilities));
  assert.equal(teacherLines[0].usage.inputTokens, 12);
  const r2 = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => { calls++; return teacherRaw('jev-1.13.0'); }, sleepImpl: async () => {} });
  assert.equal(r2.eligible, 0);
  assert.equal(calls, 2); // no re-labeling of already-labeled tasks
});

test('distill label spaces calls to respect <=50/minute and never retries a failed call automatically', async t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [
    { lang: 'en', task: 'task one' }, { lang: 'en', task: 'task two' }, { lang: 'en', task: 'task three' },
  ]) });
  let clock = 1000000;
  const calledAt = [];
  let callNumber = 0;
  const provider = async () => {
    callNumber++;
    calledAt.push(clock);
    if (callNumber === 2) fail_forcedProviderError();
    return teacherRaw('jev-1.13.0');
  };
  function fail_forcedProviderError() { throw new Error('PROVIDER_REJECTED'); }
  const sleeps = [];
  const r = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider,
    now: () => clock, sleepImpl: async ms => { sleeps.push(ms); clock += ms; } });
  assert.equal(r.attempted, 3);
  assert.equal(r.succeeded, 2);
  assert.equal(r.failed, 1);
  assert.ok(sleeps.length >= 2);
  assert.ok(sleeps.every(ms => ms >= 1199)); // >=1200ms spacing for <=50 calls/minute
  const failures = fs.readFileSync(path.join(runDir(home, 'run1'), 'teacher-failures.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(failures.length, 1);
  assert.ok(failures[0].code);
  assert.equal(Object.keys(failures[0]).includes('task'), false);
});

test('distill label resumes a failed task on the next run without re-attempting succeeded ones', async t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'flaky task' }]) });
  const r1 = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => { throw new Error('boom'); }, sleepImpl: async () => {} });
  assert.equal(r1.failed, 1);
  assert.equal(r1.succeeded, 0);
  const r2 = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  assert.equal(r2.eligible, 1); // the failed task is retried since no teacher label was written
  assert.equal(r2.succeeded, 1);
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.teacher_labels, 1);
  assert.equal(status.teacher_label_failures, 1);
});

test('distill label stops the batch after 5 consecutive failures, keeps already-recorded results, and reports abortReason', async t => {
  const home = fixture(t);
  const lines = Array.from({ length: 8 }, (_, i) => ({ lang: 'en', task: `task ${i}` }));
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, lines) });
  let call = 0;
  // succeeds twice, then fails 5 times in a row (>= the breaker threshold), then would succeed again
  const provider = async () => {
    call++;
    if (call <= 2 || call > 7) return teacherRaw('jev-1.13.0');
    throw new Error('PROVIDER_DOWN');
  };
  const r = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider, sleepImpl: async () => {} });
  assert.equal(r.aborted, true);
  assert.ok(r.abortReason);
  assert.equal(r.succeeded, 2);
  assert.equal(r.failed, 5);
  assert.equal(r.attempted, 7); // stopped after the 5th consecutive failure, task 8 never attempted
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.teacher_labels, 2); // already-recorded successes are preserved
  assert.equal(status.teacher_label_failures, 5);
  // rerun resumes: the 2 succeeded tasks are skipped, the 5 failed + 1 never-attempted are retried
  const r2 = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  assert.equal(r2.eligible, 6);
  assert.equal(r2.succeeded, 6);
  assert.equal(r2.aborted, false);
});

test('distill label does not abort when failures are not consecutive (a success resets the streak)', async t => {
  const home = fixture(t);
  const lines = Array.from({ length: 6 }, (_, i) => ({ lang: 'en', task: `task ${i}` }));
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, lines) });
  let call = 0;
  const provider = async () => {
    call++;
    if (call % 2 === 0) throw new Error('boom'); // fails on even calls, succeeds on odd: never 5 in a row
    return teacherRaw('jev-1.13.0');
  };
  const r = await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider, sleepImpl: async () => {} });
  assert.equal(r.aborted, false);
  assert.equal(r.abortReason, null);
  assert.equal(r.attempted, 6);
});

test('two concurrent distill label runs on the same run do not both call the teacher', async t => {
  const home = fixture(t);
  const lines = Array.from({ length: 4 }, (_, i) => ({ lang: 'en', task: `task ${i}` }));
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, lines) });
  let inFlight = 0, maxInFlight = 0;
  const provider = async () => {
    inFlight++; maxInFlight = Math.max(maxInFlight, inFlight);
    await new Promise(resolve => setTimeout(resolve, 10));
    inFlight--;
    return teacherRaw('jev-1.13.0');
  };
  const run = () => distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider, sleepImpl: async () => {} });
  const [a, b] = await Promise.allSettled([run(), run()]);
  const outcomes = [a, b];
  const rejected = outcomes.filter(o => o.status === 'rejected');
  const fulfilled = outcomes.filter(o => o.status === 'fulfilled');
  assert.equal(rejected.length, 1); // the second concurrent run is refused, not silently interleaved
  assert.match(rejected[0].reason.message, /DISTILL_LOCKED/);
  assert.equal(fulfilled.length, 1);
  assert.equal(maxInFlight, 1); // the teacher is never called from two runs at once
});

// ============================== review ==============================

function labeledFixture(t, count = 6) {
  const home = fixture(t);
  const lines = [];
  for (let i = 0; i < count; i++) lines.push({ lang: i % 2 === 0 ? 'ko' : 'en', task: `${i % 2 === 0 ? '\uD55C\uAD6D\uC5B4 \uC791\uC5C5' : 'english task'} number ${i}` });
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, lines) });
  return home;
}
function fakePromptQueue(answers) {
  let i = 0;
  return async () => answers[i++] ?? '';
}

test('review requires a TTY on both stdin and stdout', async t => {
  const home = labeledFixture(t, 2);
  await assert.rejects(distillReview(home, { run: 'run1', isStdinTTY: false, isStdoutTTY: true, prompt: async () => '' }), /HUMAN_TTY_REQUIRED/);
  await assert.rejects(distillReview(home, { run: 'run1', isStdinTTY: true, isStdoutTTY: false, prompt: async () => '' }), /HUMAN_TTY_REQUIRED/);
});

test('review accepts teacher defaults on Enter, overrides with a typed label, skips, and is stratified by language', async t => {
  const home = labeledFixture(t, 4); // 2 ko + 2 en
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  const written = [];
  const result = await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, write: t => written.push(t),
    prompt: fakePromptQueue(['', '', '', 'edit', '3', 'high', 's', '', '']) }); // task1: accept all; task2: override all; task3: skip; task4: accept all
  assert.equal(result.selected, 4);
  assert.ok(result.reviewed >= 2);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.ok(lines.every(l => l.reviewer === 'human-tty'));
  const overridden = lines.find(l => l.labels.intent === 'edit' && l.labels.risk === 'high');
  assert.ok(overridden);
  assert.equal(overridden.labels.difficulty, 2); // typed "3" (1-based) -> 0-based 2
});

test('review prints a progress header and a numbered option list (with teacher marker + probability) for choice and score questions', async t => {
  const home = labeledFixture(t, 2); // 1 ko + 1 en
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  const written = [];
  await distillReview(home, { run: 'run1', count: 2, isStdinTTY: true, isStdoutTTY: true, write: t => written.push(t),
    prompt: fakePromptQueue(Array(6).fill('')) });
  const out = written.join('');
  assert.match(out, /\[1\/2\] lang=ko/); // progress header: k/N + lang
  // choice question (intent): every criteria key listed in order, teacher top ('edit') marked, with a 2-decimal probability
  assert.match(out, / 1\) explain — Explain or locate existing code; do not change behavior \(p=0\.\d\d\)/);
  assert.match(out, /\*2\) edit — Write or change a bounded piece of code or documentation \(p=0\.82\)/);
  // score question (difficulty): 5 numbered options 1..5, teacher top (picks.difficulty=1 -> 0-based -> displayed as "2") marked
  assert.match(out, /\*2\) 2: Small local change with explicit requirements and known validation \(p=0\.80\)/);
  assert.match(out, / 5\) 5: Deep reasoning, unknown repository-wide impact, architecture or long-horizon planning \(p=0\.\d\d\)/);
});

test('review re-prompts the same question on invalid input instead of aborting, and skips the task after 5 consecutive invalid attempts', async t => {
  const home = labeledFixture(t, 2);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const written = [];
  // task1 (ko): 5 invalid answers to the first question ("intent") -> capped -> treated as skip.
  // task2 (en): all Enter -> accepted.
  const result = await distillReview(home, { run: 'run1', count: 2, isStdinTTY: true, isStdoutTTY: true, write: t => written.push(t),
    prompt: fakePromptQueue(['bogus', 'zzz', '99', '-1', 'nope', '', '', '']) });
  assert.equal(result.skipped, 1);
  assert.equal(result.reviewed, 1);
  assert.ok(written.some(w => /invalid/i.test(w))); // a one-line error was written, not a thrown exception
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.length, 1); // the invalid-input task was never recorded
});

test('review accepts a 1-based option number or a case-insensitive exact key for a choice question', async t => {
  const home = labeledFixture(t, 2);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  // task1 (ko): intent by 1-based number (3 -> 'debug'), difficulty accept, risk by case-insensitive key 'HIGH'
  // task2 (en): all accept
  const result = await distillReview(home, { run: 'run1', count: 2, isStdinTTY: true, isStdoutTTY: true,
    prompt: fakePromptQueue(['3', '', 'HIGH', '', '', '']) });
  assert.equal(result.reviewed, 2);
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const overridden = lines.find(l => l.labels.intent === 'debug');
  assert.ok(overridden);
  assert.equal(overridden.labels.risk, 'high');
});

test('review is resumable: an interrupted (quit) session leaves unreviewed tasks pending for the next call', async t => {
  const home = labeledFixture(t, 4);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const first = await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true,
    prompt: fakePromptQueue(['', '', '', 'q']) }); // accept task1 fully, quit mid-task2
  assert.equal(first.reviewed, 1);
  assert.equal(first.quit, true);
  const second = await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true,
    prompt: fakePromptQueue(['', '', '', '', '', '', '']) });
  assert.equal(second.reviewed, 3); // the remaining 3 tasks, not re-reviewing the first
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(new Set(lines.map(l => l.task_id)).size, 4);
});

test('review selection never selects two members of the same group (translation pair) in one pass', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uBB38\uC11C\uB97C \uAC80\uD1A0\uD574\uC918', group: 'pairA' },
    { lang: 'en', task: 'Review the document', group: 'pairA' },
    { lang: 'ko', task: '\uB2E4\uB978 \uC791\uC5C5 \uD45C\uC900', group: 'pairB' },
    { lang: 'en', task: 'another standalone task', group: 'pairB' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const result = await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, prompt: fakePromptQueue(Array(12).fill('')) });
  assert.equal(result.selected, 2); // one member per group at most, never both siblings in the same selection
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const byId = new Map(tasks.map(t => [t.task_id, t]));
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const groupsSeen = lines.map(l => byId.get(l.task_id).group);
  assert.equal(new Set(groupsSeen).size, groupsSeen.length); // no duplicate group among reviewed tasks
});

test('review never selects a task marked reviewable:false; it stays train-only via its teacher label', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC9E7\uC740 \uD569\uC131 \uBB38\uC7A5 1', reviewable: false },
    { lang: 'en', task: 'short synthetic sentence 1', reviewable: false },
    { lang: 'ko', task: '\uC2E4\uC81C \uC5D0\uC774\uC804\uD2B8 \uD504\uB86C\uD504\uD2B8\uCC98\uB7FC \uAE34 \uC791\uC5C5 \uC9C0\uC2DC\uBB38\uC785\uB2C8\uB2E4', reviewable: true },
    { lang: 'en', task: 'a realistic long agent prompt task instruction', reviewable: true },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const result = await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, prompt: fakePromptQueue(Array(12).fill('')) });
  assert.equal(result.selected, 2); // only the 2 reviewable tasks, never the reviewable:false ones
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const byId = new Map(tasks.map(t => [t.task_id, t]));
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.ok(lines.every(l => byId.get(l.task_id).reviewable === true));
});

test('distill status reports a reviewable count', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC791\uC5C5 A1', reviewable: false },
    { lang: 'en', task: 'task A2', reviewable: false },
    { lang: 'ko', task: '\uC791\uC5C5 B' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.reviewable, 1); // only the default-true task
});

test('distill status reports a groups count (distinct group keys; ungrouped tasks count as their own group)', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC791\uC5C5 A1', group: 'g1' },
    { lang: 'en', task: 'task A2', group: 'g1' },
    { lang: 'ko', task: '\uC791\uC5C5 B' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.groups, 2); // g1 (2 members) + the ungrouped task's own singleton group
});

// ============================== build ==============================

test('build: teacher-only tasks ground train with soft targets; reviewed tasks ground calibration/test and are excluded from train', async t => {
  const home = labeledFixture(t, 8); // 4 ko + 4 en
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'edit', difficulty: 2, risk: 'safe' }), sleepImpl: async () => {} });
  const written = [];
  await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, write: t => written.push(t),
    prompt: fakePromptQueue(Array(2 * 3).fill('')) }); // review only the first 2 selected (1 ko + 1 en) worth of answers
  const built = buildDistillDataset(home, { run: 'run1' });
  assert.match(built.dataset_version, /^[0-9a-f]{64}$/);
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  assert.equal(samples.length, built.sample_count);
  const trainTasks = new Set(samples.filter(s => s.split === 'train').map(s => s.task_id));
  const evalTasks = new Set(samples.filter(s => s.split !== 'train').map(s => s.task_id));
  for (const id of evalTasks) assert.equal(trainTasks.has(id), false); // no leakage: a reviewed task never also appears in train
  const trainSample = samples.find(s => s.split === 'train');
  assert.equal(trainSample.label_source, 'teacher');
  const sum = Object.values(trainSample.target.probabilities).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) <= 0.02);
  assert.notEqual(encode(trainSample.target.probabilities).includes('"1"') && Object.values(trainSample.target.probabilities).filter(v => v === 1).length, 1); // soft, not one-hot
  const evalSample = samples.find(s => s.split !== 'train');
  assert.equal(evalSample.label_source, 'human');
  assert.equal(evalSample.label_confidence, 1);
});

test('build: a non-reviewed task whose group has a reviewed member is excluded from train (no leakage via translation pair)', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC774 \uBB38\uC11C\uB97C \uC694\uC57D\uD574\uC918', group: 'pairA' },
    { lang: 'en', task: 'Summarize this document', group: 'pairA' },
    { lang: 'en', task: 'Add a retry to the fetch call' }, // ungrouped, stays teacher-only in train
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const pairKo = tasks.find(t => t.lang === 'ko');
  const pairEn = tasks.find(t => t.group === 'pairA' && t.lang === 'en');
  const lone = tasks.find(t => t.group === null);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  // Simulate a human review of ONLY the ko member of pairA (bypassing the TTY review flow, which
  // would already refuse to select both siblings at once — see the dedup test above).
  const reviewFilePath = path.join(runDir(home, 'run1'), 'review.jsonl');
  fs.writeFileSync(reviewFilePath, JSON.stringify({ task_id: pairKo.task_id, labels: { intent: 'edit', difficulty: 0, risk: 'safe' }, reviewer: 'human-tty', at: new Date().toISOString() }) + '\n', { mode: 0o600 });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  const trainSamples = samples.filter(s => s.split === 'train');
  const evalSamples = samples.filter(s => s.split !== 'train');
  assert.ok(!samples.some(s => s.raw_refs.distill.task_id === pairEn.task_id)); // teacher-labeled sibling never emitted
  assert.ok(trainSamples.some(s => s.raw_refs.distill.task_id === lone.task_id)); // unrelated teacher-only task still trains
  assert.ok(evalSamples.some(s => s.raw_refs.distill.task_id === pairKo.task_id)); // reviewed member grounds eval
  const manifest = JSON.parse(fs.readFileSync(built.manifest, 'utf8'));
  assert.equal(manifest.counts.excluded_group_leak, 1);
  const trainGroupIds = new Set(trainSamples.map(s => s.group_id));
  const evalGroupIds = new Set(evalSamples.map(s => s.group_id));
  for (const g of trainGroupIds) assert.equal(evalGroupIds.has(g), false); // no group_id crosses train/eval
  assert.equal(evalSamples.find(s => s.raw_refs.distill.task_id === pairKo.task_id).group_id, digest({ distill_group: 'pairA' }));
});

test('ungrouped datasets are unaffected by the group feature: group_id/split match the pre-group formula exactly', async t => {
  const home = labeledFixture(t, 4);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, prompt: fakePromptQueue(Array(6).fill('')) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  assert.ok(samples.length > 0);
  for (const s of samples) {
    const origTaskId = s.raw_refs.distill.task_id;
    assert.equal(s.group_id, digest({ distill_task: origTaskId })); // pre-group formula, byte-identical
    if (s.label_source === 'human') {
      const expectedSplit = parseInt(origTaskId.slice(0, 8), 16) % 2 === 0 ? 'calibration' : 'test';
      assert.equal(s.split, expectedSplit);
    }
  }
});

test('build is idempotent and writes into the shared training store root usable by export/holdout', async t => {
  const home = labeledFixture(t, 6);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  await distillReview(home, { run: 'run1', count: 4, isStdinTTY: true, isStdoutTTY: true, prompt: fakePromptQueue(Array(6).fill('')) });
  const first = buildDistillDataset(home, { run: 'run1' });
  const second = buildDistillDataset(home, { run: 'run1' });
  assert.equal(first.dataset_version, second.dataset_version);
  const store = createTrainingStore({ home });
  const holdout = freezeHoldout(home, { datasetVersion: first.dataset_version, name: 'distill-holdout', store });
  assert.ok(holdout.sample_count >= 0);
  const laya = exportDataset(store, first.dataset_version, 'laya');
  assert.equal(laya.dataset_version, first.dataset_version);
});

test('build excludes a task whose state fails the shared safe-content screen instead of failing the whole build, and reports it in counts.excluded_unsafe', async t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'Add a retry to the fetch call' }]) });
  // Simulate pre-fix data: a task that slipped past an older/weaker import screen and landed in
  // tasks.jsonl directly (distillImport itself now blocks this — see the import test above).
  const dir = runDir(home, 'run1');
  const unsafeTask = { task_id: digest({ lang: 'en', text: 'Escalate to test@company.com about the outage' }), lang: 'en', domain: null,
    task: 'Escalate to test@company.com about the outage', group: null, reviewable: true, source: 'synthetic', egress: 'allowed', added_at: new Date().toISOString() };
  fs.appendFileSync(path.join(dir, 'tasks.jsonl'), JSON.stringify(unsafeTask) + '\n');
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const built = buildDistillDataset(home, { run: 'run1' }); // must not throw TRAINING_SENSITIVE_OR_OVERSIZED for the whole build
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  assert.ok(samples.length > 0);
  assert.ok(!samples.some(s => s.raw_refs.distill.task_id === unsafeTask.task_id)); // no samples for the unsafe task at all
  const manifest = JSON.parse(fs.readFileSync(built.manifest, 'utf8'));
  assert.equal(manifest.counts.excluded_unsafe, 1);
});

test('distill status reports counts only, never task content', async t => {
  const home = labeledFixture(t, 3);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.tasks, 3);
  assert.equal(status.teacher_labels, 3);
  assert.equal(JSON.stringify(status).match(/task/gi)?.length > 0, true); // only field NAMES like "tasks"/"teacher_labels", not sentence content
  assert.equal(JSON.stringify(status).includes('\uD55C\uAD6D\uC5B4'), false);
  assert.equal(JSON.stringify(status).includes('english task'), false);
});

// ============================== readDataset extension ==============================

test('readDataset rejects a teacher-sourced label placed outside the train split', async t => {
  const home = labeledFixture(t, 2);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const canonicalPath = path.join(store.root, 'datasets', built.dataset_version, 'canonical.jsonl');
  const lines = fs.readFileSync(canonicalPath, 'utf8').trim().split('\n').map(JSON.parse);
  const tampered = lines.map(l => l.split === 'train' ? { ...l, split: 'test' } : l);
  const tamperedText = tampered.map(l => JSON.stringify(l)).join('\n') + '\n';
  atomicWrite(canonicalPath, tamperedText);
  const manifestPath = path.join(store.root, 'manifests', `${built.dataset_version}.json`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  manifest.data_sha256 = digest(tamperedText);
  atomicWrite(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  assert.throws(() => readDataset(store, built.dataset_version), /TEACHER_LABEL_IN_EVAL_SPLIT/);
});

test('every readDataset consumer (freezeHoldout, qualify) refuses a dataset whose eval rows carry a teacher label', async t => {
  const home = labeledFixture(t, 2);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const canonicalPath = path.join(store.root, 'datasets', built.dataset_version, 'canonical.jsonl');
  const lines = fs.readFileSync(canonicalPath, 'utf8').trim().split('\n').map(JSON.parse);
  const tampered = lines.map(l => l.split === 'train' ? { ...l, split: 'calibration' } : l);
  const tamperedText = tampered.map(l => JSON.stringify(l)).join('\n') + '\n';
  atomicWrite(canonicalPath, tamperedText);
  const manifestPath = path.join(store.root, 'manifests', `${built.dataset_version}.json`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  manifest.data_sha256 = digest(tamperedText);
  atomicWrite(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  assert.throws(() => freezeHoldout(home, { datasetVersion: built.dataset_version, name: 'h1', store }), /TEACHER_LABEL_IN_EVAL_SPLIT/);
  const checkpoint = 'a'.repeat(64);
  atomicWrite(path.join(home, 'laya', 'candidates', `${checkpoint}.json`),
    JSON.stringify({ python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint, runtimeVersion: '0.3.4', device: 'cpu', precision: 'fp32' }, null, 2) + '\n');
  await assert.rejects(qualifyCandidate(home, { candidateHash: checkpoint, datasetVersion: built.dataset_version, holdoutName: 'h1',
    layaClient: { infer: async () => ({}) } }), /TEACHER_LABEL_IN_EVAL_SPLIT|HOLDOUT/);
});

// ============================== import-reference (owner decision 2026-09-23) ==============================

function referenceLine(lang, task, overrides = {}) {
  return { lang, task, labels: { intent: 'edit', difficulty: 3, risk: 'safe', ...overrides } };
}

test('distill import-reference resolves a task by the same task_id rule as import, stores label_source-ready labels, and defaults role to eval', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'Add a retry to the fetch call' }]) });
  const r = distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, [referenceLine('en', 'Add a retry to the fetch call')]) });
  assert.equal(r.added, 1);
  assert.equal(r.role, 'eval');
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'reference.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].source, 'claude-sonnet-5');
  assert.equal(lines[0].role, 'eval');
  assert.equal(lines[0].labels.intent, 'edit');
  assert.equal(lines[0].labels.risk, 'safe');
  assert.equal(lines[0].labels.difficulty, 2); // human-facing 3 (1..5) stored 0-based like review
});

test('distill import-reference accepts --role train and stores it', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'ko', task: '\uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918' }]) });
  const r = distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'train',
    inputFile: writeInputFile(t, [referenceLine('ko', '\uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918')]) });
  assert.equal(r.role, 'train');
  const lines = fs.readFileSync(path.join(runDir(home, 'run1'), 'reference.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(lines[0].role, 'train');
});

test('distill import-reference rejects an invalid role', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x task' }]) });
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'bogus',
    inputFile: writeInputFile(t, [referenceLine('en', 'x task')]) }), /INVALID_DISTILL_REFERENCE_ROLE/);
});

test('distill import-reference validates --source against a safe model-id pattern', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x task' }]) });
  const file = writeInputFile(t, [referenceLine('en', 'x task')]);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'has a space', inputFile: file }), /INVALID_DISTILL_REFERENCE_SOURCE/);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: '../etc/passwd', inputFile: file }), /INVALID_DISTILL_REFERENCE_SOURCE/);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: '', inputFile: file }), /INVALID_DISTILL_REFERENCE_SOURCE/);
  const r = distillImportReference(home, { run: 'run1', source: 'claude-opus-4.6:20261001', inputFile: file });
  assert.equal(r.added, 1);
});

test('distill import-reference validates labels against ROUTE_QUESTIONS (bad choice key, out-of-range difficulty)', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'x task' }]) });
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, [referenceLine('en', 'x task', { intent: 'not-a-real-intent' })]) }), /INVALID_DISTILL_REFERENCE_LINE/);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, [referenceLine('en', 'x task', { difficulty: 0 })]) }), /INVALID_DISTILL_REFERENCE_LINE/);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, [referenceLine('en', 'x task', { difficulty: 6 })]) }), /INVALID_DISTILL_REFERENCE_LINE/);
});

test('distill import-reference fails the WHOLE import with DISTILL_REFERENCE_UNKNOWN_TASK before writing anything, when any task is unresolved', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'a known imported task' }]) });
  const file = writeInputFile(t, [
    referenceLine('en', 'a known imported task'),
    referenceLine('en', 'a task that was never imported'),
  ]);
  assert.throws(() => distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: file }), /DISTILL_REFERENCE_UNKNOWN_TASK/);
  const refPath = path.join(runDir(home, 'run1'), 'reference.jsonl');
  assert.equal(fs.existsSync(refPath), false); // nothing written, not even the resolvable line
});

test('distill import-reference skips a duplicate task_id already in reference.jsonl and counts it', t => {
  const home = fixture(t);
  distillImport(home, { run: 'run1', inputFile: writeInputFile(t, [{ lang: 'en', task: 'dup task' }]) });
  const file = writeInputFile(t, [referenceLine('en', 'dup task')]);
  const r1 = distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: file });
  assert.equal(r1.added, 1);
  const r2 = distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: file });
  assert.equal(r2.added, 0);
  assert.equal(r2.skippedDuplicate, 1);
});

// ============================== build: reference precedence ==============================

test('build: a reference eval label alone (no teacher label needed) grounds calibration/test with label_source ai_reference and a one-hot target', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC791\uC5C5 A' }, { lang: 'en', task: 'task B' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [
    referenceLine('ko', '\uC791\uC5C5 A'), referenceLine('en', 'task B'),
  ]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  assert.ok(samples.length > 0);
  assert.ok(samples.every(s => s.label_source === 'ai_reference'));
  assert.ok(samples.every(s => s.split === 'calibration' || s.split === 'test'));
  assert.ok(samples.every(s => s.label_confidence === 1));
  for (const s of samples) {
    const sum = Object.values(s.target.probabilities).reduce((a, b) => a + b, 0);
    assert.equal(sum, 1);
    assert.equal(Object.values(s.target.probabilities).filter(v => v === 1).length, 1); // one-hot, not soft
  }
});

test('build: a role:train reference label REPLACES the Jev-teacher sample for that task (teacher label never emitted)', async t => {
  const home = labeledFixture(t, 2); // 1 ko + 1 en
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const target = tasks[0];
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'debug', difficulty: 4, risk: 'high' }), sleepImpl: async () => {} });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'train', inputFile: writeInputFile(t, [
    referenceLine(target.lang, target.task, { intent: 'edit', difficulty: 1, risk: 'safe' }),
  ]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  const targetSamples = samples.filter(s => s.raw_refs.distill.task_id === target.task_id);
  assert.ok(targetSamples.length > 0);
  assert.ok(targetSamples.every(s => s.label_source === 'ai_reference'));
  assert.ok(targetSamples.every(s => s.split === 'train'));
  const intentSample = targetSamples.find(s => s.question_id === 'intent');
  assert.equal(intentSample.target.value, 'edit'); // the Claude train label, not the teacher's 'debug'
});

test('build precedence: human review wins over a role:eval reference label for the same task', async t => {
  const home = labeledFixture(t, 2);
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const target = tasks[0];
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [
    referenceLine(target.lang, target.task, { intent: 'research', difficulty: 5, risk: 'unknown' }),
  ]) });
  // Bypass the TTY review flow to record a human label directly (same technique as the existing group-leak test).
  fs.writeFileSync(path.join(runDir(home, 'run1'), 'review.jsonl'),
    JSON.stringify({ task_id: target.task_id, labels: { intent: 'edit', difficulty: 0, risk: 'safe' }, reviewer: 'human-tty', at: new Date().toISOString() }) + '\n', { mode: 0o600 });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  const targetSamples = samples.filter(s => s.raw_refs.distill.task_id === target.task_id);
  assert.ok(targetSamples.every(s => s.label_source === 'human'));
  const intentSample = targetSamples.find(s => s.question_id === 'intent');
  assert.equal(intentSample.target.value, 'edit'); // human label, not the reference's 'research'
});

test('build: a train-role reference/teacher task is excluded from train when its group has an eval member (human OR eval reference)', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC774 \uBB38\uC11C\uB97C \uC694\uC57D\uD574\uC918', group: 'pairA' },
    { lang: 'en', task: 'Summarize this document', group: 'pairA' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  const koTask = tasks.find(t => t.lang === 'ko'), enTask = tasks.find(t => t.lang === 'en');
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  // ko member gets an eval-role reference label (an eval-grounding member); en sibling only has a teacher label.
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [referenceLine('ko', koTask.task)]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const { samples } = readDataset(store, built.dataset_version);
  assert.ok(!samples.some(s => s.raw_refs.distill.task_id === enTask.task_id)); // teacher-labeled sibling excluded (group leak)
  assert.ok(samples.some(s => s.raw_refs.distill.task_id === koTask.task_id && s.label_source === 'ai_reference'));
  const manifest = JSON.parse(fs.readFileSync(built.manifest, 'utf8'));
  assert.equal(manifest.counts.excluded_group_leak, 1);
});

test('manifest counts.label_source_by_split breaks ai_reference/human/teacher out per split', async t => {
  const home = labeledFixture(t, 4); // 2 ko + 2 en
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0'), sleepImpl: async () => {} });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [
    referenceLine(tasks[0].lang, tasks[0].task), // role:eval -> calibration or test
  ]) });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'train', inputFile: writeInputFile(t, [
    referenceLine(tasks[1].lang, tasks[1].task), // role:train -> train, replaces teacher
  ]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const manifest = JSON.parse(fs.readFileSync(built.manifest, 'utf8'));
  assert.ok(manifest.counts.label_source_by_split.train.ai_reference >= 1);
  assert.ok(manifest.counts.label_source_by_split.train.teacher >= 1);
  const evalAiRef = (manifest.counts.label_source_by_split.calibration?.ai_reference ?? 0) + (manifest.counts.label_source_by_split.test?.ai_reference ?? 0);
  assert.ok(evalAiRef >= 1);
});

// ============================== compare-teacher ==============================

test('distill compare-teacher reports eval_reference and train_reference separately with agreement/confusion/per-lang/difficulty stats', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'ko', task: '\uC791\uC5C5 1' }, { lang: 'en', task: 'task 2' }, { lang: 'en', task: 'task 3' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: async () => teacherRaw('jev-1.13.0', { intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  const [t1, t2, t3] = tasks;
  // t1: eval reference that AGREES with teacher; t2: train reference that DISAGREES; t3: no reference (excluded from report).
  // teacherRaw picks difficulty 0-based (=1); referenceLine's difficulty is 1..5 human-facing, stored (n-1).
  // t1: input difficulty 2 -> stored 1, matching the teacher's stored 1 exactly (agreement).
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [referenceLine(t1.lang, t1.task, { intent: 'edit', difficulty: 2, risk: 'safe' })]) });
  // t2: input difficulty 5 -> stored 4, vs teacher's stored 1: |1-4|=3 (disagreement).
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'train', inputFile: writeInputFile(t, [referenceLine(t2.lang, t2.task, { intent: 'debug', difficulty: 5, risk: 'high' })]) });
  const report = distillCompareTeacher(home, { run: 'run1' });
  assert.equal(report.eval_reference.compared, 1);
  assert.equal(report.train_reference.compared, 1);
  assert.equal(report.eval_reference.questions.intent.agreement.count, 1);
  assert.equal(report.eval_reference.questions.intent.agreement.rate, 1);
  assert.equal(report.train_reference.questions.intent.agreement.count, 0);
  assert.equal(report.train_reference.questions.intent.agreement.rate, 0);
  // confusion matrix: teacher's argmax key -> reference key -> count
  assert.equal(report.train_reference.questions.intent.confusion.edit.debug, 1);
  // difficulty: teacher picked 1 (0-based), reference picked 5->4 (0-based); |0-4|=4
  assert.equal(report.train_reference.questions.difficulty.mean_absolute_difference, 3);
  assert.equal(report.train_reference.questions.difficulty.within_one.count, 0);
  assert.equal(report.eval_reference.questions.difficulty.mean_absolute_difference, 0);
  assert.equal(report.eval_reference.questions.difficulty.within_one.rate, 1);
  // per-language rates present
  assert.ok(Object.hasOwn(report.eval_reference.questions.intent.by_lang, t1.lang));
});

test('distill compare-teacher by_split counts sum to the eval_reference total', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'en', task: 'task alpha' }, { lang: 'en', task: 'task bravo' }, { lang: 'ko', task: '\uC791\uC5C5 \uCC30\uB9AC' },
    { lang: 'ko', task: '\uC791\uC5C5 \uB378\uD0C0' }, { lang: 'en', task: 'task echo' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: fakeProvider({ intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, tasks.map(t => referenceLine(t.lang, t.task))) });
  const report = distillCompareTeacher(home, { run: 'run1' });
  assert.equal(report.eval_reference.compared, 5);
  assert.equal(report.eval_reference.by_split.calibration.compared + report.eval_reference.by_split.test.compared, 5);
  // Every by_split section has full question coverage, not just the totals.
  for (const splitName of ['calibration', 'test']) {
    assert.ok(Object.hasOwn(report.eval_reference.by_split[splitName], 'questions'));
    assert.ok(Object.hasOwn(report.eval_reference.by_split[splitName].questions, 'intent'));
  }
  // train_reference (no eval-split concept) must not get a by_split key.
  assert.equal(report.train_reference.by_split, undefined);
});

test('distill compare-teacher by_split assigns each task to the same calibration/test split buildDistillDataset uses', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [
    { lang: 'en', task: 'task alpha' }, { lang: 'en', task: 'task bravo' }, { lang: 'ko', task: '\uC791\uC5C5 \uCC30\uB9AC' },
    { lang: 'ko', task: '\uC791\uC5C5 \uB378\uD0C0' }, { lang: 'en', task: 'task echo' }, { lang: 'en', task: 'task foxtrot' },
  ]);
  distillImport(home, { run: 'run1', inputFile: file });
  await distillLabel(home, { run: 'run1', confirmEgress: true, key: 'k', provider: fakeProvider({ intent: 'edit', difficulty: 1, risk: 'safe' }), sleepImpl: async () => {} });
  const tasks = fs.readFileSync(path.join(runDir(home, 'run1'), 'tasks.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5',
    inputFile: writeInputFile(t, tasks.map(t => referenceLine(t.lang, t.task))) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const { samples: dataset } = readDataset(createTrainingStore({ home }), built.dataset_version);
  const report = distillCompareTeacher(home, { run: 'run1' });
  // hashToUuid/group_id/split assignment are private to laya-distill.mjs, so cross-check indirectly:
  // the build's own per-split sample count (this run only ever emits label_source:'ai_reference' via
  // eval-role reference lines) must match the report's eval_reference by_split count for that split.
  const builtSplitCounts = { calibration: 0, test: 0 };
  for (const s of dataset) if (s.label_source === 'ai_reference') builtSplitCounts[s.split] = (builtSplitCounts[s.split] ?? 0) + 1;
  // Each task contributes one sample per ROUTE_QUESTIONS entry to the built dataset but one pair to
  // the report; dividing by the question count recovers per-task counts for a direct comparison.
  const questionCount = Object.keys(ROUTE_QUESTIONS).length;
  assert.equal(builtSplitCounts.calibration / questionCount, report.eval_reference.by_split.calibration.compared);
  assert.equal(builtSplitCounts.test / questionCount, report.eval_reference.by_split.test.compared);
});

// ============================== status: reference_labels ==============================

test('distill status reports reference_labels total and by role', t => {
  const home = fixture(t);
  const file = writeInputFile(t, [{ lang: 'en', task: 'task A' }, { lang: 'en', task: 'task B' }]);
  distillImport(home, { run: 'run1', inputFile: file });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [referenceLine('en', 'task A')]) });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', role: 'train', inputFile: writeInputFile(t, [referenceLine('en', 'task B')]) });
  const status = distillStatus(home, { run: 'run1' });
  assert.equal(status.reference_labels, 2);
  assert.equal(status.reference_labels_by_role.eval, 1);
  assert.equal(status.reference_labels_by_role.train, 1);
});

// ============================== holdout / qualify / compare over ai_reference ==============================

// Generic fake worker: answers ANY single-question ROUTE_QUESTIONS-shaped request (choice/score)
// with a fixed high-confidence pick, so qualify/compare can run end to end without a real model.
function genericFakeLayaClient() {
  return {
    status: () => ({ running: false }), close: () => {}, prepare: async () => ({}),
    async infer(payload, settings) {
      const laya = settings.laya;
      const [qid, q] = Object.entries(payload.questions)[0];
      let answer;
      if (q.type === 'choice') {
        const keys = Object.keys(q.criteria);
        const other = keys.length > 1 ? .1 / (keys.length - 1) : 0;
        answer = { type: 'choice', choice: keys[0], confidence: .9, probabilities: Object.fromEntries(keys.map((k, i) => [k, i === 0 ? .9 : other])) };
      } else if (q.type === 'score') {
        const n = q.criteria.length;
        // Sharply concentrated on index 0 so the expected value (sum k*p_k) stays within the
        // 0.05 tolerance of the reported score=0 (contracts.mjs normalizeResponse).
        const other = n > 1 ? .001 / (n - 1) : 0;
        answer = { type: 'score', score: 0, confidence: .9, probabilities: Object.fromEntries(Array.from({ length: n }, (_, i) => [String(i), i === 0 ? 1 - other * (n - 1) : other])) };
      } else {
        answer = { type: 'noul', noul: .9 };
      }
      return { identity: { model: laya.model, checkpoint: laya.checkpoint, runtime_version: laya.runtimeVersion, device: laya.device,
        precision: laya.precision === 'fp16' ? 'torch.float16' : 'torch.float32' },
        answers: { [qid]: answer }, usage: { input_tokens: 5, output_tokens: 0 } };
    },
  };
}
function writeCandidate(home, checkpoint) {
  fs.mkdirSync(path.join(home, 'laya', 'candidates'), { recursive: true, mode: 0o700 });
  atomicWrite(path.join(home, 'laya', 'candidates', `${checkpoint}.json`),
    JSON.stringify({ python: '/usr/bin/python3', modelPath: home, model: 'laya/base', checkpoint, runtimeVersion: '0.3.4', device: 'cpu', precision: 'fp32' }, null, 2) + '\n');
}

test('laya holdout freeze accepts ai_reference samples from the test split', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [{ lang: 'ko', task: '\uC791\uC5C5 A' }, { lang: 'en', task: 'task B' }]);
  distillImport(home, { run: 'run1', inputFile: file });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [referenceLine('ko', '\uC791\uC5C5 A'), referenceLine('en', 'task B')]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const holdout = freezeHoldout(home, { datasetVersion: built.dataset_version, name: 'ref-holdout', store });
  assert.ok(holdout.sample_count > 0);
  const holdoutLines = fs.readFileSync(path.join(home, 'laya', 'holdouts', 'ref-holdout.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.ok(holdoutLines.some(l => l.label_source === 'ai_reference'));
});

test('qualify reports evaluation_label_sources and metric_semantics:agreement-with-ai-reference when the dataset carries ai_reference eval labels', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [{ lang: 'ko', task: '\uC791\uC5C5 A' }, { lang: 'en', task: 'task B' }, { lang: 'en', task: 'task C' }]);
  distillImport(home, { run: 'run1', inputFile: file });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [
    referenceLine('ko', '\uC791\uC5C5 A'), referenceLine('en', 'task B'), referenceLine('en', 'task C'),
  ]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const holdout = freezeHoldout(home, { datasetVersion: built.dataset_version, name: 'q-holdout', store });
  assert.ok(holdout.sample_count > 0);
  const checkpoint = 'b'.repeat(64);
  writeCandidate(home, checkpoint);
  const result = await qualifyCandidate(home, { candidateHash: checkpoint, datasetVersion: built.dataset_version, holdoutName: 'q-holdout',
    layaClient: genericFakeLayaClient(), targetAccuracy: 0.5, minCoverage: 0.01, minCalibration: 1, minTest: 1, minLowerBound: 0 });
  assert.equal(result.metric_semantics, 'agreement-with-ai-reference');
  assert.ok(result.evaluation_label_sources.ai_reference > 0);
});

test('compare reports evaluation_label_sources and metric_semantics over the same holdout it evaluates', async t => {
  const home = fixture(t);
  const file = writeInputFile(t, [{ lang: 'ko', task: '\uC791\uC5C5 A' }, { lang: 'en', task: 'task B' }]);
  distillImport(home, { run: 'run1', inputFile: file });
  distillImportReference(home, { run: 'run1', source: 'claude-sonnet-5', inputFile: writeInputFile(t, [referenceLine('ko', '\uC791\uC5C5 A'), referenceLine('en', 'task B')]) });
  const built = buildDistillDataset(home, { run: 'run1' });
  const store = createTrainingStore({ home });
  const holdout = freezeHoldout(home, { datasetVersion: built.dataset_version, name: 'c-holdout', store });
  assert.ok(holdout.sample_count > 0);
  const checkpoint = 'c'.repeat(64);
  writeCandidate(home, checkpoint);
  const cmp = await compareCandidate(home, { candidateHash: checkpoint, holdoutName: 'c-holdout', layaClient: genericFakeLayaClient(), noActiveBaseline: true });
  assert.equal(cmp.metric_semantics, 'agreement-with-ai-reference');
  assert.ok(cmp.evaluation_label_sources.ai_reference > 0);
});
