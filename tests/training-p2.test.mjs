import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createTrainingStore, DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE } from '../src/training/store.mjs';
import { digest } from '../src/training/schema.mjs';
import { evaluateDecision, evaluateStore } from '../src/training/evaluate.mjs';
import { buildDataset, exportDataset } from '../src/training/dataset.mjs';
import { allowRunner, removeRunner, listRunners, verifyRunner, validateRunnerCheck } from '../src/training/runner.mjs';
import { correctDecision } from '../src/training/correct.mjs';
import { createLinkIndex, recordSubagentStop, LINK_RETENTION_MS, MAX_LINKS } from '../src/training/links.mjs';
import { fixture, request, response, trace, decision, outcome, KEY } from './training-helpers.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const cli = (f, args, input) => spawnSync(process.execPath, [path.join(ROOT, 'bin/pointsman.mjs'), ...args], {
  input, encoding: 'utf8', timeout: 8000, env: { ...process.env, HOME: f.home, POINTSMAN_HOME: f.home, TYPESAFE_API_KEY: '', POINTSMAN_DISABLE: '0' },
});

// ---- CLI outcome/MCP style ingestion can never assert runner/human authority ----
test('CLI training outcome always downgrades to host_review, even when the pipe claims runner/human', t => {
  const f = fixture(t); const d = f.save(decision());
  const claimed = { ...outcome(d), source: 'human', labels: outcome(d).labels.map(l => ({ ...l, source: 'human' })) };
  const r = cli(f, ['training', 'outcome'], JSON.stringify(claimed));
  assert.equal(r.status, 0, r.stderr);
  const stored = JSON.parse(r.stdout);
  assert.equal(stored.stored, true);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes').data;
  assert.equal(o.source, 'host_review');
  assert.equal(o.labels[0].source, 'host_review');
  assert.equal(evaluateDecision({ event_id: d.decision_id, data: d }, [{ event_id: o.decision_id, data: o }]).labels.length, 0);
});

// ---- runner.mjs registration ----
test('runner registration rejects malformed checks, route+label, and requires TTY', async t => {
  assert.throws(() => validateRunnerCheck({ argv: [], timeoutMs: 100 }), /INVALID_RUNNER_CHECK/);
  assert.throws(() => validateRunnerCheck({ argv: ['true'], timeoutMs: 0 }), /INVALID_RUNNER_CHECK/);
  assert.throws(() => validateRunnerCheck({ argv: ['true'], timeoutMs: 100, question_id: 'q' }), /INVALID_RUNNER_CHECK/);
  assert.throws(() => validateRunnerCheck({ argv: ['true'], timeoutMs: 100, pass_label: 'a' }), /INVALID_RUNNER_CHECK/);
  assert.throws(() => validateRunnerCheck({ argv: ['true'], timeoutMs: 100, purpose: 'route', question_id: 'q', pass_label: 'a', fail_label: 'b' }), /RUNNER_ROUTE_LABEL_REFUSED/);
  assert.throws(() => validateRunnerCheck({ argv: ['true'], timeoutMs: 100, purpose: 'judge', question_id: 'q', pass_label: 'a', fail_label: 'a' }), /INVALID_RUNNER_CHECK/);
  const f = fixture(t, false);
  await assert.rejects(allowRunner({ home: f.home, name: 'x', check: { argv: ['true'], timeoutMs: 100 }, isStdinTTY: false, isStdoutTTY: true, prompt: async () => 'x' }), /HUMAN_TTY_REQUIRED/);
  await assert.rejects(allowRunner({ home: f.home, name: 'x', check: { argv: ['true'], timeoutMs: 100 }, isStdinTTY: true, isStdoutTTY: false, prompt: async () => 'x' }), /HUMAN_TTY_REQUIRED/);
});
test('runner allow requires typed re-confirmation and refuses silent overwrite', async t => {
  const f = fixture(t, false);
  const check = { argv: ['true'], timeoutMs: 100 };
  await assert.rejects(allowRunner({ home: f.home, name: 'x', check, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'wrong', write: () => {} }), /RUNNER_CONFIRMATION_MISMATCH/);
  assert.deepEqual(listRunners(f.home).checks, {});
  const r = await allowRunner({ home: f.home, name: 'x', check, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'x', write: () => {} });
  assert.equal(r.stored, true);
  await assert.rejects(allowRunner({ home: f.home, name: 'x', check, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'x', write: () => {} }), /RUNNER_CHECK_EXISTS/);
  const replaced = await allowRunner({ home: f.home, name: 'x', check: { ...check, timeoutMs: 500 }, replace: true, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'x', write: () => {} });
  assert.equal(replaced.stored, true);
  assert.equal(listRunners(f.home).checks.x.timeoutMs, 500);
  await assert.rejects(removeRunner({ home: f.home, name: 'x', isStdinTTY: false, isStdoutTTY: true, prompt: async () => 'x' }), /HUMAN_TTY_REQUIRED/);
  await assert.rejects(removeRunner({ home: f.home, name: 'missing', isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'missing' }), /RUNNER_CHECK_NOT_FOUND/);
  const removed = await removeRunner({ home: f.home, name: 'x', isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'x' });
  assert.equal(removed.removed, true);
  assert.deepEqual(listRunners(f.home).checks, {});
});

// ---- runner verify: pointsman executes the pre-registered assertion directly, never a caller-chosen command ----
async function registerAndDecide(f, check, questions = { worker: { type: 'choice', instructions: 'pick', criteria: { light: 'l', strong: 's' } } }, purpose = 'judge') {
  await allowRunner({ home: f.home, name: 'chk', check, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'chk', write: () => {} });
  const d = await f.engine.decide({ purpose, risk: 'routine', state: { x: 1 }, questions, trace: { task_id: randomUUID(), snapshot_id: digest('snap') } });
  return d;
}
test('runner verify records a passing objective label and discards stdout/stderr', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'console.log("SECRET_STDOUT_MARKER");process.exit(0)'], timeoutMs: 3000, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const v = await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  assert.equal(v.passed, true); assert.equal(v.exit_status, 0);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes' && e.data.decision_id === d.id).data;
  assert.equal(o.source, 'runner'); assert.equal(o.executed, false);
  assert.equal(o.labels[0].value, 'light'); assert.equal(o.labels[0].source, 'objective');
  const label = o.checks.find(c => c.kind === 'label'); assert.equal(label.passed, true);
  assert.equal(JSON.stringify(o).includes('SECRET_STDOUT_MARKER'), false);
  const e = evaluateDecision({ event_id: d.id, data: { ...f.store.scan().events.find(x => x.kind === 'decisions' && x.data.decision_id === d.id).data } },
    f.store.scan().events.filter(x => x.kind === 'outcomes' && x.data.decision_id === d.id));
  assert.equal(e.labels[0].value, 'light');
});
test('runner verify records the fail label on nonzero exit without inventing task success', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'process.exit(1)'], timeoutMs: 3000, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const v = await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  assert.equal(v.passed, false); assert.equal(v.exit_status, 1);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes').data;
  assert.equal(o.labels[0].value, 'strong');
  assert.equal(o.checks.find(c => c.kind === 'command').passed, false);
  assert.equal(o.checks.find(c => c.kind === 'label').passed, true);
});
test('runner verify kills a timed-out command and records failure without an exit status', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'setTimeout(()=>{},5000)'], timeoutMs: 150 }, undefined, 'select');
  const v = await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  assert.equal(v.timed_out, true); assert.equal(v.passed, false); assert.equal(v.exit_status, null);
});
test('runner verify creates no label when the check timed out or could not start', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'setTimeout(()=>{},5000)'], timeoutMs: 150, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const v = await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  assert.equal(v.timed_out, true);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes' && e.data.decision_id === d.id).data;
  assert.deepEqual(o.labels, []); assert.equal(o.checks.some(c => c.kind === 'label'), false);
  const f2 = fixture(t);
  const d2 = await registerAndDecide(f2, { argv: ['/nonexistent/pointsman-runner-missing-binary'], timeoutMs: 1000, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const v2 = await verifyRunner({ store: f2.store, home: f2.home, decisionId: d2.id, checkName: 'chk' });
  assert.equal(v2.passed, false);
  const o2 = f2.store.scan().events.find(e => e.kind === 'outcomes' && e.data.decision_id === d2.id).data;
  assert.deepEqual(o2.labels, []);
});
test('runner verify does not pass the TypeSafe key to the registered command', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'process.exit(process.env.TYPESAFE_API_KEY ? 3 : 0)'], timeoutMs: 3000, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const saved = process.env.TYPESAFE_API_KEY; process.env.TYPESAFE_API_KEY = 'ts_fixture_not_a_real_key';
  t.after(() => { if (saved === undefined) delete process.env.TYPESAFE_API_KEY; else process.env.TYPESAFE_API_KEY = saved; });
  const v = await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  assert.equal(v.exit_status, 0);
});
test('runner verify enforces purpose/question consistency and cannot be steered by the caller', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: ['true'], timeoutMs: 500, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  const other = await f.engine.decide({ purpose: 'select', risk: 'routine', state: { x: 2 },
    questions: { worker: { type: 'choice', instructions: 'pick', criteria: { light: 'l', strong: 's' } } }, trace: { task_id: randomUUID(), snapshot_id: digest('s2') } });
  await assert.rejects(verifyRunner({ store: f.store, home: f.home, decisionId: other.id, checkName: 'chk' }), /RUNNER_PURPOSE_MISMATCH/);
  await assert.rejects(verifyRunner({ store: f.store, home: f.home, decisionId: randomUUID(), checkName: 'chk' }), /DECISION_NOT_FOUND/);
  await assert.rejects(verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'missing' }), /RUNNER_CHECK_NOT_FOUND/);
  const missingQuestion = await allowRunner({ home: f.home, name: 'chk2', check: { argv: ['true'], timeoutMs: 500, purpose: 'judge', question_id: 'nope', pass_label: 'light', fail_label: 'strong' },
    isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'chk2', write: () => {} });
  assert.equal(missingQuestion.stored, true);
  await assert.rejects(verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk2' }), /RUNNER_QUESTION_NOT_FOUND/);
});
test('runner verify is a no-op when capture is off', async t => {
  const f = fixture(t, false);
  await allowRunner({ home: f.home, name: 'chk', check: { argv: ['true'], timeoutMs: 500 }, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'chk', write: () => {} });
  const r = await verifyRunner({ store: f.store, home: f.home, decisionId: randomUUID(), checkName: 'chk' });
  assert.equal(r.stored, false); assert.equal(r.reason, 'CAPTURE_OFF');
});

// ---- training correct: TTY-gated human labels ----
test('training correct requires TTY, an existing decision, and records source human labels', async t => {
  const f = fixture(t); const d = f.save(decision());
  await assert.rejects(correctDecision({ store: f.store, decisionId: d.decision_id, isStdinTTY: false, isStdoutTTY: true, prompt: async () => 'light' }), /HUMAN_TTY_REQUIRED/);
  await assert.rejects(correctDecision({ store: f.store, decisionId: randomUUID(), isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'light' }), /DECISION_NOT_FOUND/);
  const r = await correctDecision({ store: f.store, decisionId: d.decision_id, isStdinTTY: true, isStdoutTTY: true, prompt: async () => 'strong', write: () => {} });
  assert.equal(r.stored, true); assert.equal(r.labels_recorded, 1);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes').data;
  assert.equal(o.source, 'human'); assert.equal(o.labels[0].value, 'strong'); assert.equal(o.labels[0].source, 'human');
});
test('training correct skips a question left blank and reports no labels when all are skipped', async t => {
  const f = fixture(t); const d = f.save(decision());
  const r = await correctDecision({ store: f.store, decisionId: d.decision_id, isStdinTTY: true, isStdoutTTY: true, prompt: async () => '', write: () => {} });
  assert.equal(r.stored, false); assert.equal(r.reason, 'NO_LABELS_PROVIDED');
});
test('training correct validates noul and score answers against the actual question', async t => {
  const f = fixture(t); const d = decision();
  d.request.questions = { flag: { type: 'noul', instructions: 'ok?' }, depth: { type: 'score', instructions: 'how deep', criteria: ['a', 'b', 'c'] } };
  d.request_hash = digest(d.request); d.answers = { flag: { type: 'noul', value: false, confidence: null, probabilityTrue: .1 }, depth: { type: 'score', value: 0, confidence: .9, probabilities: { 0: 1, 1: 0, 2: 0 } } };
  f.save(d);
  let calls = 0;
  const answers = ['true', '2'];
  const r = await correctDecision({ store: f.store, decisionId: d.decision_id, isStdinTTY: true, isStdoutTTY: true, prompt: async () => answers[calls++], write: () => {} });
  assert.equal(r.labels_recorded, 2);
  const o = f.store.scan().events.find(e => e.kind === 'outcomes').data;
  assert.deepEqual(o.labels.map(l => [l.question_id, l.value]).sort(), [['depth', 2], ['flag', true]]);
});

// ---- min-labels gate ----
test('min-labels defaults to 100, gates dataset build, and is bypassable only with --allow-small', t => {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-minlabels-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  const store = createTrainingStore({ home }); store.setCapture(true);
  assert.equal(store.status().minStrongLabelsPerPurpose, DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE);
  const d = decision(); store.decision(d); store.outcome(outcome(d));
  evaluateStore(store);
  const blocked = buildDataset(store);
  assert.equal(blocked.built, false); assert.equal(blocked.reason, 'DATASET_TOO_SMALL');
  assert.equal(blocked.shortfall.route, 1);
  assert.equal(fs.existsSync(path.join(store.root, 'manifests')), false);
  const built = buildDataset(store, { allowSmall: true });
  assert.equal(built.sample_count, 1);
  const manifest = JSON.parse(fs.readFileSync(built.manifest, 'utf8'));
  assert.equal(manifest.small_sample_override, true);
  assert.equal(manifest.min_strong_labels_per_purpose, DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE);
  // Lowering the threshold does not change dataset content or its version hash.
  store.setMinLabels(1);
  const rebuilt = buildDataset(store);
  assert.equal(rebuilt.dataset_version, built.dataset_version);
  assert.throws(() => store.setMinLabels(0), /INVALID_TRAINING_CONFIG/);
  assert.throws(() => store.setMinLabels(1.5), /INVALID_TRAINING_CONFIG/);
});
test('export refuses an empty split even with allowSmall at build time', t => {
  const f = fixture(t); const d = f.save(decision()); f.store.outcome(outcome(d));
  const built = buildDataset(f.store, { allowSmall: true });
  assert.throws(() => exportDataset(f.store, built.dataset_version), /EMPTY_SPLIT/);
  assert.ok(exportDataset(f.store, built.dataset_version, 'canonical').files.length);
});
test('CLI training min-labels reads the threshold but refuses to change it from a pipe', t => {
  const f = fixture(t, false);
  const status = cli(f, ['training', 'min-labels']); assert.equal(status.status, 0, status.stderr);
  assert.equal(JSON.parse(status.stdout).minStrongLabelsPerPurpose, DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE);
  // A non-TTY agent must not be able to lower the dataset gate.
  const set = cli(f, ['training', 'min-labels', '5']); assert.notEqual(set.status, 0);
  assert.match(set.stderr + set.stdout, /HUMAN_TTY_REQUIRED/);
  assert.equal(f.store.config().minStrongLabelsPerPurpose, DEFAULT_MIN_STRONG_LABELS_PER_PURPOSE);
});

// ---- links.mjs: tool_use/agent -> decision correlation, no transcript content ----
test('link index puts, aliases, expires, and caps without storing any content', t => {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-links-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  let clock = 1000000;
  const links = createLinkIndex({ home, now: () => clock });
  const decisionId = randomUUID();
  assert.equal(links.get({ host: 'claude', kind: 'tool_use', id: 'tu_1' }), null);
  links.put({ host: 'claude', kind: 'tool_use', id: 'tu_1', decision_id: decisionId });
  assert.equal(links.get({ host: 'claude', kind: 'tool_use', id: 'tu_1' }), decisionId);
  const aliased = links.alias({ host: 'claude', fromKind: 'tool_use', fromId: 'tu_1', toKind: 'agent', toId: 'agent_1' });
  assert.equal(aliased.stored, true);
  assert.equal(links.get({ host: 'claude', kind: 'agent', id: 'agent_1' }), decisionId);
  assert.equal(links.alias({ host: 'claude', fromKind: 'tool_use', fromId: 'no_such', toKind: 'agent', toId: 'agent_2' }).stored, false);
  for (const file of fs.readdirSync(links.root)) assert.equal(JSON.stringify(JSON.parse(fs.readFileSync(path.join(links.root, file), 'utf8'))).includes('tool_use_id'), false);
  clock += LINK_RETENTION_MS + 1;
  assert.equal(links.get({ host: 'claude', kind: 'agent', id: 'agent_1' }), null);
});
test('link index evicts the oldest entries once past its cap', t => {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-links-cap-')));
  t.after(() => fs.rmSync(home, { recursive: true, force: true }));
  let clock = 0;
  const cap = 20; // exercises the real MAX_LINKS eviction path without 10000 real fsync'd writes
  const links = createLinkIndex({ home, now: () => clock, maxLinks: cap });
  for (let i = 0; i < cap + 5; i++) { clock++; links.put({ host: 'codex', kind: 'agent', id: `a${i}`, decision_id: randomUUID() }); }
  assert.equal(links.get({ host: 'codex', kind: 'agent', id: 'a0' }), null);
  assert.notEqual(links.get({ host: 'codex', kind: 'agent', id: `a${cap + 4}` }), null);
  assert.ok(fs.readdirSync(links.root).length <= cap);
});
test('link index MAX_LINKS matches the documented default (10000)', () => {
  assert.equal(MAX_LINKS, 10000);
});
test('SubagentStop records a weak host_review outcome only when a decision is linked, never task success', t => {
  const f = fixture(t); const d = f.save(decision());
  const links = createLinkIndex({ home: f.home });
  assert.equal(recordSubagentStop(f.store, links, { host: 'claude', agent_id: 'agent_x', status: 'completed' }).stored, false);
  links.put({ host: 'claude', kind: 'agent', id: 'agent_x', decision_id: d.decision_id });
  const completed = recordSubagentStop(f.store, links, { host: 'claude', agent_id: 'agent_x', status: 'completed' });
  assert.equal(completed.stored, true);
  const failed = recordSubagentStop(f.store, links, { host: 'claude', agent_id: 'agent_x', status: 'failed' });
  assert.equal(failed.stored, true);
  const events = f.store.scan().events.filter(e => e.kind === 'outcomes');
  const byReview = Object.fromEntries(events.map(e => [e.data.host_review, e.data]));
  assert.equal(byReview.uncertain.source, 'host_review'); assert.equal(byReview.uncertain.metrics.task_succeeded, null);
  assert.equal(byReview.fail.source, 'host_review');
  assert.throws(() => recordSubagentStop(f.store, links, { host: 'claude', agent_id: 'agent_x', status: 'bogus' }), /INVALID_SUBAGENT_STATUS/);
});

// ---- end-to-end: decision -> runner verify -> evaluate -> build(--allow-small) -> export ----
test('synthetic end-to-end: judge decision verified by a registered runner check yields a strong exportable label', async t => {
  const f = fixture(t);
  const d = await registerAndDecide(f, { argv: [process.execPath, '-e', 'process.exit(0)'], timeoutMs: 2000, purpose: 'judge', question_id: 'worker', pass_label: 'light', fail_label: 'strong' });
  await verifyRunner({ store: f.store, home: f.home, decisionId: d.id, checkName: 'chk' });
  evaluateStore(f.store);
  const built = buildDataset(f.store, { allowSmall: true });
  assert.equal(built.sample_count, 1);
  const exported = exportDataset(f.store, built.dataset_version, 'canonical');
  assert.equal(exported.samples, 1);
});
