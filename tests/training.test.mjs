import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { atomicWrite, setMode } from '../src/storage.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createTrainingStore, outsideGit } from '../src/training/store.mjs';
import { digest, encode, POLICY_VERSION, MAX_DERIVED_BYTES, validateOutcome, safeContent } from '../src/training/schema.mjs';
import { evaluateDecision, evaluateStore, summarizeComparisons, indexEvents } from '../src/training/evaluate.mjs';
import { buildDataset, readDataset, exportDataset, validateDatasetSource } from '../src/training/dataset.mjs';
import { recordHost } from '../src/training/host.mjs';
import { fixture, request, response, trace, decision, outcome, REF, KEY, stateForSplit } from './training-helpers.mjs';
const evaluation = (store, d) => { const s = store.scan(); return evaluateDecision(s.events.find(e => e.kind === 'decisions' && e.data.decision_id === d.decision_id), s.events.filter(e => e.kind === 'outcomes' && e.data.decision_id === d.decision_id)); };

test('capture defaults OFF: inference and outcome create no content-bearing training files', async t => {
  const f = fixture(t, false); const r = await f.engine.decide({ ...request(), trace: trace() });
  assert.equal(r.apply, true); assert.equal(r.trainingCapture.stored, false);
  assert.equal(f.store.outcome({ bad: KEY }).stored, false);
  assert.equal(fs.existsSync(f.store.root), false);
  const logs = fs.readdirSync(path.join(f.home, 'logs')).map(p => fs.readFileSync(path.join(f.home, 'logs', p), 'utf8')).join('');
  for (const hidden of [KEY, 'bounded documentation', 'Narrow known', 'probabilities']) assert.equal(logs.includes(hidden), false);
});
test('decision stores versioned provenance separately from later outcome, without correctness', async t => {
  const f = fixture(t); const r = await f.engine.decide({ ...request(), trace: trace() });
  assert.equal(r.trainingCapture.stored, true);
  const d = f.store.scan().events[0].data;
  assert.equal(d.provenance.provider, 'jev'); assert.equal(d.provenance.model_version, 'jev-1.13.0');
  assert.equal('correct' in d, false); assert.equal('evaluation' in d, false);
  const nextProcess = createTrainingStore({ home: f.home }); nextProcess.outcome(outcome(d));
  assert.equal(evaluation(nextProcess, d).state, 'LABEL_CANDIDATE');
  assert.equal(evaluation(nextProcess, d).baseline_skip_supported, true);
});
test('OFF and capture toggles cannot capture an in-flight request without continuous consent', async t => {
  const f = fixture(t, false); let release;
  const e = createDecisionEngine({ home: f.home, env: f.env, provider: async () => { await new Promise(r => { release = r; }); return response(); } });
  t.after(() => e.close()); const pending = e.decide({ ...request(), trace: trace() });
  f.store.setCapture(true); release(); const r = await pending;
  assert.equal(r.trainingCapture.stored, false); assert.equal(f.store.scan().events.length, 0);
  const ticket = f.store.ticket(); f.store.setCapture(false); f.store.setCapture(true);
  assert.equal(f.store.decision(decision(), { expectedTicket: ticket }).stored, false);
  setMode(f.home, 'off', {}); assert.equal((await e.decide(request())).reason, 'OFF');
});
test('Jev, Laya and host provenance stays distinct and agreements do not create labels', t => {
  const f = fixture(t); const tr = { ...trace(), comparison_id: randomUUID() };
  for (const p of ['jev', 'laya']) f.save(decision(p, { trace: tr }));
  recordHost(f.store, { request: request(), trace: tr, model: 'host-model', model_version: 'host-r1', values: { worker: 'light' }, latency_ms: 9 });
  const s = summarizeComparisons(f.store.scan()); assert.equal(s.arms.length, 3);
  assert.equal(s.agreement.matched, 3); assert.equal(s.agreement.isAccuracy, false);
  assert.equal(buildDataset(f.store).sample_count, 0);
});
test('generic success and weak reviews are not labels or objective task evidence', t => {
  const f = fixture(t); const d = f.save(decision());
  f.store.outcome(outcome(d, { labels: [], checks: [{ kind: 'build', passed: true, required: true, scope: 'task', evidence_ref: REF }] }));
  assert.equal(evaluation(f.store, d).quality_observed, true); assert.equal(buildDataset(f.store).sample_count, 0);
  const d2 = f.save(decision()); f.store.outcome(outcome(d2), { trust: 'host' });
  assert.equal(evaluation(f.store, d2).labels.length, 0); assert.equal(evaluation(f.store, d2).quality_observed, null);
});
test('objective labels require a question-specific matching evidence assertion', t => {
  const f = fixture(t); const d = f.save(decision()); const o = outcome(d); o.checks = o.checks.filter(c => c.kind !== 'label');
  f.store.outcome(o); assert.equal(evaluation(f.store, d).labels.length, 0);
});
test('human corrections appended after execution are supervision, not implicit model truth', t => {
  const f = fixture(t); const d = f.save(decision()); f.store.outcome(outcome(d, { labels: [] }));
  f.store.outcome(outcome(d, { source: 'human', executed: false, executed_answers: {}, checks: [], metrics: { human_override: true },
    labels: [{ question_id: 'worker', value: 'strong', source: 'human', label_confidence: .99, evidence_ref: REF }] }));
  const e = evaluation(f.store, d); assert.equal(e.quality_observed, true); assert.equal(e.labels[0].value, 'strong');
  const b = buildDataset(f.store); assert.equal(readDataset(f.store, b.dataset_version).samples[0].target.value, 'strong');
});
test('conflicting corrections are excluded rather than resolved by voting', t => {
  const f = fixture(t); const d = f.save(decision()); const o = outcome(d);
  o.labels.push({ ...o.labels[0], value: 'strong' }); f.store.outcome(o);
  assert.equal(evaluation(f.store, d).issues.includes('CONFLICTING_LABELS'), true); assert.equal(buildDataset(f.store).sample_count, 0);
});
test('shadow cannot inherit an execution but independent annotations support calibration', t => {
  const f = fixture(t); const d = f.save(decision('laya', { arm: 'shadow', mode: 'shadow', apply: false }));
  f.store.outcome(outcome(d)); assert.equal(evaluation(f.store, d).state, 'SHADOW_HAS_NO_COUNTERFACTUAL_OUTCOME');
  assert.equal(buildDataset(f.store).sample_count, 0);
  const d2 = f.save(decision('laya', { arm: 'shadow', mode: 'shadow', apply: false }));
  f.store.outcome(outcome(d2, { executed: false, executed_answers: {}, metrics: {} }));
  const e = evaluation(f.store, d2); assert.equal(e.quality_observed, null); assert.equal(e.labels.length, 1);
  assert.equal(summarizeComparisons(f.store.scan()).calibration.length, 1);
});
test('downstream success and avoided baseline are attributed only to the executed prediction', t => {
  const f = fixture(t); const d = f.save(decision());
  f.store.outcome(outcome(d, { executed_answers: { worker: 'strong' } }));
  const e = evaluation(f.store, d); assert.equal(e.execution_matches_prediction, false); assert.equal(e.baseline_skip_supported, false);
  const s = summarizeComparisons(f.store.scan()); assert.equal(s.arms[0].measuredSuccesses, 0); assert.equal(s.arms[0].baselineCallsReportedSkipped, 0);
});
test('purpose-specific evidence remains multidimensional', async t => {
  for (const [purpose, metrics, issue] of [['route', { token_usage: 10 }, null], ['select', { artifact_created: false }, 'REQUESTED_ARTIFACT_MISSING'],
    ['retry', { retry_needed: true }, 'RETRY_WAS_NEEDED'], ['review', { serious_review_issue: true }, 'SERIOUS_ISSUE_FOUND'],
    ['judge', { runtime_error: true }, 'OBJECTIVE_FAILURE_OBSERVED'], ['escalate', { escalated: true }, 'ESCALATION_OCCURRED']]) await t.test(purpose, t => {
    const f = fixture(t); const d = decision(); d.request.purpose = purpose; d.request_hash = digest(d.request); f.save(d);
    const o = outcome(d); Object.assign(o.metrics, metrics); f.store.outcome(o); const e = evaluation(f.store, d);
    if (issue) assert.ok(e.issues.includes(issue)); if (purpose === 'route') assert.equal(e.utility.token_usage, 10);
    assert.equal('correct' in e, false);
  });
});
test('orphan and mismatched evaluation references cannot produce training samples', t => {
  const f = fixture(t); const d = decision(); f.store.outcome(outcome(d));
  assert.equal(validateDatasetSource(f.store).orphanOutcomes, 1); assert.equal(buildDataset(f.store).sample_count, 0);
  f.save(d); assert.equal(validateDatasetSource(f.store).ok, true);
  const snap = f.store.scan(); snap.events.push({ kind: 'evaluations', event_id: randomUUID(), data: { decision_id: d.decision_id, outcome_ids: [d.decision_id] } });
  assert.equal(indexEvents(snap).report.orphanEvaluations, 1);
});
test('low-confidence, unsupported, and contradictory labels are filtered', t => {
  const f = fixture(t); const d = f.save(decision()); const o = outcome(d); o.labels[0].label_confidence = .4;
  f.store.outcome(o); assert.equal(buildDataset(f.store).sample_count, 0);
  assert.throws(() => validateOutcome(outcome(d, { checks: [{ kind: 'command', passed: true, required: true, scope: 'task', evidence_ref: REF, exit_status: 1 }] })), /INVALID_TRAINING_SCHEMA/);
});
test('private storage refuses secrets, nested secret fields and oversized minimal states', t => {
  const f = fixture(t);
  for (const state of [KEY, 'ghp_12345678901234567890', 'alice@example.com', { environment: {} }, '{"credentials":"opaque-value"}', 'x'.repeat(4097)]) {
    const d = decision(); d.request.state = state; d.request_hash = digest(d.request);
    assert.equal(f.store.decision(d, { secret: KEY }).stored, false);
  }
  assert.equal(f.store.scan().events.length, 0);
});
test('every derived JSONL row is screened, not just the first', t => {
  const f = fixture(t);
  assert.throws(() => f.store.lock(() => f.store.writeDerived('exports/example/data.jsonl', '{}\n{"secret":"bad"}\n')), /TRAINING_SENSITIVE/);
  assert.equal(fs.existsSync(path.join(f.store.root, 'exports/example/data.jsonl')), false);
  assert.throws(() => safeContent({ state: '{"environment":{"THING":"value"}}' }), /TRAINING_SENSITIVE/);
});
// A JSON-encoded string (the Laya export's `state` column) must be screened on its PARSED content only:
// escaping turns a newline before an "@team.channel" mention into "\n@..." whose literal "n@" looked like
// an email, so export rejected tasks that import/build had accepted (2026-09-26, 7 tasks / 21 rows).
test('safeContent screens a JSON-encoded string by its parsed content, not by escape artefacts', () => {
  const task = 'Deploy after the review.\n@payments-oncall.team please confirm in the thread';
  assert.doesNotThrow(() => safeContent({ task }));
  assert.doesNotThrow(() => safeContent({ state: JSON.stringify({ task }) }));
  assert.throws(() => safeContent({ state: JSON.stringify({ task: 'mail ops@company.com first' }) }), /TRAINING_SENSITIVE/);
  assert.throws(() => safeContent({ state: JSON.stringify({ task: 'call 010-1234-5678' }) }), /TRAINING_SENSITIVE/);
});
// The dataset-write/read boundary in writeDerived (store.mjs) checks the SAME exported constant on
// both the write-size check and the read-back-for-immutability check, so a derived canonical dataset
// larger than the old 64 MiB cap (owner-reported: ~24,000-sample dataset already exceeded it) can be
// written and then read back without the two checks disagreeing (2026-09-26 dataset growth fix).
test('derived artifact cap is one exported MAX_DERIVED_BYTES constant, raised past the old 64 MiB limit', t => {
  const f = fixture(t);
  assert.equal(MAX_DERIVED_BYTES, 256 * 1024 * 1024);
  const OLD_CAP = 64 * 1024 * 1024;
  assert.ok(MAX_DERIVED_BYTES > OLD_CAP);
  // Many short lines (not one huge line): safeContent's per-line email-pattern regex check is
  // quadratic in a single string's length, so a realistic small-record JSONL (like an actual
  // canonical dataset) reaches the target size in seconds instead of minutes.
  const line = JSON.stringify({ pad: 'x'.repeat(40) }) + '\n';
  const lineBytes = Buffer.byteLength(line);
  const targetBytes = OLD_CAP + 5 * 1024 * 1024; // just over the old cap, well under MAX_DERIVED_BYTES
  const contents = line.repeat(Math.ceil(targetBytes / lineBytes));
  const size = Buffer.byteLength(contents);
  assert.ok(size > OLD_CAP && size < MAX_DERIVED_BYTES);
  const file = f.store.lock(() => f.store.writeDerived('exports/example/large.jsonl', contents));
  // Re-writing identical contents forces writeDerived to read the existing file back (old !== null
  // branch) with the same maxBytes: MAX_DERIVED_BYTES used on write, proving both sides agree.
  const again = f.store.lock(() => f.store.writeDerived('exports/example/large.jsonl', contents));
  assert.equal(again, file);
  assert.equal(fs.readFileSync(file, 'utf8'), contents);
});
test('outside-Git, private permissions and immutable events are enforced', t => {
  const f = fixture(t); const d = f.save(decision()); const file = path.join(f.store.root, 'raw/decisions', d.decision_id + '.json');
  assert.equal(fs.statSync(file).mode & 0o777, 0o600); assert.equal(fs.statSync(f.store.root).mode & 0o777, 0o700);
  assert.equal(f.store.decision(d).duplicate, true); assert.equal(f.store.decision({ ...d, latency_ms: 999 }).stored, false);
  const git = path.join(f.home, 'repo'); fs.mkdirSync(path.join(git, '.git'), { recursive: true }); fs.writeFileSync(path.join(git, '.git/HEAD'), 'ref: refs/heads/main');
  assert.throws(() => outsideGit(path.join(git, 'training')), /TRAINING_IN_REPOSITORY/);
  const bare = path.join(f.home, 'bare'); fs.mkdirSync(path.join(bare, 'objects'), { recursive: true }); fs.mkdirSync(path.join(bare, 'refs')); fs.writeFileSync(path.join(bare, 'HEAD'), 'ref');
  assert.throws(() => outsideGit(bare), /TRAINING_IN_REPOSITORY/);
});
test('lock conflicts are explicit and corrupt raw events do not poison good samples', t => {
  const f = fixture(t); const d = f.save(decision()); f.store.outcome(outcome(d));
  fs.mkdirSync(path.join(f.store.root, '.lock')); assert.equal(f.store.decision(decision()).reason, 'TRAINING_LOCKED'); fs.rmdirSync(path.join(f.store.root, '.lock'));
  atomicWrite(path.join(f.store.root, 'raw/decisions', randomUUID() + '.json'), '{broken');
  const b = buildDataset(f.store); assert.equal(b.sample_count, 1); assert.equal(b.exclusions.invalidEvents, 1);
  assert.equal(validateDatasetSource(f.store).ok, false);
});
test('raw events and evaluations remain separate and dataset rebuild is byte reproducible', t => {
  const f = fixture(t); const d = f.save(decision()); f.store.outcome(outcome(d));
  const first = buildDataset(f.store); const before = readDataset(f.store, first.dataset_version);
  evaluateStore(f.store); evaluateStore(f.store);
  const next = buildDataset(f.store); assert.equal(next.dataset_version, first.dataset_version);
  assert.equal(readDataset(f.store, next.dataset_version).contents, before.contents);
  assert.equal(f.store.scan().events.filter(e => e.kind === 'evaluations').length, 1);
  assert.equal(before.manifest.evaluation_policy_version, POLICY_VERSION);
});
test('dedup retains all task groups and provider provenance before splitting', t => {
  const f = fixture(t); const one = f.save(decision('jev')); const two = decision('laya'); f.save(two);
  const third = decision(); third.trace = two.trace; third.request.state = { task: 'A second distinct task state' }; third.request_hash = digest(third.request); f.save(third);
  for (const d of [one, two, third]) f.store.outcome(outcome(d));
  const b = buildDataset(f.store), { samples, manifest } = readDataset(f.store, b.dataset_version);
  assert.equal(samples.length, 2); assert.equal(new Set(samples.map(s => s.group_id)).size, 1);
  const merged = samples.find(s => s.provenance.length === 2); assert.equal(merged.task_ids.length, 2);
  assert.equal(manifest.provider_distribution.laya, 1);
});
test('model changes and raw changes create new dataset versions, not mutable checkpoints', t => {
  const f = fixture(t); const one = f.save(decision('laya')); f.store.outcome(outcome(one)); const a = buildDataset(f.store);
  const two = decision('laya'); two.provenance.model = 'laya/jadeon-v2'; two.provenance.checkpoint = two.provenance.model_version = 'b'.repeat(64);
  f.save(two); f.store.outcome(outcome(two)); const b = buildDataset(f.store);
  assert.notEqual(a.dataset_version, b.dataset_version); assert.equal(readDataset(f.store, b.dataset_version).samples[0].provenance.length, 2);
});
test('Laya export has official three JSON-string columns with isolated holdouts', t => {
  const f = fixture(t);
  const buildRequest = i => { const r = request(); r.state = { task: `A bounded documentation change ${i}` }; r.questions.worker.instructions = '\uB2F4\uB2F9\uC790\uB97C \uC120\uD0DD\uD558\uC138\uC694.'; return r; };
  // A laya export refuses an empty split, so exercise it with one isolated sample per split.
  let cursor = 0;
  for (const split of ['train', 'calibration', 'test']) {
    const i = stateForSplit(buildRequest, split, cursor); cursor = i + 1;
    const d = decision(); d.request = buildRequest(i); d.request_hash = digest(d.request);
    f.save(d); f.store.outcome(outcome(d));
  }
  const b = buildDataset(f.store), e = exportDataset(f.store, b.dataset_version);
  let count = 0;
  for (const file of e.files.filter(p => p.endsWith('.jsonl'))) for (const line of fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean)) {
    const row = JSON.parse(line); assert.deepEqual(Object.keys(row).sort(), ['gold', 'questions', 'state']);
    for (const field of Object.values(row)) assert.equal(typeof field, 'string');
    const q = JSON.parse(row.questions).worker; assert.equal(JSON.parse(q.instructions)[1], '\uB2F4\uB2F9\uC790\uB97C \uC120\uD0DD\uD558\uC138\uC694.');
    assert.deepEqual(JSON.parse(row.gold).worker.probabilities, { light: 1, strong: 0 }); count++;
  }
  assert.equal(count, 3); assert.equal(e.trained, false);
  assert.deepEqual(exportDataset(f.store, b.dataset_version).files, e.files);
  assert.ok(exportDataset(f.store, b.dataset_version, 'canonical').files.some(p => p.endsWith('canonical.jsonl')));
});
test('tampered canonical contents are refused before export', t => {
  const f = fixture(t); const d = f.save(decision()); f.store.outcome(outcome(d)); const b = buildDataset(f.store);
  const p = path.join(f.store.root, 'datasets', b.dataset_version, 'canonical.jsonl'); fs.appendFileSync(p, '{}\n');
  assert.throws(() => exportDataset(f.store, b.dataset_version), /DATASET_MANIFEST_MISMATCH/);
});
