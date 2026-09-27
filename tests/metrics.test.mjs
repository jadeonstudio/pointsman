import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { appendEvent } from '../src/storage.mjs';
import { readMetrics } from '../src/metrics.mjs';
import { evaluateStore } from '../src/training/evaluate.mjs';
import { fixture, decision, outcome, REF } from './training-helpers.mjs';

const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));

function hash(file) { return fs.readFileSync(file, 'utf8'); }
function allLogFiles(home) {
  const dir = path.join(home, 'logs');
  return fs.existsSync(dir) ? fs.readdirSync(dir).map(n => path.join(dir, n)) : [];
}
function allTrainingRawFiles(home) {
  const root = path.join(home, 'training', 'raw');
  if (!fs.existsSync(root)) return [];
  const out = [];
  for (const kind of fs.readdirSync(root)) for (const name of fs.readdirSync(path.join(root, kind))) out.push(path.join(root, kind, name));
  return out;
}

test('P4 metrics: hook reason/role distribution, mismatch/applied, mode split, latency percentiles', t => {
  const f = fixture(t);
  const d1 = decision('jev'); f.save(d1);
  appendEvent(f.home, { kind: 'decision', at: new Date().toISOString(), id: d1.decision_id, mode: 'on', provider: 'jev',
    purpose: 'route', reason: 'ACCEPTED', apply: true, eligible: true, inputBytes: 10, questionCount: 1,
    networkCalls: 1, inferenceCalls: 1, elapsedMs: 5, usage: { inputTokens: 50, outputTokens: 10 } });
  appendEvent(f.home, { kind: 'route', at: new Date().toISOString(), id: d1.decision_id, mode: 'on', apply: true,
    reason: 'ACCEPTED', networkCalls: 1, inferenceCalls: 1, elapsedMs: 7, model: 'jev-1', policy: 'x' });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ACCEPTED',
    mode: 'on', applied: true, original_role: 'general-purpose', recommended_role: 'implementer', decision_id: d1.decision_id, elapsedMs: 2.1 });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'post-spawn', reason: 'LINKED',
    mode: 'on', applied: false, original_role: null, recommended_role: null, decision_id: null, elapsedMs: 0.5 });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'subagent-stop', reason: 'RECORDED',
    mode: 'on', applied: false, original_role: null, recommended_role: null, decision_id: null, elapsedMs: 0.3 });
  // shadow: recommendation differs from the caller's original role, but shadow never rewrites (applied stays false).
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'codex', event: 'pre-spawn', reason: 'SHADOW',
    mode: 'shadow', applied: false, original_role: 'implementer', recommended_role: 'specialist', decision_id: randomUUID(), elapsedMs: 3.4 });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'codex', event: 'pre-spawn', reason: 'ROLE_NOT_ROUTABLE',
    mode: 'on', applied: false, original_role: 'verifier', recommended_role: null, decision_id: null, elapsedMs: 1.1 });

  const m = readMetrics(f.home, 7);
  assert.equal(m.hooks.total, 5);
  assert.equal(m.hooks.byHostEvent['claude:pre-spawn'], 1);
  assert.equal(m.hooks.byHostEvent['codex:pre-spawn'], 2);
  assert.equal(m.hooks.reasons.ROLE_NOT_ROUTABLE, 1);
  assert.equal(m.hooks.reasons.ACCEPTED, 1);
  assert.equal(m.hooks.recommendations.on.total, 1);
  assert.equal(m.hooks.recommendations.on.byRole.implementer, 1);
  assert.equal(m.hooks.recommendations.on.mismatch, 1);
  assert.equal(m.hooks.recommendations.on.applied, 1);
  assert.equal(m.hooks.recommendations.shadow.total, 1);
  assert.equal(m.hooks.recommendations.shadow.byRole.specialist, 1);
  assert.equal(m.hooks.recommendations.shadow.mismatch, 1);
  assert.equal(m.hooks.recommendations.shadow.applied, 0);
  assert.equal(m.hooks.hookLatencyMs['pre-spawn'].n, 3);
  assert.equal(m.hooks.hookLatencyMs['pre-spawn'].max, 3.4);
  assert.ok(Number.isFinite(m.hooks.hookLatencyMs['pre-spawn'].p50));
  assert.equal(m.hooks.hookLatencyMs['post-spawn'].n, 1);
  // decision_id join: only the claude pre-spawn hook carries a decision_id with a matching route event.
  assert.equal(m.hooks.routeDecisionLatencyMs.n, 1);
  assert.equal(m.hooks.routeDecisionLatencyMs.p50, 7);
  assert.equal(m.hooks.networkCallsFromHooks.total, 1);
  assert.equal(m.hooks.networkCallsFromHooks.decisionsReferenced, 2); // claude pre-spawn + shadow codex pre-spawn decision ids
  assert.equal(m.hooks.networkCallsFromHooks.decisionsLinkedToRoute, 1);
  assert.equal(typeof m.hooks.networkCallsFromHooks.limitation, 'string');
});

test('P4 metrics: TypeSafe usage from hooks is summed with no dollar estimate', t => {
  const f = fixture(t);
  const d1 = decision('jev'); f.save(d1);
  appendEvent(f.home, { kind: 'decision', at: new Date().toISOString(), id: d1.decision_id, mode: 'on', provider: 'jev',
    purpose: 'route', reason: 'ACCEPTED', apply: true, eligible: true, inputBytes: 10, questionCount: 1,
    networkCalls: 1, inferenceCalls: 1, elapsedMs: 5, usage: { inputTokens: 123, outputTokens: 45 } });
  appendEvent(f.home, { kind: 'route', at: new Date().toISOString(), id: d1.decision_id, mode: 'on', apply: true, reason: 'ACCEPTED', networkCalls: 1, inferenceCalls: 1, elapsedMs: 4 });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ACCEPTED',
    mode: 'on', applied: true, original_role: 'general-purpose', recommended_role: 'implementer', decision_id: d1.decision_id, elapsedMs: 1 });

  const m = readMetrics(f.home, 7);
  assert.equal(m.hooks.typeSafeUsage.inputTokens, 123);
  assert.equal(m.hooks.typeSafeUsage.outputTokens, 45);
  assert.equal(m.hooks.typeSafeUsage.decisionsMatched, 1);
  assert.equal(m.hooks.costUsd, null);
  assert.equal(typeof m.hooks.costReason, 'string');
  assert.equal(m.tokenSavings, null);
  assert.equal(m.costSavings, null);
});

test('P4 metrics: labels report outcome sources, strong labels per purpose and trainingCandidateReady', t => {
  const f = fixture(t);
  f.store.setMinLabels(1);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1)); // source 'runner', objective label -> strong label for purpose 'route'
  evaluateStore(f.store);
  const d2 = decision('jev'); f.save(d2);
  f.store.outcome({ decision_id: d2.decision_id, execution_id: randomUUID(), executed: false, final: true, source: 'host_review',
    executed_answers: {}, metrics: { task_succeeded: null }, checks: [], labels: [], host_review: 'uncertain' }, { trust: 'host' });

  const m = readMetrics(f.home, 7);
  assert.equal(m.labels.bySource.runner, 1);
  assert.equal(m.labels.bySource.host_review, 1);
  assert.equal(m.labels.minStrongLabelsPerPurpose, 1);
  assert.equal(m.labels.strongLabelsByPurpose.route, 1);
  assert.equal(m.labels.trainingCandidateReady.route, true);
  assert.equal(m.labels.evaluationsAvailable, true);
});

test('P4 metrics: trainingCandidateReady is false below the configured minimum', t => {
  const f = fixture(t);
  f.store.setMinLabels(5);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1));
  evaluateStore(f.store);
  const m = readMetrics(f.home, 7);
  assert.equal(m.labels.strongLabelsByPurpose.route, 1);
  assert.equal(m.labels.trainingCandidateReady.route, false);
});

test('P4 metrics: followedOutcomes links applied/not-applied hooks to outcomes by decision_id, with n<30 flagged', t => {
  const f = fixture(t);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1)); // executed runner outcome, task_succeeded true
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ACCEPTED',
    mode: 'on', applied: true, original_role: 'general-purpose', recommended_role: 'implementer', decision_id: d1.decision_id, elapsedMs: 1 });

  const d2 = decision('jev'); f.save(d2);
  f.store.outcome({ decision_id: d2.decision_id, execution_id: randomUUID(), executed: false, final: true, source: 'host_review',
    executed_answers: {}, metrics: { task_succeeded: null }, checks: [], labels: [], host_review: 'fail' }, { trust: 'host' });
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ROLE_NOT_ROUTABLE',
    mode: 'on', applied: false, original_role: 'verifier', recommended_role: null, decision_id: d2.decision_id, elapsedMs: 1 });

  const m = readMetrics(f.home, 7);
  assert.equal(m.followedOutcomes.applied.runnerPassed, 1);
  assert.equal(m.followedOutcomes.applied.n, 1);
  assert.equal(m.followedOutcomes.applied.insufficientSample, true);
  assert.equal(m.followedOutcomes.notApplied.fail, 1);
  assert.equal(m.followedOutcomes.notApplied.n, 1);
  assert.equal(m.followedOutcomes.notApplied.insufficientSample, true);
  assert.equal(typeof m.followedOutcomes.note, 'string');
  assert.ok(!/causal|caused|because it was (applied|followed)/i.test(m.followedOutcomes.note));
});

test('P4 metrics: no prompt/state/marker content leaks into the metrics output', t => {
  const f = fixture(t);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1));
  evaluateStore(f.store);
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ACCEPTED',
    mode: 'on', applied: true, original_role: 'general-purpose', recommended_role: 'implementer', decision_id: d1.decision_id, elapsedMs: 1 });

  const serialized = JSON.stringify(readMetrics(f.home, 7));
  for (const forbidden of ['A bounded documentation change', 'scope=local', 'complete=yes', 'failures=0', REF]) {
    assert.equal(serialized.includes(forbidden), false, `leaked: ${forbidden}`);
  }
});

test('P4 metrics: readMetrics never mutates training raw files or log files', t => {
  const f = fixture(t);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1));
  evaluateStore(f.store);
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'ACCEPTED',
    mode: 'on', applied: true, original_role: 'general-purpose', recommended_role: 'implementer', decision_id: d1.decision_id, elapsedMs: 1 });

  const before = new Map([...allLogFiles(f.home), ...allTrainingRawFiles(f.home)].map(file => [file, hash(file)]));
  readMetrics(f.home, 7);
  readMetrics(f.home, 30);
  const afterFiles = [...allLogFiles(f.home), ...allTrainingRawFiles(f.home)];
  assert.deepEqual(new Set(afterFiles), new Set(before.keys()));
  for (const file of afterFiles) assert.equal(hash(file), before.get(file), `mutated: ${file}`);
});

test('P4 metrics: a training store error surfaces only as labels.error, never breaking the rest of metrics', t => {
  const f = fixture(t);
  appendEvent(f.home, { kind: 'hook', at: new Date().toISOString(), host: 'claude', event: 'pre-spawn', reason: 'NO_CONTEXT_ANNOTATION',
    mode: 'shadow', applied: false, original_role: 'implementer', recommended_role: null, decision_id: null, elapsedMs: 1 });
  fs.mkdirSync(path.join(f.home, 'training'), { recursive: true });
  fs.writeFileSync(path.join(f.home, 'training.json'), 'not json', { mode: 0o600 });

  const m = readMetrics(f.home, 7);
  assert.equal(typeof m.labels.error, 'string');
  assert.equal(m.hooks.total, 1);
  assert.equal(m.followedOutcomes.applied.n, 0);
});

test('doctor warns TRAINING_CANDIDATE_READY when a purpose reaches its strong-label minimum, without failing on metrics errors', t => {
  const f = fixture(t);
  f.store.setMinLabels(1);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1));
  evaluateStore(f.store);

  const result = spawnSync(process.execPath, [bin, 'doctor'], { env: { ...f.env, POINTSMAN_HOME: f.home, HOME: f.home }, encoding: 'utf8' });
  const out = JSON.parse(result.stdout);
  assert.ok(out.warnings.some(w => w.startsWith('TRAINING_CANDIDATE_READY:route')));
  assert.ok(out.warnings.find(w => w.startsWith('TRAINING_CANDIDATE_READY:route')).includes('does not start training'));
});

test('doctor does not warn when no purpose has reached its strong-label minimum', t => {
  const f = fixture(t);
  f.store.setMinLabels(100);
  const d1 = decision('jev'); f.save(d1);
  f.store.outcome(outcome(d1));
  evaluateStore(f.store);

  const result = spawnSync(process.execPath, [bin, 'doctor'], { env: { ...f.env, POINTSMAN_HOME: f.home, HOME: f.home }, encoding: 'utf8' });
  const out = JSON.parse(result.stdout);
  assert.equal(out.warnings.some(w => w.startsWith('TRAINING_CANDIDATE_READY')), false);
});
