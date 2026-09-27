import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { evaluateDecision, evaluateStore, summarizeComparisons, EVALUATION_POLICY } from '../src/training/evaluate.mjs';
import { buildDataset, readDataset } from '../src/training/dataset.mjs';
import { fixture, decision, outcome } from './training-helpers.mjs';

function evaluate(d, o) {
  return evaluateDecision({ event_id: d.decision_id, kind: 'decisions', data: d },
    [{ event_id: randomUUID(), kind: 'outcomes', data: o }]);
}
for (const name of ['tests_passed', 'build_passed', 'lint_passed', 'typecheck_passed', 'artifact_created']) {
  test(`explicit ${name}=false cannot be counted as downstream success`, () => {
    const d = decision(), o = outcome(d); o.metrics[name] = false;
    const result = evaluate(d, o);
    assert.equal(result.quality_observed, false);
    assert.equal(result.utility.quality, false);
    assert.ok(result.issues.includes('CONTRADICTORY_OUTCOME_EVIDENCE'));
    // A separate, question-specific assertion can still supply a valid correction.
    assert.equal(result.labels.length, 1);
    assert.equal(o.metrics[name], false);
  });
}
test('a serious review issue defeats a broad success claim', () => {
  const d = decision(), o = outcome(d); o.metrics.serious_review_issue = true;
  assert.equal(evaluate(d, o).quality_observed, false);
});
test('unknown metrics are not invented failures and absent required checks remain unknown', () => {
  const d = decision(), o = outcome(d); o.metrics.tests_passed = null;
  assert.equal(evaluate(d, o).quality_observed, true);
  o.checks = o.checks.filter(c => c.kind === 'label');
  assert.equal(evaluate(d, o).quality_observed, null);
});
test('a direct evaluator cannot borrow another decision outcome', () => {
  const d = decision(), other = decision(), result = evaluate(d, outcome(other));
  assert.equal(result.state, 'INVALID_OUTCOME_LINK');
  assert.equal(result.quality_observed, null);
  assert.equal(result.baseline_skip_supported, false);
  assert.deepEqual(result.labels, []);
});
test('contradictory assertions for the same label evidence do not create training labels', () => {
  const d = decision(), o = outcome(d);
  o.checks.push({ ...o.checks.find(c => c.kind === 'label'), passed: false });
  const result = evaluate(d, o);
  assert.equal(result.labels.length, 0);
  // Label validity is separate from the task-scoped test result.
  assert.equal(result.quality_observed, true);
});
test('weak host reports do not become objective failures or labels', () => {
  const d = decision(), o = outcome(d, { source: 'host_review', labels: [] });
  o.metrics.build_passed = false;
  const result = evaluate(d, o);
  assert.equal(result.quality_observed, null);
  assert.deepEqual(result.labels, []);
});
test('failed-quality arms are excluded from successful route preferences and success statistics', t => {
  const f = fixture(t), one = f.save(decision()), two = decision('laya');
  two.trace = structuredClone(one.trace);
  two.answers.worker = { type: 'choice', value: 'strong', confidence: .97, selectedProbability: .99,
    probabilities: { light: .01, strong: .99 } };
  f.save(two);
  const first = outcome(one), second = outcome(two);
  first.metrics.build_passed = false;
  second.metrics.latency_ms = 12000; second.metrics.token_usage = 3200;
  f.store.outcome(first); f.store.outcome(second);
  const built = buildDataset(f.store);
  assert.equal(built.preference_count, 0);
  const stats = summarizeComparisons(f.store.scan());
  const failed = stats.arms.find(a => a.provider === 'jev');
  assert.equal(failed.measuredSuccesses, 0);
  assert.equal(failed.measuredFailures, 1);
});
test('enforcement revision is versioned in derived evidence and reproducible manifests', t => {
  const f = fixture(t), d = f.save(decision()); f.store.outcome(outcome(d));
  evaluateStore(f.store);
  const ev = f.store.scan().events.find(e => e.kind === 'evaluations');
  assert.equal(ev.data.derived.implementation_revision, EVALUATION_POLICY.implementation_revision);
  const first = buildDataset(f.store), second = buildDataset(f.store);
  assert.equal(first.dataset_version, second.dataset_version);
  const dataset = readDataset(f.store, first.dataset_version);
  assert.equal(dataset.manifest.filter_rules.implementation_revision, EVALUATION_POLICY.implementation_revision);
});
