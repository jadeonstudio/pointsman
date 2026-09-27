import test from 'node:test';
import assert from 'node:assert/strict';
import {
  questionAgreement, withinOneRate, selectiveAgreement, summarizeLatencies,
  aggregateModel, limitByTask, fixedStates, formatMarkdownTable,
} from '../scripts/laya-benchmark.mjs';

// ============================== questionAgreement ==============================

test('questionAgreement counts n/agreement/refused over synthetic records, refused never counted correct', () => {
  const records = [
    { correct: true, refused: false }, { correct: true, refused: false },
    { correct: false, refused: false }, { correct: false, refused: true },
  ];
  const r = questionAgreement(records);
  assert.equal(r.n, 4);
  assert.equal(r.agreement, 0.5);
  assert.equal(r.refused, 1);
});
test('questionAgreement returns null agreement for an empty set instead of dividing by zero', () => {
  assert.deepEqual(questionAgreement([]), { n: 0, agreement: null, refused: 0 });
});

// ============================== withinOneRate ==============================

test('withinOneRate counts |predicted-target|<=1 among answered (non-refused) records only', () => {
  const records = [
    { predicted: 2, target: 2, refused: false }, // diff 0
    { predicted: 1, target: 2, refused: false }, // diff 1
    { predicted: 4, target: 0, refused: false }, // diff 4
    { predicted: 0, target: 4, refused: true },  // refused: excluded from denominator
  ];
  const r = withinOneRate(records);
  assert.equal(r.n, 3);
  assert.equal(r.rate, 2 / 3);
});
test('withinOneRate returns null rate when nothing was answered', () => {
  assert.deepEqual(withinOneRate([{ predicted: 0, target: 0, refused: true }]), { n: 0, rate: null });
});

// ============================== selectiveAgreement ==============================

test('selectiveAgreement covers only metric>=threshold, non-refused records; coverage is over the FULL set', () => {
  const records = [
    { correct: true, refused: false, metric: 0.9 },
    { correct: false, refused: false, metric: 0.95 },
    { correct: true, refused: false, metric: 0.5 }, // below threshold: not covered
    { correct: false, refused: true, metric: -Infinity }, // refused: never covered even if metric were high
  ];
  const r = selectiveAgreement(records, 0.8);
  assert.equal(r.n, 2);
  assert.equal(r.coverage, 2 / 4);
  assert.equal(r.agreement, 0.5);
});
test('selectiveAgreement reports null agreement and zero coverage when nothing is covered', () => {
  const r = selectiveAgreement([{ correct: true, refused: false, metric: 0.1 }], 0.9);
  assert.equal(r.n, 0);
  assert.equal(r.coverage, 0);
  assert.equal(r.agreement, null);
});

// ============================== summarizeLatencies ==============================

test('summarizeLatencies computes p50/p95 over unsorted input without mutating it', () => {
  const values = [50, 10, 30, 90, 20, 80, 40, 70, 60, 100];
  const copy = [...values];
  const r = summarizeLatencies(values);
  assert.deepEqual(values, copy); // no mutation
  assert.equal(r.n, 10);
  assert.equal(r.p50, 60); // sorted[floor(0.5*10)] = sorted[5] = 60
  assert.equal(r.p95, 100); // sorted[floor(0.95*10)] = sorted[9] = 100
});
test('summarizeLatencies returns null percentiles for an empty series', () => {
  assert.deepEqual(summarizeLatencies([]), { n: 0, p50: null, p95: null });
});

// ============================== aggregateModel ==============================

function record(question_id, { lang = 'en', correct = true, refused = false, metric = 0.9, predicted = null, target = null } = {}) {
  return { question_id, lang, correct, refused, metric, predicted, target };
}
test('aggregateModel splits by question and by lang, and reports high_to_safe only for the risk question', () => {
  const records = [
    record('intent', { lang: 'ko', correct: true }),
    record('intent', { lang: 'en', correct: false }),
    record('difficulty', { lang: 'ko', correct: true, predicted: 2, target: 2 }),
    record('difficulty', { lang: 'en', correct: false, predicted: 0, target: 4 }),
    record('risk', { lang: 'ko', correct: false, predicted: 'safe', target: 'high' }), // high_to_safe
    record('risk', { lang: 'en', correct: true, predicted: 'safe', target: 'safe' }),
  ];
  const agg = aggregateModel(records);
  assert.equal(agg.n, 6);
  assert.equal(agg.by_question.intent.n, 2);
  assert.equal(agg.by_question.intent.agreement, 0.5);
  assert.equal(agg.by_question.difficulty.within_one.n, 2);
  assert.equal(agg.by_question.difficulty.within_one.rate, 0.5); // diff 0 within, diff 4 not
  assert.equal(agg.by_lang.ko.intent.n, 1);
  assert.equal(agg.by_lang.en.risk.n, 1);
  assert.equal(agg.high_to_safe, 1);
  assert.equal(agg.selective, null); // no qualifiedThreshold given
});
test('aggregateModel computes selective agreement/coverage only when a qualified threshold is given', () => {
  const records = [
    record('intent', { correct: true, metric: 0.95 }),
    record('intent', { correct: false, metric: 0.5 }),
  ];
  const agg = aggregateModel(records, { qualifiedThreshold: 0.9 });
  assert.ok(agg.selective);
  assert.equal(agg.selective.n, 1);
  assert.equal(agg.selective.coverage, 0.5);
  assert.equal(agg.selective.agreement, 1);
});

// ============================== limitByTask ==============================

test('limitByTask keeps every question row for the first N distinct tasks (by request_hash) in dataset order', () => {
  const samples = [
    { request_hash: 'a', question_id: 'intent' }, { request_hash: 'a', question_id: 'difficulty' }, { request_hash: 'a', question_id: 'risk' },
    { request_hash: 'b', question_id: 'intent' }, { request_hash: 'b', question_id: 'difficulty' }, { request_hash: 'b', question_id: 'risk' },
    { request_hash: 'c', question_id: 'intent' },
  ];
  const limited = limitByTask(samples, 2);
  assert.equal(limited.length, 6);
  assert.ok(limited.every(s => s.request_hash === 'a' || s.request_hash === 'b'));
});
test('limitByTask returns every sample unchanged when limit is not an integer', () => {
  const samples = [{ request_hash: 'a' }, { request_hash: 'b' }];
  assert.deepEqual(limitByTask(samples, undefined), samples);
});

// ============================== fixedStates ==============================

test('fixedStates deduplicates identical states and stops at `count`', () => {
  const samples = [
    { state: { task: 'x' } }, { state: { task: 'x' } }, { state: { task: 'y' } },
    { state: { task: 'z' } }, { state: { task: 'w' } },
  ];
  const states = fixedStates(samples, 3);
  assert.equal(states.length, 3);
  assert.deepEqual(states, [{ task: 'x' }, { task: 'y' }, { task: 'z' }]);
});
test('fixedStates returns an empty array when given no samples', () => {
  assert.deepEqual(fixedStates([]), []);
});

// ============================== formatMarkdownTable ==============================

test('formatMarkdownTable includes the reference-label header note, one row per model, and n/a for missing selective/latency data', () => {
  const modelsReport = {
    checkpointA: {
      by_question: { intent: { agreement: 0.9, refused: 1 }, difficulty: { agreement: 0.7, refused: 0, within_one: { rate: 0.8 } }, risk: { agreement: 0.95, refused: 0 } },
      high_to_safe: 2, selective: { agreement: 0.99, coverage: 0.3 },
      latency: { cold_start_ms: 1234.5, warm_p50_ms: 12.3, warm_p95_ms: 45.6 },
    },
    checkpointB: {
      by_question: { intent: { agreement: null, refused: 0 }, difficulty: { agreement: null, refused: 0, within_one: { rate: null } }, risk: { agreement: null, refused: 0 } },
      high_to_safe: 0, selective: null,
      latency: { cold_start_ms: null, warm_p50_ms: null, warm_p95_ms: null },
    },
  };
  const table = formatMarkdownTable(modelsReport);
  assert.ok(table.startsWith('Claude reference-label agreement, not accuracy'));
  assert.match(table, /\| checkpointA \| 90\.0% \| 70\.0% \| 80\.0% \| 95\.0% \| 2 \| 1 \| 99\.0%@30\.0% \| 1235ms \| 12ms\/46ms \|/);
  assert.match(table, /\| checkpointB \| n\/a \| n\/a \| n\/a \| n\/a \| 0 \| 0 \| n\/a \| n\/a \| n\/a\/n\/a \|/);
});
