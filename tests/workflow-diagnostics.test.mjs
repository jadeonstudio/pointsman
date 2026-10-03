import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { runDiagnostics } from '../src/workflow-diagnostics.mjs';
import { createWorkflowRunner } from '../src/workflows.mjs';
import { isCompletedWorkflow } from '../src/workflow-hosts.mjs';

function record(result, id) {
  if (typeof id !== 'string') return id;
  assert.equal(result.details.recordTableVersion, 1);
  assert(Object.hasOwn(result.details.records, id), `Missing record ${id}`);
  return result.details.records[id];
}

function context(files, extras = {}) {
  const reads = [];
  return { root: '/project', signal: undefined, limits: {}, stats: {}, reads, check() {},
    async read(path, role) {
      reads.push({ path, role });
      if (!(path in files)) return { path, reason: 'FILE_MISSING' };
      return { path, text: files[path], hash: createHash('sha256').update(files[path]).digest('hex'), ref: `source:${path}` };
    }, async listFiles() { return Object.keys(files); }, async decide() { throw new Error('deterministic diagnostics must not classify every record'); }, ...extras };
}
const request = (workflow, inputs, acceptance) => ({ workflow, inputs, ...(acceptance ? { acceptance } : {}) });

test('NDJSON triage retains singleton/first errors, success, correlations and UTF-8 offsets', async () => {
  const text = [
    { timestamp: '2026-10-03T00:00:00Z', level: 'error', message: '연결 실패', traceId: 'a' },
    { timestamp: '2026-10-03T00:00:01Z', level: 'error', message: '연결 실패', traceId: 'a' },
    { timestamp: '2026-10-03T00:00:02Z', level: 'error', message: 'rare cause', traceId: 'a' },
    { timestamp: '2026-10-03T00:00:03Z', status: 'recovered', message: 'success', traceId: 'a' }
  ].map(JSON.stringify).join('\n');
  const result = await runDiagnostics(request('log-triage', { paths: ['run.ndjson'] }), context({ 'run.ndjson': text }));
  assert.equal(result.status, 'done'); assert.equal(result.acceptance.first_errors, 'met');
  assert.deepEqual(result.details.groups.map(g => g.count), [2, 1, 1]);
  assert.equal(record(result, result.details.groups[0].last).ref.byteStart, Buffer.byteLength(text.split('\n')[0]) + 1);
  assert.equal(result.details.contrary.length, 1);
  assert.equal(result.details.correlations[0].errors.length, 3); assert.equal(result.details.correlations[0].successes.length, 1);
  assert.ok(result.evidence.some(e => e.text.includes('rare cause') && e.hash && e.ref && e.startLine === 3));
});

test('diagnostic reference tables share records and preserve source identity after JSON transport', async () => {
  const text = [
    { timestamp: '2026-10-03T00:00:00Z', level: 'error', message: 'failure' },
    { timestamp: '2026-10-03T00:00:01Z', status: 'recovered', message: 'ready' },
  ].map(JSON.stringify).join('\n');
  const result = JSON.parse(JSON.stringify(await runDiagnostics(request('log-triage', { paths: ['events.ndjson'] }), context({ 'events.ndjson': text }))));
  const d = result.details;
  assert.equal(Object.keys(d.records).length, 2);
  assert.equal(d.groups[0].first, d.groups[0].last); assert.equal(d.groups[0].first, d.timeline[0]);
  assert.equal(d.contrary[0], d.timeline[1]);
  for (const value of Object.values(d.records)) {
    const span = result.evidence.find(e => e.ref === value.ref.evidenceRef);
    assert(span); assert.equal(span.hash, createHash('sha256').update(text).digest('hex'));
    assert.equal(span.text, value.snippet); assert(result.evidence.some(e => e.ref === span.sourceRef && e.sourceRef));
  }
  assert.throws(() => record(result, 'missing'), /Missing record/);
});

test('time window retains uncertain clocks and reports malformed records and backwards time', async () => {
  const input = '{"timestamp":"2026-10-03T00:00:02Z","level":"error","message":"first"}\n' +
    '{"timestamp":"2026-10-03T00:00:01Z","level":"error","message":"earlier"}\n' +
    '{"timestamp":"2026-10-02T00:00:00Z","message":"outside"}\n' +
    '{"time":"2026-10-03T00:00:00","message":"unknown timezone"}\n{broken\nplain failure';
  const result = await runDiagnostics(request('log-triage', { paths: ['a.log'], timeWindow: { from: '2026-10-03T00:00:00Z' } }), context({ 'a.log': input }));
  assert.equal(result.status, 'needs_parent'); assert.equal(result.coverage.excludedByWindow, 1);
  assert.equal(result.details.parseFailures.length, 1); assert.equal(record(result, result.details.parseFailures[0]).ref.lineStart, 5);
  assert.ok(result.details.clockUncertainty.some(x => x.reason === 'source_clock_moved_backwards'));
  assert.ok(result.details.groups.some(g => record(result, g.first).message === 'unknown timezone'));
  assert.ok(result.evidence.some(e => e.text === '{broken'));
});

test('JSON containers report record indexes and unsupported containers never become complete', async () => {
  const ctx = context({ 'pretty.json': JSON.stringify({ events: [{ level: 'fatal', message: 'cause' }, { status: 'ok', message: 'ready' }] }, null, 2), 'unknown.json': '{"thing": 1}' });
  const result = await runDiagnostics(request('log-triage', { paths: ['pretty.json'] }), ctx);
  assert.equal(record(result, result.details.groups[0].first).recordIndex, 0);
  assert.equal(record(result, result.details.groups[1].first).recordIndex, 1);
  const unsupported = await runDiagnostics(request('test-diagnose', { resultPaths: ['unknown.json'] }), ctx);
  assert.equal(unsupported.status, 'needs_parent'); assert.equal(record(unsupported, unsupported.details.parseFailures[0]).reason, 'unsupported_test_status');
});

test('TAP failure signatures dedupe, preserve passes and fetch referenced fixture/implementation spans', async () => {
  const tap = `TAP version 13
# Subtest: cash
not ok 1 - cash
  ---
  error: 'expected 10, actual 0'
  code: 'ERR_ASSERTION'
  stack: |-
    at cash (/project/src/cash.mjs:4:2)
    at test (tests/cash.test.mjs:2:1)
  ...
not ok 2 - cash retry
  ---
  error: 'expected 10, actual 0'
  code: 'ERR_ASSERTION'
  stack: |-
    at cash (/project/src/cash.mjs:4:2)
    at test (tests/cash.test.mjs:2:1)
  ...
ok 3 - unchanged contract
1..3
# tests 3
# pass 1
# fail 2`;
  const ctx = context({ 'tap.log': tap, 'src/cash.mjs': 'a\nb\nc\nthrow new Error()\ne', 'tests/cash.test.mjs': 'assert.equal(1, 1)\nfixture()' });
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['tap.log'] }), ctx);
  assert.equal(result.status, 'needs_parent'); assert.equal(result.reason, 'FAILURE_CAUSE_UNRESOLVED');
  assert.equal(result.details.failures.length, 1); assert.equal(result.details.failures[0].count, 2);
  assert.equal(record(result, result.details.passed[0]).name, 'unchanged contract');
  assert.deepEqual(result.details.rerunSet, ['cash', 'cash retry']);
  assert.deepEqual(result.details.sourceEvidence.map(e => record(result, e).path), ['src/cash.mjs', 'tests/cash.test.mjs']);
  assert.ok(result.evidence.some(e => e.role === 'diagnostic_source' && e.text.includes('throw new Error')));
  assert.equal(result.acceptance.unresolved_cause, 'met');
});

test('Jest result and node JSON reporter retain contradictory attempts without declaring a flake', async () => {
  const files = { 'jest.json': JSON.stringify({ testResults: [{ name: 'tests/a.test.mjs', assertionResults: [
    { fullName: 'a retries', status: 'failed', failureMessages: ['AssertionError: mismatch'] },
    { fullName: 'a retries', status: 'passed' }, { fullName: 'not run', status: 'pending' }
  ] }] }), 'tests/a.test.mjs': 'test()', 'node.json': '{"type":"test:pass","data":{"name":"other","file":"tests/a.test.mjs"}}\n{"type":"test:fail","data":{"name":"failed","file":"tests/a.test.mjs","details":{"error":{"message":"bad"}}}}' };
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['jest.json', 'node.json'] }), context(files));
  assert.deepEqual(result.details.possibleFlakes, ['a retries']); assert.equal(result.details.passed.length, 2);
  assert.equal(result.details.skipped.length, 1); assert.equal(result.details.failures.length, 2);
  assert.equal(result.details.cause, 'unresolved_from_reporter_evidence');
});

test('JUnit subset preserves skipped/passed evidence; malformed XML remains unresolved', async () => {
  const xml = '<testsuite tests="3">\n<testcase name="passes"/>\n<testcase name="skipped"><skipped/></testcase>\n<testcase name="bad"><failure message="expected &lt; value">stack src/a.mjs:1:1</failure></testcase>\n</testsuite>';
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['results.xml'] }), context({ 'results.xml': xml, 'src/a.mjs': 'a()' }));
  assert.equal(record(result, result.details.failures[0].first).ref.lineStart, 4);
  assert.ok(record(result, result.details.failures[0].first).details.includes('expected < value'));
  assert.equal(result.details.passed.length, 1); assert.equal(result.details.skipped.length, 1);
  const broken = await runDiagnostics(request('test-diagnose', { resultPaths: ['broken.xml'] }), context({ 'broken.xml': '<testsuite><testcase name="bad"><failure/></testcase>' }));
  assert.equal(broken.status, 'needs_parent'); assert.equal(broken.details.parseFailures.length, 1);
});

test('missing sources/outside-root stacks and unsupported reporter prevent completeness', async () => {
  const ctx = context({ 'bad.tap': 'not ok 1 - failure\n  at wrapper (/elsewhere/secret.mjs:1:1)', 'unsupported.log': 'some wrapper exit code 1' });
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['bad.tap', 'missing.log', 'unsupported.log'] }), ctx);
  assert.equal(result.status, 'needs_parent'); assert.equal(result.reason, 'INCOMPLETE_DIAGNOSTIC_EVIDENCE');
  assert.ok(result.coverage.omissions.some(x => x.reason === 'outside_root_reference'));
  assert.ok(result.coverage.omissions.some(x => x.reason === 'FILE_MISSING'));
  assert.equal(ctx.reads.some(x => x.path.includes('secret')), false);
  assert.equal(result.details.parseFailures.length, 1);
});

test('registered test names require injected authority and run exactly once', async () => {
  let calls = 0;
  const req = request('test-diagnose', { registeredTest: 'authorized' });
  const unavailable = await runDiagnostics(req, context({})); assert.equal(unavailable.reason, 'REGISTERED_TEST_UNAVAILABLE');
  const ctx = context({ 'result.tap': 'ok 1 - exact test\n1..1' }, { async runRegisteredTest(name) { assert.equal(name, 'authorized'); calls++; return { resultPaths: ['result.tap'] }; } });
  const result = await runDiagnostics(req, ctx); assert.equal(calls, 1); assert.equal(result.status, 'done');
  assert.equal(result.details.passed.length, 1);
});

test('checks propagate cancellation/budget and arbitrary acceptance stays unresolved', async () => {
  const ctx = context({ 'ok.tap': 'ok 1 - passes' }); let checks = 0;
  ctx.check = () => { if (++checks === 4) throw Object.assign(new Error('CANCELLED'), { code: 'CANCELLED' }); };
  await assert.rejects(runDiagnostics(request('test-diagnose', { resultPaths: ['ok.tap'] }), ctx), { code: 'CANCELLED' });
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['ok.tap'] }, ['prove_product_root_cause']), context({ 'ok.tap': 'ok 1 - passes' }));
  assert.equal(result.acceptance.prove_product_root_cause, 'unresolved'); assert.equal(result.status, 'needs_parent');
});

test('long failure tail still fetches stack sources while output excerpts stay bounded', async () => {
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['long.tap'] }), context({
    'long.tap': 'not ok 1 - assertion\n' + 'detail '.repeat(180) + '\n at source (/project/src/tail.mjs:2:1)',
    'src/tail.mjs': 'start\nassert(false)\nend'
  }));
  assert.equal(record(result, result.details.sourceEvidence[0]).path, 'src/tail.mjs');
  assert.equal(record(result, result.details.failures[0].first).details.length, 700);
  assert.equal(JSON.stringify(result).includes('referenceText'), false);
});

test('partial JSON reporter schema support does not hide unparsed test records', async () => {
  const result = await runDiagnostics(request('test-diagnose', { resultPaths: ['partial.json'] }), context({
    'partial.json': '[{"status":"passed","name":"known"},null,{"status":"new-status"},{"assertionResults":{}}]'
  }));
  assert.equal(result.status, 'needs_parent'); assert.equal(result.details.passed.length, 1);
  assert.equal(result.details.parseFailures.length, 3);
  assert.equal(result.evidence.find(e => e.text === '' && e.role === 'diagnostic_observation')?.locator, 'json_record_index_in_source');
});

test('JSON container checks never swallow budget or cancellation failures', async () => {
  let count = 0;
  const ctx = context({ 'events.json': '[{"message":"a"},{"message":"b"}]' }, { check() { if (++count === 4) throw Object.assign(new Error('budget'), { code: 'WORKFLOW_ACTION_BUDGET' }); } });
  await assert.rejects(runDiagnostics(request('log-triage', { paths: ['events.json'] }), ctx), { code: 'WORKFLOW_ACTION_BUDGET' });
});

test('shared runner diagnostic packets satisfy native completion only with complete source evidence', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-diagnostic-native-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.writeFile(path.join(root, 'result.tap'), 'ok 1 - unchanged contract\n1..1\n# tests 1\n# pass 1\n# fail 0');
  await fs.writeFile(path.join(root, 'events.ndjson'), '{"timestamp":"2026-10-03T00:00:00Z","level":"error","message":"first incident"}\n{"timestamp":"2026-10-03T00:00:01Z","status":"recovered","message":"success"}');
  const runner = createWorkflowRunner({ root,
    engine: { status: () => ({ mode: 'on', policyRevision: 'native-fixture' }), decide: () => assert.fail('no decision inference') },
    getPolicy: () => ({ mode: 'on', maxActions: 24, maxMs: 5000, maxDecisionCalls: 0, maxOutputBytes: 24000 }),
    capabilities: { registeredTests: { inline: async () => ({ text: 'ok 1 - injected test\n1..1' }) } }
  });
  for (const req of [request('test-diagnose', { resultPaths: ['result.tap'] }, []), request('log-triage', { paths: ['events.ndjson'] }, [])]) {
    const result = await runner.run(req);
    assert.equal(result.status, 'done'); assert.ok(Object.keys(result.acceptance).length > 0);
    assert.equal(result.coverage.complete, true); assert.equal(result.coverage.scope, 'provided_inputs_only');
    assert.equal(result.coverage.exhaustive, false); assert.equal(isCompletedWorkflow(req, result), true);
    assert.equal(result.stats.inferenceCalls, 0); assert.equal(result.stats.networkCalls, 0);
  }
  const missing = request('test-diagnose', { resultPaths: ['result.tap', 'missing.tap'] }, []);
  const result = await runner.run(missing);
  assert.equal(result.status, 'needs_parent'); assert.equal(result.coverage.complete, false);
  assert.equal(result.acceptance.sources, 'unresolved'); assert.equal(isCompletedWorkflow(missing, result), false);
  const inline = request('test-diagnose', { registeredTest: 'inline' }, []);
  const injected = await runner.run(inline);
  assert.equal(injected.status, 'done'); assert.equal(injected.coverage.nativeUnsupported, 'inline_registered_test_not_file_snapshot');
  assert.equal(isCompletedWorkflow(inline, injected), false);
});

test('line-only read capability aggregates repeated logs before returning metadata', async () => {
  const text = Array.from({ length: 400 }, () => '{"timestamp":"2026-10-03T00:00:00Z","level":"error","message":"common failure"}').join('\n') + '\nrare failure\nrecovered success';
  let streamed = false;
  const ctx = context({}, { async read(file, role, { onLine } = {}) {
    assert.equal(file, 'stream.ndjson'); assert.equal(role, 'diagnostic_input'); assert.equal(typeof onLine, 'function');
    let byteStart = 0;
    for (const [i, textLine] of text.split('\n').entries()) {
      const byteEnd = byteStart + Buffer.byteLength(textLine);
      onLine({ text: textLine, lineStart: i + 1, lineEnd: i + 1, byteStart, byteEnd }); byteStart = byteEnd + 1;
    }
    streamed = true; return { path: file, hash: createHash('sha256').update(text).digest('hex'), ref: 'stream-source', streamed: true };
  } });
  const result = await runDiagnostics(request('log-triage', { paths: ['stream.ndjson'] }), ctx);
  assert.equal(streamed, true); assert.equal(result.status, 'done'); assert.equal(result.details.parsedCount, 402);
  assert.equal(result.details.groups[0].count, 400); assert.equal(result.details.timeline.length, 3);
  assert.equal(record(result, result.details.groups[0].first).ref.lineStart, 1);
  assert.equal(record(result, result.details.groups[0].last).ref.lineStart, 400);
  assert.equal(result.details.contrary.length, 1);
  assert.ok(JSON.stringify(result).length < 12000);
  assert.ok(result.evidence.every(e => e.hash === createHash('sha256').update(text).digest('hex')));
});

test('bounded pretty JSON buffering retains container records and malformed NDJSON singleton offsets', async () => {
  const result = await runDiagnostics(request('log-triage', { paths: ['blank.json', 'array.json', 'malformed.ndjson'] }), context({
    'blank.json': '{\n\n  "events": [{"message":"first error"}]\n}',
    'array.json': '{"events": [\n{"message":"second error"}\n]}',
    'malformed.ndjson': '{"message": broken\n{"message":"rare error"}\n{"message":"success"}'
  }));
  assert.equal(result.status, 'needs_parent'); assert.equal(result.details.parseFailures.length, 1);
  assert.ok(result.details.groups.some(g => record(result, g.first).message === 'first error'));
  assert.ok(result.details.groups.some(g => record(result, g.first).message === 'second error'));
  const rare = result.details.groups.find(g => record(result, g.first).message === 'rare error');
  assert.equal(record(result, rare.first).ref.lineStart, 2); assert.equal(result.details.contrary.length, 1);
});

test('recurring errors after recovery remain incident timeline transitions', async () => {
  const text = [
    { timestamp: '2026-10-03T00:00:00Z', message: 'same failure' },
    { timestamp: '2026-10-03T00:00:01Z', message: 'same failure' },
    { timestamp: '2026-10-03T00:00:02Z', message: 'recovered success' },
    { timestamp: '2026-10-03T00:00:03Z', message: 'same failure' }
  ].map(JSON.stringify).join('\n');
  const result = await runDiagnostics(request('log-triage', { paths: ['recurring.ndjson'] }), context({ 'recurring.ndjson': text }));
  assert.deepEqual(result.details.timeline.map(ref => record(result, ref).kind), ['error', 'success', 'error']);
  assert.deepEqual(result.details.timeline.map(ref => record(result, ref).ref.lineStart), [1, 3, 4]);
  assert.equal(result.details.groups[0].count, 3); assert.equal(result.details.contrary.length, 1);
  assert.equal(record(result, result.details.groups[0].last).ref.lineStart, 4);
});
