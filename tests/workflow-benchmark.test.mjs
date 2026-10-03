import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { runWorkflowBenchmark } from '../scripts/workflow-benchmark.mjs';

test('actual code batch and executor pass independent frozen checks in randomized repeated order', async () => {
  const report = await runWorkflowBenchmark({ rounds: 2, seed: 17 });
  assert.equal(report.observations.length, 12);
  assert.ok(report.observations.every(row => row.independentCheck === 'PASS'), JSON.stringify(report.observations.filter(row => row.failure)));
  assert.ok(report.observations.every(row => row.elapsedMs >= 0 && Number.isInteger(row.actions) && row.outputBytes > 0));
  assert.ok(report.observations.every(row => row.inferenceCalls === 0 && row.networkCalls === 0));
  assert.equal(report.modelArm.status, 'NONEXECUTABLE');
  for (const field of ['parentModelRequests', 'promptCache', 'endUserE2E', 'cost']) assert.equal(report[field], 'UNKNOWN');
  const orders = [0, 1].map(round => report.observations.filter(row => row.round === round).map(row => `${row.task}/${row.arm}`));
  assert.equal(new Set(orders[0]).size, 6); assert.equal(new Set(orders[1]).size, 6);
  assert.notDeepEqual(orders[0], orders[1]);
  assert.equal(Object.keys(report.fixtureHashes).length, 6);
  assert.ok(report.summary.every(row => row.passed === 2));
});
test('unqualified model arm never executes and cannot create model performance evidence', async () => {
  const report = await runWorkflowBenchmark({ rounds: 1, modelArm: { run: () => assert.fail('unqualified arm must not execute') } });
  assert.equal(report.modelArm.status, 'NONEXECUTABLE');
  assert.equal(report.observations.some(row => row.arm === 'executor_with_model'), false);
});
test('injected future arm receives no golden facts; inert decision is not executable model evidence', async () => {
  const report = await runWorkflowBenchmark({ rounds: 1, modelArm: {
    qualification: { status: 'qualified', consumer: 'fixture-bound branch', revision: 'test-only', evidenceRef: 'test-only' },
    run: async ({ task, request }) => {
      assert.equal(task.expected, undefined); assert.equal(request.expected, undefined);
      return { result: {}, actions: 0, decisionCalls: 0, inferenceCalls: 0, networkCalls: 0, decisionConsumerUsed: false };
    },
  } });
  assert.equal(report.modelArm.status, 'NONEXECUTABLE');
  const failed = report.observations.filter(row => row.arm === 'executor_with_model');
  assert.equal(failed.length, 3);
  assert.ok(failed.every(row => row.failure === 'MODEL_CONSUMER_NOT_EXECUTED' && row.independentCheck === 'FAIL'));
});
test('injected arm errors retain UNKNOWN missing measurements and useful failure evidence', async () => {
  const report = await runWorkflowBenchmark({ rounds: 1, modelArm: {
    qualification: { status: 'qualified', consumer: 'fixture-bound branch', revision: 'test-only', evidenceRef: 'test-only' },
    run: async () => { throw new Error('MEASURED_ARM_FAILED'); },
  } });
  const failed = report.observations.filter(row => row.arm === 'executor_with_model');
  assert.ok(failed.every(row => row.failure === 'MEASURED_ARM_FAILED' && row.actions === 'UNKNOWN' && row.inferenceCalls === 'UNKNOWN'));
  assert.equal(report.modelArm.status, 'NONEXECUTABLE');
  assert.ok(report.summary.filter(row => row.arm === 'executor_with_model').every(row => row.outputBytes === 'UNKNOWN'));
});
test('frozen fixture mutation stops the comparison and owned temporary fixture is removed', async () => {
  let ownedRoot;
  await assert.rejects(runWorkflowBenchmark({ rounds: 1, modelArm: {
    qualification: { status: 'qualified', consumer: 'fixture-bound branch', revision: 'test-only', evidenceRef: 'test-only' },
    run: async ({ root }) => { ownedRoot = root; await fs.writeFile(path.join(root, 'src/math.mjs'), 'changed'); return { result: {}, actions: 1, decisionCalls: 0, inferenceCalls: 0, networkCalls: 0 }; },
  } }), /FROZEN_FIXTURE_CHANGED/);
  await assert.rejects(fs.stat(ownedRoot), { code: 'ENOENT' });
});
test('invalid budgets or unsupported host metric claims fail before measurement', async () => {
  await assert.rejects(runWorkflowBenchmark({ rounds: 0 }), /INVALID_BENCHMARK_OPTIONS/);
  await assert.rejects(runWorkflowBenchmark({ hostTrace: { cost: 0 } }), /UNVERIFIED_HOST_TRACE/);
});
