// Installed Claude plugin-test host; no authenticated query loop or provider exists.
// The native testing kit has no bottom fs/network/process implementation.
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { createWorkflowRunner } from '../../src/workflows.mjs';
import { WORKFLOW_HOST_CONTRACTS } from '../../src/workflow-hosts.mjs';

const binary = process.argv[2] ?? path.join(os.homedir(), '.local', 'bin', 'claude');
const version = spawnSync(binary, ['--version'], { encoding: 'utf8' }).stdout.trim().split(' ')[0];
assert.equal(version, '2.1.288', 'Revalidate the installed native testing API after a version change.');
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-claude-native-')));
const plugin = path.join(root, 'plugin');
const sourcePlugin = new URL('../../mods/pointsman-workflows/', import.meta.url);
try {
  await fs.writeFile(path.join(root, 'source.mjs'), 'export function sum(a,b) { return a+b; }\n');
  await fs.writeFile(path.join(root, 'source.test.mjs'), 'sum(1,2);\n');
  const request = { workflow: 'repo-evidence', inputs: { symbols: ['sum'] }, acceptance: ['definition', 'direct_callers', 'tests'] };
  const runner = createWorkflowRunner({ root, engine: { status: () => ({ mode: 'on', policyRevision: 'frozen-native-probe' }), decide: () => assert.fail('No model branch is needed') }, getPolicy: () => ({ mode: 'on' }) });
  const packet = await runner.run(request);
  assert.equal(packet.status, 'done');
  assert.equal(packet.stats.inferenceCalls, 0);
  assert.equal(packet.stats.networkCalls, 0);
  await fs.cp(sourcePlugin, plugin, { recursive: true });
  await fs.mkdir(path.join(plugin, 'tests'));
  const testSource = `import { describe, expect, mock, test } from 'claude-code/testing';
const request = ${JSON.stringify(request)};
const packet = ${JSON.stringify(packet)};
const response = { apply: true, text: JSON.stringify(packet), contractRevision: ${JSON.stringify(WORKFLOW_HOST_CONTRACTS.claude.revision)} };
const event = { turnId: 'owned-native-fixture', index: 0, model: 'unchanged-model', messageCount: 1 };
function fixture(on, output = response) {
  const calls = { bridge: 0, delegate: 0, network: 0 };
  mock.env(on, { HOME: '/owned-probe-home' });
  on('prompt.submit', ($, e) => ({ text: e.text, origin: e.origin }));
  on('turn.complete', () => ({ text: '' }));
  on('http.fetch', () => { calls.network++; throw new Error('NETWORK_FORBIDDEN'); });
  on('process.run', ($, e) => {
    calls.bridge++;
    expect(e.argv).toEqual(['/owned-probe-home/.local/bin/pointsman', 'workflow-native', '--host', 'claude', '--event', 'turn-step']);
    expect(JSON.parse(e.init.stdin).request).toEqual(request);
    return { value: { exitCode: 0, stdout: JSON.stringify(output), stderr: '' } };
  });
  on('turn.step', async function* ($, e) {
    calls.delegate++;
    yield { kind: 'text', index: 0, text: 'fallback-sentinel' };
    yield { kind: 'stop', stopReason: 'end_turn', usage: null };
    return { turnId: e.turnId, index: e.index, answer: 'fallback-sentinel', toolUses: [], stopReason: 'end_turn', usage: null };
  });
  return calls;
}
async function prompt($, text) { await $.prompt.submit({ text, origin: { kind: 'composer' }, wait: false }); }
async function consume($, e = event) {
  const stream = $.turn.step(e), chunks = [];
  while (true) {
    const item = await stream.next();
    if (item.done) return { chunks, result: item.value };
    chunks.push(item.value);
  }
}
describe('pointsman-workflows installed native component', () => {
  test('completed segment streams text and final result without bottom model request; ordinary continuation delegates once', async ($, on) => {
    const calls = fixture(on);
    await prompt($, '/pointsman-workflow ' + JSON.stringify(request));
    const out = await consume($);
    expect(out.chunks).toEqual([{ kind: 'text', index: 0, text: JSON.stringify(packet) }, { kind: 'stop', stopReason: 'end_turn', usage: null }]);
    expect(out.result).toEqual({ turnId: event.turnId, index: 0, answer: JSON.stringify(packet), toolUses: [], stopReason: 'end_turn', usage: null });
    expect(calls).toEqual({ bridge: 1, delegate: 0, network: 0 });
    await prompt($, 'Continue normally.');
    const continued = await consume($, { ...event, turnId: 'next-turn', messageCount: 3 });
    expect(continued.result.answer).toBe('fallback-sentinel');
    expect(calls).toEqual({ bridge: 1, delegate: 1, network: 0 });
  });
  test('unapplied bridge result delegates once and preserves fallback chunks/result', async ($, on) => {
    const calls = fixture(on, { apply: false, reason: 'off' });
    await prompt($, '/pointsman-workflow ' + JSON.stringify(request));
    const out = await consume($);
    expect(out.chunks[0].text).toBe('fallback-sentinel');
    expect(out.result.answer).toBe('fallback-sentinel');
    expect(calls).toEqual({ bridge: 1, delegate: 1, network: 0 });
  });
  test('subagent and later step bypass the bridge unchanged', async ($, on) => {
    const calls = fixture(on);
    await prompt($, '/pointsman-workflow ' + JSON.stringify(request));
    expect((await consume($, { ...event, agentId: 'child' })).result.answer).toBe('fallback-sentinel');
    expect((await consume($, { ...event, index: 1 })).result.answer).toBe('fallback-sentinel');
    expect(calls).toEqual({ bridge: 0, delegate: 2, network: 0 });
  });
  test('closing before stream consumption calls neither bridge nor bottom model', async ($, on) => {
    const calls = fixture(on);
    await prompt($, '/pointsman-workflow ' + JSON.stringify(request));
    const stream = $.turn.step(event);
    await stream.return();
    expect(calls).toEqual({ bridge: 0, delegate: 0, network: 0 });
  });
});
`;
  await fs.writeFile(path.join(plugin, 'tests/pointsman-workflows.test.ts'), testSource);
  // No auth/API environment is copied to the native test process.
  const env = { PATH: process.env.PATH, HOME: os.homedir() };
  const validation = spawnSync(binary, ['plugin', 'validate', plugin], { env, encoding: 'utf8', timeout: 15000 });
  assert.equal(validation.status, 0, validation.stdout + validation.stderr);
  const run = spawnSync(binary, ['plugin', 'test', plugin], { env, encoding: 'utf8', timeout: 30000 });
  const evidence = { observedAt: new Date().toISOString(), host: 'claude', version,
    grade: 'installed_native_plugin_test_host_with_injected_bottom_hooks',
    binaryHash: createHash('sha256').update(await fs.readFile(binary)).digest('hex'),
    moduleHash: createHash('sha256').update(await fs.readFile(new URL('hooks/pointsman-workflows.ts', sourcePlugin))).digest('hex'),
    contractRevision: WORKFLOW_HOST_CONTRACTS.claude.revision,
    officialTypes: { commit: '684800b206824dfd0cc8a876e8604b20f72c3617', generatedByVersion: '2.1.277', url: 'https://github.com/anthropics/claude-code/blob/684800b206824dfd0cc8a876e8604b20f72c3617/mods/types/claude-code.d.ts' },
    command: ['claude', 'plugin', 'test', '<owned temporary plugin copy>'],
    productionManifestValidation: validation.status === 0 ? 'PASS' : 'FAIL',
    nativeTestExitCode: run.status, nativeTestOutput: (run.stdout + run.stderr).split('\n').filter(line => /\(pass\)|\(fail\)| pass$| fail$|Ran \d|TypeError:|Received:/.test(line)).join('\n'),
    installedTestingStream: 'Async generator: capture done.value through next(); installed test stream has no .result property.',
    actualProviderPresentInTestHost: false, actualExternalModelRequests: 0,
    providerCountScope: 'No provider implementation exists beneath native testing hooks; this does not measure an authenticated CLI session.',
    workflowInferenceCalls: packet.stats.inferenceCalls, workflowNetworkCalls: packet.stats.networkCalls,
    globalSettingsChanged: false, activationChanged: false,
    streamAndResult: run.status === 0 ? 'PASS_NATIVE_TEST_HOST' : 'FAIL',
    ordinaryContinuation: run.status === 0 ? 'PASS_NATIVE_HOOK_CHAIN_WITH_SENTINEL' : 'FAIL',
    cancellationBeforeConsumption: run.status === 0 ? 'PASS_NATIVE_TEST_HOST' : 'FAIL',
    visibleUi: 'UNKNOWN', assistantHistoryRetention: 'UNKNOWN', authenticatedContinuation: 'UNKNOWN', inFlightCancellation: 'UNKNOWN',
    adoption: 'UNQUALIFIED_FULL_NATIVE_SESSION' };
  await fs.writeFile(new URL('claude-component-evidence.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify(evidence));
  assert.equal(run.status, 0, run.stdout + run.stderr);
} finally { await fs.rm(root, { recursive: true, force: true }); }
