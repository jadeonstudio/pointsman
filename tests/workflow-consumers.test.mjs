import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { PassThrough } from 'node:stream';
import { startMcp } from '../src/mcp.mjs';
import { createWorkflowRunner } from '../src/workflows.mjs';
import { setMode } from '../src/storage.mjs';
import { setFeatureMode, setWorkflowNativeMode, workflowPolicy, validateFeaturePolicy } from '../src/feature-policy.mjs';
import { fixture, tempHome, delay } from './helpers.mjs';

const request = { workflow: 'repo-evidence', inputs: { symbols: ['hello'] }, acceptance: ['definition', 'tests'], budget: { maxDecisionCalls: 0 } };
function source(t) {
  const root = tempHome(t);
  fs.writeFileSync(path.join(root, 'hello.mjs'), 'export function hello() { return 7; }\n');
  fs.mkdirSync(path.join(root, 'tests'));
  fs.writeFileSync(path.join(root, 'tests/hello.test.mjs'), 'hello();\n');
  return root;
}
function cli(home, args, input) {
  const p = spawnSync(process.execPath, ['bin/pointsman.mjs', ...args, '--home', home], {
    env: { PATH: process.env.PATH, HOME: home }, input: JSON.stringify(input), encoding: 'utf8', timeout: 10000,
  });
  assert.equal(p.status, 0, p.stderr); assert.equal(p.stderr, ''); return JSON.parse(p.stdout);
}
function comparable(result) {
  const { id, stats, ...contract } = result;
  return contract;
}

test('legacy policy gains OFF defaults; native and workflow gates cannot override global OFF', t => {
  const home = tempHome(t);
  assert.equal(validateFeaturePolicy({ version: 2 }).workflow.mode, 'off');
  setFeatureMode(home, 'workflow', 'on'); setWorkflowNativeMode(home, 'on');
  assert.equal(workflowPolicy(home, 'off').mode, 'off');
  assert.equal(workflowPolicy(home, 'off').nativeMode, 'off');
  assert.equal(workflowPolicy(home, 'shadow').nativeMode, 'shadow');
  assert.throws(() => validateFeaturePolicy({ version: 2, workflow: { maxActions: 1025 } }));
  assert.throws(() => validateFeaturePolicy({ version: 2, workflow: { maxOutputBytes: 49153 } }));
});

test('CLI and JavaScript consume the same root-bound recipe without a provider', async t => {
  const home = tempHome(t), root = source(t);
  const disabled = cli(home, ['run', '--root', root], request);
  assert.equal(disabled.reason, 'OFF'); assert.equal(disabled.stats.actions, 0);
  cli(home, ['workflow', 'on']);
  const enabled = cli(home, ['on']);
  assert.equal(enabled.mode, 'on'); assert.equal(enabled.ready, false);
  const js = await createWorkflowRunner({ root, engine: { status: () => ({ mode: 'on' }) },
    getPolicy: () => workflowPolicy(home, 'on') }).run(request);
  const viaCli = cli(home, ['run', '--root', root], request);
  assert.equal(viaCli.status, 'done'); assert.equal(viaCli.stats.inferenceCalls, 0);
  const decision = cli(home, ['decide'], { purpose: 'select', risk: 'routine', state: 'Choose a supplied item.', questions: {
    pick: { type: 'choice', instructions: 'Choose.', criteria: { a: 'first', b: 'second' } },
  } });
  assert.equal(decision.apply, false); assert.equal(decision.reason, 'NO_API_KEY');
  assert.deepEqual(comparable(viaCli), comparable(js));
  const outside = cli(home, ['run', '--root', root], { ...request, inputs: { symbols: ['hello'], requiredPaths: ['../external.mjs'] } });
  assert.notEqual(outside.status, 'done');
  assert.equal(outside.evidence.some(e => e.path === '../external.mjs'), false);
});

test('MCP run returns actual evidence and cancellation uses the same abort signal', async t => {
  const f = fixture(t), root = source(t), input = new PassThrough(), output = new PassThrough(), messages = [];
  setFeatureMode(f.home, 'workflow', 'on');
  const realRunner = createWorkflowRunner({ root, engine: f.engine, getPolicy: () => workflowPolicy(f.home, 'on') });
  let cancelled = false;
  const runner = { run: (args, options) => args.wait ? new Promise(resolve => {
    options.signal.addEventListener('abort', () => { cancelled = true; resolve({ status: 'cancelled' }); }, { once: true });
  }) : realRunner.run(args, options) };
  output.on('data', chunk => messages.push(...chunk.toString().trim().split('\n').map(JSON.parse)));
  const server = startMcp(f.engine, { input, output, root, workflowRunner: runner });
  t.after(() => { server.close(); input.destroy(); output.destroy(); });
  const send = value => input.write(JSON.stringify({ jsonrpc: '2.0', ...value }) + '\n');
  const wait = async id => { for (let i = 0; i < 200; i++) { const m = messages.find(x => x.id === id); if (m) return m; await delay(5); } throw new Error('MCP timeout'); };
  send({ id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', clientInfo: { name: 'workflow-test', version: '1' } } });
  send({ method: 'notifications/initialized' });
  send({ id: 2, method: 'tools/call', params: { name: 'run', arguments: request } });
  const result = JSON.parse((await wait(2)).result.content[0].text);
  assert.equal(result.status, 'done'); assert.equal(result.evidence[0].path, 'hello.mjs'); assert.equal(f.calls(), 0);
  send({ id: 3, method: 'tools/call', params: { name: 'run', arguments: { wait: true } } });
  send({ method: 'notifications/cancelled', params: { requestId: 3 } });
  assert.equal(JSON.parse((await wait(3)).result.content[0].text).status, 'cancelled'); assert.equal(cancelled, true);
});

test('native CLI observes actual host version and separate gates before synthetic completion', t => {
  const home = tempHome(t), root = source(t), bin = path.join(home, 'bin'); fs.mkdirSync(bin);
  const host = path.join(bin, 'claude'); fs.writeFileSync(host, '#!/bin/sh\nprintf "2.1.287\\n"\n', { mode: 0o700 });
  const run = envelope => {
    const p = spawnSync(process.execPath, ['bin/pointsman.mjs', 'workflow-native', '--host', 'claude', '--event', 'turn-step', '--root', root, '--home', home], {
      env: { PATH: `${bin}:${process.env.PATH}`, HOME: home }, input: JSON.stringify(envelope), encoding: 'utf8', timeout: 10000,
    });
    assert.equal(p.status, 0, p.stderr); return JSON.parse(p.stdout);
  };
  const envelope = { scope: 'workflow', request };
  setMode(home, 'on', {}); setFeatureMode(home, 'workflow', 'on');
  assert.equal(run(envelope).apply, false);
  setWorkflowNativeMode(home, 'on');
  assert.equal(run(envelope).apply, true);
  fs.writeFileSync(host, '#!/bin/sh\nprintf "2.1.280\\n"\n', { mode: 0o700 });
  assert.equal(run({ ...envelope, runtime: { version: '2.1.287' } }).apply, false);
  setMode(home, 'off', {});
  assert.equal(run(envelope).apply, false);
});

test('workflow CLI cancellation closes held-open stdin and kills an in-flight host-version child', async t => {
  for (const phase of ['stdin', 'version']) await t.test(phase, async t => {
    const home = tempHome(t), root = source(t), bin = path.join(home, 'bin');
    fs.mkdirSync(bin);
    const marker = path.join(home, 'version-child.json');
    fs.writeFileSync(path.join(bin, 'claude'), `#!${process.execPath}\nimport fs from 'node:fs';\nfs.writeFileSync(${JSON.stringify(marker)}, JSON.stringify({pid:process.pid}));\nsetTimeout(() => console.log('2.1.288'), 12000);\n`, { mode: 0o700 });
    setMode(home, 'on', {}); setFeatureMode(home, 'workflow', 'on'); setWorkflowNativeMode(home, 'on');
    // A test-only readiness marker avoids sending SIGINT before the real handler exists.
    const readiness = "process.on('newListener', name => { if (name === 'SIGTERM' && process.listenerCount('SIGINT')) process.stderr.write('WORKFLOW_CANCEL_READY\\n'); });";
    const child = spawn(process.execPath, ['--import', `data:text/javascript,${encodeURIComponent(readiness)}`,
      'bin/pointsman.mjs', 'workflow-native', '--host', 'claude', '--event', 'turn-step', '--root', root, '--home', home],
    { env: { PATH: `${bin}:${process.env.PATH}`, HOME: home }, stdio: ['pipe', 'pipe', 'pipe'] });
    const closed = once(child, 'close');
    let stdout = '', stderr = '', versionPid;
    child.stdout.on('data', b => { stdout += b; }); child.stderr.on('data', b => { stderr += b; });
    child.stdin.on('error', () => {});
    t.after(() => {
      child.stdin.destroy();
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
      if (versionPid) { try { process.kill(versionPid, 'SIGKILL'); } catch (e) { if (e.code !== 'ESRCH') throw e; } }
    });
    child.stdin.write(JSON.stringify({ scope: 'workflow', request }));
    if (phase === 'version') child.stdin.end();
    for (let i = 0; i < 400 && !(phase === 'stdin' ? stderr.includes('WORKFLOW_CANCEL_READY') : fs.existsSync(marker)); i++) await delay(5);
    if (phase === 'version') { assert.ok(fs.existsSync(marker), stderr); versionPid = JSON.parse(fs.readFileSync(marker)).pid; }
    else assert.match(stderr, /WORKFLOW_CANCEL_READY/);
    child.kill(phase === 'stdin' ? 'SIGINT' : 'SIGTERM');
    let timer;
    try {
      await Promise.race([closed, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${phase} did not stop after cancellation`)), 4000); })]);
    } finally { clearTimeout(timer); }
    assert.doesNotMatch(stdout, /"apply"\s*:\s*true/);
    if (versionPid) {
      assert.throws(() => process.kill(versionPid, 0), { code: 'ESRCH' });
      versionPid = undefined;
    }
  });
});
