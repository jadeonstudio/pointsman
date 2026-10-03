import test from 'node:test';
import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { stripTypeScriptTypes } from 'node:module';
import { createWorkflowRunner } from '../src/workflows.mjs';
import { WORKFLOW_HOST_CONTRACTS, matchesWorkflowHost, prepareNativeWorkflow, claudeWorkflowStep, geminiWorkflowResponse, createCodexWorkflowClient, createWorkflowRpc } from '../src/workflow-hosts.mjs';

const request = { workflow: 'repo-evidence', inputs: { symbols: ['hello'] } };
const completed = () => ({ version: 1, recipeRevision: 'workflow-v1', workflow: 'repo-evidence', mode: 'on', status: 'done', needsParent: null,
  acceptance: { source: 'met' }, coverage: { complete: true, omissions: [] }, evidence: [{ ref: 'src/a:1', path: 'src/a', text: 'source', role: 'definition', hash: 'a'.repeat(64) }],
  snapshot: { revision: 'b'.repeat(40), files: { 'src/a': 'a'.repeat(64) } }, authorizesExecution: false, stats: { inferenceCalls: 0, networkCalls: 0 } });
const runtime = host => ({ version: host === 'claude' ? '2.1.287' : host === 'codex' ? '0.154.0' : '0.42.0', contractRevision: WORKFLOW_HOST_CONTRACTS[host].revision });
const opts = (host, result = completed()) => ({ runtime: runtime(host), getMode: () => ({ globalMode: 'on', mode: 'on' }), runner: { run: async () => result } });
const envelope = { scope: 'workflow', request, threadId: 'thread-1', input: [{ type: 'text', text: 'Do this bounded workflow' }] };

test('version-bound contract and default/OFF/SHADOW pass-through avoid execution', async () => {
  assert.equal(matchesWorkflowHost('claude', runtime('claude')), true);
  assert.equal(matchesWorkflowHost('claude', { ...runtime('claude'), version: '2.1.280' }), false);
  assert.equal(matchesWorkflowHost('codex', { ...runtime('codex'), version: '0.155.0' }), false);
  assert.equal(matchesWorkflowHost('claude', runtime('claude'), '2.1.280'), false);
  for (const mode of [{ globalMode: 'off', mode: 'on' }, { globalMode: 'shadow', mode: 'on' }, { globalMode: 'on', mode: 'shadow' }, {}]) {
    const out = await prepareNativeWorkflow({ ...opts('claude'), host: 'claude', getMode: () => mode, runner: { run: () => assert.fail('OFF must not execute') } }, envelope);
    assert.equal(out.apply, false);
  }
});
test('malformed, unresolved, incomplete evidence and changed policy fail through', async () => {
  for (const mutate of [r => { r.status = 'needs_parent'; }, r => { r.acceptance.source = 'unresolved'; }, r => { r.workflow = 'wrong'; }, r => { r.evidence[0].hash = 'bad'; }, r => { r.coverage.omissions.push('caller'); }]) {
    const result = completed(); mutate(result);
    assert.equal((await prepareNativeWorkflow({ ...opts('claude', result), host: 'claude' }, envelope)).apply, false);
  }
  let calls = 0;
  assert.equal((await prepareNativeWorkflow({ ...opts('claude'), host: 'claude', getMode: () => ({ globalMode: 'on', mode: ++calls === 1 ? 'on' : 'off' }) }, envelope)).apply, false);
  assert.equal((await prepareNativeWorkflow({ ...opts('claude'), host: 'claude' }, { ...envelope, request: { ...request, snapshot: { revision: 'c'.repeat(40) } } })).apply, false);
  assert.equal((await prepareNativeWorkflow({ ...opts('claude'), host: 'claude' }, { ...envelope, request: { ...request, snapshot: { files: { 'src/a': 'd'.repeat(64) } } } })).apply, false);
});
test('Claude yields native chunks and preserves fallback stream/result', async () => {
  let modelRequests = 0;
  const event = { turnId: 't', index: 0, model: 'unchanged' };
  const next = async function* (e) { assert.equal(e, event); modelRequests++; yield { kind: 'text', index: 0, text: 'model' }; return { answer: 'model' }; };
  const stream = claudeWorkflowStep(event, next, opts('claude'), envelope);
  assert.deepEqual((await stream.next()).value.kind, 'text');
  assert.deepEqual((await stream.next()).value, { kind: 'stop', stopReason: 'end_turn', usage: null });
  const final = await stream.next(); assert.equal(final.done, true); assert.equal(final.value.turnId, 't'); assert.equal(modelRequests, 0);
  const fallback = claudeWorkflowStep(event, next, opts('claude', { status: 'needs_parent' }), envelope);
  assert.equal((await fallback.next()).value.text, 'model'); assert.equal((await fallback.next()).value.answer, 'model'); assert.equal(modelRequests, 1);
});
test('Gemini uses string parts with no synthetic functionCall or fabricated token usage', async () => {
  const out = await geminiWorkflowResponse(opts('gemini'), envelope);
  assert.equal(out.decision, 'deny');
  assert.equal(typeof out.hookSpecificOutput.llm_response.candidates[0].content.parts[0], 'string');
  assert.equal(out.hookSpecificOutput.llm_response.usageMetadata, undefined);
  assert.deepEqual(await geminiWorkflowResponse(opts('gemini', {}), envelope), {});
});
test('Codex direct MCP completed path starts no turn; unresolved starts one preserving input', async () => {
  const calls = [];
  let result = completed();
  const rpc = { request: async (method, params) => { calls.push({ method, params }); return method === 'mcpServer/tool/call' ? { structuredContent: result } : { turn: { id: 't' } }; } };
  const client = createCodexWorkflowClient({ rpc, runtime: runtime('codex'), getMode: opts('codex').getMode });
  assert.equal((await client.run(envelope)).turnsStarted, 0);
  assert.deepEqual(calls.map(c => c.method), ['mcpServer/tool/call']);
  assert.equal(calls[0].params.tool, 'run');
  result = { ...completed(), status: 'needs_parent', needsParent: 'reason' };
  assert.equal((await client.run(envelope)).turnsStarted, 1);
  assert.deepEqual(calls.map(c => c.method), ['mcpServer/tool/call', 'mcpServer/tool/call', 'turn/start']);
  assert.deepEqual(calls[2].params.input[0], envelope.input[0]);
});
test('Codex fixed registry only, restrictive sandbox, cancellation prevents continuation', async () => {
  const calls = [], controller = new AbortController();
  const rpc = { request: async (method, params) => { calls.push({ method, params }); return { exitCode: 0 }; } };
  const settings = { rpc, runtime: runtime('codex'), getMode: opts('codex').getMode, cwd: '/repo', commands: { check: { command: ['node', '--version'], timeoutMs: 1000 } } };
  assert.throws(() => createCodexWorkflowClient({ ...settings, sandboxPolicy: { type: 'dangerFullAccess' } }));
  const client = createCodexWorkflowClient(settings);
  await assert.rejects(client.runCommand('user-argv'), /UNREGISTERED/);
  await client.runCommand('check'); assert.equal(calls[0].method, 'command/exec'); assert.deepEqual(calls[0].params.sandboxPolicy, { type: 'readOnly' });
  const cancelClient = createCodexWorkflowClient({ ...settings, rpc: { request: async () => { controller.abort(); return { structuredContent: completed() }; } } });
  await assert.rejects(cancelClient.run(envelope, { signal: controller.signal }), /abort/i);
});
test('cancelled turn/start ACK is interrupted before returning', async () => {
  const controller = new AbortController(), calls = [];
  const client = createCodexWorkflowClient({ getMode: () => ({ globalMode: 'off', mode: 'off' }), runtime: runtime('codex'), rpc: { request: async method => {
    calls.push(method);
    if (method === 'turn/start') { controller.abort(); return { turn: { id: 'late-ack' } }; }
    return {};
  } } });
  await assert.rejects(client.run(envelope, { signal: controller.signal }), /abort/i);
  assert.deepEqual(calls, ['turn/start', 'turn/interrupt']);
});
test('real shared runner completion is consumed with zero inference calls', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-native-fixture-'));
  try {
    await fs.writeFile(path.join(root, 'source.mjs'), 'export function hello() { return 1; }\n');
    await fs.writeFile(path.join(root, 'source.test.mjs'), 'hello();\n');
    const runner = createWorkflowRunner({ root, getPolicy: () => ({ mode: 'on' }), engine: { status: () => ({ mode: 'on', policyRevision: 'fixture' }), decide: () => assert.fail('unique definition needs no provider') } });
    const out = await prepareNativeWorkflow({ ...opts('claude'), host: 'claude', runner }, envelope);
    assert.equal(out.apply, true, JSON.stringify(out));
    assert.equal(out.result.stats.inferenceCalls, 0);
    assert.equal(out.result.stats.networkCalls, 0);
    assert.equal(out.result.coverage.complete, true);
  } finally { await fs.rm(root, { recursive: true, force: true }); }
});
test('stdio JSON-RPC handshake, request/notification and approval forwarding', async () => {
  const input = new PassThrough(), output = new PassThrough(), sent = [];
  output.on('data', data => sent.push(JSON.parse(String(data))));
  const rpc = createWorkflowRpc({ input, output, onRequest: async msg => { assert.equal(msg.method, 'item/commandExecution/requestApproval'); return { decision: 'decline' }; } });
  const handshake = rpc.request('initialize', { clientInfo: { name: 'pointsman-workflow', version: '0.1.0' }, capabilities: { experimentalApi: true } });
  input.write(JSON.stringify({ id: sent[0].id, result: { userAgent: 'fixture' } }) + '\n');
  assert.equal((await handshake).userAgent, 'fixture'); rpc.notify('initialized');
  input.write(JSON.stringify({ id: 99, method: 'item/commandExecution/requestApproval', params: {} }) + '\n');
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(sent.at(-1), { id: 99, result: { decision: 'decline' } });
  rpc.close();
});
test('actual Claude mod registers bounded explicit entry and yields synthetic response chunks', async () => {
  const source = await fs.readFile(new URL('../mods/pointsman-workflows/hooks/pointsman-workflows.ts', import.meta.url), 'utf8');
  const module = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);
  const hooks = {}, calls = [];
  module.register((name, fn) => { hooks[name] = fn; }, { pointsmanPath: '/owned/pointsman' });
  const $ = { process: { run: async (argv, options) => {
    calls.push({ argv, options }); return { exitCode: 0, stdout: JSON.stringify({ apply: true, text: 'verified bounded evidence', contractRevision: WORKFLOW_HOST_CONTRACTS.claude.revision }) };
  } } };
  const step = { turnId: 'native-fixture', index: 0, model: 'locked-original' };
  let modelCalls = 0;
  const next = async function* (event) { assert.equal(event, step); modelCalls++; yield { kind: 'text', index: 0, text: 'normal' }; return { answer: 'normal' }; };
  next.signal = new AbortController().signal;
  await hooks['prompt.submit']($, { text: 'ordinary prompt' }, async event => event);
  await hooks['turn.step']($, step, next).next();
  assert.equal(modelCalls, 1); assert.equal(calls.length, 0);
  await hooks['prompt.submit']($, { text: `pointsman-workflow ${JSON.stringify(request)}` }, async event => event);
  const stream = hooks['turn.step']($, step, next);
  assert.deepEqual((await stream.next()).value, { kind: 'text', index: 0, text: 'verified bounded evidence' });
  assert.equal((await stream.next()).value.kind, 'stop');
  assert.equal((await stream.next()).value.turnId, step.turnId);
  assert.deepEqual(calls[0].argv, ['/owned/pointsman', 'workflow-native', '--host', 'claude', '--event', 'turn-step']);
  assert.deepEqual(JSON.parse(calls[0].options.stdin).request, request);
  assert.equal(modelCalls, 1);
});
test('Claude ordinary entry is explicit; slash, malformed JSON and plugin submissions preserve fallback', async () => {
  const source = await fs.readFile(new URL('../mods/pointsman-workflows/hooks/pointsman-workflows.ts', import.meta.url), 'utf8');
  const module = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);
  const hooks = {};
  module.register((name, fn) => { hooks[name] = fn; }, {});
  const $ = { process: { run: () => assert.fail('Non-entry text must not invoke the bridge') } };
  const next = async function* () { yield { kind: 'text', index: 0, text: 'normal' }; return { answer: 'normal' }; };
  next.signal = new AbortController().signal;
  for (const event of [
    { text: `/pointsman-workflow ${JSON.stringify(request)}` },
    { text: `Please run pointsman-workflow ${JSON.stringify(request)}` },
    { text: 'pointsman-workflow {broken' },
    { text: `pointsman-workflow ${JSON.stringify(request)}`, origin: { kind: 'plugin' } },
  ]) {
    assert.equal(await hooks['prompt.submit']($, event, async received => received), event);
    assert.equal((await hooks['turn.step']($, { turnId: 'entry-guard', index: 0 }, next).next()).value.text, 'normal');
  }
});
test('Claude cancelled bridge suppresses late success and rejection fallback; preabort starts nothing', async () => {
  const source = await fs.readFile(new URL('../mods/pointsman-workflows/hooks/pointsman-workflows.ts', import.meta.url), 'utf8');
  const module = await import(`data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString('base64')}`);
  for (const scenario of ['resolve', 'reject', 'preabort']) {
    const hooks = {}, controller = new AbortController();
    let settle, started, bridgeCalls = 0, modelCalls = 0;
    const entered = new Promise(resolve => { started = resolve; });
    module.register((name, fn) => { hooks[name] = fn; }, { pointsmanPath: '/owned/pointsman' });
    const $ = { process: { run: () => { bridgeCalls++; started(); return new Promise((resolve, reject) => {
      settle = () => scenario === 'reject' ? reject(new Error('cancelled process')) : resolve({ exitCode: 0, stdout: JSON.stringify({ apply: true, text: 'late packet', contractRevision: WORKFLOW_HOST_CONTRACTS.claude.revision }) });
    }); } } };
    const next = async function* () { modelCalls++; yield { kind: 'text', index: 0, text: 'normal' }; };
    next.signal = controller.signal;
    await hooks['prompt.submit']($, { text: `pointsman-workflow ${JSON.stringify(request)}` }, async event => event);
    if (scenario === 'preabort') controller.abort();
    const result = hooks['turn.step']($, { turnId: 'cancelled-entry', index: 0 }, next).next();
    if (scenario !== 'preabort') { await entered; controller.abort(); settle(); }
    assert.deepEqual(await result, { value: undefined, done: true });
    assert.equal(bridgeCalls, scenario === 'preabort' ? 0 : 1);
    assert.equal(modelCalls, 0);
  }
});
