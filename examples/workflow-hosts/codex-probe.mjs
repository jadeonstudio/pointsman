// Wire stdin/stdout to a user-owned `codex app-server` process. This probe starts no model turn.
// stdout is protocol only; the observation packet is stderr. No subprocess is launched here.
import { createWorkflowRpc, createCodexWorkflowClient, WORKFLOW_HOST_CONTRACTS } from '../../src/workflow-hosts.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const version = process.argv[2];
if (version !== '0.154.0') throw new Error('Pass the actual observed `codex --version` value (supported: 0.154.0).');
const observed = [], methods = [];
const transport = createWorkflowRpc({ input: process.stdin, output: process.stdout, onNotification: event => observed.push(event.method), timeoutMs: 15000 });
const rpc = {
  request(method, params, options) {
    if (method === 'turn/start') throw new Error('MODEL_TURN_FORBIDDEN_IN_PROTOCOL_PROBE');
    methods.push(method); return transport.request(method, params, options);
  },
  notify(method, params) { methods.push(method); transport.notify(method, params); },
  close: transport.close,
};
try {
  await rpc.request('initialize', { clientInfo: { name: 'pointsman-native-probe', version: '0.1.0' }, capabilities: { experimentalApi: true } });
  rpc.notify('initialized');
  if (process.argv[3] === '--workflow') {
    const manifest = JSON.parse(await readFile(process.argv[4], 'utf8'));
    const thread = await rpc.request('thread/start', { cwd: manifest.fixtureRoot, ephemeral: true, sandbox: 'read-only', approvalPolicy: 'on-request' });
    const threadId = thread.thread.id;
    const inventory = await rpc.request('mcpServerStatus/list', { threadId, detail: 'toolsAndAuthOnly', limit: 100 });
    const registered = inventory.data.find(server => server.name === manifest.server);
    if (!registered?.tools?.run) throw new Error(`WORKFLOW_TOOL_NOT_REGISTERED:${JSON.stringify({ registered: registered?.name, tools: Object.keys(registered?.tools ?? {}), error: registered?.toolsError })}`);
    const client = createCodexWorkflowClient({ rpc, runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS.codex.revision },
      getMode: () => ({ globalMode: 'on', mode: 'on' }), server: manifest.server, tool: 'run' });
    const envelope = { scope: 'workflow', threadId, request: manifest.request, input: [] };
    const before = methods.length, cancelled = new AbortController(); cancelled.abort();
    let cancelledBeforeDispatch = false;
    try { await client.run(envelope, { signal: cancelled.signal }); } catch (error) { cancelledBeforeDispatch = error.name === 'AbortError'; }
    if (!cancelledBeforeDispatch || methods.length !== before) throw new Error('CANCELLATION_DISPATCHED_A_REQUEST');
    const result = await client.run(envelope);
    const packet = result.result;
    const paths = role => [...new Set(packet.evidence.filter(item => (item.roles ?? [item.role]).includes(role)).map(item => item.path))].sort();
    const facts = { definitions: paths('definition'), callers: paths('direct_caller'), tests: paths('test') };
    for (const [file, expected] of Object.entries(manifest.request.snapshot.files)) {
      const actual = createHash('sha256').update(await readFile(`${manifest.fixtureRoot}/${file}`)).digest('hex');
      if (actual !== expected || packet.snapshot.files[file] !== expected) throw new Error(`FIXTURE_CHANGED:${file}`);
    }
    const evidence = { observedAt: new Date().toISOString(), host: 'codex', version, evidence: 'installed_direct_mcp_workflow_probe',
      frozenAt: manifest.frozenAt, runtimeSourceHashes: manifest.runtimeSourceHashes,
      ephemeralThread: true, sandbox: thread.sandbox, approvalsReviewer: thread.approvalsReviewer,
      registration: { server: registered.name, runtimeStatus: registered.runtimeStatus, run: registered.tools.run.name, serverInfo: registered.serverInfo },
      methodsSent: methods, notifications: observed, turnStartRequests: methods.filter(method => method === 'turn/start').length,
      providerRequests: 'UNKNOWN', workflowStats: packet.stats, independentCompletion: JSON.stringify(facts) === JSON.stringify(manifest.expected) ? 'PASS' : 'FAIL', facts, packet,
      cancellation: { beforeDispatch: 'PASS', nativeInFlight: 'UNKNOWN: no direct MCP cancellation method in installed schema' }, visibleHistoryContinuation: 'UNKNOWN' };
    if (manifest.outputPath) await writeFile(manifest.outputPath, JSON.stringify(evidence, null, 2) + '\n');
    process.stderr.write(JSON.stringify({ ...evidence, packet: undefined }) + '\n');
    if (evidence.independentCompletion !== 'PASS') throw new Error(`INDEPENDENT_ORACLE_FAILED:${JSON.stringify(facts)}`);
  } else {
  const client = createCodexWorkflowClient({ rpc, runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS.codex.revision },
    getMode: () => ({ globalMode: 'on', mode: 'on' }), cwd: process.cwd(), commands: { identity: { command: ['node', '--version'], timeoutMs: 5000 } } });
  const command = await client.runCommand('identity');
  process.stderr.write(JSON.stringify({ host: 'codex', version, evidence: 'installed_direct_command_probe',
    turnStartRequests: 0, providerRequests: 'UNKNOWN', notifications: observed, command }) + '\n');
  }
} finally { rpc.close(); process.stdin.pause(); }
