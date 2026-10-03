// Wire stdin/stdout to a user-owned `codex app-server` process. This probe starts no model turn.
// stdout is protocol only; the observation packet is stderr. No subprocess is launched here.
import { createWorkflowRpc, createCodexWorkflowClient, WORKFLOW_HOST_CONTRACTS } from '../../src/workflow-hosts.mjs';

const version = process.argv[2];
if (version !== '0.154.0') throw new Error('Pass the actual observed `codex --version` value (supported: 0.154.0).');
const observed = [];
const rpc = createWorkflowRpc({ input: process.stdin, output: process.stdout, onNotification: event => observed.push(event.method), timeoutMs: 10000 });
try {
  await rpc.request('initialize', { clientInfo: { name: 'pointsman-native-probe', version: '0.1.0' }, capabilities: { experimentalApi: true } });
  rpc.notify('initialized');
  const client = createCodexWorkflowClient({ rpc, runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS.codex.revision },
    getMode: () => ({ globalMode: 'on', mode: 'on' }), cwd: process.cwd(), commands: { identity: { command: ['node', '--version'], timeoutMs: 5000 } } });
  const command = await client.runCommand('identity');
  process.stderr.write(JSON.stringify({ host: 'codex', version, evidence: 'installed_direct_command_probe',
    turnStartRequests: 0, providerRequests: 'UNKNOWN', notifications: observed, command }) + '\n');
} finally { rpc.close(); process.stdin.pause(); }
