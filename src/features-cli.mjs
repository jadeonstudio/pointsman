import { parseArgs } from 'node:util';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { addAbortSignal } from 'node:stream';
import { MAX_FRAME_BYTES, fail } from './constants.mjs';
import { resolveHome } from './storage.mjs';
import { initializeFeaturePolicy, loadFeaturePolicy, setFeatureMode, presetHostRoles, setAbControlShare, setWorkflowNativeMode, workflowPolicy } from './feature-policy.mjs';
import { createDecisionEngine } from './engine.mjs';
import { createControlLayer } from './control-layer.mjs';
import { evaluatePairedRuns } from './evaluation.mjs';

const print = value => process.stdout.write(JSON.stringify(value, null, 2) + '\n');
async function jsonStdin(limit = MAX_FRAME_BYTES, signal) {
  if (process.stdin.isTTY) fail('PIPE_JSON_TO_STDIN');
  signal?.throwIfAborted();
  const input = signal ? addAbortSignal(signal, process.stdin) : process.stdin;
  const chunks = []; let size = 0;
  try {
    for await (const chunk of input) {
      size += Buffer.byteLength(chunk); if (size > limit) fail('INPUT_TOO_LARGE'); chunks.push(Buffer.from(chunk));
    }
  } catch (error) {
    if (signal?.aborted) fail('CANCELLED');
    throw error;
  }
  if (signal?.aborted) fail('CANCELLED');
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { fail('INVALID_JSON'); }
}
/** Handles only new commands. The existing CLI, key handling and installer remain unchanged. */
export async function featureMain(argv = process.argv.slice(2), env = process.env) {
  const peek = parseArgs({ args: argv, allowPositionals: true, strict: false, options: { home: { type: 'string' } } });
  const command = peek.positionals[0];
  if (!['router', 'bulk', 'effort', 'workflow', 'run', 'workflow-native', 'route', 'filter', 'policy', 'evaluate', 'status'].includes(command)) return false;
  if (Number(process.versions.node.split('.')[0]) < 22) fail('NODE_22_REQUIRED');
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, strict: true,
    options: { home: { type: 'string' }, help: { type: 'boolean' }, host: { type: 'string' }, root: { type: 'string' }, event: { type: 'string' }, 'dry-run': { type: 'boolean' }, replace: { type: 'boolean' } } });
  if (values.help) { process.stdout.write(FEATURE_HELP); return true; }
  const sub = positionals[1];
  const abOnly = ['router', 'effort'].includes(command) && sub === 'ab';
  const workflowNativeMode = command === 'workflow' && sub === 'native';
  if (positionals.length > (abOnly || workflowNativeMode ? 3 : (['router', 'bulk', 'effort', 'workflow', 'policy'].includes(command) ? 2 : 1))) fail('UNEXPECTED_ARGUMENTS');
  if (abOnly && positionals.length !== 3) fail('UNEXPECTED_ARGUMENTS');
  const rolesOnly = command === 'policy' && sub === 'roles';
  if (!rolesOnly && command !== 'workflow-native' && values.host !== undefined) fail('UNEXPECTED_OPTION');
  if (!rolesOnly && (values['dry-run'] !== undefined || values.replace !== undefined)) fail('UNEXPECTED_OPTION');
  if (!['run', 'workflow-native'].includes(command) && values.root !== undefined) fail('UNEXPECTED_OPTION');
  if (command !== 'workflow-native' && values.event !== undefined) fail('UNEXPECTED_OPTION');
  const home = resolveHome({ ...env, ...(values.home ? { POINTSMAN_HOME: values.home } : {}) });
  const engine = createDecisionEngine({ home, env });
  const layer = createControlLayer({ home, env, engine });
  try {
  if (abOnly) { setAbControlShare(home, positionals[2], command); print(layer.status()); }
  else if (workflowNativeMode) {
    setWorkflowNativeMode(home, positionals[2]); print(layer.status());
  } else if (command === 'workflow' && sub === 'status') print(layer.status());
  else if (command === 'router' || command === 'bulk' || command === 'effort' || command === 'workflow') {
    if (!['off', 'shadow', 'on'].includes(sub)) fail('INVALID_FEATURE_MODE');
    setFeatureMode(home, command, sub); print(layer.status());
  } else if (command === 'status') print(layer.status());
  else if (command === 'policy') {
    if (sub === 'init') print(initializeFeaturePolicy(home));
    else if (sub === 'check') print({ valid: true, policy: loadFeaturePolicy(home), nativeTargetsVerified: false });
    else if (sub === 'roles') {
      if (!['codex', 'claude'].includes(values.host)) fail('INVALID_ROLES_HOST');
      print(presetHostRoles(home, values.host, { env, dryRun: Boolean(values['dry-run']), replace: Boolean(values.replace) }));
    }
    else fail('INVALID_POLICY_COMMAND');
  } else if (command === 'route') { const { trace, ...request } = await jsonStdin(); print(await layer.route(request, { trace })); }
  else if (command === 'filter') {
    const { trace, ...request } = await jsonStdin();
    const result = await layer.filter(request, { trace }); print(result);
    if (!result.valid) process.exitCode = 2;
  } else if (command === 'run' || command === 'workflow-native') {
    const { createWorkflowRunner } = await import('./workflows.mjs');
    const getPolicy = () => workflowPolicy(home, engine.status().mode);
    const runner = createWorkflowRunner({ engine, root: values.root ?? process.cwd(), getPolicy });
    const controller = new AbortController();
    const cancel = () => controller.abort();
    process.once('SIGINT', cancel); process.once('SIGTERM', cancel);
    try {
      const request = await jsonStdin(MAX_FRAME_BYTES, controller.signal);
      if (command === 'run') print(await runner.run(request, { signal: controller.signal }));
      else {
        if (!['claude', 'gemini'].includes(values.host) ||
            !(values.host === 'claude' ? ['turn-step'] : ['before-agent', 'before-model']).includes(values.event)) fail('INVALID_WORKFLOW_HOST');
        const policy = getPolicy();
        const publicGemini = values.host === 'gemini' && (values.event === 'before-agent' || request?.hook_event_name !== undefined || request?.llm_request !== undefined || request?.scope !== 'workflow');
        const { prepareNativeWorkflow, geminiWorkflowResponse, geminiWorkflowEnvelope, WORKFLOW_HOST_CONTRACTS } = await import('./workflow-hosts.mjs');
        if (policy.nativeMode !== 'on') {
          if (publicGemini) geminiWorkflowEnvelope({ home, root: values.root ?? process.cwd(), getMode: () => ({ mode: 'off' }) }, request);
          print(publicGemini ? {} : { apply: false, reason: policy.nativeMode.toUpperCase() });
        }
        else {
          let version;
          try {
            const result = await promisify(execFile)(values.host, ['--version'], { timeout: 5000, maxBuffer: 4096, env, signal: controller.signal });
            version = result.stdout.match(/\b\d+\.\d+\.\d+\b/)?.[0];
          } catch { /* unavailable host delegates without running a workflow */ }
          if (publicGemini && (!version || controller.signal.aborted)) geminiWorkflowEnvelope({ home, root: values.root ?? process.cwd(), getMode: () => ({ mode: 'off' }) }, request);
          if (controller.signal.aborted) print(publicGemini ? {} : { apply: false, reason: 'cancelled' });
          else if (!version) print(publicGemini ? {} : { apply: false, reason: 'HOST_VERSION_UNAVAILABLE' });
          else {
            const native = values.host === 'gemini' ? geminiWorkflowResponse : prepareNativeWorkflow;
            const options = { host: values.host,
              runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS[values.host].revision },
              getMode: () => ({ globalMode: engine.status().mode, mode: getPolicy().nativeMode }), runner };
            if (publicGemini && request?.hook_event_name !== (values.event === 'before-agent' ? 'BeforeAgent' : 'BeforeModel')) {
              geminiWorkflowEnvelope({ home, root: values.root ?? process.cwd(), getMode: () => ({ mode: 'off' }) }, request);
            }
            const envelope = publicGemini ? (request?.hook_event_name === (values.event === 'before-agent' ? 'BeforeAgent' : 'BeforeModel') ?
              geminiWorkflowEnvelope({ ...options, home, root: values.root ?? process.cwd() }, request, { signal: controller.signal }) : null) : request;
            print(envelope ? await native(options, envelope, { signal: controller.signal }) : {});
          }
        }
      }
    } finally { process.removeListener('SIGINT', cancel); process.removeListener('SIGTERM', cancel); }
  } else if (command === 'evaluate') print(evaluatePairedRuns(await jsonStdin(1048576)));
  return true;
  } finally { engine.close(); }
}
export const FEATURE_HELP = `\nClassifier-inspired features (explicit Jev or local Laya provider):\n  policy init|check   Create/validate private features.json; no automatic targets\n  policy roles --host codex|claude [--dry-run] [--replace]\n                      Preset router.profiles[host] from roles the host actually has (~/.codex/agents/*.toml, ~/.claude/agents/*.md); refuses to overwrite an existing profile unless --replace\n  router off|shadow|on  Cap routing independently of the global switch\n  router ab SHARE|off Randomised control/treatment split (0..0.5) of otherwise-rewritten spawns; see README "Measuring cost and time"\n  bulk off|shadow|on    Cap prefiltering independently of the global switch\n  effort off|shadow|on  Cap the Claude Code main-loop reasoning-effort mod independently of the global switch (see mods/pointsman-effort)\n  effort ab SHARE|off Randomised control/treatment split (0..0.5) of otherwise-applied ON-mode effort changes\n  workflow off|shadow|on|status  Gate fixed recipes independently of the global switch\n  workflow native off|shadow|on Gate opt-in native completion independently\n  run [--root PATH]  Read one repo-evidence/test-diagnose/log-triage request from stdin\n  workflow-native --host claude|gemini --event turn-step|before-agent|before-model [--root PATH]\n                      Version-checked native bridge; requires both workflow gates ON\n  route|filter       Read bounded JSON from stdin; results contain no raw text\n  evaluate           Offline paired-execution report from JSON stdin\n`;
