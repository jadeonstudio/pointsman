import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { parseArgs } from 'node:util';
import { VERSION, MAX_FRAME_BYTES, ControlError, fail } from './constants.mjs';
import { resolveHome, setMode, getCredential, saveCredential, removeCredential, loadConfig } from './storage.mjs';
import { createDecisionEngine } from './engine.mjs';
import { createControlLayer } from './control-layer.mjs';
import { startMcp } from './mcp.mjs';
import { installationPlan, applyInstallation, describeHookStatus, layaAgentPlistPath, layaAgentPlist, REPO_ROOT } from './installer.mjs';
import { readMetrics } from './metrics.mjs';
import { layaSocketPath, layaSocketExists, createLayaSocketClient } from './inference.mjs';
import { readText } from './storage.mjs';

export const SMOKE_REQUEST = { purpose: 'route', risk: 'routine', state: { change: 'Documentation typo fix only; no executable code changed.', tests: { exitCode: 0 } }, questions: {
  category: { type: 'choice', instructions: 'What kind of change is described?', criteria: { documentation: 'Only documentation changed', implementation: 'Executable code changed' } },
  passed: { type: 'noul', instructions: 'Did the supplied test process exit successfully?' },
  complexity: { type: 'score', instructions: 'Rate the implementation complexity of this change.', criteria: ['No executable code change', 'Small implementation change', 'Large architectural change'] },
} };
export const SMOKE_RESPONSE = { model: 'jev-offline-fixture', usage: { input_tokens: 200, output_tokens: 0 }, answers: {
  category: { type: 'choice', choice: 'documentation', confidence: 0.98, probabilities: { documentation: 0.99, implementation: 0.01 } },
  passed: { type: 'noul', noul: 0.99 },
  complexity: { type: 'score', score: 0.02, confidence: 0.98, probabilities: { 0: 0.98, 1: 0.02, 2: 0 }, legend: { 0: null, 1: null, 2: null } },
} };
const output = value => process.stdout.write(JSON.stringify(value, null, 2) + '\n');
function executableFound(name, env) {
  return (env.PATH || '').split(path.delimiter).some(dir => { try { fs.accessSync(path.join(dir, name), fs.constants.X_OK); return true; } catch { return false; } });
}
export async function hiddenKeyPrompt() {
  if (!process.stdin.isTTY || !process.stderr.isTTY) fail('KEY_REQUIRES_YOUR_INTERACTIVE_TERMINAL');
  process.stderr.write('TypeSafe API key (hidden; stored outside repositories): ');
  const previousRaw = process.stdin.isRaw;
  process.stdin.setRawMode(true); process.stdin.resume();
  try {
    return await new Promise((resolve, reject) => {
      let value = '';
      const finish = (error, result) => { process.stdin.off('data', onData); error ? reject(error) : resolve(result); };
      const onData = buffer => {
        for (const ch of buffer.toString('utf8')) {
          if (ch === '\u0003' || ch === '\u0004') { finish(new ControlError('KEY_INPUT_CANCELLED')); return; }
          if (ch === '\r' || ch === '\n') { finish(null, value); return; }
          if (ch === '\u007f' || ch === '\b') { value = value.slice(0, -1); continue; }
          if (ch >= ' ' && ch <= '~') value += ch;
          if (value.length > 512) { finish(new ControlError('INVALID_API_KEY')); return; }
        }
      };
      process.stdin.on('data', onData);
    });
  } finally { process.stdin.setRawMode(Boolean(previousRaw)); process.stdin.pause(); process.stderr.write('\n'); }
}
async function readStdin() {
  if (process.stdin.isTTY) fail('PIPE_JSON_TO_STDIN');
  const chunks = []; let bytes = 0;
  for await (const chunk of process.stdin) {
    bytes += chunk.length; if (bytes > MAX_FRAME_BYTES) fail('INPUT_TOO_LARGE'); chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch { fail('INVALID_JSON'); }
}
export async function main(argv = process.argv.slice(2), env = process.env) {
  if (Number(process.versions.node.split('.')[0]) < 22) fail('NODE_22_REQUIRED');
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, strict: true, options: {
    home: { type: 'string' }, target: { type: 'string' }, scope: { type: 'string' }, project: { type: 'string' },
    'dry-run': { type: 'boolean' }, live: { type: 'boolean' }, days: { type: 'string' }, help: { type: 'boolean' },
    hooks: { type: 'boolean' }, 'hooks-only': { type: 'boolean' }, 'no-skills': { type: 'boolean' }, 'laya-agent': { type: 'boolean' },
  } });
  const [command = 'help', subcommand] = positionals;
  if (positionals.length > (command === 'key' ? 2 : 1)) fail('UNEXPECTED_ARGUMENTS');
  const home = resolveHome({ ...env, ...(values.home ? { POINTSMAN_HOME: values.home } : {}) });
  const engine = createDecisionEngine({ home, env });
  try {
  if (values.help || command === 'help') {
    process.stdout.write(`pointsman ${VERSION}\n\nCommands:\n  install|uninstall [--target both|codex|claude] [--scope user|project] [--project PATH] [--dry-run]\n    install --hooks          Also install owned host hooks and the short managed instruction block\n    uninstall --hooks-only   Remove only owned host hooks and the instruction block (MCP/skills/shim/mode unchanged)\n    --no-skills              Leave the host skills directory untouched (e.g. a symlinked skills root)\n    --laya-agent              macOS only: also install/remove the user launchd agent that runs \`laya serve\` at login\n  off|shadow|on       Shared switch, reread on every decision\n  status|doctor      Offline diagnostics; never prints a key\n  key set|remove     Run set yourself in an interactive terminal\n  smoke [--live]     Offline by default; live needs a key and active mode\n  decide             Read one JSON request from stdin\n  metrics [--days 7] Local metadata, not inferred savings\n  hook --host codex|claude --event pre-spawn|post-spawn|subagent-start|subagent-stop\n                      Fail-open host hook executor; reads one hook JSON from stdin, never denies\n  mcp                Local stdio server\n  migrate --from OLD_HOME [--dry-run]\n                      Move an existing jev-agent-control data home to POINTSMAN_HOME and fix its absolute paths\n\nGlobal: --home ABSOLUTE_PATH. No command accepts a key as an argument.\n`); return;
  }
  const allowed = { install: ['target', 'scope', 'project', 'dry-run', 'hooks', 'no-skills', 'laya-agent'], uninstall: ['target', 'scope', 'project', 'dry-run', 'hooks-only', 'no-skills', 'laya-agent'], smoke: ['live'], metrics: ['days'] };
  if (Object.keys(values).some(k => k !== 'home' && !(allowed[command] || []).includes(k))) fail('UNEXPECTED_OPTION');
  if (command === 'install' || command === 'uninstall') {
    const plan = installationPlan({ home, env, target: values.target || 'both', scope: values.scope || 'user', project: values.project || process.cwd(),
      remove: command === 'uninstall', hooks: Boolean(values.hooks), hooksOnly: Boolean(values['hooks-only']), skills: !values['no-skills'],
      layaAgent: Boolean(values['laya-agent']) });
    output(applyInstallation(plan, { dryRun: values['dry-run'] })); return;
  }
  if (['off', 'shadow', 'on'].includes(command)) {
    if (command !== 'off' && !engine.status().ready) fail(engine.status().provider === 'laya' ? 'LAYA_NOT_READY' : 'NO_API_KEY');
    setMode(home, command, env); output(engine.status()); return;
  }
  if (command === 'status') { output(engine.status()); return; }
  if (command === 'doctor') {
    const status = engine.status();
    const clients = { codex: executableFound('codex', env), claude: executableFound('claude', env) };
    const routerWarnings = createControlLayer({ home, env, engine }).status().features.router.warnings ?? [];
    // Metrics is read-only and best-effort here: a metrics read failure (bad logs dir, training
    // store error, etc.) must never fail `doctor` itself, only skip the training-candidate warning.
    let trainingCandidateWarnings = [];
    try {
      const ready = readMetrics(home, 7).labels?.trainingCandidateReady ?? {};
      trainingCandidateWarnings = Object.entries(ready).filter(([, isReady]) => isReady)
        .map(([purpose]) => `TRAINING_CANDIDATE_READY:${purpose} (observed strong-label count reached the configured minimum; this does not start training or a checkpoint promotion automatically)`);
    } catch { /* metrics read failure must not fail doctor */ }
    let layaServer = { plist: 'not-applicable', socketExists: layaSocketExists(home), responding: false, ready: false, loading: false };
    if (process.platform === 'darwin') {
      const plistPath = layaAgentPlistPath(env.HOME || os.homedir());
      const existing = readText(plistPath, { optional: true });
      const expected = layaAgentPlist({ execPath: process.execPath, cli: path.join(REPO_ROOT, 'bin/pointsman.mjs'), home });
      layaServer.plist = existing === null ? 'missing' : (existing === expected ? 'installed' : 'changed');
    }
    if (layaServer.socketExists) {
      try {
        const s = await createLayaSocketClient({ socketPath: layaSocketPath(home) }).status({ timeoutMs: 100 });
        layaServer = { ...layaServer, responding: true, ready: s.ready, loading: s.loading };
      } catch { /* server not responding; leave defaults */ }
    }
    output({ ...status, node: process.versions.node, platform: process.platform, clients,
      hooks: describeHookStatus({ home, env }), layaServer,
      checksPerformed: ['local-config', 'credential-readiness', 'client-path'], checksNotPerformed: ['native-client-e2e', 'live-api', 'native-hook-execution'],
      warnings: [status.credential === 'missing' ? 'Configure a key locally before shadow/on.' : null,
        !clients.codex && !clients.claude ? 'No host CLI found in PATH; MCP config can still be prepared.' : null,
        env.POINTSMAN_DISABLE === '1' ? 'POINTSMAN_DISABLE=1 overrides persistent mode. Restart inherited processes after changing environment.' : null,
        ...routerWarnings, ...trainingCandidateWarnings].filter(Boolean) });
    if (status.configError || status.credential === 'invalid') process.exitCode = 2;
    return;
  }
  if (command === 'key') {
    if (subcommand === 'set') { const key = await hiddenKeyPrompt(); saveCredential(home, key); output({ stored: true, source: 'managed-file', liveApiVerified: false }); }
    else if (subcommand === 'remove') { removeCredential(home); output({ removedManagedCredential: true, environmentUnchanged: true }); }
    else fail('INVALID_KEY_COMMAND'); return;
  }
  if (command === 'decide') { output(await engine.decide(await readStdin())); return; }
  if (command === 'metrics') { output(readMetrics(home, values.days ? Number(values.days) : 7)); return; }
  if (command === 'mcp') {
    const server = startMcp(engine);
    process.once('SIGINT', () => { server.close(); process.exit(0); });
    process.once('SIGTERM', () => { server.close(); process.exit(0); });
    return;
  }
  if (command === 'smoke') {
    if (values.live) {
      if (loadConfig(home, env).mode === 'off') fail('ENABLE_SHADOW_BEFORE_LIVE_SMOKE');
      if (!engine.status().ready) fail(engine.status().provider === 'laya' ? 'LAYA_NOT_READY' : 'NO_API_KEY');
      const result = await engine.decide(SMOKE_REQUEST);
      const ok = ['ACCEPTED', 'LOW_CONFIDENCE', 'UNQUALIFIED_PROVIDER', 'SHADOW'].includes(result.reason);
      const comparison = ok ? engine.feedback({ id: result.id, baseline: { category: 'documentation', passed: true, complexity: 0.02 } }) : null;
      output({ ok, live: true, result, comparison }); if (!ok) process.exitCode = 2;
    } else {
      const temp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-smoke-')));
      try {
        setMode(temp, 'on');
        const mockEngine = createDecisionEngine({ home: temp, env: { TYPESAFE_API_KEY: 'offline-fixture-not-a-real-key' }, provider: async () => structuredClone(SMOKE_RESPONSE) });
        const result = await mockEngine.decide(SMOKE_REQUEST);
        output({ ok: result.apply, live: false, fixture: true, externalApiCalls: 0, result }); if (!result.apply) process.exitCode = 2;
      } finally { fs.rmSync(temp, { recursive: true, force: true }); }
    }
    return;
  }
  fail('UNKNOWN_COMMAND');
  } finally { if (command !== 'mcp') engine.close(); }
}
