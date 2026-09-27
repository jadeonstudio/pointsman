import { parseArgs } from 'node:util';
import { MAX_FRAME_BYTES, fail } from './constants.mjs';
import { resolveHome } from './storage.mjs';
import { initializeFeaturePolicy, loadFeaturePolicy, setFeatureMode, presetHostRoles } from './feature-policy.mjs';
import { createDecisionEngine } from './engine.mjs';
import { createControlLayer } from './control-layer.mjs';
import { evaluatePairedRuns } from './evaluation.mjs';

const print = value => process.stdout.write(JSON.stringify(value, null, 2) + '\n');
async function jsonStdin(limit = MAX_FRAME_BYTES) {
  if (process.stdin.isTTY) fail('PIPE_JSON_TO_STDIN');
  const chunks = []; let size = 0;
  for await (const chunk of process.stdin) {
    size += Buffer.byteLength(chunk); if (size > limit) fail('INPUT_TOO_LARGE'); chunks.push(Buffer.from(chunk));
  }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { fail('INVALID_JSON'); }
}
/** Handles only new commands. The existing CLI, key handling and installer remain unchanged. */
export async function featureMain(argv = process.argv.slice(2), env = process.env) {
  const peek = parseArgs({ args: argv, allowPositionals: true, strict: false, options: { home: { type: 'string' } } });
  const command = peek.positionals[0];
  if (!['router', 'bulk', 'route', 'filter', 'policy', 'evaluate', 'status'].includes(command)) return false;
  if (Number(process.versions.node.split('.')[0]) < 22) fail('NODE_22_REQUIRED');
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, strict: true,
    options: { home: { type: 'string' }, help: { type: 'boolean' }, host: { type: 'string' }, 'dry-run': { type: 'boolean' }, replace: { type: 'boolean' } } });
  if (values.help) { process.stdout.write(FEATURE_HELP); return true; }
  const sub = positionals[1];
  if (positionals.length > (['router', 'bulk', 'policy'].includes(command) ? 2 : 1)) fail('UNEXPECTED_ARGUMENTS');
  const rolesOnly = command === 'policy' && sub === 'roles';
  if (!rolesOnly && (values.host !== undefined || values['dry-run'] !== undefined || values.replace !== undefined)) fail('UNEXPECTED_OPTION');
  const home = resolveHome({ ...env, ...(values.home ? { POINTSMAN_HOME: values.home } : {}) });
  const engine = createDecisionEngine({ home, env });
  const layer = createControlLayer({ home, env, engine });
  try {
  if (command === 'router' || command === 'bulk') {
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
  } else if (command === 'evaluate') print(evaluatePairedRuns(await jsonStdin(1048576)));
  return true;
  } finally { engine.close(); }
}
export const FEATURE_HELP = `\nClassifier-inspired features (explicit Jev or local Laya provider):\n  policy init|check   Create/validate private features.json; no automatic targets\n  policy roles --host codex|claude [--dry-run] [--replace]\n                      Preset router.profiles[host] from roles the host actually has (~/.codex/agents/*.toml, ~/.claude/agents/*.md); refuses to overwrite an existing profile unless --replace\n  router off|shadow|on  Cap routing independently of the global switch\n  bulk off|shadow|on    Cap prefiltering independently of the global switch\n  route|filter       Read bounded JSON from stdin; results contain no raw text\n  evaluate           Offline paired-execution report from JSON stdin\n`;
