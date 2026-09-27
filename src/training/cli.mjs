import { parseArgs } from 'node:util';
import path from 'node:path';
import readline from 'node:readline';
import { resolveHome } from '../storage.mjs';
import { fail } from '../constants.mjs';
import { createTrainingStore } from './store.mjs';
import { recordHost } from './host.mjs';
import { evaluateStore } from './evaluate.mjs';
import { buildDataset, exportDataset, validateDatasetSource, datasetStats } from './dataset.mjs';
import { allowRunner, removeRunner, listRunners, verifyRunner } from './runner.mjs';
import { correctDecision } from './correct.mjs';
import { selectProvider } from '../inference.mjs';
import { createDecisionEngine } from '../engine.mjs';
import { registerCheckpoint, activateCandidate, freezeHoldout, listHoldouts, qualifyCandidate, compareCandidate,
  promoteCandidate, rollbackLaya, layaStatus, adoptCandidate } from './laya-lifecycle.mjs';
import { packageCandidate, pullCandidate } from './laya-package.mjs';
import { startLayaServer } from '../laya-server.mjs';
import { layaSocketPath, createLayaSocketClient } from '../inference.mjs';
import { distillImport, distillImportReference, distillImportShadow, distillLabel, distillReview, buildDistillDataset, distillCompareTeacher, distillStatus } from './laya-distill.mjs';

export const TRAINING_COMMANDS = ['training', 'dataset', 'provider', 'compare', 'runner', 'laya'];
export const TRAINING_HELP = `\nProvider and offline dataset commands:\n  provider status|jev|laya       Select an explicitly configured provider; no download\n  training capture status|on|off  Content capture is OFF by default and separate from telemetry\n  training min-labels [N]        Minimum strong labels per purpose for dataset build; changing it requires a TTY\n  training outcome|host          Read minimal evidence/baseline JSON from stdin; outcome is always weak host_review\n  training correct --decision ID Interactive TTY-only human correction; never accepted from a pipe\n  training evaluate              Append derived evaluations without changing raw evidence\n  runner allow --name N --timeout-ms MS [--cwd DIR] [--purpose P --question Q --pass-label V --fail-label V] [--replace] -- ARGV...\n                                  TTY-only pre-registration; an agent later selects only the name\n  runner list|remove --name N    Show or remove a registered pre-approved check\n  runner verify --decision ID --check N  Run the pre-registered check and record its own source:'runner' outcome\n  dataset stats|validate|build [--allow-small]\n  dataset export --version HASH [--format laya|canonical]\n  compare --live                 Explicitly authorize Jev/Laya comparison; only one active arm\n  laya register --checkpoint DIR [--python ABS] [--model NAME] [--device cpu|mps|cuda] [--precision fp32|fp16] [--input-fit lossless|task-head]  Copy and fingerprint a prepared local checkpoint\n  laya activate --candidate HASH   Install an unqualified candidate for shadow/data collection (never applied in ON; rollback undoes)\n  laya holdout freeze --dataset HASH [--name ID]  Freeze the dataset's test split as an immutable regression holdout\n  laya holdout list              List holdout metadata only (no sample content)\n  laya qualify --candidate HASH --dataset HASH --holdout ID [--target-accuracy N] [--min-coverage N] [--min-calibration N] [--min-test N] [--min-lower-bound N] [--min-threshold N]\n                [--route-max-error N] [--route-max-error-upper N] [--route-min-applied N]\n                                  Calibrate a conservative acceptance threshold and check it on test+holdout;\n                                  also fits a decision-level route gate (tierCoverage/maxHostProbability) independently\n  laya compare --candidate HASH --holdout ID [--no-active-baseline]  Shadow-compare the candidate against the active checkpoint\n  laya promote --candidate HASH --holdout ID [--max-regression N]  Explicit operator promotion; requires qualification + non-regressing comparison\n  laya rollback                  Restore the laya block active immediately before the last promote/rollback/adopt\n  laya status                    Active checkpoint, candidates, qualification state, published records, holdout metadata\n  laya package --candidate HASH --out DIR [--repo OWNER/NAME]  Build a local folder ready for \`hf upload\`; never uploads\n  laya pull --repo OWNER/NAME --revision SHA40 [--python ABS] [--device cpu|mps|cuda] [--dry-run]\n                                  Fetch a pinned published manifest+files from Hugging Face, verify sha256, then register (never applied until 'laya adopt')\n  laya adopt --candidate HASH    Activate a pulled candidate using its published qualification (no local holdout to run qualify/compare)\n  laya serve [--home DIR] [--no-preload]  Resident server on a local Unix socket (POINTSMAN_HOME/run/laya.sock); idle-unloads the worker\n  laya server-status             Query the resident server over the socket (content-free)\n  laya distill import --run R --input FILE.jsonl   Import synthetic {lang,domain?,task} lines (egress:'allowed')\n  laya distill import-shadow --run R               Import captured route task text from training capture (egress:'forbidden', never sent out)\n  laya distill import-reference --run R --input FILE.jsonl --source MODEL [--role eval|train]\n                                  Owner decision (2026-09-23): import AI-reference (e.g. Claude) labels {lang,task,labels}; recorded label_source:'ai_reference', never 'human'.\n                                  role:'eval' (default) grounds calibration/test like human review; role:'train' grounds train, replacing that task's Jev-teacher sample. Unknown tasks fail the whole import first.\n  laya distill label --run R --confirm-egress [--limit N]  Explicit remote teacher (Jev/TypeSafe) calls; synthetic-only, rate-limited, no auto-retry\n  laya distill review --run R [--count 200]        TTY-only human correction of teacher labels; language-stratified, resumable\n  laya distill build --run R                       Build a standard datasets/<version>: precedence human(eval) > reference eval > reference train > Jev teacher(train)\n  laya distill compare-teacher --run R             Numbers-only Jev-teacher vs reference-label agreement report (never 'accuracy'), eval_reference and train_reference reported separately\n  laya distill status --run R                      Counts only (no task content)\nNo command trains, promotes a model, reads keys into output, or uploads a dataset.\n`;
const output = x => process.stdout.write(JSON.stringify(x, null, 2) + '\n');
async function stdin() {
  if (process.stdin.isTTY) fail('PIPE_JSON_TO_STDIN');
  const chunks = []; let bytes = 0;
  for await (const b of process.stdin) { bytes += b.length; if (bytes > 49152) fail('INPUT_TOO_LARGE'); chunks.push(b); }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { fail('INVALID_JSON'); }
}
async function confirmPrompt(text) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  try { return await new Promise(resolve => rl.question(text, resolve)); } finally { rl.close(); }
}
function splitDoubleDash(argv) {
  const at = argv.indexOf('--');
  return at === -1 ? { head: argv, rest: null } : { head: argv.slice(0, at), rest: argv.slice(at + 1) };
}
function parseFlags(head, allowed, boolFlags = ['replace']) {
  const out = {};
  for (let i = 0; i < head.length; i++) {
    const token = head[i];
    if (!token.startsWith('--')) fail('INVALID_RUNNER_ARGS');
    const name = token.slice(2);
    if (boolFlags.includes(name)) { out[name] = true; continue; }
    if (!allowed.includes(name)) fail('UNEXPECTED_OPTION');
    const value = head[++i];
    if (value === undefined) fail('INVALID_RUNNER_ARGS');
    out[name] = value;
  }
  return out;
}
function coerceLabel(value) {
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (/^-?\d+$/.test(value)) return Number(value);
  return value;
}
function runnerHome(flags, env) { return resolveHome({ ...env, ...(flags.home ? { POINTSMAN_HOME: flags.home } : {}) }); }
async function runnerMain(argv, env) {
  const sub = argv[0];
  const tty = { isStdinTTY: Boolean(process.stdin.isTTY), isStdoutTTY: Boolean(process.stdout.isTTY), prompt: confirmPrompt };
  if (sub === 'list') {
    const { head, rest } = splitDoubleDash(argv.slice(1)); if (rest !== null) fail('UNEXPECTED_OPTION');
    const flags = parseFlags(head, ['home']);
    output(listRunners(runnerHome(flags, env))); return;
  }
  if (sub === 'allow') {
    const { head, rest } = splitDoubleDash(argv.slice(1));
    if (!rest || !rest.length) fail('RUNNER_ARGV_REQUIRED');
    const flags = parseFlags(head, ['home', 'name', 'timeout-ms', 'cwd', 'purpose', 'question', 'pass-label', 'fail-label']);
    if (!flags.name) fail('RUNNER_NAME_REQUIRED');
    if (!flags['timeout-ms']) fail('RUNNER_TIMEOUT_REQUIRED');
    const check = { argv: rest, timeoutMs: Number(flags['timeout-ms']) };
    if (flags.cwd !== undefined) check.cwd = flags.cwd;
    if (flags.purpose !== undefined) check.purpose = flags.purpose;
    if (flags.question !== undefined) check.question_id = flags.question;
    if (flags['pass-label'] !== undefined) check.pass_label = coerceLabel(flags['pass-label']);
    if (flags['fail-label'] !== undefined) check.fail_label = coerceLabel(flags['fail-label']);
    output(await allowRunner({ home: runnerHome(flags, env), name: flags.name, check, replace: Boolean(flags.replace),
      ...tty, write: t => process.stdout.write(t) })); return;
  }
  if (sub === 'remove') {
    const { head, rest } = splitDoubleDash(argv.slice(1)); if (rest !== null) fail('UNEXPECTED_OPTION');
    const flags = parseFlags(head, ['home', 'name']);
    if (!flags.name) fail('RUNNER_NAME_REQUIRED');
    output(await removeRunner({ home: runnerHome(flags, env), name: flags.name, ...tty })); return;
  }
  if (sub === 'verify') {
    const { head, rest } = splitDoubleDash(argv.slice(1)); if (rest !== null) fail('UNEXPECTED_OPTION');
    const flags = parseFlags(head, ['home', 'decision', 'check']);
    if (!flags.decision || !flags.check) fail('RUNNER_VERIFY_ARGS_REQUIRED');
    const home = runnerHome(flags, env);
    const store = createTrainingStore({ home });
    output(await verifyRunner({ store, home, decisionId: flags.decision, checkName: flags.check })); return;
  }
  fail('INVALID_TRAINING_COMMAND');
}
function layaHome(flags, env) { return resolveHome({ ...env, ...(flags.home ? { POINTSMAN_HOME: flags.home } : {}) }); }
async function layaMain(argv, env) {
  const sub = argv[0];
  if (sub === 'activate') {
    const flags = parseFlags(argv.slice(1), ['home', 'candidate'], []);
    if (!flags.candidate) fail('LAYA_CANDIDATE_REQUIRED');
    output(activateCandidate(layaHome(flags, env), { candidateHash: flags.candidate })); return;
  }
  if (sub === 'register') {
    const flags = parseFlags(argv.slice(1), ['home', 'checkpoint', 'model', 'device', 'precision', 'python', 'input-fit'], []);
    if (!flags.checkpoint) fail('LAYA_CHECKPOINT_PATH_REQUIRED');
    // --python bootstraps the first registration, before providers.json has a laya block to inherit it from.
    output(registerCheckpoint(layaHome(flags, env), { checkpointDir: flags.checkpoint, model: flags.model, device: flags.device, precision: flags.precision, python: flags.python, inputFit: flags['input-fit'] })); return;
  }
  if (sub === 'holdout') {
    const action = argv[1];
    if (action === 'freeze') {
      const flags = parseFlags(argv.slice(2), ['home', 'dataset', 'name'], []);
      if (!flags.dataset) fail('DATASET_VERSION_REQUIRED');
      output(freezeHoldout(layaHome(flags, env), { datasetVersion: flags.dataset, name: flags.name })); return;
    }
    if (action === 'list') { const flags = parseFlags(argv.slice(2), ['home'], []); output(listHoldouts(layaHome(flags, env))); return; }
    fail('INVALID_TRAINING_COMMAND');
  }
  if (sub === 'qualify') {
    const flags = parseFlags(argv.slice(1),
      ['home', 'candidate', 'dataset', 'holdout', 'target-accuracy', 'min-coverage', 'min-calibration', 'min-test', 'min-lower-bound', 'min-threshold',
        'route-max-error', 'route-max-error-upper', 'route-min-applied'], []);
    if (!flags.candidate || !flags.dataset || !flags.holdout) fail('LAYA_QUALIFY_ARGS_REQUIRED');
    const opts = { candidateHash: flags.candidate, datasetVersion: flags.dataset, holdoutName: flags.holdout };
    if (flags['target-accuracy'] !== undefined) opts.targetAccuracy = Number(flags['target-accuracy']);
    if (flags['min-coverage'] !== undefined) opts.minCoverage = Number(flags['min-coverage']);
    if (flags['min-calibration'] !== undefined) opts.minCalibration = Number(flags['min-calibration']);
    if (flags['min-test'] !== undefined) opts.minTest = Number(flags['min-test']);
    if (flags['min-lower-bound'] !== undefined) opts.minLowerBound = Number(flags['min-lower-bound']);
    if (flags['min-threshold'] !== undefined) opts.minThreshold = Number(flags['min-threshold']);
    if (flags['route-max-error'] !== undefined) opts.routeMaxError = Number(flags['route-max-error']);
    if (flags['route-max-error-upper'] !== undefined) opts.routeMaxErrorUpper = Number(flags['route-max-error-upper']);
    if (flags['route-min-applied'] !== undefined) opts.routeMinApplied = Number(flags['route-min-applied']);
    output(await qualifyCandidate(layaHome(flags, env), opts)); return;
  }
  if (sub === 'compare') {
    const flags = parseFlags(argv.slice(1), ['home', 'candidate', 'holdout'], ['no-active-baseline']);
    if (!flags.candidate || !flags.holdout) fail('LAYA_COMPARE_ARGS_REQUIRED');
    output(await compareCandidate(layaHome(flags, env), { candidateHash: flags.candidate, holdoutName: flags.holdout,
      noActiveBaseline: Boolean(flags['no-active-baseline']) })); return;
  }
  if (sub === 'promote') {
    const flags = parseFlags(argv.slice(1), ['home', 'candidate', 'holdout', 'max-regression'], []);
    if (!flags.candidate || !flags.holdout) fail('LAYA_PROMOTE_ARGS_REQUIRED');
    const opts = { candidateHash: flags.candidate, holdoutName: flags.holdout };
    if (flags['max-regression'] !== undefined) opts.maxRegression = Number(flags['max-regression']);
    output(promoteCandidate(layaHome(flags, env), opts)); return;
  }
  if (sub === 'rollback') { const flags = parseFlags(argv.slice(1), ['home'], []); output(rollbackLaya(layaHome(flags, env))); return; }
  if (sub === 'status') { const flags = parseFlags(argv.slice(1), ['home'], []); output(layaStatus(layaHome(flags, env))); return; }
  if (sub === 'adopt') {
    const flags = parseFlags(argv.slice(1), ['home', 'candidate'], []);
    if (!flags.candidate) fail('LAYA_CANDIDATE_REQUIRED');
    output(adoptCandidate(layaHome(flags, env), { candidateHash: flags.candidate })); return;
  }
  if (sub === 'package') {
    const flags = parseFlags(argv.slice(1), ['home', 'candidate', 'out', 'repo'], []);
    if (!flags.candidate) fail('LAYA_CANDIDATE_REQUIRED');
    if (!flags.out) fail('LAYA_PACKAGE_OUT_REQUIRED');
    output(packageCandidate(layaHome(flags, env), { candidateHash: flags.candidate, outDir: path.resolve(flags.out), repo: flags.repo })); return;
  }
  if (sub === 'pull') {
    const flags = parseFlags(argv.slice(1), ['home', 'repo', 'revision', 'python', 'device'], ['dry-run']);
    if (!flags.repo || !flags.revision) fail('LAYA_PULL_ARGS_REQUIRED');
    output(await pullCandidate(layaHome(flags, env), { repo: flags.repo, revision: flags.revision,
      python: flags.python, device: flags.device, dryRun: Boolean(flags['dry-run']), env })); return;
  }
  if (sub === 'distill') {
    const action = argv[1];
    if (action === 'import') {
      const flags = parseFlags(argv.slice(2), ['home', 'run', 'input'], []);
      if (!flags.run || !flags.input) fail('DISTILL_IMPORT_ARGS_REQUIRED');
      output(distillImport(layaHome(flags, env), { run: flags.run, inputFile: flags.input })); return;
    }
    if (action === 'import-shadow') {
      const flags = parseFlags(argv.slice(2), ['home', 'run'], []);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      output(distillImportShadow(layaHome(flags, env), { run: flags.run })); return;
    }
    if (action === 'import-reference') {
      const flags = parseFlags(argv.slice(2), ['home', 'run', 'input', 'source', 'role'], []);
      if (!flags.run || !flags.input || !flags.source) fail('DISTILL_IMPORT_REFERENCE_ARGS_REQUIRED');
      const opts = { run: flags.run, inputFile: flags.input, source: flags.source };
      if (flags.role !== undefined) opts.role = flags.role;
      output(distillImportReference(layaHome(flags, env), opts)); return;
    }
    if (action === 'label') {
      const flags = parseFlags(argv.slice(2), ['home', 'run', 'limit'], ['confirm-egress']);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      const opts = { run: flags.run, confirmEgress: Boolean(flags['confirm-egress']), env };
      if (flags.limit !== undefined) opts.limit = Number(flags.limit);
      output(await distillLabel(layaHome(flags, env), opts)); return;
    }
    if (action === 'review') {
      const flags = parseFlags(argv.slice(2), ['home', 'run', 'count'], []);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      const opts = { run: flags.run };
      if (flags.count !== undefined) opts.count = Number(flags.count);
      output(await distillReview(layaHome(flags, env), { ...opts,
        isStdinTTY: Boolean(process.stdin.isTTY), isStdoutTTY: Boolean(process.stdout.isTTY),
        prompt: confirmPrompt, write: t => process.stdout.write(t) })); return;
    }
    if (action === 'build') {
      const flags = parseFlags(argv.slice(2), ['home', 'run'], []);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      output(buildDistillDataset(layaHome(flags, env), { run: flags.run })); return;
    }
    if (action === 'compare-teacher') {
      const flags = parseFlags(argv.slice(2), ['home', 'run'], []);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      output(distillCompareTeacher(layaHome(flags, env), { run: flags.run })); return;
    }
    if (action === 'status') {
      const flags = parseFlags(argv.slice(2), ['home', 'run'], []);
      if (!flags.run) fail('DISTILL_RUN_REQUIRED');
      output(distillStatus(layaHome(flags, env), { run: flags.run })); return;
    }
    fail('INVALID_TRAINING_COMMAND');
  }
  if (sub === 'serve') {
    const flags = parseFlags(argv.slice(1), ['home'], ['no-preload']);
    const home = layaHome(flags, env);
    const server = await startLayaServer({ home, env, preload: !flags['no-preload'] });
    const shutdown = () => { server.close().finally(() => process.exit(0)); };
    process.once('SIGINT', shutdown);
    process.once('SIGTERM', shutdown);
    return;
  }
  if (sub === 'server-status') {
    const flags = parseFlags(argv.slice(1), ['home'], []);
    const home = layaHome(flags, env);
    const client = createLayaSocketClient({ socketPath: layaSocketPath(home) });
    try { output({ running: true, ...await client.status({ timeoutMs: 2000 }) }); }
    catch (e) { output({ running: false, reason: e?.code ?? 'LAYA_SERVER_UNAVAILABLE' }); }
    return;
  }
  fail('INVALID_TRAINING_COMMAND');
}
export async function trainingMain(argv, env = process.env) {
  if (argv[0] === 'runner') { await runnerMain(argv.slice(1), env); return; }
  if (argv[0] === 'laya') { await layaMain(argv.slice(1), env); return; }
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, strict: true, options: {
    home: { type: 'string' }, version: { type: 'string' }, format: { type: 'string' }, live: { type: 'boolean' }, help: { type: 'boolean' },
    decision: { type: 'string' }, 'allow-small': { type: 'boolean' },
  } });
  const [command, sub, action] = positionals;
  if (!TRAINING_COMMANDS.includes(command) || positionals.length > 3) fail('INVALID_TRAINING_COMMAND');
  if (values.help) { process.stdout.write(TRAINING_HELP); return; }
  const allowed = command === 'dataset' && sub === 'export' ? ['home', 'version', 'format']
    : command === 'dataset' && sub === 'build' ? ['home', 'allow-small']
    : command === 'compare' ? ['home', 'live']
    : command === 'training' && sub === 'correct' ? ['home', 'decision']
    : ['home'];
  if (Object.keys(values).some(k => !allowed.includes(k))) fail('UNEXPECTED_OPTION');
  const home = resolveHome({ ...env, ...(values.home ? { POINTSMAN_HOME: values.home } : {}) });
  const store = createTrainingStore({ home });
  if (command === 'training') {
    if (sub === 'capture') {
      if (!['on', 'off', 'status'].includes(action)) fail('INVALID_TRAINING_COMMAND');
      output(action === 'status' ? store.status() : store.setCapture(action === 'on')); return;
    }
    if (sub === 'min-labels') {
      if (action === undefined) { output(store.status()); return; }
      // Lowering the dataset gate is an operator decision; a piped agent may read it but not change it.
      if (!process.stdin.isTTY || !process.stdout.isTTY) fail('HUMAN_TTY_REQUIRED');
      output(store.setMinLabels(Number(action))); return;
    }
    if (sub === 'correct') {
      if (action !== undefined) fail('INVALID_TRAINING_COMMAND');
      if (!values.decision) fail('DECISION_ID_REQUIRED');
      output(await correctDecision({ store, home, decisionId: values.decision,
        isStdinTTY: Boolean(process.stdin.isTTY), isStdoutTTY: Boolean(process.stdout.isTTY),
        prompt: confirmPrompt, write: t => process.stdout.write(t) })); return;
    }
    if (action !== undefined) fail('INVALID_TRAINING_COMMAND');
    if (sub === 'outcome') { output(store.outcome(await stdin(), { trust: 'host' })); return; }
    if (sub === 'host') { output(recordHost(store, await stdin())); return; }
    if (sub === 'evaluate') { output(evaluateStore(store)); return; }
  }
  if (command === 'dataset' && !action) {
    if (sub === 'stats') { output(datasetStats(store)); return; }
    if (sub === 'validate') { const r = validateDatasetSource(store); output(r); if (!r.ok) process.exitCode = 2; return; }
    if (sub === 'build') { output(buildDataset(store, { allowSmall: Boolean(values['allow-small']) })); return; }
    if (sub === 'export') { if (!values.version) fail('DATASET_VERSION_REQUIRED'); output(exportDataset(store, values.version, values.format ?? 'laya')); return; }
  }
  if (command === 'provider' && !action) {
    if (sub === 'status') { const e = createDecisionEngine({ home, env }); try { output(e.status()); } finally { e.close(); } return; }
    if (['jev', 'laya'].includes(sub)) { output(selectProvider(home, sub)); return; }
  }
  if (command === 'compare' && !sub && !action) {
    if (!values.live) fail('EXPLICIT_REMOTE_COMPARISON_CONSENT_REQUIRED');
    const engine = createDecisionEngine({ home, env });
    try { output(await engine.compare(await stdin(), { remoteConsent: true })); } finally { engine.close(); }
    return;
  }
  fail('INVALID_TRAINING_COMMAND');
}
