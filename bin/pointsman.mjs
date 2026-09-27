#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { main } from '../src/cli.mjs';
import { featureMain, FEATURE_HELP } from '../src/features-cli.mjs';
import { trainingMain, TRAINING_COMMANDS, TRAINING_HELP } from '../src/training/cli.mjs';
import { errorCode, fail } from '../src/constants.mjs';
import { runHookCli } from '../src/hooks.mjs';
import { runMigrateCli } from '../src/migrate.mjs';
// Host hooks must never exit non-zero: Claude treats exit 2 from PreToolUse as a block of the tool call.
// `hook` is therefore dispatched before any strict parsing, and every failure ends in exit 0 with no output.
if (process.argv[2] === 'hook') {
  try {
    const { values } = parseArgs({ args: process.argv.slice(3), strict: false, options: { home: { type: 'string' }, host: { type: 'string' }, event: { type: 'string' } } });
    const pick = v => (typeof v === 'string' ? v : undefined);
    await runHookCli({ host: pick(values.host), event: pick(values.event), home: pick(values.home) });
  } catch { /* fail-open */ }
  process.exitCode = 0;
} else try {
  if (Number(process.versions.node.split('.')[0]) < 22) fail('NODE_22_REQUIRED');
  const args = process.argv.slice(2);
  const peek = parseArgs({ args, allowPositionals: true, strict: false, options: { home: { type: 'string' } } });
  const command = peek.positionals[0];
  if (command === 'migrate') await runMigrateCli(args.slice(1), process.env);
  else if (TRAINING_COMMANDS.includes(command)) await trainingMain(args);
  else if (!(await featureMain(args))) {
    await main(args);
    if (!command || command === 'help' || args.includes('--help')) process.stdout.write(FEATURE_HELP + TRAINING_HELP);
  }
} catch (error) {
  process.stderr.write(JSON.stringify({ ok: false, error: errorCode(error) }) + '\n');
  process.exitCode = 2;
}
