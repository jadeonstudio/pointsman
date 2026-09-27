// `pointsman migrate --from <oldHome> [--dry-run]`: moves an existing installation's data home to
// the current POINTSMAN_HOME and rewrites the old home's absolute path prefix inside the small set of
// JSON/JSONL records that store it. This command exists ONLY to help users (including the owner)
// coming from the jev-agent-control name migrate their existing local data home; it is the one place
// that name is expected to still appear in code.
import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { execFileSync } from 'node:child_process';
import { fail } from './constants.mjs';
import { readText, atomicWrite, resolveHome } from './storage.mjs';
import { layaSocketPath, createLayaSocketClient, validateProviderConfig } from './inference.mjs';

const OLD_REPOSITORY_URL = 'https://github.com/jadeonstudio/jev-agent-control.git';
const NEW_REPOSITORY_URL = 'https://github.com/jadeonstudio/pointsman.git';
const SOCKET_PROBE_TIMEOUT_MS = 200;
// Matches reality checked against a real installed home (2026-09-27): top-level JSON records, the
// laya per-checkpoint/per-candidate/per-holdout records, the append-only history log and the
// installer's own ownership records. Deliberately excludes training/ (raw evidence) and repository/
// (a git clone, handled separately below).
const REWRITE_TOP_FILES = ['providers.json', 'config.json', 'features.json', 'training.json'];
const REWRITE_SINGLE_FILES = ['laya/history.jsonl'];
const REWRITE_DIRS = [
  { dir: 'laya/candidates', suffix: '.json' },
  { dir: 'laya/qualifications', suffix: '.json' },
  { dir: 'laya/holdouts', suffix: '.json' },
  { dir: 'installations', suffix: '.json' },
];

function deviceOf(target) {
  let current = path.resolve(target);
  for (;;) {
    try { return fs.statSync(current).dev; } catch (e) { if (e.code !== 'ENOENT') throw e; }
    const parent = path.dirname(current);
    if (parent === current) fail('MIGRATE_TARGET_UNRESOLVABLE');
    current = parent;
  }
}
async function socketIsLive(sock) {
  let st;
  try { st = fs.lstatSync(sock); } catch { return false; }
  if (!st.isSocket()) return false;
  try { await createLayaSocketClient({ socketPath: sock }).status({ timeoutMs: SOCKET_PROBE_TIMEOUT_MS }); return true; }
  catch { return false; }
}
/** Read-only: resolves and validates `--from`/target and lists the relative paths that would be
 * rewritten. Throws a clear ControlError code and changes nothing. */
export function planMigration({ from, to }) {
  if (typeof from !== 'string' || !from) fail('MIGRATE_FROM_REQUIRED');
  if (!path.isAbsolute(from)) fail('MIGRATE_FROM_NOT_ABSOLUTE');
  const fromResolved = path.resolve(from);
  let fromStat;
  try { fromStat = fs.lstatSync(fromResolved); } catch (e) { if (e.code === 'ENOENT') fail('MIGRATE_FROM_NOT_FOUND'); throw e; }
  if (fromStat.isSymbolicLink()) fail('MIGRATE_FROM_SYMLINK');
  if (!fromStat.isDirectory()) fail('MIGRATE_FROM_NOT_DIRECTORY');
  if (typeof process.getuid === 'function' && fromStat.uid !== process.getuid()) fail('MIGRATE_FROM_WRONG_OWNER');
  const toResolved = path.resolve(to);
  if (fromResolved === toResolved) fail('MIGRATE_SAME_PATH');
  let targetStat;
  try { targetStat = fs.lstatSync(toResolved); } catch (e) { if (e.code !== 'ENOENT') throw e; targetStat = null; }
  if (targetStat) {
    if (targetStat.isSymbolicLink() || !targetStat.isDirectory() || fs.readdirSync(toResolved).length) fail('MIGRATE_TARGET_EXISTS');
  }
  if (deviceOf(fromResolved) !== deviceOf(toResolved)) fail('MIGRATE_CROSS_FILESYSTEM');
  return { from: fromResolved, to: toResolved };
}
/** Lists rewrite-candidate files relative to `root` (which may be the old or the new home; only
 * existence is checked, never content, so this is read-only either way). */
function listRewriteTargets(root) {
  const targets = [];
  for (const name of [...REWRITE_TOP_FILES, ...REWRITE_SINGLE_FILES]) {
    if (fs.existsSync(path.join(root, name))) targets.push(name);
  }
  for (const { dir, suffix } of REWRITE_DIRS) {
    let entries;
    try { entries = fs.readdirSync(path.join(root, dir)); } catch { continue; }
    for (const entry of entries.sort()) if (entry.endsWith(suffix)) targets.push(path.join(dir, entry));
  }
  return targets;
}
function repositoryOriginStatus(repositoryDir) {
  if (!fs.existsSync(path.join(repositoryDir, '.git'))) return null;
  try {
    const current = execFileSync('git', ['-C', repositoryDir, 'remote', 'get-url', 'origin'], { encoding: 'utf8' }).trim();
    return current === OLD_REPOSITORY_URL;
  } catch { return null; }
}
/** Moves `from` to `to` (same-filesystem rename only) and rewrites the old absolute path prefix
 * inside the bounded file set above. `dryRun` performs every precondition check and lists the
 * planned rewrite paths/counts but touches nothing. Returns a content-free JSON summary. */
export async function migrateHome({ from, to, dryRun = false }) {
  const plan = planMigration({ from, to });
  if (await socketIsLive(layaSocketPath(plan.from))) fail('MIGRATE_OLD_SERVER_LIVE');
  const relativeTargets = listRewriteTargets(plan.from);
  const repositoryDir = path.join(plan.from, 'repository');
  const originMatchesOld = repositoryOriginStatus(repositoryDir);
  if (dryRun) {
    return { ok: true, dryRun: true, from: plan.from, to: plan.to,
      rewrite: { count: relativeTargets.length, files: relativeTargets },
      repository: { present: fs.existsSync(repositoryDir), originUpdatePlanned: originMatchesOld === true } };
  }
  try { fs.renameSync(plan.from, plan.to); }
  catch (e) { if (e.code === 'EXDEV') fail('MIGRATE_CROSS_FILESYSTEM'); throw e; }
  const rewritten = [];
  for (const relative of relativeTargets) {
    const file = path.join(plan.to, relative);
    const mode = fs.statSync(file).mode & 0o777;
    const before = readText(file, { privateFile: true, maxBytes: 64 * 1024 * 1024 });
    if (!before.includes(plan.from)) continue;
    const after = before.split(plan.from).join(plan.to);
    atomicWrite(file, after, { expected: before, mode });
    rewritten.push(relative);
  }
  const newRepositoryDir = path.join(plan.to, 'repository');
  let originUpdated = null;
  if (fs.existsSync(path.join(newRepositoryDir, '.git'))) {
    originUpdated = repositoryOriginStatus(newRepositoryDir) === true;
    if (originUpdated) execFileSync('git', ['-C', newRepositoryDir, 'remote', 'set-url', 'origin', NEW_REPOSITORY_URL]);
  }
  // Validate the final providers.json (whether or not it needed a path rewrite) so a migrate never
  // silently leaves a broken/invalid provider config behind.
  let providersValidated = null;
  if (relativeTargets.includes('providers.json')) {
    validateProviderConfig(JSON.parse(readText(path.join(plan.to, 'providers.json'), { privateFile: true, maxBytes: 8192 })));
    providersValidated = true;
  }
  return { ok: true, dryRun: false, from: plan.from, to: plan.to,
    rewrite: { count: rewritten.length, files: rewritten }, providersValidated,
    repository: { present: fs.existsSync(newRepositoryDir), originUpdated } };
}
export async function runMigrateCli(argv, env = process.env) {
  const { values, positionals } = parseArgs({ args: argv, allowPositionals: true, strict: true, options: {
    from: { type: 'string' }, 'dry-run': { type: 'boolean' }, home: { type: 'string' },
  } });
  if (positionals.length) fail('UNEXPECTED_ARGUMENTS');
  if (!values.from) fail('MIGRATE_FROM_REQUIRED');
  const to = resolveHome({ ...env, ...(values.home ? { POINTSMAN_HOME: values.home } : {}) });
  const result = await migrateHome({ from: values.from, to, dryRun: Boolean(values['dry-run']) });
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
}
