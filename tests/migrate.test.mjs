import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { migrateHome, planMigration } from '../src/migrate.mjs';
import { tempHome } from './helpers.mjs';

function makeOldHome(t) {
  const root = tempHome(t);
  const oldHome = path.join(root, 'old-home');
  fs.mkdirSync(oldHome, { mode: 0o700 });
  fs.mkdirSync(path.join(oldHome, 'laya', 'candidates'), { recursive: true, mode: 0o700 });
  fs.mkdirSync(path.join(oldHome, 'installations'), { mode: 0o700 });
  const write = (rel, text, mode = 0o600) => { fs.writeFileSync(path.join(oldHome, rel), text, { mode }); };
  const checkpoint = 'a'.repeat(64);
  write('providers.json', JSON.stringify({ version: 1, provider: 'jev', laya: {
    python: path.join(oldHome, 'laya', 'python', 'bin', 'python3'),
    modelPath: path.join(oldHome, 'laya', 'checkpoints', checkpoint),
    model: 'laya/my-checkpoint', checkpoint, runtimeVersion: '0.3.4', device: 'cpu',
  } }, null, 2) + '\n');
  write('config.json', JSON.stringify({ version: 1, mode: 'off', model: 'jev-latest' }, null, 2) + '\n');
  write(path.join('laya', 'history.jsonl'),
    JSON.stringify({ at: 'x', home: oldHome }) + '\n', 0o600);
  write(path.join('laya', 'candidates', 'abc.json'),
    JSON.stringify({ checkpoint: 'abc', path: path.join(oldHome, 'laya', 'checkpoints', 'abc') }, null, 2) + '\n');
  write(path.join('installations', 'x.json'),
    JSON.stringify({ owner: 'pointsman', path: 'k', value: { cwd: oldHome } }, null, 2) + '\n');
  // A file with no reference to the old home path at all: must be left untouched byte-for-byte.
  write('features.json', JSON.stringify({ version: 1 }, null, 2) + '\n');
  return { root, oldHome };
}

test('migrate: happy path rewrites paths, keeps modes, leaves providers.json valid', async t => {
  const { root, oldHome } = makeOldHome(t);
  const newHome = path.join(root, 'new-home');
  const before = fs.statSync(path.join(oldHome, 'providers.json')).mode & 0o777;
  const result = await migrateHome({ from: oldHome, to: newHome });
  assert.equal(result.ok, true);
  assert.equal(result.dryRun, false);
  assert.equal(fs.existsSync(oldHome), false);
  assert.equal(fs.existsSync(newHome), true);
  assert.equal(result.providersValidated, true);
  assert.ok(result.rewrite.files.includes('providers.json'));
  assert.ok(result.rewrite.files.includes(path.join('laya', 'history.jsonl')));
  assert.ok(result.rewrite.files.includes(path.join('laya', 'candidates', 'abc.json')));
  assert.ok(result.rewrite.files.includes(path.join('installations', 'x.json')));
  assert.ok(!result.rewrite.files.includes('features.json'), 'a file with no old-path reference is not rewritten');

  const providers = JSON.parse(fs.readFileSync(path.join(newHome, 'providers.json'), 'utf8'));
  assert.equal(providers.provider, 'jev');
  assert.equal(providers.laya.python, path.join(newHome, 'laya', 'python', 'bin', 'python3'));
  assert.equal(providers.laya.modelPath, path.join(newHome, 'laya', 'checkpoints', 'a'.repeat(64)));
  const history = JSON.parse(fs.readFileSync(path.join(newHome, 'laya', 'history.jsonl'), 'utf8').trim());
  assert.equal(history.home, newHome);
  const candidate = JSON.parse(fs.readFileSync(path.join(newHome, 'laya', 'candidates', 'abc.json'), 'utf8'));
  assert.equal(candidate.path, path.join(newHome, 'laya', 'checkpoints', 'abc'));
  const installation = JSON.parse(fs.readFileSync(path.join(newHome, 'installations', 'x.json'), 'utf8'));
  assert.equal(installation.value.cwd, newHome);

  const after = fs.statSync(path.join(newHome, 'providers.json')).mode & 0o777;
  assert.equal(after, before);
  const featuresText = fs.readFileSync(path.join(newHome, 'features.json'), 'utf8');
  assert.equal(featuresText, JSON.stringify({ version: 1 }, null, 2) + '\n');
});

test('migrate: --dry-run changes nothing', async t => {
  const { root, oldHome } = makeOldHome(t);
  const newHome = path.join(root, 'new-home');
  const result = await migrateHome({ from: oldHome, to: newHome, dryRun: true });
  assert.equal(result.ok, true);
  assert.equal(result.dryRun, true);
  assert.equal(fs.existsSync(oldHome), true, 'old home is untouched');
  assert.equal(fs.existsSync(newHome), false, 'target is never created by a dry run');
  assert.ok(result.rewrite.count >= 4);
  assert.ok(result.rewrite.files.includes('providers.json'));
  const originalProviders = fs.readFileSync(path.join(oldHome, 'providers.json'), 'utf8');
  assert.ok(originalProviders.includes(oldHome), 'dry run never rewrites the source file');
});

test('migrate: refuses when the target already exists and is non-empty', async t => {
  const { root, oldHome } = makeOldHome(t);
  const newHome = path.join(root, 'new-home');
  fs.mkdirSync(newHome, { mode: 0o700 });
  fs.writeFileSync(path.join(newHome, 'marker.json'), '{}');
  await assert.rejects(migrateHome({ from: oldHome, to: newHome }), /MIGRATE_TARGET_EXISTS/);
  assert.equal(fs.existsSync(oldHome), true);
});

test('migrate: an empty existing target directory is accepted', async t => {
  const { root, oldHome } = makeOldHome(t);
  const newHome = path.join(root, 'new-home');
  fs.mkdirSync(newHome, { mode: 0o700 });
  const result = await migrateHome({ from: oldHome, to: newHome });
  assert.equal(result.ok, true);
  assert.equal(fs.existsSync(path.join(newHome, 'providers.json')), true);
});

test('migrate: refuses when the old home\'s Laya socket is live', async t => {
  // A Unix socket path is capped (~104 bytes on macOS/BSD), so this uses its own short-prefixed
  // temp dir instead of makeOldHome()'s longer nested layout.
  const oldHome = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pm-mig-')));
  t.after(() => fs.rmSync(oldHome, { recursive: true, force: true }));
  fs.mkdirSync(path.join(oldHome, 'run'), { mode: 0o700 });
  const sock = path.join(oldHome, 'run', 'laya.sock');
  const server = net.createServer(socket => { socket.on('data', () => socket.end('{}\n')); });
  await new Promise((resolve, reject) => { server.listen(sock, resolve); server.on('error', reject); });
  t.after(() => new Promise(resolve => server.close(resolve)));
  await assert.rejects(migrateHome({ from: oldHome, to: path.join(oldHome, '..', 'pm-mig-new') }), /MIGRATE_OLD_SERVER_LIVE/);
  assert.equal(fs.existsSync(oldHome), true);
});

test('migrate: refuses a symlinked --from', async t => {
  const { root, oldHome } = makeOldHome(t);
  const link = path.join(root, 'old-home-link');
  fs.symlinkSync(oldHome, link);
  assert.throws(() => planMigration({ from: link, to: path.join(root, 'new-home') }), /MIGRATE_FROM_SYMLINK/);
});

test('migrate: refuses a relative --from', async t => {
  const { root } = makeOldHome(t);
  assert.throws(() => planMigration({ from: 'relative/path', to: path.join(root, 'new-home') }), /MIGRATE_FROM_NOT_ABSOLUTE/);
});

test('migrate: refuses when --from does not exist', async t => {
  const { root } = makeOldHome(t);
  assert.throws(() => planMigration({ from: path.join(root, 'does-not-exist'), to: path.join(root, 'new-home') }), /MIGRATE_FROM_NOT_FOUND/);
});
