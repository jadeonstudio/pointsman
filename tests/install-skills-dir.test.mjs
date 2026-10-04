import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { installationPlan, applyInstallation, REPO_ROOT } from '../src/installer.mjs';
import { atomicWrite, loadConfig } from '../src/storage.mjs';
import { tempHome } from './helpers.mjs';

function fixture(t) {
  const user = tempHome(t), home = path.join(user, 'state'), skillsDir = path.join(user, 'real skills');
  fs.mkdirSync(skillsDir);
  const env = { PATH: process.env.PATH, HOME: user, POINTSMAN_HOME: home, CODEX_HOME: path.join(user, '.codex') };
  const options = { home, env, project: user, target: 'claude', skillsDirectory: skillsDir };
  return { user, home, skillsDir, env, options,
    plan: changes => installationPlan({ ...options, ...changes }),
    install: changes => applyInstallation(installationPlan({ ...options, ...changes })),
    cli: (...args) => spawnSync(process.execPath, [path.join(REPO_ROOT, 'bin/pointsman.mjs'), ...args, '--home', home], { env, encoding: 'utf8', timeout: 5000 }),
  };
}
const ownedSkills = ['pointsman-control', 'pointsman-decisions'];

test('single-host CLI installs into a reviewed real skills directory and uninstall preserves foreign files', async t => {
  for (const target of ['codex', 'claude']) await t.test(target, t => {
    const f = fixture(t), defaultDir = path.join(f.user, target === 'codex' ? '.agents/skills' : '.claude/skills');
    const linked = path.join(f.user, 'linked-default'); fs.mkdirSync(linked);
    fs.mkdirSync(path.dirname(defaultDir), { recursive: true }); fs.symlinkSync(linked, defaultDir);
    const foreign = path.join(f.skillsDir, 'foreign/SKILL.md'); atomicWrite(foreign, '# Keep this foreign skill\n');
    const baseline = target === 'codex' ? 'model = "existing-model"\n' : '{"theme":"existing"}\n';
    const hostConfig = target === 'codex' ? path.join(f.env.CODEX_HOME, 'config.toml') : path.join(f.user, '.claude.json');
    atomicWrite(hostConfig, baseline);
    const args = ['--target', target, '--skills-dir', f.skillsDir];
    const dry = f.cli('install', ...args, '--dry-run'); assert.equal(dry.status, 0, dry.stderr);
    const review = JSON.parse(dry.stdout);
    for (const skill of ownedSkills) assert.ok(review.changes.some(c => c.path === path.join(f.skillsDir, skill, 'SKILL.md')));
    assert.equal(fs.readFileSync(hostConfig, 'utf8'), baseline); assert.deepEqual(fs.readdirSync(linked), []);
    assert.deepEqual(fs.readdirSync(f.skillsDir), ['foreign']); assert.equal(fs.existsSync(f.home), false);
    const installed = f.cli('install', ...args); assert.equal(installed.status, 0, installed.stderr);
    for (const skill of ownedSkills) assert.equal(fs.existsSync(path.join(f.skillsDir, skill, 'SKILL.md')), true);
    assert.equal(loadConfig(f.home, {}).mode, 'off'); assert.deepEqual(fs.readdirSync(linked), []);
    const second = f.cli('install', ...args); assert.equal(second.status, 0, second.stderr);
    assert.deepEqual(JSON.parse(second.stdout).changes, []);
    const removed = f.cli('uninstall', ...args); assert.equal(removed.status, 0, removed.stderr);
    for (const skill of ownedSkills) for (const name of ['SKILL.md', 'scripts/run.mjs', '.pointsman-managed.json']) {
      assert.equal(fs.existsSync(path.join(f.skillsDir, skill, name)), false);
    }
    assert.equal(fs.readFileSync(foreign, 'utf8'), '# Keep this foreign skill\n');
    if (target === 'codex') assert.equal(fs.readFileSync(hostConfig, 'utf8'), baseline);
    else assert.deepEqual(JSON.parse(fs.readFileSync(hostConfig, 'utf8')), JSON.parse(baseline));
    assert.deepEqual(fs.readdirSync(linked), []);
    assert.equal(fs.lstatSync(defaultDir).isSymbolicLink(), true);
  });
});

test('explicit skills directory refuses ambiguous or contradictory options before any installation writes', t => {
  const f = fixture(t), before = fs.readdirSync(f.user);
  for (const args of [
    ['install', '--target', 'claude', '--skills-dir', 'relative/skills'],
    ['install', '--target', 'both', '--skills-dir', f.skillsDir],
    ['install', '--target', 'claude', '--skills-dir', f.skillsDir, '--no-skills'],
    ['uninstall', '--target', 'claude', '--skills-dir', f.skillsDir, '--hooks-only'],
  ]) {
    const out = f.cli(...args); assert.notEqual(out.status, 0, out.stdout);
    assert.deepEqual(fs.readdirSync(f.user), before); assert.deepEqual(fs.readdirSync(f.skillsDir), []);
    assert.equal(fs.existsSync(f.home), false);
  }
});

test('default, explicit and ancestor symlink paths remain refused without writing through them', t => {
  const f = fixture(t), link = path.join(f.user, '.claude/skills');
  fs.mkdirSync(path.dirname(link), { recursive: true }); fs.symlinkSync(f.skillsDir, link);
  const foreign = path.join(f.skillsDir, 'foreign/SKILL.md'); atomicWrite(foreign, '# Preserved\n');
  const ancestor = path.join(f.user, 'linked-parent'); fs.symlinkSync(f.user, ancestor);
  for (const options of [{ skillsDirectory: undefined }, { skillsDirectory: link }, { skillsDirectory: path.join(ancestor, 'real skills') }]) {
    assert.throws(() => f.install(options), /SYMLINK/);
    assert.deepEqual(fs.readdirSync(f.skillsDir), ['foreign']); assert.equal(fs.readFileSync(foreign, 'utf8'), '# Preserved\n');
    assert.equal(fs.existsSync(f.home), false); assert.equal(fs.existsSync(path.join(f.user, '.claude.json')), false);
  }
});

test('alternate skills directory preserves marker ownership, edited files and atomic rollback', async t => {
  for (const collision of ['foreign-marker', 'edited-owned-file', 'unmarked-file']) await t.test(collision, t => {
    const f = fixture(t), skill = path.join(f.skillsDir, 'pointsman-decisions'), file = path.join(skill, 'SKILL.md');
    if (collision === 'edited-owned-file') { f.install(); fs.appendFileSync(file, '\nlocal edit\n'); }
    else {
      atomicWrite(file, '# Foreign file\n');
      if (collision === 'foreign-marker') atomicWrite(path.join(skill, '.pointsman-managed.json'), '{"owner":"someone-else","files":{}}\n');
    }
    const before = fs.readFileSync(file, 'utf8'), host = path.join(f.user, '.claude.json');
    const hostBefore = fs.existsSync(host) ? fs.readFileSync(host, 'utf8') : null;
    for (const remove of [false, true]) assert.throws(() => f.install({ remove }), /SKILL_COLLISION|SKILL_CHANGED_OR_COLLISION/);
    assert.equal(fs.readFileSync(file, 'utf8'), before);
    assert.equal(fs.existsSync(host) ? fs.readFileSync(host, 'utf8') : null, hostBefore);
  });
  await t.test('failed commit rolls back files in the alternate root', t => {
    const f = fixture(t), plan = f.plan(), original = fs.renameSync;
    const firstSkill = plan.actions.find(a => a.file.startsWith(f.skillsDir)); let failed = false;
    t.mock.method(fs, 'renameSync', (...args) => {
      if (!failed && args[1] === firstSkill.file) { failed = true; throw new Error('simulated alternate skill write failure'); }
      return original(...args);
    });
    assert.throws(() => applyInstallation(plan), /simulated alternate skill write failure/); assert.equal(failed, true);
    for (const action of plan.actions) assert.equal(fs.existsSync(action.file), false, action.file);
  });
});
