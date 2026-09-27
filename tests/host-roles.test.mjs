import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { discoverHostRoles } from '../src/host-roles.mjs';

function tempHome() {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-host-roles-')));
  return { home, cleanup: () => fs.rmSync(home, { recursive: true, force: true }) };
}

test('missing agents directory yields an empty role list, not an error', () => {
  const { home, cleanup } = tempHome();
  try {
    assert.deepEqual(discoverHostRoles('codex', { env: {}, userHome: home }), []);
    assert.deepEqual(discoverHostRoles('claude', { env: {}, userHome: home }), []);
  } finally { cleanup(); }
});

test('codex roles come from ~/.codex/agents/*.toml file names only; content is never read', () => {
  const { home, cleanup } = tempHome();
  try {
    const dir = path.join(home, '.codex', 'agents');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'implementer.toml'), 'this is not valid TOML and must never be parsed {{{');
    fs.writeFileSync(path.join(dir, 'scout.toml'), '');
    fs.writeFileSync(path.join(dir, 'notes.txt'), 'ignored: wrong extension');
    assert.deepEqual(discoverHostRoles('codex', { env: {}, userHome: home }), ['implementer', 'scout']);
  } finally { cleanup(); }
});

test('claude roles come from ~/.claude/agents/*.md file names', () => {
  const { home, cleanup } = tempHome();
  try {
    const dir = path.join(home, '.claude', 'agents');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'implementer.md'), '# implementer');
    fs.writeFileSync(path.join(dir, 'lightweight-worker.md'), '# lightweight-worker');
    assert.deepEqual(discoverHostRoles('claude', { env: {}, userHome: home }), ['implementer', 'lightweight-worker']);
  } finally { cleanup(); }
});

test('CODEX_HOME/CLAUDE_CONFIG_DIR override the default ~/.codex or ~/.claude base', () => {
  const { home, cleanup } = tempHome();
  try {
    const codexDir = path.join(home, 'alt-codex', 'agents');
    fs.mkdirSync(codexDir, { recursive: true });
    fs.writeFileSync(path.join(codexDir, 'specialist.toml'), '');
    assert.deepEqual(discoverHostRoles('codex', { env: { CODEX_HOME: path.join(home, 'alt-codex') }, userHome: home }), ['specialist']);
    assert.deepEqual(discoverHostRoles('codex', { env: {}, userHome: home }), []);

    const claudeDir = path.join(home, 'alt-claude', 'agents');
    fs.mkdirSync(claudeDir, { recursive: true });
    fs.writeFileSync(path.join(claudeDir, 'verifier.md'), '');
    assert.deepEqual(discoverHostRoles('claude', { env: { CLAUDE_CONFIG_DIR: path.join(home, 'alt-claude') }, userHome: home }), ['verifier']);
  } finally { cleanup(); }
});

test('a broken symlink is skipped instead of throwing; a followed symlink still counts', () => {
  const { home, cleanup } = tempHome();
  try {
    const dir = path.join(home, '.codex', 'agents');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'real-role.toml'), '');
    fs.symlinkSync(path.join(dir, 'real-role.toml'), path.join(dir, 'linked-role.toml'));
    fs.symlinkSync(path.join(dir, 'does-not-exist.toml'), path.join(dir, 'broken-link.toml'));
    assert.deepEqual(discoverHostRoles('codex', { env: {}, userHome: home }), ['linked-role', 'real-role']);
  } finally { cleanup(); }
});

test('malformed stems and directory entries beyond the cap are rejected or bounded', () => {
  const { home, cleanup } = tempHome();
  try {
    const dir = path.join(home, '.codex', 'agents');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, '__proto__.toml'), '');
    fs.writeFileSync(path.join(dir, '1-starts-with-digit.toml'), '');
    fs.mkdirSync(path.join(dir, 'a-directory.toml'));
    for (let i = 0; i < 140; i++) fs.writeFileSync(path.join(dir, `role-${String(i).padStart(3, '0')}.toml`), '');
    const roles = discoverHostRoles('codex', { env: {}, userHome: home });
    assert.ok(roles.length <= 128);
    assert.ok(!roles.includes('__proto__'));
    assert.ok(!roles.includes('1-starts-with-digit'));
    assert.ok(!roles.includes('a-directory'));
    assert.deepEqual(roles, [...roles].sort());
  } finally { cleanup(); }
});
