import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { execFileSync } from 'node:child_process';
import { callTypeSafe } from '../src/provider.mjs';
import { API_URL, MAX_RESPONSE_BYTES } from '../src/constants.mjs';
import { resolveHome, saveCredential, getCredential, removeCredential, readText, atomicWrite, setMode, loadConfig, validateConfig } from '../src/storage.mjs';
import { tempHome, KEY, response } from './helpers.mjs';

test('credentials use env first; managed file is 0600 and removable', t => {
  const home = tempHome(t); saveCredential(home, KEY);
  assert.equal(fs.statSync(path.join(home, 'credentials.env')).mode & 0o777, 0o600);
  assert.equal(getCredential(home, {}).source, 'managed-file'); assert.equal(getCredential(home, { TYPESAFE_API_KEY: 'another-test-key' }).key, 'another-test-key');
  removeCredential(home); assert.equal(getCredential(home, {}).source, 'missing');
});
test('worktree credentials and permissive managed files/directories are refused', t => {
  const home = tempHome(t); fs.mkdirSync(path.join(home, '.git')); fs.writeFileSync(path.join(home, '.git', 'HEAD'), 'ref: refs/heads/main\n');
  assert.throws(() => saveCredential(home, KEY), /KEY_IN_REPOSITORY_REFUSED/);
  fs.rmSync(path.join(home, '.git'), { recursive: true }); saveCredential(home, KEY);
  fs.chmodSync(path.join(home, 'credentials.env'), 0o644); assert.throws(() => getCredential(home, {}), /PRIVATE_FILE_REQUIRED/);
  fs.chmodSync(path.join(home, 'credentials.env'), 0o600); fs.chmodSync(home, 0o755); assert.throws(() => getCredential(home, {}), /PRIVATE_DIRECTORY_REQUIRED/); fs.chmodSync(home, 0o700);
});
test('non-repository .git cache ancestors do not prevent private credential storage', t => {
  const root = tempHome(t), home = path.join(root, '.local', 'share', 'jev');
  fs.mkdirSync(path.join(root, '.git', 'gk'), { recursive: true });
  saveCredential(home, KEY);
  assert.equal(getCredential(home, {}).key, KEY);
  assert.equal(fs.statSync(path.join(home, 'credentials.env')).mode & 0o777, 0o600);
});
test('ancestor Git metadata, linked-worktree files and symlink markers stay blocked', t => {
  const root = tempHome(t), marker = path.join(root, '.git'), home = path.join(root, 'nested', 'jev');
  execFileSync('git', ['init', '--quiet', root]);
  assert.throws(() => saveCredential(home, KEY), /KEY_IN_REPOSITORY_REFUSED/);
  assert.equal(fs.existsSync(path.join(home, 'credentials.env')), false);
  fs.rmSync(marker, { recursive: true });
  fs.writeFileSync(marker, 'gitdir: /fixture/worktrees/linked\n');
  assert.throws(() => saveCredential(home, KEY), /KEY_IN_REPOSITORY_REFUSED/);
  fs.unlinkSync(marker); fs.symlinkSync(path.join(root, 'missing'), marker);
  assert.throws(() => saveCredential(home, KEY), /KEY_IN_REPOSITORY_REFUSED/);
});
test('key file is parsed as one assignment, never shell-evaluated', t => {
  const home = tempHome(t); const f = path.join(home, 'credentials.env');
  for (const text of ['export TYPESAFE_API_KEY=abc123456', 'TYPESAFE_API_KEY=abc123456\nEVIL=1\n', 'TYPESAFE_API_KEY="quoted-value"']) {
    atomicWrite(f, text); assert.throws(() => getCredential(home, {}));
  }
  atomicWrite(f, 'TYPESAFE_API_KEY=$(touch-no-command-is-run)');
  assert.equal(getCredential(home, {}).key, '$(touch-no-command-is-run)'); assert.equal(fs.readdirSync(home).length, 1);
});
test('symlink and hardlink files are refused', t => {
  const home = tempHome(t); const real = path.join(home, 'real'); atomicWrite(real, 'value');
  const link = path.join(home, 'link'); fs.symlinkSync(real, link); assert.throws(() => readText(link), /UNSAFE_SYMLINK/); assert.throws(() => atomicWrite(link, 'bad'), /UNSAFE_SYMLINK/);
  fs.unlinkSync(link); fs.linkSync(real, link); assert.throws(() => readText(link), /UNSAFE_FILE/);
});
test('symlink parent and file-size limits are refused', t => {
  const home = tempHome(t); const dir = path.join(home, 'dir'); fs.mkdirSync(dir); fs.symlinkSync(dir, path.join(home, 'alias'));
  assert.throws(() => atomicWrite(path.join(home, 'alias', 'x'), 'no'), /UNSAFE_SYMLINK/);
  atomicWrite(path.join(dir, 'large'), '123456'); assert.throws(() => readText(path.join(dir, 'large'), { maxBytes: 3 }), /FILE_TOO_LARGE/);
});
test('atomic conflict does not overwrite another writer', t => {
  const home = tempHome(t); const file = path.join(home, 'test'); atomicWrite(file, 'other');
  assert.throws(() => atomicWrite(file, 'new', { expected: 'old' }), /WRITE_CONFLICT/); assert.equal(readText(file), 'other');
});
test('configuration rejects unknown/prototype keys and emergency OFF repairs JSON', t => {
  const home = tempHome(t); assert.throws(() => validateConfig(JSON.parse('{"__proto__":{}}')), /INVALID_CONFIG/);
  assert.throws(() => validateConfig({ maxQuestions: 9 }), /INVALID_CONFIG/); assert.throws(() => resolveHome({ POINTSMAN_HOME: 'relative' }), /HOME_MUST_BE_ABSOLUTE/);
  atomicWrite(path.join(home, 'config.json'), '{broken'); assert.throws(() => loadConfig(home, {}), /INVALID_CONFIG/); setMode(home, 'off', {}); assert.equal(loadConfig(home, {}).mode, 'off');
});
test('transport matches the official origin/path/authorization and forbids redirects', async () => {
  let n = 0;
  const raw = await callTypeSafe({ state: 'public', questions: {} }, KEY, { timeoutMs: 1000, fetchImpl: async (url, options) => {
    n++; assert.equal(url, API_URL); assert.equal(options.headers.Authorization, `Bearer ${KEY}`); assert.equal(options.redirect, 'error'); assert.equal(options.method, 'POST');
    return Response.json(response());
  } });
  assert.equal(raw.model, 'jev-offline-fixture'); assert.equal(n, 1);
});
test('HTTP errors map to bounded codes and no retry or response-body leak', async t => {
  for (const [status, expected] of [[401, 'AUTH_ERROR'], [403, 'AUTH_ERROR'], [429, 'RATE_LIMITED'], [500, 'PROVIDER_UNAVAILABLE'], [400, 'PROVIDER_REJECTED']]) await t.test(String(status), async () => {
    let n = 0; await assert.rejects(callTypeSafe({}, KEY, { timeoutMs: 1000, fetchImpl: async () => { n++; return new Response(KEY, { status }); } }), e => e.message === expected && !e.message.includes(KEY)); assert.equal(n, 1);
  });
});
test('transport rejects invalid JSON, content type and excessive response size', async t => {
  const cases = [() => new Response('bad', { headers: { 'content-type': 'text/html' } }), () => new Response('{bad', { headers: { 'content-type': 'application/json' } }), () => new Response('x'.repeat(MAX_RESPONSE_BYTES + 1), { headers: { 'content-type': 'application/json' } })];
  for (let i = 0; i < cases.length; i++) await t.test(String(i), async () => { await assert.rejects(callTypeSafe({}, KEY, { timeoutMs: 1000, fetchImpl: async () => cases[i]() }), /MALFORMED_RESPONSE|RESPONSE_TOO_LARGE/); });
});
test('timeout covers fetch and response-body wait; cancellation is propagated', async () => {
  await assert.rejects(callTypeSafe({}, KEY, { timeoutMs: 20, fetchImpl: async () => new Promise(() => {}) }), /TIMEOUT/);
  await assert.rejects(callTypeSafe({}, KEY, { timeoutMs: 20, fetchImpl: async () => new Response(new ReadableStream({ start() {} }), { headers: { 'content-type': 'application/json' } }) }), /TIMEOUT/);
  const c = new AbortController(); const pending = callTypeSafe({}, KEY, { timeoutMs: 1000, signal: c.signal, fetchImpl: async () => new Promise(() => {}) }); c.abort(); await assert.rejects(pending, /CANCELLED/);
});
test('real fetch integration on loopback rejects a redirect without forwarding Authorization', async t => {
  let redirected = 0;
  const server = http.createServer((req, res) => {
    if (req.url === '/redirect') { res.writeHead(302, { Location: '/secret-sink' }); res.end(); }
    else { redirected++; res.end('unexpected'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  await assert.rejects(callTypeSafe({}, KEY, { timeoutMs: 1000, fetchImpl: (url, options) => { assert.equal(url, API_URL); return fetch(`${origin}/redirect`, options); } }), /NETWORK_ERROR/);
  assert.equal(redirected, 0);
});
