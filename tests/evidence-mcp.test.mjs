import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { PassThrough } from 'node:stream';
import { startMcp } from '../src/mcp.mjs';
import { setMode } from '../src/storage.mjs';
import { setFeatureMode, validateFeaturePolicy } from '../src/feature-policy.mjs';
import { setup, response } from './features-helpers.mjs';
import { tempHome, deferred } from './helpers.mjs';

const sha = text => createHash('sha256').update(text).digest('hex');
const collect = extra => ({ query: 'Find transaction behavior and its exceptions', terms: ['transaction'], ...extra });
const ref = ({ path, hash, startLine, endLine }) => ({ path, hash, startLine, endLine });
function source(t, files = { 'main.mjs': 'before\ntransaction starts\ntransaction commits\nafter\noutside\n' }) {
  const root = tempHome(t);
  for (const [name, text] of Object.entries(files)) fs.writeFileSync(path.join(root, name), text);
  return root;
}
function largeSource(t) {
  return source(t, Object.fromEntries(['keep', 'drop', 'uncertain', 'required', 'contrary', 'protected'].map(name =>
    [`${name}.txt`, `${name === 'drop' || name === 'uncertain' ? name + ':' : name} transaction ${'context '.repeat(190)}\n`])));
}
function fixture(t, options = {}) {
  const s = setup(options); t.after(s.cleanup);
  if (options.evidenceMode !== undefined) { s.policy.evidence = { mode: options.evidenceMode }; s.save(); }
  return s;
}
function transport(t, input, output, close) {
  let buffer = '', nextId = 1;
  const replies = new Map(), waiting = new Map();
  output.on('data', chunk => {
    buffer += chunk.toString();
    let end;
    while ((end = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
      if (!line.trim()) continue;
      const message = JSON.parse(line), pending = waiting.get(message.id);
      if (pending) { waiting.delete(message.id); pending(message); }
      else replies.set(message.id, message);
    }
  });
  t.after(close);
  const send = (method, params, id) => input.write(JSON.stringify({ jsonrpc: '2.0', ...(id === undefined ? {} : { id }), method, ...(params ? { params } : {}) }) + '\n');
  const request = (method, params) => {
    const id = nextId++;
    send(method, params, id);
    let timer;
    const reply = new Promise((resolve, reject) => {
      if (replies.has(id)) { resolve(replies.get(id)); replies.delete(id); return; }
      timer = setTimeout(() => { waiting.delete(id); reject(new Error(`MCP response timeout: ${method}`)); }, 5000);
      waiting.set(id, resolve);
    }).finally(() => clearTimeout(timer));
    return { id, reply };
  };
  const value = async reply => {
    const message = await reply;
    assert.equal(message.error, undefined, JSON.stringify(message));
    return JSON.parse(message.result.content[0].text);
  };
  return { send, request, value, call: (name, args) => value(request('tools/call', { name, arguments: args }).reply),
    async ready() {
      await request('initialize', { protocolVersion: '2025-06-18', clientInfo: { name: 'evidence-test', version: '1' }, capabilities: {} }).reply;
      send('notifications/initialized');
    } };
}
async function client(t, s, root) {
  const input = new PassThrough(), output = new PassThrough();
  const server = startMcp(s.engine, { input, output, layer: s.layer, root });
  const c = transport(t, input, output, () => { server.close(); input.destroy(); output.destroy(); });
  await c.ready(); return c;
}
function selectedResponse(payload) {
  const result = response(payload);
  for (const [name, text] of Object.entries(payload.state.candidates)) {
    const label = text.includes('drop:') ? 'exclude' : text.includes('uncertain:') ? 'uncertain' : 'include';
    result.answers[name] = { type: 'choice', choice: label, confidence: .99,
      probabilities: Object.fromEntries(Object.keys(payload.questions[name].criteria).map(key => [key, key === label ? 1 : 0])) };
  }
  return result;
}

test('evidence defaults OFF independently; global OFF and SHADOW perform no source or model action', async t => {
  const policy = validateFeaturePolicy({ version: 2, bulk: { mode: 'on' }, workflow: { mode: 'on' } });
  assert.equal(policy.evidence.mode, 'off');
  const s = fixture(t), c = await client(t, s, path.join(s.home, 'nonexistent-source'));
  const tools = (await c.request('tools/list').reply).result.tools;
  for (const name of ['collect_evidence', 'read_evidence']) {
    const tool = tools.find(tool => tool.name === name); assert.ok(tool, name);
    assert.equal(tool.inputSchema.additionalProperties, false);
    assert.equal(tool.inputSchema.properties.root, undefined);
    assert.equal(tool.inputSchema.properties.command, undefined);
    assert.equal(tool.inputSchema.properties.commands, undefined);
  }
  for (const [globalMode, featureMode, reason] of [['on', 'off', 'OFF'], ['off', 'on', 'OFF'], ['shadow', 'on', 'SHADOW'], ['on', 'shadow', 'SHADOW']]) {
    setMode(s.home, globalMode, s.env); setFeatureMode(s.home, 'evidence', featureMode);
    for (const [name, args] of [['collect_evidence', collect()], ['read_evidence', { refs: [{ path: 'main.mjs', hash: '0'.repeat(64), startLine: 1, endLine: 1 }] }]]) {
      const result = await c.call(name, args);
      assert.equal(result.reason, reason); assert.deepEqual(result.evidence, []);
      assert.equal(result.stats.actions, 0); assert.equal(result.stats.inferenceCalls, 0);
    }
  }
  assert.equal(s.calls.length, 0);
});

test('local collection merges overlapping literal spans and exact reads recover their source', async t => {
  const s = fixture(t, { evidenceMode: 'on' }), root = source(t), c = await client(t, s, root);
  setFeatureMode(s.home, 'bulk', 'off'); setFeatureMode(s.home, 'workflow', 'off');
  const result = await c.call('collect_evidence', collect({ contextLines: 1 }));
  assert.equal(result.status, 'done', JSON.stringify(result)); assert.equal(result.evidence.length, 1);
  const snippet = result.evidence[0];
  assert.deepEqual(ref(snippet), { path: 'main.mjs', hash: sha(fs.readFileSync(path.join(root, 'main.mjs'))), startLine: 1, endLine: 4 });
  assert.equal(snippet.text.replace(/\n$/, ''), 'before\ntransaction starts\ntransaction commits\nafter');
  const read = await c.call('read_evidence', { refs: [ref(snippet)] });
  assert.equal(read.status, 'done', JSON.stringify(read)); assert.deepEqual(read.evidence.map(ref), [ref(snippet)]);
  assert.equal(read.evidence[0].text, snippet.text); assert.equal(s.calls.length, 0);
  const small = await c.call('collect_evidence', collect({ semantic: true, contextLines: 1 }));
  assert.deepEqual(small.evidence.map(ref), result.evidence.map(ref)); assert.equal(s.calls.length, 0);
});

test('semantic selection uses shared Jev batching with bulk OFF, protects mandatory evidence and permits exact recovery', async t => {
  const s = fixture(t, { evidenceMode: 'on', provider: selectedResponse }), root = largeSource(t), c = await client(t, s, root);
  setFeatureMode(s.home, 'bulk', 'off');
  const result = await c.call('collect_evidence', collect({ semantic: true, contextLines: 0,
    requiredPaths: ['required.txt'], uncertainPaths: ['protected.txt'], counterevidencePaths: ['contrary.txt'] }));
  assert.equal(result.status, 'done', JSON.stringify(result)); assert.ok(s.calls.length > 0);
  assert.equal(result.details.selection.applied, true);
  assert.deepEqual(result.evidence.map(e => e.path).sort(), ['contrary.txt', 'keep.txt', 'protected.txt', 'required.txt', 'uncertain.txt']);
  const omitted = result.details.omitted.find(item => item.path === 'drop.txt'); assert.ok(omitted);
  assert.equal(omitted.hash, sha(fs.readFileSync(path.join(root, 'drop.txt'))));
  const before = s.calls.length, read = await c.call('read_evidence', { refs: [ref(omitted)] });
  assert.equal(read.status, 'done', JSON.stringify(read)); assert.equal(read.evidence[0].path, 'drop.txt');
  assert.equal(read.evidence[0].text.replace(/\n$/, ''), fs.readFileSync(path.join(root, 'drop.txt'), 'utf8').replace(/\n$/, ''));
  assert.equal(s.calls.length, before);
  for (const payload of s.calls) {
    assert.equal(payload.model, 'jev-1.13.0'); assert.ok(Object.keys(payload.questions).length <= 8);
    assert.ok(Buffer.byteLength(JSON.stringify(payload)) <= 24000);
    assert.equal(JSON.stringify(payload.state).includes('required transaction'), false);
  }
});

test('provider failure keeps every candidate and exhaustive coverage bypasses inference', async t => {
  const s = fixture(t, { evidenceMode: 'on', provider: () => { throw new Error('offline provider failure'); } });
  const root = largeSource(t), c = await client(t, s, root);
  const failed = await c.call('collect_evidence', collect({ semantic: true, contextLines: 0 }));
  assert.equal(s.calls.length, 1); assert.equal(failed.details.selection.applied, false);
  assert.equal(failed.evidence.length, 6); assert.deepEqual(failed.details.omitted, []);
  fs.writeFileSync(path.join(root, 'notes.txt'), 'unrelated news without the search term\n');
  const exhaustive = await c.call('collect_evidence', collect({ semantic: true, coverage: 'exhaustive', contextLines: 0 }));
  assert.equal(exhaustive.evidence.length, 7); assert.equal(exhaustive.coverage.complete, true);
  assert.deepEqual(exhaustive.details.omitted, []); assert.equal(s.calls.length, 1);
});

test('Git ignored, secret and symlink sources are refused by both collection and recovery', async t => {
  const secret = ['password', '=abcdefghi; transaction\n'].join('');
  const root = source(t, { 'safe.txt': 'transaction safe\n', '.gitignore': 'ignored.txt\n', 'ignored.txt': 'transaction ignored private value\n', 'unsafe.txt': secret });
  execFileSync('git', ['init', '-q', root]);
  const outside = source(t, { 'outside.txt': 'transaction symlink private value\n' });
  fs.symlinkSync(path.join(outside, 'outside.txt'), path.join(root, 'linked.txt'));
  const s = fixture(t, { evidenceMode: 'on' }), c = await client(t, s, root);
  const result = await c.call('collect_evidence', collect({ contextLines: 0 }));
  assert.deepEqual(result.evidence.map(e => e.path), ['safe.txt']);
  assert.equal(result.coverage.complete, false);
  for (const [name, text] of [['ignored.txt', 'transaction ignored private value\n'], ['unsafe.txt', secret], ['linked.txt', 'transaction symlink private value\n']]) {
    const read = await c.call('read_evidence', { refs: [{ path: name, hash: sha(text), startLine: 1, endLine: 1 }] });
    assert.equal(read.evidence.length, 0); assert.notEqual(read.status, 'done');
    assert.equal(JSON.stringify(read).includes('abcdefghi'), false);
    assert.equal(JSON.stringify(read).includes('private value'), false);
  }
  assert.equal(s.calls.length, 0);
});

test('stale hashes and supplied snapshots never return changed exact data', async t => {
  const s = fixture(t, { evidenceMode: 'on' }), root = source(t), c = await client(t, s, root);
  const original = await c.call('collect_evidence', collect());
  const oldRef = ref(original.evidence[0]);
  fs.appendFileSync(path.join(root, 'main.mjs'), 'transaction changed exact data\n');
  for (const [name, args] of [['read_evidence', { refs: [oldRef] }],
    ['collect_evidence', collect({ snapshot: { files: { 'main.mjs': oldRef.hash } } })]]) {
    const result = await c.call(name, args);
    assert.notEqual(result.status, 'done'); assert.deepEqual(result.evidence, []);
    assert.match(result.reason, /STALE|CHANGED/); assert.equal(JSON.stringify(result).includes('changed exact data'), false);
  }
  assert.equal(s.calls.length, 0);
});

test('cancellation, source mutation and feature mode flips invalidate in-flight semantic selection', async t => {
  for (const change of ['cancel', 'source', 'mode']) await t.test(change, async t => {
    const entered = deferred(), release = deferred();
    const s = fixture(t, { evidenceMode: 'on', provider: async payload => { entered.resolve(); await release.promise; return selectedResponse(payload); } });
    t.after(() => release.resolve());
    const root = largeSource(t), c = await client(t, s, root);
    const pending = c.request('tools/call', { name: 'collect_evidence', arguments: collect({ semantic: true, contextLines: 0 }) });
    const timer = setTimeout(() => entered.resolve('timeout'), 4000);
    const timedOut = await entered.promise; clearTimeout(timer); assert.notEqual(timedOut, 'timeout', 'semantic provider must be reached');
    if (change === 'cancel') c.send('notifications/cancelled', { requestId: pending.id });
    if (change === 'source') fs.appendFileSync(path.join(root, 'keep.txt'), 'changed\n');
    if (change === 'mode') setFeatureMode(s.home, 'evidence', 'off');
    release.resolve();
    const result = await c.value(pending.reply);
    assert.notEqual(result.status, 'done'); assert.deepEqual(result.evidence, []);
    assert.match(result.reason, change === 'cancel' ? /CANCELLED/ : /CHANGED|STALE/);
    assert.equal(result.details?.selection?.applied ?? false, false);
  });
});

test('protocol requests cannot inject roots, commands or exceed admission bounds', async t => {
  const s = fixture(t, { evidenceMode: 'on' }), root = source(t), c = await client(t, s, root);
  for (const extra of [{ root }, { commands: ['pwd'] }, { maxFiles: 97 }, { maxSnippets: 65 }, { contextLines: 13 }]) {
    const result = await c.call('collect_evidence', collect(extra));
    assert.notEqual(result.status, 'done'); assert.deepEqual(result.evidence ?? [], []);
    assert.match(result.reason ?? result.error, /^INVALID_/);
  }
  const read = await c.call('read_evidence', { refs: [{ path: '../outside.txt', hash: '0'.repeat(64), startLine: 1, endLine: 1 }] });
  assert.notEqual(read.status, 'done'); assert.deepEqual(read.evidence ?? [], []); assert.equal(s.calls.length, 0);
  setFeatureMode(s.home, 'workflow', 'on'); setFeatureMode(s.home, 'evidence', 'off');
  const bypass = await c.call('run', { workflow: 'collect-evidence', inputs: { query: 'transaction', terms: ['transaction'] } });
  assert.match(bypass.reason, /^INVALID_/); assert.deepEqual(bypass.evidence, []); assert.equal(s.calls.length, 0);
});

test('real stdio collects and reads local evidence with no provider readiness or network call', async t => {
  const home = tempHome(t), root = source(t);
  setMode(home, 'on', {}); setFeatureMode(home, 'evidence', 'on');
  const guard = 'globalThis.fetch=()=>{process.stderr.write("UNEXPECTED_NETWORK\\n");throw new Error("offline stdio test");};';
  const child = spawn(process.execPath, ['--import', `data:text/javascript,${encodeURIComponent(guard)}`,
    path.resolve('bin/pointsman.mjs'), 'mcp', '--root', root, '--home', home],
  { env: { PATH: process.env.PATH, HOME: home, POINTSMAN_HOME: home }, stdio: ['pipe', 'pipe', 'pipe'] });
  let stderr = ''; child.stderr.on('data', chunk => { stderr += chunk; });
  const closed = once(child, 'close');
  const c = transport(t, child.stdin, child.stdout, () => { child.stdin.destroy(); if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL'); });
  await c.ready();
  const status = await c.call('status', {}); assert.equal(status.mode, 'on'); assert.equal(status.ready, false);
  const result = await c.call('collect_evidence', collect({ semantic: false }));
  assert.equal(result.status, 'done', JSON.stringify(result)); assert.equal(result.stats.inferenceCalls, 0);
  const read = await c.call('read_evidence', { refs: result.evidence.map(ref) });
  assert.equal(read.status, 'done', JSON.stringify(read)); assert.equal(read.stats.inferenceCalls, 0);
  assert.equal(read.evidence[0].text, result.evidence[0].text);
  child.stdin.end();
  const [code] = await closed; assert.equal(code, 0, stderr); assert.equal(stderr, '');
});
