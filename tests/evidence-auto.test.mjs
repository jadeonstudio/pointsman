import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { PassThrough } from 'node:stream';
import { startMcp } from '../src/mcp.mjs';
import { setMode, atomicWrite } from '../src/storage.mjs';
import { loadFeaturePolicy, validateFeaturePolicy } from '../src/feature-policy.mjs';
import { setup, response } from './features-helpers.mjs';
import { tempHome, deferred } from './helpers.mjs';

const collect = extra => ({ query: 'Find transaction behavior and exceptions', terms: ['transaction'], contextLines: 0, ...extra });
function fixture(t, automatic = 'off', options = {}) {
  const s = setup({ featureMode: 'off', ...options });
  t.after(s.cleanup);
  s.policy.evidence = { mode: 'on', automatic }; s.save();
  return s;
}
function sources(t) {
  const root = tempHome(t);
  for (let i = 0; i < 6; i++) fs.writeFileSync(path.join(root, `source-${i}.txt`), `transaction case ${i} ${'context '.repeat(190)}\n`);
  return root;
}
async function client(t, s, root) {
  const input = new PassThrough(), output = new PassThrough();
  const server = startMcp(s.engine, { input, output, layer: s.layer, root });
  let buffer = '', nextId = 1;
  const waiting = new Map();
  output.on('data', chunk => {
    buffer += chunk.toString();
    let end;
    while ((end = buffer.indexOf('\n')) !== -1) {
      const message = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1);
      waiting.get(message.id)?.(message);
    }
  });
  t.after(() => { server.close(); input.destroy(); output.destroy(); });
  const send = (method, params, id) => input.write(JSON.stringify({ jsonrpc: '2.0', ...(id === undefined ? {} : { id }), method, params }) + '\n');
  const request = async (method, params) => {
    const id = nextId++; let timer;
    try {
      return await new Promise((resolve, reject) => {
        waiting.set(id, resolve);
        timer = setTimeout(() => reject(new Error(`MCP timeout: ${method}`)), 5000);
        send(method, params, id);
      });
    } finally { clearTimeout(timer); waiting.delete(id); }
  };
  const initialized = await request('initialize', { protocolVersion: '2025-06-18', clientInfo: { name: 'evidence-auto-test', version: '1' }, capabilities: {} });
  assert.equal(initialized.error, undefined); send('notifications/initialized');
  return { instructions: initialized.result.instructions,
    async call(name, args = {}) {
      const message = await request('tools/call', { name, arguments: args });
      assert.equal(message.error, undefined, JSON.stringify(message));
      return JSON.parse(message.result.content[0].text);
    } };
}
function cli(home, ...args) {
  const guard = 'globalThis.fetch=()=>{throw new Error("UNEXPECTED_NETWORK");};';
  return spawnSync(process.execPath, ['--import', `data:text/javascript,${encodeURIComponent(guard)}`,
    path.resolve('bin/pointsman.mjs'), ...args, '--home', home], {
    env: { PATH: process.env.PATH, HOME: home, POINTSMAN_HOME: home }, encoding: 'utf8', timeout: 5000,
  });
}

test('legacy policies default automatic OFF; local CLI activation needs no provider and preserves other settings', t => {
  for (const raw of [{}, { version: 2, evidence: { mode: 'on' } }]) {
    assert.equal(validateFeaturePolicy(raw).evidence.automatic, 'off');
  }
  const home = tempHome(t), policy = validateFeaturePolicy({ router: { mode: 'shadow' }, effort: { mode: 'shadow' } });
  atomicWrite(path.join(home, 'features.json'), JSON.stringify(policy)); setMode(home, 'off', {});
  for (const automatic of ['local', 'semantic', 'off', 'local']) {
    const result = cli(home, 'evidence', 'auto', automatic);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(loadFeaturePolicy(home), { ...policy, evidence: { ...policy.evidence, automatic } });
    const status = cli(home, 'status'); assert.equal(status.status, 0, status.stderr);
    assert.equal(JSON.parse(status.stdout).mode, 'off');
  }
  const before = fs.readFileSync(path.join(home, 'features.json'), 'utf8');
  const invalid = cli(home, 'evidence', 'auto', 'on');
  assert.notEqual(invalid.status, 0); assert.equal(fs.readFileSync(path.join(home, 'features.json'), 'utf8'), before);
  const evidence = cli(home, 'evidence', 'on'); assert.equal(evidence.status, 0, evidence.stderr);
  const enabled = cli(home, 'on'); assert.equal(enabled.status, 0, enabled.stderr);
  const status = JSON.parse(enabled.stdout); assert.equal(status.mode, 'on'); assert.equal(status.ready, false);
  assert.deepEqual(loadFeaturePolicy(home), { ...policy, evidence: { mode: 'on', automatic: 'local' } });
});

test('MCP advertises its bound absolute root and current automatic mode without exposing fixture credentials', async t => {
  for (const automatic of ['off', 'local', 'semantic']) await t.test(automatic, async t => {
    const s = fixture(t, automatic), root = sources(t), c = await client(t, s, path.relative(process.cwd(), root));
    const status = await c.call('status');
    assert.equal(status.server.root, root); assert.equal(path.isAbsolute(status.server.root), true);
    assert.ok(c.instructions.includes(root), c.instructions); assert.ok(c.instructions.includes(`Evidence automatic mode: ${automatic}.`));
    assert.equal(status.features.evidence.automatic, automatic);
    assert.equal(JSON.stringify(status).includes(s.env.TYPESAFE_API_KEY), false);
    assert.equal(c.instructions.includes(s.env.TYPESAFE_API_KEY), false);
    if (automatic !== 'off') assert.match(c.instructions, /ordinary multi-file.*collect_evidence/i);
    if (automatic === 'off') assert.doesNotMatch(c.instructions, /(?:automatically|ordinary task).*collect_evidence/i);
    assert.equal(s.calls.length, 0);
  });
});

test('automatic semantic changes omitted semantic only; local and explicit false never call the provider', async t => {
  for (const [automatic, semantic, expectedCalls] of [['off', undefined, 0], ['local', undefined, 0], ['semantic', undefined, 1], ['semantic', false, 0]]) {
    await t.test(`${automatic}, semantic=${semantic}`, async t => {
      const s = fixture(t, automatic), c = await client(t, s, sources(t));
      const packet = await c.call('collect_evidence', collect(semantic === undefined ? {} : { semantic }));
      assert.equal(packet.status, 'done', JSON.stringify(packet)); assert.ok(packet.details.candidateBytes > 4096);
      assert.equal(packet.evidence.length, 6); assert.equal(s.calls.length, expectedCalls);
      assert.equal(packet.details.selection.requested, expectedCalls > 0);
      assert.equal(packet.details.selection.applied, expectedCalls > 0);
    });
  }
});

test('global or evidence OFF prevents all collection and recovery actions regardless of automatic mode', async t => {
  for (const automatic of ['off', 'local', 'semantic']) for (const disabled of ['global', 'evidence']) await t.test(`${automatic}, ${disabled} OFF`, async t => {
    const s = fixture(t, automatic);
    if (disabled === 'global') setMode(s.home, 'off', s.env);
    else { s.policy.evidence.mode = 'off'; s.save(); }
    const c = await client(t, s, path.join(s.home, 'nonexistent-source'));
    assert.ok(c.instructions.includes('Evidence automatic mode: off.'));
    assert.doesNotMatch(c.instructions, /ordinary multi-file.*collect_evidence/i);
    for (const [name, args] of [['collect_evidence', collect()], ['read_evidence', { refs: [{ path: 'source.txt', hash: '0'.repeat(64), startLine: 1, endLine: 1 }] }]]) {
      const result = await c.call(name, args);
      assert.equal(result.reason, 'OFF'); assert.deepEqual(result.evidence, []);
      assert.equal(result.stats.actions, 0); assert.equal(result.stats.inferenceCalls, 0);
    }
    assert.equal(s.calls.length, 0);
  });
});

test('changing automatic mode during semantic inference invalidates the in-flight evidence packet', async t => {
  const entered = deferred(), release = deferred();
  const s = fixture(t, 'semantic', { provider: async payload => { entered.resolve(); await release.promise; return response(payload); } });
  t.after(() => release.resolve());
  const c = await client(t, s, sources(t)), pending = c.call('collect_evidence', collect());
  const timer = setTimeout(() => entered.resolve('timeout'), 4000);
  const entry = await entered.promise; clearTimeout(timer);
  assert.notEqual(entry, 'timeout', 'automatic semantic must reach the provider');
  const switched = cli(s.home, 'evidence', 'auto', 'local'); assert.equal(switched.status, 0, switched.stderr);
  release.resolve();
  const packet = await pending;
  assert.notEqual(packet.status, 'done'); assert.match(packet.reason, /POLICY_CHANGED|STALE/);
  assert.deepEqual(packet.evidence, []); assert.equal(packet.details?.selection?.applied ?? false, false);
  assert.equal(s.calls.length, 1);
});
