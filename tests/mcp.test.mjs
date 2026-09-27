import test from 'node:test';
import assert from 'node:assert/strict';
import { PassThrough } from 'node:stream';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { startMcp } from '../src/mcp.mjs';
import { REPO_ROOT } from '../src/installer.mjs';
import { fixture, request, response, deferred, tempHome, delay } from './helpers.mjs';
function client(t, engine) {
  const input = new PassThrough(), output = new PassThrough(), lines = [];
  output.on('data', chunk => lines.push(...chunk.toString().trim().split('\n').map(JSON.parse)));
  const server = startMcp(engine, { input, output }); t.after(() => { server.close(); input.destroy(); output.destroy(); });
  const send = (id, method, params) => input.write(JSON.stringify({ jsonrpc: '2.0', ...(id === undefined ? {} : { id }), method, ...(params ? { params } : {}) }) + '\n');
  const wait = async id => { for (let i = 0; i < 100; i++) { const found = lines.find(x => x.id === id); if (found) return found; await delay(5); } throw new Error('MCP response timeout'); };
  const ready = () => { send(1, 'initialize', { protocolVersion: '2025-06-18', clientInfo: { name: 'test', version: '1' }, capabilities: {} }); send(undefined, 'notifications/initialized'); };
  return { input, output, lines, send, wait, ready };
}
test('MCP negotiates version, advertises six tools, and serves a typed decision', async t => {
  const f = fixture(t), c = client(t, f.engine); c.ready(); const init = await c.wait(1); assert.equal(init.result.protocolVersion, '2025-06-18');
  c.send(2, 'tools/list'); assert.deepEqual((await c.wait(2)).result.tools.map(x => x.name), ['decide', 'status', 'feedback', 'route', 'filter', 'observe', 'record']);
  c.send(3, 'tools/call', { name: 'decide', arguments: request() }); const r = JSON.parse((await c.wait(3)).result.content[0].text); assert.equal(r.apply, true);
  c.send(4, 'tools/call', { name: 'feedback', arguments: { id: r.id, baseline: { category: 'documentation', passed: true, complexity: .02 } } }); assert.equal(JSON.parse((await c.wait(4)).result.content[0].text).matched, 3);
});
test('MCP rejects pre-initialization requests and negotiates an unsupported version', async t => {
  const f = fixture(t), c = client(t, f.engine); c.send(9, 'tools/list'); assert.equal((await c.wait(9)).error.code, -32002);
  c.send(1, 'initialize', { protocolVersion: 'future-version', clientInfo: { name: 'test', version: '1' }, capabilities: {} }); assert.equal((await c.wait(1)).result.protocolVersion, '2025-06-18');
});
test('MCP handles malformed JSON, unknown tools/methods and invalid status arguments', async t => {
  const f = fixture(t), c = client(t, f.engine); c.ready(); c.input.write('{broken\n');
  assert.equal(c.lines.some(x => x.error?.code === -32700), true);
  c.send(2, 'unknown'); assert.equal((await c.wait(2)).error.code, -32601);
  c.send(3, 'tools/call', { name: 'missing' }); assert.equal((await c.wait(3)).error.code, -32602);
  c.send(4, 'tools/call', { name: 'status', arguments: { secret: 'not-valid' } }); assert.equal((await c.wait(4)).result.isError, true);
});
test('MCP fragmented UTF-8 frames and CRLF are parsed correctly', async t => {
  const f = fixture(t), c = client(t, f.engine); c.ready();
  const frame = Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'decide', arguments: { ...request(), state: '\uBB38\uC11C \uBCC0\uACBD' } } }) + '\r\n');
  for (let i = 0; i < frame.length; i += 3) c.input.write(frame.subarray(i, i + 3));
  assert.equal(JSON.parse((await c.wait(2)).result.content[0].text).apply, true);
});
test('MCP cancellation prevents application even if a provider ignores the abort', async t => {
  const d = deferred(); const f = fixture(t, { provider: async () => { await d.promise; return response(); } }), c = client(t, f.engine);
  c.ready(); c.send(2, 'tools/call', { name: 'decide', arguments: request() }); c.send(undefined, 'notifications/cancelled', { requestId: 2 }); d.resolve();
  const r = JSON.parse((await c.wait(2)).result.content[0].text); assert.equal(r.reason, 'CANCELLED'); assert.equal(r.apply, false);
});
test('MCP oversized frame closes transport safely', t => {
  const f = fixture(t), c = client(t, f.engine); c.input.write('x'.repeat(65537)); assert.equal(c.lines[0].error.message, 'Frame too large');
});
test('real stdio subprocess exchanges newline JSON, with no startup chatter or secret output', async t => {
  const home = tempHome(t); const child = spawn(process.execPath, [path.join(REPO_ROOT, 'bin/pointsman.mjs'), 'mcp'], { env: { ...process.env, POINTSMAN_HOME: home, TYPESAFE_API_KEY: 'never-print-this-test-key' }, stdio: ['pipe', 'pipe', 'pipe'] });
  t.after(() => child.kill()); let out = '', stderr = ''; child.stdout.on('data', b => { out += b; }); child.stderr.on('data', b => { stderr += b; });
  const send = object => child.stdin.write(JSON.stringify(object) + '\n');
  send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', clientInfo: { name: 'stdio-test', version: '1' }, capabilities: {} } });
  send({ jsonrpc: '2.0', method: 'notifications/initialized' }); send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'status', arguments: {} } });
  for (let i = 0; i < 200 && out.trim().split('\n').length < 2; i++) await delay(5);
  const messages = out.trim().split('\n').map(JSON.parse); assert.equal(messages.length, 2); assert.equal(JSON.parse(messages[1].result.content[0].text).mode, 'off');
  assert.equal(stderr, ''); assert.equal(out.includes('never-print-this-test-key'), false);
});
