import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { setup, response, ROUTE_INPUT, FILTER_INPUT } from './features-helpers.mjs';
import { atomicWrite, loadConfig, setMode } from '../src/storage.mjs';
import { setFeatureMode, validateFeaturePolicy } from '../src/feature-policy.mjs';
import { readMetrics } from '../src/metrics.mjs';
import { startMcp } from '../src/mcp.mjs';
import { evaluatePairedRuns } from '../src/evaluation.mjs';
const copy = structuredClone;
const patch = (s, change) => atomicWrite(path.join(s.home, 'config.json'), JSON.stringify({ ...loadConfig(s.home, s.env), ...change }));
const dropping = () => ({ ...copy(FILTER_INPUT), items: [{ id: 'first', text: 'drop: first' }, { id: 'second', text: 'drop: second' }, { id: 'third', text: 'drop: third' }] });

test('batch packing honors global maxQuestions and actual UTF-8 wire budget', async t => {
  const s = setup(); t.after(s.cleanup); patch(s, { maxQuestions: 2, maxInputBytes: 2500 });
  const req = { ...dropping(), items: Array.from({ length: 5 }, (_, i) => ({ id: `item${i}`, text: 'drop: ' + '\uD55C'.repeat(150) })) };
  const r = await s.layer.filter(req);
  assert.equal(r.rejectIds.length, 5); assert.ok(s.calls.length >= 3);
  for (const call of s.calls) { assert.ok(Object.keys(call.questions).length <= 2); assert.ok(Buffer.byteLength(JSON.stringify(call)) <= 2500); }
});
test('oversized singleton is retained without truncation or invalid provider call', async t => {
  const s = setup(); t.after(s.cleanup); patch(s, { maxInputBytes: 1600 });
  const req = dropping(); req.items[0].text = 'drop: ' + 'x'.repeat(2000);
  const r = await s.layer.filter(req); assert.ok(r.keepIds.includes('first')); assert.ok(r.reviewIds.includes('first'));
  assert.equal(s.calls.some(p => JSON.stringify(p).includes('x'.repeat(2000))), false);
});
for (const change of ['global-policy', 'feature-off', 'cancel', 'version']) {
  test(`a prior exclusion is rolled back after ${change} during later chunk`, async t => {
    const controller = new AbortController(); let s;
    s = setup({ provider: (p, k, opts, count) => {
      if (count === 2) {
        if (change === 'global-policy') patch(s, { minConfidence: .91 });
        if (change === 'feature-off') setFeatureMode(s.home, 'bulk', 'off');
        if (change === 'cancel') controller.abort();
        if (change === 'version') return { ...response(p), model: 'jev-1.14.0' };
      }
      return response(p);
    } }); t.after(s.cleanup); s.policy.bulk.batchSize = 1; s.save();
    const r = await s.layer.filter(dropping(), { signal: controller.signal });
    assert.equal(r.apply, false); assert.deepEqual(r.rejectIds, []); assert.deepEqual(r.keepIds, ['first', 'second', 'third']); assert.equal(s.calls.length, 2);
  });
}
test('shared provider quota covers route and subsequent bulk calls without bypass', async t => {
  const s = setup(); t.after(s.cleanup); patch(s, { maxCallsPerMinute: 1 });
  assert.equal((await s.layer.route(copy(ROUTE_INPUT))).apply, true);
  const r = await s.layer.filter(dropping()); assert.equal(r.apply, false); assert.equal(r.keepIds.length, 3); assert.equal(s.calls.length, 1);
});
test('metadata wrappers do not double-count API usage', async t => {
  const s = setup(); t.after(s.cleanup);
  await s.layer.route(copy(ROUTE_INPUT)); await s.layer.filter(copy(FILTER_INPUT));
  const m = readMetrics(s.home); assert.equal(m.networkCalls, 2); assert.equal(m.jevReportedTokens.input, 246);
  assert.equal(m.tokenSavings, null); assert.equal(m.costSavings, null);
});
test('null policy sections and coerced numeric values are rejected', () => {
  for (const raw of [{ version: null }, { router: null }, { bulk: null }, { router: { economyMaxDifficulty: '2' } }]) assert.throws(() => validateFeaturePolicy(raw));
});
test('paired evaluator reports quality regressions and negative savings without approving deployment', () => {
  const run = { executed: true, passed: true, elapsedMs: 100, outputTokens: 100 };
  const r = evaluatePairedRuns({ cases: [{ downgraded: true, baseline: run, routed: { ...run, passed: false, elapsedMs: 200, outputTokens: 150 } }] });
  assert.equal(r.wrongDowngrades, 1); assert.equal(r.qualityRegressions, 1); assert.equal(r.elapsedReduction, -1); assert.equal(r.outputTokenReduction, -.5); assert.equal(r.zeroErrorUpper95, null); assert.equal(r.automaticApproval, false);
});

test('new MCP tools use real layer dispatch, preserve shadow and deny option injection', async t => {
  const s = setup(); t.after(s.cleanup);
  const input = new PassThrough(), output = new PassThrough(), messages = [];
  output.on('data', data => messages.push(...data.toString().trim().split('\n').map(JSON.parse)));
  const server = startMcp(s.engine, { input, output, layer: s.layer });
  t.after(() => { server.close(); input.destroy(); output.destroy(); });
  const send = (id, method, params) => input.write(JSON.stringify({ jsonrpc: '2.0', ...(id == null ? {} : { id }), method, params }) + '\n');
  const wait = async id => {
    for (let i = 0; i < 100; i++) { const m = messages.find(v => v.id === id); if (m) return m; await new Promise(r => setTimeout(r, 5)); }
    throw new Error('MCP timeout');
  };
  const value = async id => JSON.parse((await wait(id)).result.content[0].text);
  send(1, 'initialize', { protocolVersion: '2025-06-18', clientInfo: { name: 'feature-test', version: '1' } }); send(null, 'notifications/initialized');
  send(2, 'tools/call', { name: 'route', arguments: copy(ROUTE_INPUT) });
  { const v = await value(2); assert.equal(v.route.model, 'fixture-economy'); assert.equal(v.route.role, 'fixture-economy-role'); }
  send(3, 'tools/call', { name: 'filter', arguments: copy(FILTER_INPUT) }); assert.deepEqual((await value(3)).rejectIds, ['drop_1']);
  setFeatureMode(s.home, 'router', 'shadow');
  send(4, 'tools/call', { name: 'route', arguments: copy(ROUTE_INPUT) }); const r = await value(4); assert.equal(r.route, null); assert.equal(r.apply, false);
  send(5, 'tools/call', { name: 'observe', arguments: { id: r.id, tier: 'economy' } }); assert.equal((await value(5)).taskQualityMeasured, false);
  const before = s.calls.length;
  send(6, 'tools/call', { name: 'route', arguments: { ...copy(ROUTE_INPUT), modelOverride: 'attacker', modeLimit: 'on' } });
  assert.equal((await value(6)).reason, 'INVALID_ROUTE_REQUEST'); assert.equal(s.calls.length, before);
  setMode(s.home, 'off', s.env); send(7, 'tools/call', { name: 'filter', arguments: copy(FILTER_INPUT) }); assert.equal((await value(7)).keepIds.length, 4); assert.equal(s.calls.length, before);
});
