import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { setup, response } from './features-helpers.mjs';
import { atomicWrite, setMode } from '../src/storage.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { processHookEvent, runHookCli } from '../src/hooks.mjs';

const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
function effortFixture(t, mode = 'on') {
  const s = setup({ mode, featureMode: 'off' }); t.after(s.cleanup); t.after(() => s.engine.close());
  s.policy.effort = { ...s.policy.effort, mode: 'on', mainLoop: 'cold-only', subagents: 'on' }; s.save();
  const laya = { python: '/usr/bin/python3', modelPath: s.home, model: 'laya/base', checkpoint: 'b'.repeat(64),
    runtimeVersion: '0.3.4', device: 'cpu', qualification: { checkpoint: 'b'.repeat(64), calibrationVersion: 'v1', purposes: ['route'],
      minConfidence: .9, minChoiceProbability: .9, noulCertainty: .9 } };
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify({ version: 1, provider: 'laya', laya }));
  return { ...s, laya };
}
test('runHookCli effort ON/router OFF calls local provider and emits an applicable effort result', async t => {
  const s = effortFixture(t); let output = '', calls = 0;
  await runHookCli({ host: 'claude', event: 'turn-effort', home: s.home, env: s.env,
    stdin: Readable.from([JSON.stringify({ text: 'Rename this variable.', loop: 'subagent' })]), write: text => { output += text; },
    engineFactory: options => { assert.equal(options.layaSpawn, false); assert.equal(options.layaWait, false);
      return createDecisionEngine({ ...options, provider: async payload => { calls++;
        return { ...response(payload), identity: { model: s.laya.model, checkpoint: s.laya.checkpoint,
          runtime_version: s.laya.runtimeVersion, device: s.laya.device, precision: 'torch.float32' } }; } }); } });
  assert.equal(calls, 1); const result = JSON.parse(output);
  assert.equal(result.mode, 'on'); assert.equal(result.apply, true); assert.equal(result.level, s.policy.effort.lowerTo);
  assert.equal(result.reason, 'LOWER');
  setMode(s.home, 'off', s.env); calls = 0; output = '';
  await runHookCli({ host: 'claude', event: 'turn-effort', home: s.home, env: s.env,
    stdin: Readable.from([JSON.stringify({ text: 'Rename this variable.', loop: 'subagent' })]), write: text => { output += text; },
    engineFactory: options => createDecisionEngine({ ...options, provider: async () => { calls++; throw Error('OFF must not infer'); } }) });
  assert.equal(calls, 0); assert.equal(output, '');
});
test('real hook executable processes effort while router OFF and global OFF suppresses outcome writes', t => {
  const s = effortFixture(t);
  const env = { ...process.env, ...s.env };
  const input = JSON.stringify({ turn_id: 'turn-1', loop: 'main', steps: 1, duration_ms: 10, tool_uses: 0,
    usage: { input: 1, output: 1, cache_creation: 0, cache_read: 0 } });
  const run = () => spawnSync(process.execPath, [bin, 'hook', '--host', 'claude', '--event', 'turn-outcome'], { env, input, encoding: 'utf8' });
  const first = run(); assert.equal(first.status, 0); assert.equal(first.stdout, '');
  const events = () => fs.readdirSync(path.join(s.home, 'logs')).flatMap(name => fs.readFileSync(path.join(s.home, 'logs', name), 'utf8').trim().split('\n').map(JSON.parse));
  assert.ok(events().some(e => e.reason === 'TURN_OUTCOME' && e.event === 'turn-outcome'));
  const count = events().length; setMode(s.home, 'off', s.env);
  const off = run(); assert.equal(off.status, 0); assert.equal(off.stdout, ''); assert.equal(events().length, count);
});

test('late effort completion leaves no output, successful event or pending effort decision', async t => {
  const s = effortFixture(t), controller = new AbortController(); let received;
  const layer = { ...s.layer, effort: async (_input, options) => {
    received = options.signal; controller.abort();
    return { id: 'late-effort', mode: 'on', apply: true, level: 'medium', arm: 'treatment', reason: 'LOWER' };
  } };
  const result = await processHookEvent({ host: 'claude', event: 'turn-effort', home: s.home, env: s.env,
    input: { text: 'Rename this variable.', loop: 'subagent' }, layer, signal: controller.signal });
  assert.equal(received, controller.signal); assert.equal(result.output, null);
  assert.equal(fs.existsSync(path.join(s.home, 'links', 'pending-effort')), false);
  assert.equal(fs.existsSync(path.join(s.home, 'logs')), false);
});
