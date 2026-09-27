import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { setup, response } from './features-helpers.mjs';
import { processHookEvent, runHookCli, parseContextAnnotation, truncateUtf8, resolveSnapshotId,
  HOSTS, HOOK_EVENTS } from '../src/hooks.mjs';
import { createDecisionEngine } from '../src/engine.mjs';
import { createControlLayer } from '../src/control-layer.mjs';
import { createTrainingStore } from '../src/training/store.mjs';
import { createLinkIndex } from '../src/training/links.mjs';
import { atomicWrite } from '../src/storage.mjs';

const bin = fileURLToPath(new URL('../bin/pointsman.mjs', import.meta.url));
const ANNOTATION = '[route scope=local complete=yes failures=0]';
const claudeInput = ({ tool_input = {}, ...rest } = {}) => ({
  session_id: 'sess-1', tool_use_id: 'tu-1', prompt_id: 'p1', cwd: '/tmp/pointsman-hook-fixture', tool_name: 'Agent', ...rest,
  tool_input: { subagent_type: 'fixture-standard-role', description: 'desc', prompt: `${ANNOTATION}\nDo the work.`, ...tool_input },
});
const codexInput = ({ tool_input = {}, ...rest } = {}) => ({
  session_id: 'sess-1', cwd: '/tmp/pointsman-hook-fixture', tool_name: 'spawn_agent', ...rest,
  tool_input: { agent_type: 'fixture-standard-role', message: `${ANNOTATION}\nDo the work.`, ...tool_input },
});
function readEvents(home) {
  const dir = path.join(home, 'logs');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).flatMap(name => fs.readFileSync(path.join(dir, name), 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l)));
}
// features-helpers' fixture role names (fixture-economy-role, ...) are policy-only; hooks call
// discoverHostRoles(), which reads real ~/.claude/agents or ~/.codex/agents files, so tests that
// need a route to actually be attempted must also create those role files under the fixture HOME.
function installFixtureRoles(home, host) {
  const dir = host === 'claude' ? path.join(home, '.claude', 'agents') : path.join(home, '.codex', 'agents');
  const ext = host === 'claude' ? '.md' : '.toml';
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  for (const role of ['fixture-economy-role', 'fixture-standard-role', 'fixture-strong-role']) fs.writeFileSync(path.join(dir, `${role}${ext}`), '', { mode: 0o600 });
}
function readLinkFiles(home) {
  const dir = path.join(home, 'links');
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) { for (const n of fs.readdirSync(full)) out.push(fs.readFileSync(path.join(full, n), 'utf8')); }
    else out.push(fs.readFileSync(full, 'utf8'));
  }
  return out;
}

// --- pure helpers ---
test('parseContextAnnotation extracts fields and rejects malformed markers', () => {
  assert.deepEqual(parseContextAnnotation('noise before [route scope=repository complete=no failures=3 impact=high exhaustive=yes] noise after'),
    { scope: 'repository', complete: false, previousFailures: 3, highImpact: true, exhaustive: true });
  assert.equal(parseContextAnnotation('plain text with no marker'), null);
  assert.equal(parseContextAnnotation('[route scope=local complete=yes failures=101]'), null);
  assert.equal(parseContextAnnotation(42), null);
});
test('truncateUtf8 never splits a multi-byte character', () => {
  const text = '\uAC00'.repeat(10); // each char is 3 bytes in UTF-8
  const cut = truncateUtf8(text, 7);
  assert.ok(Buffer.byteLength(cut) <= 7);
  assert.equal(Buffer.from(cut, 'utf8').toString('utf8'), cut); // round-trips cleanly, no replacement chars
  assert.equal(truncateUtf8('short', 100), 'short');
});
test('resolveSnapshotId falls back to a cwd hash outside a git repo', () => {
  const id = resolveSnapshotId('/tmp/definitely-not-a-git-repo-xyz');
  assert.match(id, /^[0-9a-f]{64}$/);
  assert.equal(resolveSnapshotId('/tmp/definitely-not-a-git-repo-xyz'), id); // deterministic
});

// --- pre-spawn: Claude ON rewrite ---
test('Claude ON: routable role + annotation rewrites subagent_type and model, preserves other tool_input fields', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude');
  const input = claudeInput();
  const result = await processHookEvent({ host: 'claude', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  assert.equal(s.calls.length, 1);
  assert.equal(result.output.hookSpecificOutput.hookEventName, 'PreToolUse');
  assert.equal(result.output.hookSpecificOutput.permissionDecision, 'allow');
  const updated = result.output.hookSpecificOutput.updatedInput;
  assert.equal(updated.subagent_type, 'fixture-economy-role');
  assert.equal(updated.model, 'haiku');
  assert.equal(updated.description, 'desc');
  assert.equal(updated.prompt, input.tool_input.prompt);
  assert.equal(Object.keys(result.output).length, 1);
  assert.deepEqual(Object.keys(result.output.hookSpecificOutput).sort(), ['hookEventName', 'permissionDecision', 'updatedInput']);
});

// --- pre-spawn: Codex ON rewrite (agent_type only, model untouched) ---
test('Codex ON: rewrites agent_type only, never touches model', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'codex');
  // Codex silently ignores a spawn `model` argument (R1), so a realistic payload omits it; the hook
  // must never introduce one on rewrite.
  const input = codexInput();
  const result = await processHookEvent({ host: 'codex', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  const updated = result.output.hookSpecificOutput.updatedInput;
  assert.equal(updated.agent_type, 'fixture-economy-role');
  assert.equal(Object.hasOwn(updated, 'model'), false);
  assert.equal(Object.hasOwn(updated, 'reasoning'), false);
});
test('Codex ON with a model already fixed by the caller (modelLocked) never rewrites', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'codex');
  const input = codexInput({ tool_input: { model: 'some-model-id' } });
  const result = await processHookEvent({ host: 'codex', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  assert.equal(result.output, null);
  assert.equal(result.telemetry.reason, 'MODEL_LOCKED');
  assert.equal(s.calls.length, 0);
});

// --- SHADOW ---
test('SHADOW mode: no output, but link and hook event are still recorded', async t => {
  const s = setup({ featureMode: 'shadow', provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude');
  const result = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: claudeInput(), home: s.home, env: s.env, layer: s.layer });
  assert.equal(result.output, null);
  assert.equal(s.calls.length, 1);
  assert.ok(readLinkFiles(s.home).length >= 1);
  const events = readEvents(s.home).filter(e => e.kind === 'hook');
  assert.ok(events.length >= 1);
  assert.equal(events[0].mode, 'shadow');
});

// --- immediate pass-through ---
for (const [name, options] of [['global OFF', { mode: 'off' }], ['router OFF (featureMode off)', { featureMode: 'off' }]]) {
  test(`immediate pass-through: ${name} -> no output, no provider call, no file writes`, async t => {
    const s = setup(options);
    t.after(s.cleanup);
    const before = fs.existsSync(s.home) ? fs.readdirSync(s.home, { recursive: true }).length : 0;
    const result = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: claudeInput(), home: s.home, env: s.env, layer: s.layer });
    assert.equal(result.output, null);
    assert.equal(s.calls.length, 0);
    const after = fs.readdirSync(s.home, { recursive: true }).length;
    assert.equal(after, before);
  });
}
test('immediate pass-through: POINTSMAN_DISABLE=1 -> no output, no provider call, no file writes', async t => {
  const s = setup(); t.after(s.cleanup);
  const disabledEnv = { ...s.env, POINTSMAN_DISABLE: '1' };
  const engine = createDecisionEngine({ home: s.home, env: disabledEnv, provider: async () => { throw new Error('must not be called'); } });
  const layer = createControlLayer({ home: s.home, env: disabledEnv, engine });
  const before = fs.readdirSync(s.home, { recursive: true }).length;
  const result = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: claudeInput(), home: s.home, env: disabledEnv, layer });
  assert.equal(result.output, null);
  const after = fs.readdirSync(s.home, { recursive: true }).length;
  assert.equal(after, before);
  engine.close();
});

// --- gating before any network call ---
test('ROLE_NOT_ROUTABLE: default general-purpose and an unlisted role make no provider call', async t => {
  const s = setup(); t.after(s.cleanup);
  const withoutType = claudeInput(); delete withoutType.tool_input.subagent_type;
  const r1 = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: withoutType, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r1.output, null); assert.equal(r1.telemetry.reason, 'ROLE_NOT_ROUTABLE'); assert.equal(r1.telemetry.original_role, 'general-purpose');
  const withVerifier = claudeInput({ tool_input: { subagent_type: 'verifier' } });
  const r2 = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: withVerifier, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r2.output, null); assert.equal(r2.telemetry.reason, 'ROLE_NOT_ROUTABLE');
  assert.equal(s.calls.length, 0);
});
test('NO_CONTEXT_ANNOTATION: missing marker makes no provider call', async t => {
  const s = setup(); t.after(s.cleanup);
  const input = claudeInput({ tool_input: { prompt: 'Just do the work, no marker here.' } });
  const r = await processHookEvent({ host: 'claude', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r.output, null); assert.equal(r.telemetry.reason, 'NO_CONTEXT_ANNOTATION');
  assert.equal(s.calls.length, 0);
});
test('modelLocked (tool_input.model present) blocks rewrite with no provider call', async t => {
  const s = setup(); t.after(s.cleanup);
  const input = claudeInput({ tool_input: { model: 'sonnet' } });
  const r = await processHookEvent({ host: 'claude', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r.output, null); assert.equal(r.telemetry.reason, 'MODEL_LOCKED');
  assert.equal(s.calls.length, 0);
});
test('recommended role equals original role: decision made, but no output', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude');
  const input = claudeInput({ tool_input: { subagent_type: 'fixture-economy-role' } }); // already the economy target
  const r = await processHookEvent({ host: 'claude', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r.output, null);
  assert.equal(s.calls.length, 1); // route was still called
});

// --- fail-open on provider errors, malformed JSON, timeout, oversized stdin ---
test('provider throwing does not propagate and produces no output', async t => {
  const s = setup({ provider: async () => { throw new Error('boom'); } }); t.after(s.cleanup);
  const r = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: claudeInput(), home: s.home, env: s.env, layer: s.layer });
  assert.equal(r.output, null);
});
test('runHookCli: malformed JSON on stdin yields no output', async t => {
  const s = setup({ mode: 'on', featureMode: 'on' }); t.after(s.cleanup);
  let wrote = '';
  await runHookCli({ host: 'claude', event: 'pre-spawn', home: s.home, env: s.env, stdin: Readable.from(['not json {{']), write: t2 => { wrote += t2; } });
  assert.equal(wrote, '');
});
test('runHookCli: oversized stdin yields no output', async t => {
  const s = setup({ mode: 'on', featureMode: 'on' }); t.after(s.cleanup);
  let wrote = '';
  const bigChunk = Buffer.alloc(300 * 1024, 65);
  await runHookCli({ host: 'claude', event: 'pre-spawn', home: s.home, env: s.env, stdin: Readable.from([bigChunk]), write: t2 => { wrote += t2; } });
  assert.equal(wrote, '');
});
test('runHookCli: a stdin that never ends times out and yields no output', async t => {
  const s = setup({ mode: 'on', featureMode: 'on' }); t.after(s.cleanup);
  let wrote = '';
  const neverEnding = new Readable({ read() { /* never pushes, never ends */ } });
  const start = Date.now();
  await runHookCli({ host: 'claude', event: 'pre-spawn', home: s.home, env: s.env, stdin: neverEnding, write: t2 => { wrote += t2; }, timeoutMs: 100 });
  assert.equal(wrote, '');
  assert.ok(Date.now() - start < 2000);
  neverEnding.destroy();
});

// --- no deny/ask ever appears ---
test('no branch ever produces deny/ask in output', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude'); installFixtureRoles(s.home, 'codex');
  const scenarios = [
    ['claude', 'pre-spawn', claudeInput()],
    ['codex', 'pre-spawn', codexInput()],
    ['claude', 'pre-spawn', claudeInput({ tool_input: { model: 'sonnet' } })],
    ['claude', 'pre-spawn', claudeInput({ tool_input: { prompt: 'no marker' } })],
    ['claude', 'subagent-start', { agent_id: 'a1', agent_type: 'x', session_id: 's1' }],
    ['codex', 'subagent-start', { agent_id: 'a1', agent_type: 'fixture-economy-role', session_id: 'sess-1' }],
    ['claude', 'post-spawn', { tool_use_id: 'tu-1', tool_response: { agentId: 'agent-1' } }],
    ['claude', 'subagent-stop', { agent_id: 'agent-1' }],
  ];
  for (const [host, event, input] of scenarios) {
    const r = await processHookEvent({ host, event, input, home: s.home, env: s.env, layer: s.layer });
    const text = JSON.stringify(r.output ?? {});
    assert.ok(!/"permissionDecision":"(deny|ask)"/.test(text), `${host}/${event} must not deny/ask`);
  }
});

// --- post-spawn alias -> subagent-stop host_review outcome ---
test('post-spawn aliases tool_use_id to agent_id; subagent-stop then records a host_review outcome', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude');
  const store = createTrainingStore({ home: s.home }); store.setCapture(true);
  const links = createLinkIndex({ home: s.home });
  const pre = await processHookEvent({ host: 'claude', event: 'pre-spawn', input: claudeInput(), home: s.home, env: s.env, layer: s.layer, links, store });
  assert.ok(pre.telemetry.decision_id);
  await processHookEvent({ host: 'claude', event: 'post-spawn', input: { tool_use_id: 'tu-1', tool_response: { agentId: 'agent-42' } }, home: s.home, env: s.env, layer: s.layer, links, store });
  const stopResult = await processHookEvent({ host: 'claude', event: 'subagent-stop', input: { agent_id: 'agent-42' }, home: s.home, env: s.env, layer: s.layer, links, store });
  assert.equal(stopResult.telemetry.reason, 'RECORDED');
  const scan = store.scan();
  const outcome = scan.events.find(e => e.kind === 'outcomes' && e.data.decision_id === pre.telemetry.decision_id);
  assert.ok(outcome, 'expected an outcome linked to the pre-spawn decision');
  assert.equal(outcome.data.source, 'host_review');
  assert.equal(outcome.data.host_review, 'uncertain');
});

// --- Codex pending heuristic ---
test('Codex pending: matches same session+role within 60s, ignores other sessions, expires after 60s', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'codex');
  let clock = 1000;
  const now = () => clock;
  const links = createLinkIndex({ home: s.home, now });
  const pre = await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput(), home: s.home, env: s.env, layer: s.layer, now, links });
  assert.ok(pre.telemetry.decision_id);
  const wrongSession = await processHookEvent({ host: 'codex', event: 'subagent-start',
    input: { agent_id: 'agent-x', agent_type: 'fixture-economy-role', session_id: 'other-session' }, home: s.home, env: s.env, layer: s.layer, now, links });
  assert.equal(wrongSession.telemetry.reason, 'NO_PENDING_MATCH');
  clock += 61000; // past the 60s retention window
  const expired = await processHookEvent({ host: 'codex', event: 'subagent-start',
    input: { agent_id: 'agent-y', agent_type: 'fixture-economy-role', session_id: 'sess-1' }, home: s.home, env: s.env, layer: s.layer, now, links });
  assert.equal(expired.telemetry.reason, 'NO_PENDING_MATCH');
});
test('Codex pending: matching session+role links with the heuristic marker', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'codex');
  let clock = 1000; const now = () => clock;
  const links = createLinkIndex({ home: s.home, now });
  const pre = await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput(), home: s.home, env: s.env, layer: s.layer, now, links });
  clock += 1000;
  const start = await processHookEvent({ host: 'codex', event: 'subagent-start',
    input: { agent_id: 'agent-z', agent_type: 'fixture-economy-role', session_id: 'sess-1' }, home: s.home, env: s.env, layer: s.layer, now, links });
  assert.equal(start.telemetry.reason, 'LINKED');
  assert.equal(start.telemetry.link, 'heuristic');
  assert.equal(start.telemetry.decision_id, pre.telemetry.decision_id);
});
test('Claude subagent-start is a no-op', async t => {
  const s = setup(); t.after(s.cleanup);
  const r = await processHookEvent({ host: 'claude', event: 'subagent-start', input: { agent_id: 'a1', agent_type: 'x', session_id: 's1' }, home: s.home, env: s.env, layer: s.layer });
  assert.equal(r.output, null); assert.equal(r.telemetry.reason, 'NOOP');
});

// --- content-free storage: no prompt text ever lands in logs, links or pending files ---
test('logs, links and pending files never contain prompt content', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude'); installFixtureRoles(s.home, 'codex');
  const marker = 'SECRET_MARKER_PROMPT_TEXT_98216';
  const input = claudeInput({ tool_input: { prompt: `${ANNOTATION}\n${marker}` } });
  await processHookEvent({ host: 'claude', event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
  const codexIn = codexInput({ tool_input: { message: `${ANNOTATION}\n${marker}` } });
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexIn, home: s.home, env: s.env, layer: s.layer });
  const allText = [...readEvents(s.home).map(e => JSON.stringify(e)), ...readLinkFiles(s.home)].join('\n');
  assert.ok(!allText.includes(marker));
});

// --- CLI subprocess: global off means no output at all, exit 0 ---
test('CLI subprocess: hook with global off produces no stdout and exits 0', () => {
  const home = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'pointsman-hook-cli-'));
  fs.chmodSync(home, 0o700);
  const input = JSON.stringify(claudeInput());
  const result = spawnSync(process.execPath, [bin, 'hook', '--host', 'claude', '--event', 'pre-spawn', '--home', home], {
    input, encoding: 'utf8', env: { PATH: process.env.PATH, HOME: home },
  });
  assert.equal(result.status, 0);
  assert.equal(result.stdout, '');
  fs.rmSync(home, { recursive: true, force: true });
});

test('pre-spawn only acts on subagent spawn tools; any other tool_name makes no provider call', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude'); installFixtureRoles(s.home, 'codex');
  for (const [host, input] of [['claude', claudeInput({ tool_name: 'Bash' })], ['claude', claudeInput({ tool_name: undefined })], ['codex', codexInput({ tool_name: 'shell' })]]) {
    const result = await processHookEvent({ host, event: 'pre-spawn', input, home: s.home, env: s.env, layer: s.layer });
    assert.equal(result.output, null);
  }
  assert.equal(s.calls.length, 0);
  const ok = await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'agents.spawn_agent' }), home: s.home, env: s.env, layer: s.layer });
  assert.equal(s.calls.length, 1); assert.ok(ok.telemetry);
  // codex-cli 0.154.0 actually sends the namespace concatenated without a separator (measured 2026-09-23).
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'agentsspawn_agent' }), home: s.home, env: s.env, layer: s.layer });
  assert.equal(s.calls.length, 2);
});
test('CLI subprocess: bad hook arguments or an old runtime never exit non-zero (exit 2 would block the host tool call)', () => {
  const home = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'pointsman-hook-cli-'));
  fs.chmodSync(home, 0o700);
  const input = JSON.stringify(claudeInput());
  for (const args of [['hook', '--host', 'claude', '--event', 'pre-spawn', '--bogus', '1'], ['hook', '--host', 'nope', '--event', 'x'], ['hook'], ['hook', '--home', 'relative/path', '--host', 'claude', '--event', 'pre-spawn']]) {
    const result = spawnSync(process.execPath, [bin, ...args], { input, encoding: 'utf8', env: { PATH: process.env.PATH, HOME: home } });
    assert.equal(result.status, 0, args.join(' ') + ' ' + result.stderr);
    assert.equal(result.stdout, '');
  }
  fs.rmSync(home, { recursive: true, force: true });
});

test('runHookCli records content-free diagnostics for rejected stdin and timeouts (no output, no content)', async t => {
  const s = setup({ mode: 'on', featureMode: 'on' }); t.after(s.cleanup);
  await runHookCli({ host: 'codex', event: 'pre-spawn', home: s.home, env: s.env, stdin: Readable.from(['not json SECRET_MARKER {{']), write: () => {} });
  const neverEnding = new Readable({ read() {} });
  await runHookCli({ host: 'codex', event: 'pre-spawn', home: s.home, env: s.env, stdin: neverEnding, write: () => {}, timeoutMs: 100 });
  neverEnding.destroy();
  const hookEvents = readEvents(s.home).filter(e => e.kind === 'hook');
  assert.deepEqual(hookEvents.map(e => e.reason), ['INVALID_STDIN', 'HOOK_TIMEOUT']);
  assert.equal(JSON.stringify(hookEvents).includes('SECRET_MARKER'), false);
});
test('pre-spawn with an unexpected tool_name records a sanitized tool_name for diagnosis', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'weird/tool name<script>' }), home: s.home, env: s.env, layer: s.layer });
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'collab.spawn' }), home: s.home, env: s.env, layer: s.layer });
  const ev = readEvents(s.home).filter(e => e.kind === 'hook');
  assert.equal(ev[0].reason, 'INVALID_HOOK_INPUT'); assert.equal(ev[0].tool_name, null);
  assert.equal(ev[1].tool_name, 'collab.spawn');
});

test('NO_CONTEXT_ANNOTATION records only the prompt form (plain/opaque), never content', async t => {
  const s = setup({ provider: p => response(p, { intent: 'edit', difficulty: [1, 0, 0, 0, 0] }) }); t.after(s.cleanup);
  installFixtureRoles(s.home, 'codex');
  // codex-cli 0.154.0 passes spawn_agent.message as an opaque Fernet-like token (measured 2026-09-23).
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'agentsspawn_agent', tool_input: { message: 'gAAAAABqSECRETtokenAbc_-=' } }), home: s.home, env: s.env, layer: s.layer });
  await processHookEvent({ host: 'codex', event: 'pre-spawn', input: codexInput({ tool_name: 'agentsspawn_agent', tool_input: { message: 'plain text SECRET_MARKER' } }), home: s.home, env: s.env, layer: s.layer });
  const ev = readEvents(s.home).filter(e => e.kind === 'hook');
  assert.deepEqual(ev.map(e => [e.reason, e.prompt_form]), [['NO_CONTEXT_ANNOTATION', 'opaque'], ['NO_CONTEXT_ANNOTATION', 'plain']]);
  assert.equal(JSON.stringify(ev).includes('SECRET'), false);
  assert.equal(s.calls.length, 0);
});

test('L3: runHookCli with provider=laya and no resident server passes through fast, no output, no direct spawn', async t => {
  const s = setup(); t.after(s.cleanup);
  installFixtureRoles(s.home, 'claude');
  atomicWrite(path.join(s.home, 'providers.json'), JSON.stringify({ version: 1, provider: 'laya', laya: {
    python: '/usr/bin/python3', modelPath: s.home, model: 'laya/base', checkpoint: 'a'.repeat(64), runtimeVersion: '0.3.4', device: 'cpu' } }));
  let wrote = '';
  const t0 = Date.now();
  await runHookCli({ host: 'claude', event: 'pre-spawn', home: s.home, env: s.env, stdin: Readable.from([JSON.stringify(claudeInput())]), write: t2 => { wrote += t2; } });
  assert.equal(wrote, '');
  assert.ok(Date.now() - t0 < 100, 'a hook must never wait for a worker to load');
  const hookEvent = readEvents(s.home).find(e => e.kind === 'hook' && e.event === 'pre-spawn');
  assert.ok(hookEvent);
  assert.equal(hookEvent.applied, false);
});
