import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { resolveHome, ensureDir, noSymlinks, readText, atomicWrite } from '../storage.mjs';
import { ID, PURPOSES, fail, isObject } from '../constants.mjs';
import { digest, encode, only, text, id as validId, validateTarget } from './schema.mjs';

// Pre-registered assertion commands. An agent may only select a name registered by a human
// in an interactive terminal; it never supplies argv, labels, or the pass/fail outcome itself.
export const RUNNER_CONFIG_VERSION = 1;
const MAX_CHECKS = 256;

function configPath(home) { return path.join(home, 'runner.json'); }

export function validateRunnerCheckName(name) {
  if (typeof name !== 'string' || !ID.test(name)) fail('INVALID_RUNNER_CHECK_NAME');
}
export function validateRunnerCheck(check) {
  only(check, ['argv', 'cwd', 'timeoutMs', 'purpose', 'question_id', 'pass_label', 'fail_label'], ['argv', 'timeoutMs']);
  if (!Array.isArray(check.argv) || check.argv.length < 1 || check.argv.length > 32) fail('INVALID_RUNNER_CHECK');
  for (const arg of check.argv) if (typeof arg !== 'string' || !arg.length || Buffer.byteLength(arg) > 1024 || /[\x00-\x1f]/.test(arg)) fail('INVALID_RUNNER_CHECK');
  if (check.cwd !== undefined && (typeof check.cwd !== 'string' || !path.isAbsolute(check.cwd) || check.cwd.length > 4096)) fail('INVALID_RUNNER_CHECK');
  if (!Number.isInteger(check.timeoutMs) || check.timeoutMs < 100 || check.timeoutMs > 600000) fail('INVALID_RUNNER_CHECK');
  if (check.purpose !== undefined && !PURPOSES.includes(check.purpose)) fail('INVALID_RUNNER_CHECK');
  const hasQuestion = check.question_id !== undefined;
  if (hasQuestion) {
    text(check.question_id, 64);
    if (check.purpose === undefined || check.pass_label === undefined || check.fail_label === undefined) fail('INVALID_RUNNER_CHECK');
    // route questions are intent/difficulty/risk, not task outcome; only human correction may label them.
    if (check.purpose === 'route') fail('RUNNER_ROUTE_LABEL_REFUSED');
    for (const v of [check.pass_label, check.fail_label]) if (!['string', 'number', 'boolean'].includes(typeof v)) fail('INVALID_RUNNER_CHECK');
    if (encode(check.pass_label) === encode(check.fail_label)) fail('INVALID_RUNNER_CHECK');
  } else if (check.pass_label !== undefined || check.fail_label !== undefined) fail('INVALID_RUNNER_CHECK');
  return check;
}
export function validateRunnerConfig(c) {
  only(c, ['version', 'checks'], ['version', 'checks']);
  if (c.version !== RUNNER_CONFIG_VERSION || !isObject(c.checks) || Object.keys(c.checks).length > MAX_CHECKS) fail('INVALID_RUNNER_CONFIG');
  for (const [name, check] of Object.entries(c.checks)) { validateRunnerCheckName(name); validateRunnerCheck(check); }
  return c;
}
export function readRunnerConfig(home = resolveHome()) {
  const raw = readText(configPath(home), { optional: true, privateFile: true, maxBytes: 131072 });
  if (raw === null) return { version: RUNNER_CONFIG_VERSION, checks: {} };
  let c; try { c = JSON.parse(raw); } catch { fail('INVALID_RUNNER_CONFIG'); }
  return validateRunnerConfig(c);
}
function writeRunnerConfig(home, config) {
  validateRunnerConfig(config);
  ensureDir(home, true);
  const file = configPath(home); noSymlinks(file);
  const old = readText(file, { optional: true, privateFile: true, maxBytes: 131072 });
  atomicWrite(file, JSON.stringify(config, null, 2) + '\n', { expected: old, mode: 0o600 });
}
function summarize(name, check) {
  return `runner allow\n  name: ${name}\n  argv: ${JSON.stringify(check.argv)}\n  cwd: ${check.cwd ?? '(default)'}\n  timeoutMs: ${check.timeoutMs}\n  purpose: ${check.purpose ?? '(none)'}\n  question_id: ${check.question_id ?? '(none)'}\n  pass_label: ${check.question_id !== undefined ? JSON.stringify(check.pass_label) : '(none)'}\n  fail_label: ${check.question_id !== undefined ? JSON.stringify(check.fail_label) : '(none)'}\n`;
}
// TTY checks are a same-user local-authorship signal only, not proof of a human operator;
// a PTY-attached agent also has a TTY. See SECURITY.md threat model.
export async function allowRunner({ home = resolveHome(), name, check, replace = false, isStdinTTY, isStdoutTTY, prompt, write = () => {} }) {
  if (!isStdinTTY || !isStdoutTTY) fail('HUMAN_TTY_REQUIRED');
  validateRunnerCheckName(name); validateRunnerCheck(check);
  const config = readRunnerConfig(home);
  if (Object.hasOwn(config.checks, name) && !replace) fail('RUNNER_CHECK_EXISTS');
  write(summarize(name, check));
  const typed = await prompt(`Type the check name "${name}" to confirm registration: `);
  if (typed !== name) fail('RUNNER_CONFIRMATION_MISMATCH');
  config.checks[name] = check;
  writeRunnerConfig(home, config);
  return { stored: true, name };
}
export async function removeRunner({ home = resolveHome(), name, isStdinTTY, isStdoutTTY, prompt }) {
  if (!isStdinTTY || !isStdoutTTY) fail('HUMAN_TTY_REQUIRED');
  validateRunnerCheckName(name);
  const config = readRunnerConfig(home);
  if (!Object.hasOwn(config.checks, name)) fail('RUNNER_CHECK_NOT_FOUND');
  const typed = await prompt(`Type the check name "${name}" to confirm removal: `);
  if (typed !== name) fail('RUNNER_CONFIRMATION_MISMATCH');
  delete config.checks[name];
  writeRunnerConfig(home, config);
  return { stored: true, name, removed: true };
}
export function listRunners(home = resolveHome()) {
  const config = readRunnerConfig(home);
  return { version: config.version, checks: config.checks };
}
function runCommand(spawnImpl, check) {
  return new Promise(resolve => {
    let settled = false;
    let child;
    // The registered check needs the caller's ordinary environment (PATH, project variables) but never Jev's own credential.
    const { TYPESAFE_API_KEY, ...env } = process.env;
    try { child = spawnImpl(check.argv[0], check.argv.slice(1), { cwd: check.cwd, env, shell: false, stdio: 'ignore' }); }
    catch { resolve({ exitCode: null, timedOut: false, spawnError: true }); return; }
    const timer = setTimeout(() => {
      if (settled) return; settled = true;
      try { child.kill('SIGKILL'); } catch { /* best effort */ }
      resolve({ exitCode: null, timedOut: true, spawnError: false });
    }, check.timeoutMs);
    child.once('error', () => { if (settled) return; settled = true; clearTimeout(timer); resolve({ exitCode: null, timedOut: false, spawnError: true }); });
    child.once('exit', code => { if (settled) return; settled = true; clearTimeout(timer); resolve({ exitCode: code, timedOut: false, spawnError: false }); });
  });
}
export async function verifyRunner({ store, home = resolveHome(), decisionId, checkName, spawnImpl = spawn }) {
  if (!store.config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
  validId(decisionId); validateRunnerCheckName(checkName);
  const config = readRunnerConfig(home);
  const check = config.checks[checkName];
  if (!check) fail('RUNNER_CHECK_NOT_FOUND');
  const snapshot = store.scan();
  const decisionEvent = snapshot.events.find(e => e.kind === 'decisions' && e.data.decision_id === decisionId);
  if (!decisionEvent) fail('DECISION_NOT_FOUND');
  const d = decisionEvent.data;
  if (check.purpose !== undefined && check.purpose !== d.request.purpose) fail('RUNNER_PURPOSE_MISMATCH');
  if (check.question_id !== undefined) {
    const question = d.request.questions[check.question_id];
    if (!question) fail('RUNNER_QUESTION_NOT_FOUND');
    try { validateTarget(question, check.pass_label); validateTarget(question, check.fail_label); }
    catch { fail('RUNNER_LABEL_INVALID_FOR_QUESTION'); }
  }
  const started_at = new Date().toISOString();
  const start = performance.now();
  const { exitCode, timedOut } = await runCommand(spawnImpl, check);
  const latency_ms = Math.round(performance.now() - start);
  const commandPassed = exitCode === 0 && !timedOut;
  // A timeout, spawn failure or signal death is not evidence for either answer; only a normal exit maps to a label.
  const exited = Number.isInteger(exitCode) && !timedOut;
  const evidence_ref = 'sha256:' + digest({ argv: check.argv, cwd: check.cwd ?? null, decision_id: decisionId, started_at, exit_status: timedOut || exitCode === null ? null : exitCode, check_name: checkName });
  const checks = [{ kind: 'command', passed: commandPassed, required: true, scope: 'task', evidence_ref, ...(Number.isInteger(exitCode) ? { exit_status: exitCode } : {}) }];
  const labels = [];
  if (check.question_id !== undefined && exited) {
    // The label assertion attests the check RAN and produced a value, not that the value is "pass".
    checks.push({ kind: 'label', passed: true, required: true, scope: 'task', evidence_ref, question_id: check.question_id });
    labels.push({ question_id: check.question_id, value: commandPassed ? check.pass_label : check.fail_label, source: 'objective', label_confidence: 1, evidence_ref });
  }
  const outcome = { decision_id: decisionId, execution_id: randomUUID(), executed: false, final: true, source: 'runner',
    executed_answers: {}, metrics: { latency_ms, timeout: timedOut }, checks, labels };
  const result = store.outcome(outcome);
  return { ...result, decision_id: decisionId, check: checkName, passed: commandPassed, exit_status: Number.isInteger(exitCode) ? exitCode : null, timed_out: timedOut, latency_ms };
}
