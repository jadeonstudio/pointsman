#!/usr/bin/env node
// L1 real-weights end-to-end reproduction script.
// Offline, no network, no writes outside a throwaway temp POINTSMAN_HOME. Never touches the real
// POINTSMAN_HOME/~/.codex/~/.claude and never modifies the checkpoint directory it is pointed at.
//
//   <laya-venv>/bin/python scripts/laya-bench.py is the load-time/agreement micro-benchmark;
//   this script instead drives the real pointsman decision engine end to end (engine.prepare
//   for the resident worker + cold start, engine.decide for warm route decisions) against a real
//   checkpoint, so it exercises the exact code path Codex/Claude would use, not a standalone harness.
//
// Usage:
//   node scripts/laya-e2e.mjs --python /abs/laya-venv/bin/python --model-path /abs/checkpoint-dir \
//     [--precision fp32|fp16] [--device mps|cpu|cuda] [--warm 10]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createDecisionEngine } from '../src/engine.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

// pointsman route questions (src/routing.mjs ROUTE_QUESTIONS), identical wording to scripts/laya-bench.py.
const ROUTE_QUESTIONS = {
  intent: { type: 'choice', instructions: 'Classify the work actually required. Use other when the request is unclear or outside these categories.', criteria: {
    explain: 'Explain or locate existing code; do not change behavior', edit: 'Write or change a bounded piece of code or documentation',
    debug: 'Investigate the cause of a failure', operate: 'Operate infrastructure, deploy, or modify live data',
    research: 'Research information outside the repository', architecture: 'Cross-module design, whole-repository audit or major refactoring',
    other: 'Insufficient evidence or none of these intents' } },
  difficulty: { type: 'score', instructions: 'Estimate the reasoning needed to finish, not prompt length. Account for unresolved dependencies and unknown scope. Do not infer that a short request is easy.', criteria: [
    '1: Mechanical, exact edit with explicit location and no behavioral change', '2: Small local change with explicit requirements and known validation',
    '3: Moderate implementation requiring several related steps', '4: Difficult debugging or interacting components needing substantial investigation',
    '5: Deep reasoning, unknown repository-wide impact, architecture or long-horizon planning'] },
  risk: { type: 'choice', instructions: 'Classify consequence and uncertainty, not permission. Safe requires evidence of local, reversible impact. Missing context is unknown. Risk is independent of difficulty.', criteria: {
    safe: 'Known local, reversible work with no security, production, payment or financial impact',
    caution: 'Material uncertainty or changes needing additional review',
    high: 'Irreversible operation, production, credentials, permissions, financial or payment consequences',
    unknown: 'Not enough context to assess impact' } },
};
const TASKS = [
  "Fix the typo 'teh' -> 'the' in README.md line 12. No code changes.",
  'Explain where the retry policy for HTTP requests is implemented.',
  'Add a --dry-run flag to the export command and a unit test for it.',
  'The nightly build fails with a segfault in the image decoder; find the cause.',
  'Rotate the production database credentials and redeploy all services.',
  'Redesign the plugin system so modules can be loaded lazily across the whole repo.',
  "README.md 12\uBC88\uC9F8 \uC904\uC758 \uC624\uD0C0 'teh'\uB97C 'the'\uB85C \uACE0\uCCD0\uC918. \uCF54\uB4DC\uB294 \uBC14\uAFB8\uC9C0 \uC54A\uC74C.",
  'HTTP \uC694\uCCAD \uC7AC\uC2DC\uB3C4 \uC815\uCC45\uC774 \uC5B4\uB514\uC5D0 \uAD6C\uD604\uB3FC \uC788\uB294\uC9C0 \uC124\uBA85\uD574\uC918.',
  'export \uBA85\uB839\uC5D0 --dry-run \uC635\uC158\uACFC \uB2E8\uC704 \uD14C\uC2A4\uD2B8\uB97C \uCD94\uAC00\uD574\uC918.',
  '\uC57C\uAC04 \uBE4C\uB4DC\uAC00 \uC774\uBBF8\uC9C0 \uB514\uCF54\uB354\uC5D0\uC11C segfault\uB85C \uC2E4\uD328\uD574. \uC6D0\uC778\uC744 \uCC3E\uC544\uC918.',
  '\uC6B4\uC601 DB \uBE44\uBC00\uBC88\uD638\uB97C \uAD50\uCCB4\uD558\uACE0 \uBAA8\uB4E0 \uC11C\uBE44\uC2A4\uB97C \uC7AC\uBC30\uD3EC\uD574\uC918.',
  '\uD50C\uB7EC\uADF8\uC778 \uC2DC\uC2A4\uD15C\uC744 \uC800\uC7A5\uC18C \uC804\uCCB4\uC5D0\uC11C \uC9C0\uC5F0 \uB85C\uB529\uB418\uB3C4\uB85D \uC7AC\uC124\uACC4\uD574\uC918.',
];
const stateFor = task => ({ task, context: { complete: true, scope: 'local', previousFailures: 0, highImpact: false, modelLocked: false, exhaustive: false } });
const requestFor = task => ({ purpose: 'route', risk: 'routine', state: stateFor(task), questions: ROUTE_QUESTIONS });
const summarize = ts => {
  const sorted = [...ts].sort((a, b) => a - b);
  const n = sorted.length;
  return { n, p50: sorted[Math.floor(0.5 * n)], p95: sorted[Math.min(n - 1, Math.floor(0.95 * n))], min: sorted[0], max: sorted[n - 1] };
};
const topAnswer = a => (a.value !== undefined ? a.value : null);

async function main() {
  const { values } = parseArgs({ options: {
    python: { type: 'string' }, 'model-path': { type: 'string' }, precision: { type: 'string', default: 'fp32' },
    device: { type: 'string', default: 'mps' }, warm: { type: 'string', default: '10' },
  } });
  if (!values.python || !values['model-path']) {
    process.stderr.write('usage: node scripts/laya-e2e.mjs --python /abs/path --model-path /abs/checkpoint-dir [--precision fp32|fp16] [--device mps|cpu|cuda] [--warm 10]\n');
    process.exitCode = 2; return;
  }
  if (!['fp32', 'fp16'].includes(values.precision)) { process.stderr.write('--precision must be fp32 or fp16\n'); process.exitCode = 2; return; }
  const python = path.resolve(values.python), modelPath = path.resolve(values['model-path']);

  const fp = spawnSync(python, ['-I', path.join(ROOT, 'workers/laya_worker.py'), '--fingerprint', modelPath], { encoding: 'utf8', timeout: 60000 });
  if (fp.status !== 0) { process.stderr.write(`fingerprint failed: ${fp.stderr}\n`); process.exitCode = 1; return; }
  const checkpoint = JSON.parse(fp.stdout.trim().split('\n').pop()).checkpoint;

  // storage.mjs refuses any symlink component; os.tmpdir() itself is a symlink on macOS (/var -> private/var).
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'pointsman-laya-e2e-')));
  try {
    fs.writeFileSync(path.join(home, 'providers.json'), JSON.stringify({ version: 1, provider: 'laya', laya: {
      python, modelPath, model: 'laya/e2e', checkpoint, runtimeVersion: '0.3.4', device: values.device,
      startupTimeoutMs: 60000, idleTimeoutMs: 60000, precision: values.precision } }, null, 2) + '\n', { mode: 0o600 });
    fs.writeFileSync(path.join(home, 'config.json'), JSON.stringify({ version: 1, mode: 'on', model: 'jev-latest', timeoutMs: 10000,
      maxInputBytes: 24000, maxQuestions: 8, minConfidence: 0.85, minChoiceProbability: 0.8, noulCertainty: 0.95,
      maxCallsPerMinute: 60, maxInFlight: 4, circuitFailureThreshold: 3, circuitCooldownMs: 30000, telemetry: true }, null, 2) + '\n', { mode: 0o600 });

    const env = { ...process.env, POINTSMAN_HOME: home, HOME: home };
    const engine = createDecisionEngine({ home, env });
    try {
      const t0 = Date.now();
      const identity = await engine.prepare({ timeoutMs: 60000, resident: true, env });
      const coldMs = Date.now() - t0;

      const warmCount = Number(values.warm);
      const latencies = []; let firstMs = null; const answersByTask = [];
      for (let i = 0; i < warmCount + 1; i++) {
        const task = TASKS[i % TASKS.length];
        const t = Date.now();
        let captured = null;
        const result = await engine.decide(requestFor(task), { onEvaluated: n => { captured = n; } });
        const ms = Date.now() - t;
        if (result.reason === 'OFF' || result.answers === undefined) throw new Error(`unexpected decide() reason: ${result.reason}`);
        if (i === 0) { firstMs = ms; } else { latencies.push(ms); }
        if (i < TASKS.length) answersByTask.push({ task, answers: Object.fromEntries(Object.entries(captured.answers).map(([q, a]) => [q, topAnswer(a)])) });
      }
      process.stdout.write(JSON.stringify({
        python, modelPath, checkpoint, precision: values.precision, device: values.device,
        cold_ms: coldMs, first_decide_ms: firstMs, warm_latency_ms: summarize(latencies),
        identity, answers: answersByTask,
      }, null, 2) + '\n');
    } finally { engine.close(); }
  } finally {
    fs.rmSync(home, { recursive: true, force: true });
  }
}
await main();
