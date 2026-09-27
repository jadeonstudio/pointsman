import { fail, isObject } from './constants.mjs';

// Offline aggregation of paired EXECUTIONS. A SHADOW prediction is not an execution.
export function evaluatePairedRuns(input) {
  if (!isObject(input) || Object.keys(input).some(k => k !== 'cases') || !Array.isArray(input.cases) || !input.cases.length || input.cases.length > 10000) fail('INVALID_EVALUATION');
  const totals = { baseline: { elapsedMs: 0, outputTokens: 0, successes: 0 }, routed: { elapsedMs: 0, outputTokens: 0, successes: 0 } };
  let downgrades = 0, wrongDowngrades = 0, regressions = 0;
  for (const row of input.cases) {
    if (!isObject(row) || Object.keys(row).some(k => !['baseline', 'routed', 'downgraded'].includes(k)) || typeof row.downgraded !== 'boolean') fail('INVALID_EVALUATION');
    for (const name of ['baseline', 'routed']) {
      const run = row[name];
      if (!isObject(run) || Object.keys(run).some(k => !['executed', 'passed', 'elapsedMs', 'outputTokens'].includes(k)) ||
          run.executed !== true || typeof run.passed !== 'boolean' || !Number.isFinite(run.elapsedMs) || run.elapsedMs < 0 || run.elapsedMs > 86400000 ||
          !Number.isSafeInteger(run.outputTokens) || run.outputTokens < 0) fail('EXECUTION_EVIDENCE_REQUIRED');
      totals[name].elapsedMs += run.elapsedMs;
      totals[name].outputTokens += run.outputTokens;
      if (!Number.isSafeInteger(totals[name].outputTokens)) fail('EVALUATION_TOTAL_OVERFLOW');
      totals[name].successes += Number(run.passed);
    }
    if (row.baseline.passed && !row.routed.passed) regressions++;
    if (row.downgraded && row.baseline.passed) { downgrades++; if (!row.routed.passed) wrongDowngrades++; }
  }
  const saving = (baseline, routed) => baseline > 0 ? 1 - routed / baseline : null;
  // Exact one-sided 95% bound for ZERO observed errors, under independent Bernoulli trials.
  // Correlated turns from one task do not satisfy that assumption.
  const zeroErrorUpper95 = downgrades > 0 && wrongDowngrades === 0 ? 1 - Math.pow(0.05, 1 / downgrades) : null;
  return { cases: input.cases.length, totals, qualityRegressions: regressions,
    evaluatedDowngrades: downgrades, wrongDowngrades, wrongDowngradeRate: downgrades ? wrongDowngrades / downgrades : null,
    zeroErrorUpper95, elapsedReduction: saving(totals.baseline.elapsedMs, totals.routed.elapsedMs),
    outputTokenReduction: saving(totals.baseline.outputTokens, totals.routed.outputTokens),
    monetarySavings: null, automaticApproval: false,
    requirements: 'Use paired executions on identical snapshots and independent tasks; routed totals must include classification, fallback, verification and retry costs. Shadow agreements cannot establish task quality.' };
}
