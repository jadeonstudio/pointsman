import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseRoute, chooseRouteByDecision, labelTier, tierDistribution, ROUTE_QUESTIONS } from '../src/routing.mjs';
import { FEATURE_DEFAULTS, INTENTS } from '../src/feature-policy.mjs';

const POLICY = { ...structuredClone(FEATURE_DEFAULTS.router),
  profiles: { codex: { economy: { role: 'economy-role' }, standard: { role: 'standard-role' }, strong: { role: 'strong-role' } }, claude: {} } };
const INPUT = { host: 'codex', availableRoles: ['economy-role', 'standard-role', 'strong-role'], availableSkills: [] };
const RISK_KEYS = Object.keys(ROUTE_QUESTIONS.risk.criteria);
const HOST_REASONS = new Set(['UNKNOWN_INTENT', 'RISK_REQUIRES_HOST', 'INTENT_REQUIRES_HOST']);

function oneHotChoice(criteria, value) {
  const probabilities = Object.fromEntries(Object.keys(criteria).map(k => [k, k === value ? 1 : 0]));
  return { type: 'choice', value, confidence: 1, selectedProbability: 1, probabilities };
}
function oneHotScore(index) {
  const probabilities = Object.fromEntries([0, 1, 2, 3, 4].map(i => [String(i), i === index ? 1 : 0]));
  return { type: 'score', value: index, confidence: 1, probabilities };
}

test('labelTier agrees with chooseRoute on every one-hot (intent, difficulty, risk) combination', () => {
  for (const intent of INTENTS) {
    for (let difficulty = 0; difficulty <= 4; difficulty++) {
      for (const risk of RISK_KEYS) {
        const answers = { intent: oneHotChoice(ROUTE_QUESTIONS.intent.criteria, intent), difficulty: oneHotScore(difficulty), risk: oneHotChoice(ROUTE_QUESTIONS.risk.criteria, risk) };
        const result = chooseRoute(answers, INPUT, POLICY);
        const expected = labelTier(intent, difficulty, risk, POLICY);
        if (expected === 'host') {
          assert.ok(HOST_REASONS.has(result.reason), `${intent}/${difficulty}/${risk} expected a host reason, got ${result.reason}`);
        } else {
          assert.equal(result.route?.tier, expected, `${intent}/${difficulty}/${risk} expected tier ${expected}, got ${JSON.stringify(result)}`);
        }
      }
    }
  }
});

test('tierDistribution sums independent intent x difficulty x risk probabilities into the matching labelTier bucket', () => {
  // Two-way split on each axis so the hand computation stays checkable: intent {explain:.6, debug:.4},
  // difficulty {0:.7, 4:.3}, risk {safe:.9, high:.1}.
  const answers = {
    intent: { probabilities: { explain: .6, edit: 0, debug: .4, operate: 0, research: 0, architecture: 0, other: 0 } },
    difficulty: { probabilities: { '0': .7, '1': 0, '2': 0, '3': 0, '4': .3 } },
    risk: { probabilities: { safe: .9, caution: 0, high: .1, unknown: 0 } },
  };
  const dist = tierDistribution(answers, POLICY);
  // (explain,0,safe)=.6*.7*.9=.378 -> economy (displayed 1<=2)
  // (explain,4,safe)=.6*.3*.9=.162 -> displayed 5<=2? no; <=3.5? no -> strong
  // (debug,0,safe)=.4*.7*.9=.252 -> intent debug excludes economy+standard -> strong
  // (debug,4,safe)=.4*.3*.9=.108 -> strong
  // any *high(.1) -> host: .6*.1=.06 + .4*.1=.04 = .1
  assert.ok(Math.abs(dist.economy - .378) < 1e-9);
  assert.ok(Math.abs(dist.host - .1) < 1e-9);
  assert.ok(Math.abs(dist.strong - (.162 + .252 + .108)) < 1e-9);
  assert.ok(Math.abs(dist.standard - 0) < 1e-9);
  const total = dist.host + dist.economy + dist.standard + dist.strong;
  assert.ok(Math.abs(total - 1) < 1e-9);
});

function answers({ intent = 'edit', intentP, difficultyP, risk = 'safe', riskP } = {}) {
  return {
    intent: { value: intent, probabilities: intentP ?? Object.fromEntries(Object.keys(ROUTE_QUESTIONS.intent.criteria).map(k => [k, k === intent ? 1 : 0])) },
    difficulty: { value: 0, probabilities: difficultyP ?? { '0': 1, '1': 0, '2': 0, '3': 0, '4': 0 } },
    risk: { value: risk, probabilities: riskP ?? Object.fromEntries(RISK_KEYS.map(k => [k, k === risk ? 1 : 0])) },
  };
}
const GATE = { method: 'decision-v1', tierCoverage: 0.9, maxHostProbability: 0.05 };

test('chooseRouteByDecision: HOST_PROBABLE when the host share exceeds maxHostProbability', () => {
  const a = answers({ riskP: { safe: .9, caution: 0, high: .1, unknown: 0 } }); // host share = .1 > .05
  const r = chooseRouteByDecision(a, INPUT, POLICY, GATE);
  assert.equal(r.reason, 'HOST_PROBABLE');
  assert.equal(r.route, undefined);
  assert.ok(r.features.tierProbabilities.host > GATE.maxHostProbability);
});

test('chooseRouteByDecision picks the cheapest tier whose cumulative probability reaches tierCoverage', () => {
  // At difficulty index 0 and risk 'safe': 'research' labels 'standard' (not economy/debug) and
  // 'debug' labels 'strong' (see labelTier). So this joint distribution is economy 0, standard .95,
  // strong .05, host 0 -> cumulative economy 0 (<.9), +standard .95 (>=.9) -> DECISION_STANDARD.
  const a = answers({ intentP: { explain: 0, edit: 0, debug: .05, operate: 0, research: .95, architecture: 0, other: 0 },
    difficultyP: { '0': 1, '1': 0, '2': 0, '3': 0, '4': 0 } });
  const r = chooseRouteByDecision(a, INPUT, POLICY, GATE);
  assert.equal(r.reason, 'DECISION_STANDARD');
  assert.equal(r.route.tier, 'standard');
});

test('chooseRouteByDecision: UNCERTAIN_TIER when no cumulative tier reaches tierCoverage', () => {
  // At difficulty index 0, risk safe maps explain->economy(.34), research->standard(.33), debug->strong(.33);
  // risk high(.04) maps everything to host regardless of intent. With tierCoverage=.999 the best
  // attainable cumulative across economy+standard+strong is 1-host=.96, which never reaches .999,
  // while host(.04) stays under maxHostProbability(.05) so this is genuine uncertainty, not HOST_PROBABLE.
  const a = answers({ intentP: { explain: .34, edit: 0, debug: .33, operate: 0, research: .33, architecture: 0, other: 0 },
    riskP: { safe: .96, caution: 0, high: .04, unknown: 0 } });
  const strict = { method: 'decision-v1', tierCoverage: 0.999, maxHostProbability: 0.05 };
  const r = chooseRouteByDecision(a, INPUT, POLICY, strict);
  assert.equal(r.reason, 'UNCERTAIN_TIER');
  assert.equal(r.route, undefined);
});

test('chooseRouteByDecision never applies a profile intent override to the strong tier', () => {
  const policy = { ...POLICY, profiles: { codex: { ...POLICY.profiles.codex, intents: { edit: { role: 'economy-role' } } }, claude: {} } };
  // Force a strong decision: intent edit, but push all mass past standardMaxDifficulty so only 'strong' can satisfy tierCoverage.
  const a = { intent: { value: 'edit', probabilities: Object.fromEntries(Object.keys(ROUTE_QUESTIONS.intent.criteria).map(k => [k, k === 'edit' ? 1 : 0])) },
    difficulty: { value: 4, probabilities: { '0': 0, '1': 0, '2': 0, '3': 0, '4': 1 } },
    risk: { value: 'safe', probabilities: Object.fromEntries(RISK_KEYS.map(k => [k, k === 'safe' ? 1 : 0])) } };
  const r = chooseRouteByDecision(a, INPUT, policy, GATE);
  assert.equal(r.reason, 'DECISION_STRONG');
  assert.equal(r.route.tier, 'strong');
  assert.equal(r.route.role, 'strong-role');
  assert.equal(r.route.intentOverride, undefined);
});

test('chooseRouteByDecision: TARGET_UNAVAILABLE when the decided tier has no configured/available role', () => {
  const input = { ...INPUT, availableRoles: ['standard-role', 'strong-role'] }; // economy role missing from availableRoles
  const a = answers(); // economy-heavy one-hot answers -> decided tier economy
  const r = chooseRouteByDecision(a, input, POLICY, GATE);
  assert.equal(r.reason, 'TARGET_UNAVAILABLE');
  assert.equal(r.route, undefined);
});

test('chooseRouteByDecision: MISSING_DIMENSIONS when any answer is absent', () => {
  const a = answers(); delete a.risk;
  assert.equal(chooseRouteByDecision(a, INPUT, POLICY, GATE).reason, 'MISSING_DIMENSIONS');
});
