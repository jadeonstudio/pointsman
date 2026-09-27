// pointsman-effort (phase 1, shadow): attaches pointsman's EXISTING route judgment (no new
// training, no new heads) to Claude Code's reasoning effort. Never changes a prompt, never changes
// a model, never touches a subagent's role -- the only output this mod ever produces is
// `next({ ...e, effort })` on turn.step. Everything decision-making lives in `pointsman hook
// --host claude --event turn-effort|turn-outcome` (src/hooks.mjs); this file is a thin, fail-open
// conduit: on any missing binary, bad JSON, non-zero exit or timeout, it makes no decision and
// changes nothing. See docs/plan/2026-09-27-pointsman-effort-mod.md.
//
// Cache-cost gate (measured 2026-09-27, Claude Code 2.1.280): changing effort rewrites the whole
// cached MESSAGES block. A subagent starts with a fresh context, so fixing its effort from its own
// first step (agent.spawn -> its first turn.step) costs nothing extra. The main loop's prompt cache
// can already be warm from the previous turn, so a main-loop change may only ever apply when
// `pointsman effort`'s `mainLoop` policy is 'cold-only' AND the previous main-loop request of this
// session was long enough ago -- that arithmetic is done by src/control-layer.mjs's effort(), not
// here; this file only measures and forwards how long it has been since the previous main-loop
// turn.step (`since_last_main_ms`).
import type { Register } from 'claude-code';

type EffortLevel = 'low' | 'medium' | 'high' | 'xhigh' | 'max';
type Loop = 'main' | 'subagent';
type Decision = { decision_id: string | null; mode: string; apply: boolean; level: EffortLevel | null; arm: string; reason: string };
type PointsmanOptions = { pointsmanPath: string; pointsmanHome: string; timeoutMs: number; logDecisions: boolean };
type TurnAgg = {
  loop: Loop;
  steps: number;
  toolUses: number;
  effortUsed: unknown;
  decisionId: string | null;
  arm: string;
  usage: { input: number; output: number; cache_creation: number; cache_read: number };
};

// Module-scoped state: one mod instance runs per Claude Code session process, so plain module
// state (never functions holding `$`) is enough -- no cross-session persistence is needed or
// attempted. A single pending-decision slot for the main loop: a second prompt.submit before the
// first's turn.step consumes it simply replaces it (the same "last one wins" idea the reference
// mod (davila7/claude-code-templates jev-model-router) uses for its own pending slot).
let pendingMain: Decision | null = null;
let lastMainRequestAtMs: number | null = null;
const appliedMainLevel = new Map<string, EffortLevel>(); // turnId -> level, once decided at step 0
const subagentDecision = new Map<string, { level: EffortLevel | null; decisionId: string | null; arm: string }>(); // agentId -> agent.spawn decision
const turnAgg = new Map<string, TurnAgg>(); // turnId -> running totals, flushed on turn.complete

/**
 * The mod compiler only allows `$` to be passed to a function declared at this file's top level
 * (a function declaration or a const bound to one), spelled `$.noun.event(...)` at its call sites
 * -- a nested helper closing over `$` fails to load. This is the one such helper every hook below
 * calls; it never throws (any failure -> null, i.e. "no decision").
 */
async function callPointsman($: any, options: PointsmanOptions, event: 'turn-effort' | 'turn-outcome', stdin: unknown): Promise<any | null> {
  try {
    const bin = options.pointsmanPath && options.pointsmanPath.length > 0 ? options.pointsmanPath : `${await $.env.get('HOME')}/.local/bin/pointsman`;
    const argv = [bin, 'hook', '--host', 'claude', '--event', event, ...(options.pointsmanHome ? ['--home', options.pointsmanHome] : [])];
    const r = await $.process.run(argv, { stdin: `${JSON.stringify(stdin)}\n`, timeoutMs: options.timeoutMs });
    if (r.exitCode !== 0 || !r.stdout) return null;
    const line = r.stdout.trim().split('\n').filter(Boolean).pop();
    if (!line) return null;
    return JSON.parse(line);
  } catch {
    return null; // fail open: no decision
  }
}

export const register: Register = (on, rawOptions) => {
  const options: PointsmanOptions = {
    pointsmanPath: typeof rawOptions.pointsmanPath === 'string' ? rawOptions.pointsmanPath : '',
    pointsmanHome: typeof rawOptions.pointsmanHome === 'string' ? rawOptions.pointsmanHome : '',
    timeoutMs: typeof rawOptions.timeoutMs === 'number' && rawOptions.timeoutMs > 0 ? rawOptions.timeoutMs : 1500,
    logDecisions: rawOptions.logDecisions === true,
  };

  // Main loop: decide once per user turn, never touching the prompt itself.
  on('prompt.submit', async ($, e, next) => {
    try {
      if (e.origin?.kind !== 'plugin' && typeof e.text === 'string' && e.text.length > 0) {
        const now = await $.clock.now();
        const sinceLastMainMs = lastMainRequestAtMs === null ? null : now - lastMainRequestAtMs;
        let context: string | undefined;
        try {
          // The newest assistant message with text (SessionMessage: { role, text, toolUses }).
          const messages = await $.session.messages();
          const last = Array.isArray(messages) ? [...messages].reverse().find(m => m?.role === 'assistant' && typeof m.text === 'string' && m.text.length > 0) : undefined;
          if (last) context = last.text.slice(-2000);
        } catch {
          // No prior-turn context available; the hook still decides on text alone.
        }
        const decision = await callPointsman($, options, 'turn-effort', {
          text: e.text, ...(context ? { context } : {}), loop: 'main', since_last_main_ms: sinceLastMainMs,
        });
        pendingMain = decision;
        if (options.logDecisions && decision) {
          await $.ui.log(`pointsman-effort: reason=${decision.reason} apply=${decision.apply} level=${decision.level ?? ''} arm=${decision.arm}`);
        }
      }
    } catch {
      // fail open: no decision, no prompt change
    }
    return next(e);
  });

  // A subagent starts a fresh context, so its decision is made once, at spawn, from exactly the
  // text the route checkpoint was trained on (prompt + description) -- never from a rewritten one:
  // `e` reaches `next(e)` unchanged, so this never changes the subagent's model or role. The
  // started subagent's `agentId` (race-free: `next(e)` only resolves once it exists, before that
  // subagent's own first turn.step) is the key turn.step looks the decision up by.
  on('agent.spawn', async ($, e, next) => {
    let decision: Decision | null = null;
    try {
      // Same text shape the spawn-time route judgment reads (src/hooks.mjs preSpawn: description + prompt).
      decision = await callPointsman($, options, 'turn-effort', { text: `${e.description ?? ''}\n${e.prompt}`, loop: 'subagent' });
    } catch {
      // fail open: no decision
    }
    const r = await next(e);
    try {
      if (decision && r?.agentId) subagentDecision.set(r.agentId, { level: decision.apply ? decision.level : null, decisionId: decision.decision_id ?? null, arm: decision.arm ?? 'none' });
    } catch {
      // fail open: the subagent still runs, just without an effort override
    }
    return r;
  });

  // Applies a decided effort from the loop's own first step onward (main: the pending prompt.submit
  // decision; subagent: agent.spawn's), and accumulates per-turn totals for turn.complete to report.
  on('turn.step', async function* ($, e, next) {
    const isSubagent = Boolean(e.agentId);
    let level: EffortLevel | undefined;
    if (!isSubagent) {
      if (e.index === 0) {
        const now = await $.clock.now();
        lastMainRequestAtMs = now; // this request IS the "previous main-loop request" for the NEXT one
        const decision = pendingMain;
        pendingMain = null;
        if (decision?.apply && decision.level) appliedMainLevel.set(e.turnId, decision.level);
        turnAgg.set(e.turnId, { loop: 'main', steps: 0, toolUses: 0, effortUsed: null,
          decisionId: decision?.decision_id ?? null, arm: decision?.arm ?? 'none',
          usage: { input: 0, output: 0, cache_creation: 0, cache_read: 0 } });
      }
      level = appliedMainLevel.get(e.turnId);
    } else {
      const spawned = subagentDecision.get(e.agentId as string);
      if (e.index === 0) {
        turnAgg.set(e.turnId, { loop: 'subagent', steps: 0, toolUses: 0, effortUsed: null, decisionId: spawned?.decisionId ?? null, arm: spawned?.arm ?? 'none',
          usage: { input: 0, output: 0, cache_creation: 0, cache_read: 0 } });
      }
      level = spawned?.level ?? undefined;
    }
    const stepInput = level && level !== e.effort ? { ...e, effort: level } : e;
    const r = yield* next(stepInput as typeof e);
    try {
      const agg = turnAgg.get(e.turnId);
      if (agg) {
        agg.steps += 1;
        agg.toolUses += r?.toolUses?.length ?? 0;
        agg.effortUsed = stepInput.effort ?? agg.effortUsed;
        if (r?.usage) {
          agg.usage.input += r.usage.input_tokens ?? 0;
          agg.usage.output += r.usage.output_tokens ?? 0;
          agg.usage.cache_creation += r.usage.cache_creation_input_tokens ?? 0;
          agg.usage.cache_read += r.usage.cache_read_input_tokens ?? 0;
        }
      }
    } catch {
      // Accumulation is best-effort; a miscount here must never affect the response returned above.
    }
    return r;
  });

  // Fires exactly once per turn (main or subagent), whatever the reason it ended -- a cleaner
  // end-of-turn signal than guessing from turn.step's stopReason. Reports the accumulated totals
  // and clears this turn's/subagent's state either way.
  on('turn.complete', async ($, e, next) => {
    try {
      const agg = turnAgg.get(e.turnId);
      turnAgg.delete(e.turnId);
      if (agg) {
        await callPointsman($, options, 'turn-outcome', {
          decision_id: agg.decisionId, turn_id: e.turnId, loop: agg.loop, steps: agg.steps,
          duration_ms: e.durationMs, effort_used: agg.effortUsed ?? undefined, stop_reason: e.reason,
          tool_uses: agg.toolUses, usage: agg.usage,
        });
      }
      if (e.agentId) subagentDecision.delete(e.agentId);
      else appliedMainLevel.delete(e.turnId);
    } catch {
      // Outcome reporting is best-effort observability; it must never affect the turn's own answer.
    }
    return next(e);
  });
};
