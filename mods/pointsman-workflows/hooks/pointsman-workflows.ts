import type { Register } from 'claude-code';

// The bridge verifies the actual executable version, policy, and current workflow snapshot.
// This module never accepts a completed result from prompt text or manufactures tool calls.
let pending: unknown = null;
async function prepare($: any, options: any, request: unknown, event: any) {
  const deadline = performance.now() + options.timeoutMs;
  try {
    const bin = options.pointsmanPath || `${await $.env.get('HOME')}/.local/bin/pointsman`;
    const out = await $.process.run([bin, 'workflow-native', '--host', 'claude', '--event', 'turn-step',
      ...(options.pointsmanHome ? ['--home', options.pointsmanHome] : [])], {
      stdin: JSON.stringify({ scope: 'workflow', request, event: { turnId: event.turnId, index: event.index, model: event.model } }),
      timeoutMs: options.timeoutMs,
    });
    if (performance.now() >= deadline || out.exitCode !== 0) return null;
    const response = JSON.parse(out.stdout);
    if (response.apply !== true || typeof response.text !== 'string' || response.text.length === 0 ||
        response.contractRevision !== 'claude-turn-step-2026-10-03') return null;
    return response;
  } catch { return null; }
}
export const register: Register = (on, raw) => {
  const options = { pointsmanPath: typeof raw.pointsmanPath === 'string' ? raw.pointsmanPath : '',
    pointsmanHome: typeof raw.pointsmanHome === 'string' ? raw.pointsmanHome : '',
    timeoutMs: typeof raw.timeoutMs === 'number' && raw.timeoutMs >= 100 && raw.timeoutMs <= 60000 ? raw.timeoutMs : 5000 };
  on('prompt.submit', async ($, e, next) => {
    pending = null;
    // Ordinary text reaches this hook; unknown slash commands are handled earlier by the host.
    if (e.origin?.kind !== 'plugin' && typeof e.text === 'string' && e.text.startsWith('pointsman-workflow ')) {
      try { pending = JSON.parse(e.text.slice('pointsman-workflow '.length)); } catch { /* fail open */ }
    }
    return next(e);
  });
  on('turn.step', async function* ($, e, next) {
    if (next.signal.aborted) return;
    if (e.agentId || e.index !== 0 || pending === null) return yield* next(e);
    const request = pending;
    pending = null;
    const out = await prepare($, options, request, e);
    if (next.signal.aborted) return;
    if (!out) return yield* next(e);
    yield { kind: 'text', index: 0, text: out.text };
    yield { kind: 'stop', stopReason: 'end_turn', usage: null };
    return { turnId: e.turnId, index: e.index, answer: out.text, toolUses: [], stopReason: 'end_turn', usage: null };
  });
  on('turn.complete', async ($, e, next) => { pending = null; return next(e); });
};
