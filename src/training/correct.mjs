import { randomUUID } from 'node:crypto';
import { resolveHome } from '../storage.mjs';
import { fail } from '../constants.mjs';
import { digest, id as validId, validateTarget } from './schema.mjs';

// TTY checks are a same-user local-authorship signal only, not proof of a human operator;
// a PTY-attached agent also has a TTY. See SECURITY.md threat model.
export async function correctDecision({ store, home = resolveHome(), decisionId, isStdinTTY, isStdoutTTY, prompt, write = () => {} }) {
  if (!isStdinTTY || !isStdoutTTY) fail('HUMAN_TTY_REQUIRED');
  validId(decisionId);
  if (!store.config().trainingCapture) return { stored: false, reason: 'CAPTURE_OFF' };
  const snapshot = store.scan();
  const decisionEvent = snapshot.events.find(e => e.kind === 'decisions' && e.data.decision_id === decisionId);
  if (!decisionEvent) fail('DECISION_NOT_FOUND');
  const d = decisionEvent.data;
  write(`purpose: ${d.request.purpose}\nstate (minimized, <=4KiB): ${JSON.stringify(d.request.state)}\n`);
  const labels = [];
  const makeLabel = (question_id, value) => {
    const at = new Date().toISOString();
    return { question_id, value, source: 'human', label_confidence: 1, evidence_ref: 'sha256:' + digest({ decision_id: decisionId, question_id, value, at }) };
  };
  for (const [name, q] of Object.entries(d.request.questions)) {
    write(`\nquestion ${name}: ${q.instructions}\n`);
    if (q.type === 'choice') {
      for (const [key, description] of Object.entries(q.criteria)) write(`  ${key}: ${description ?? ''}\n`);
      const answer = (await prompt(`choose value for "${name}" (blank to skip): `)).trim();
      if (answer) { validateTarget(q, answer); labels.push(makeLabel(name, answer)); }
    } else if (q.type === 'noul') {
      const answer = (await prompt(`value for "${name}" true/false (blank to skip): `)).trim();
      if (answer) {
        if (!['true', 'false'].includes(answer)) fail('INVALID_TRAINING_TARGET');
        labels.push(makeLabel(name, answer === 'true'));
      }
    } else {
      q.criteria.forEach((description, i) => write(`  ${i}: ${description ?? ''}\n`));
      const answer = (await prompt(`choose index for "${name}" (blank to skip): `)).trim();
      if (answer) {
        const index = Number(answer);
        validateTarget(q, index);
        labels.push(makeLabel(name, index));
      }
    }
  }
  if (!labels.length) return { stored: false, reason: 'NO_LABELS_PROVIDED' };
  const outcome = { decision_id: decisionId, execution_id: randomUUID(), executed: false, final: true, source: 'human',
    executed_answers: {}, metrics: {}, checks: [], labels };
  return { ...store.outcome(outcome), decision_id: decisionId, labels_recorded: labels.length };
}
