export const VERSION = '0.3.0';
export const SERVER_NAME = 'pointsman';
export const API_URL = 'https://api.typesafe.ai/v1/systemone';
export const PURPOSES = Object.freeze(['route', 'select', 'retry', 'review', 'judge', 'escalate']);
export const MODES = Object.freeze(['off', 'shadow', 'on']);
// Versioned admission profiles, not a claim that a checkpoint supports these heads.
// TypeSafe /api verifies Choice 255 and Score 10; no verified question-count uplift.
export const CAPABILITY_PROFILES = Object.freeze({
  portable: Object.freeze({ maxChoiceOptions: 16, maxScoreLevels: 10, maxQuestions: 8, maxInputBytes: 48000 }),
  'jev-expanded-v1': Object.freeze({ maxChoiceOptions: 255, maxScoreLevels: 10, maxQuestions: 8, maxInputBytes: 48000 }),
});
export const DEFAULTS = Object.freeze({
  version: 1, mode: 'off', model: 'jev-latest', timeoutMs: 2000,
  maxInputBytes: 24000, maxQuestions: 8, capabilityProfile: 'portable', maxChoiceOptions: 16, minConfidence: 0.85,
  minChoiceProbability: 0.80, noulCertainty: 0.95,
  maxCallsPerMinute: 60, maxInFlight: 4,
  circuitFailureThreshold: 3, circuitCooldownMs: 30000, telemetry: true,
});
export const MAX_FRAME_BYTES = 65536;
export const MAX_RESPONSE_BYTES = 131072;
export const ID = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;
export const MODEL_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,100}$/;
export const RESERVED = new Set(['__proto__', 'prototype', 'constructor']);
export class ControlError extends Error {
  constructor(code) { super(code); this.name = 'ControlError'; this.code = code; }
}
export function fail(code) { throw new ControlError(code); }
export function errorCode(error) { return error instanceof ControlError ? error.code : 'INTERNAL_ERROR'; }
export const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
