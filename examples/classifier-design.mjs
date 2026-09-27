import assert from 'node:assert/strict';
import { setup, ROUTE_INPUT, FILTER_INPUT } from '../tests/features-helpers.mjs';
import { routeOrDelegate } from '../src/routing.mjs';
// SYNTHETIC fixtures only. No API calls, native host execution, quality or savings benchmark.
const fixture = setup();
try {
  let hostCalls = 0;
  const route = await routeOrDelegate(fixture.layer, structuredClone(ROUTE_INPUT), {
    use: selected => selected, delegate: () => { hostCalls++; return null; },
  });
  const filter = await fixture.layer.filter(structuredClone(FILTER_INPUT));
  assert.equal(hostCalls, 0); assert.equal(route.model, 'fixture-economy');
  assert.deepEqual(filter.rejectIds, ['drop_1']);
  console.log(JSON.stringify({ synthetic: true, externalApiCalls: 0, hostDecisionCalls: hostCalls, route,
    filter: { kept: filter.keepIds, rejected: filter.rejectIds }, taskQualityVerified: false }, null, 2));
} finally { fixture.cleanup(); }
