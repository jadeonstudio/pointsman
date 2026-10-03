// Installed native component contract probe, not an authenticated CLI/UI run.
// The only ContentGenerator is an injected throw-only sentinel. No provider is created.
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { createWorkflowRunner } from '../../src/workflows.mjs';
import { geminiWorkflowResponse, WORKFLOW_HOST_CONTRACTS } from '../../src/workflow-hosts.mjs';

const installation = '/opt/homebrew/lib/node_modules/@google/gemini-cli';
const bundle = `${installation}/bundle/chunk-7VVHSNDQ.js`;
const version = JSON.parse(await fs.readFile(`${installation}/package.json`, 'utf8')).version;
assert.equal(version, '0.42.0', 'Use the verified installed build; never fake a host version.');
const { GeminiChat, HookSystem, HookType, HookEventName, Turn, AgentExecutionStoppedError } = await import(pathToFileURL(bundle));
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-gemini-native-')));
const observed = [];
try {
  await fs.writeFile(path.join(root, 'source.mjs'), 'export function sum(a,b) { return a+b; }\n');
  await fs.writeFile(path.join(root, 'source.test.mjs'), 'sum(1,2);\n');
  const request = { workflow: 'repo-evidence', inputs: { symbols: ['sum'] }, acceptance: ['definition', 'direct_callers', 'tests'] };
  const runner = createWorkflowRunner({ root, engine: { status: () => ({ mode: 'on', policyRevision: 'frozen-native-probe' }), decide: () => assert.fail('No model branch is needed') }, getPolicy: () => ({ mode: 'on' }) });
  const packet = await runner.run(request);
  assert.equal(packet.status, 'done');
  assert.deepEqual(Object.keys(packet.acceptance).filter(key => packet.acceptance[key] === 'met').sort(), ['definition', 'direct_callers', 'tests']);
  assert.equal(packet.stats.inferenceCalls, 0);
  async function runCase(name, mode, result = packet, signal = new AbortController().signal, legacy = false) {
    let providerInvocations = 0, hookCalls = 0;
    const model = 'pointsman-local-throw-only-sentinel';
    const config = {
      promptId: name, getProjectRoot: () => root, getWorkingDir: () => root, getSessionId: () => name,
      getUsageStatisticsEnabled: () => false, getProjectHooks: () => undefined, getHooks: () => undefined,
      getExtensions: () => [], isTrustedFolder: () => false, getDisabledHooks: () => [],
      getMaxAttempts: () => 1, getRetryFetchErrors: () => false, isContextManagementEnabled: () => false,
      getModel: () => model, getActiveModel: () => model, setActiveModel: () => {}, getApprovalMode: () => 'default',
      getModelAvailabilityService: () => ({ selectFirstAvailable: () => ({ selectedModel: model }), consumeStickyAttempt: () => {} }),
      modelConfigService: { getResolvedConfig: () => ({ model, generateContentConfig: {} }) },
      getContentGeneratorConfig: () => undefined,
      getContentGenerator: () => ({ generateContentStream() { providerInvocations++; throw new AgentExecutionStoppedError('CONTENT_GENERATOR_SENTINEL'); } }),
    };
    config.config = config;
    const hooks = new HookSystem(config);
    config.getHookSystem = () => hooks;
    hooks.getRegistry().registerHook({ type: HookType.Runtime, name: 'pointsman-component-workflow', timeout: 2000, action: async () => {
      hookCalls++;
      const response = await geminiWorkflowResponse({ runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS.gemini.revision }, getMode: () => ({ globalMode: mode, mode }), runner: { run: async () => result } }, { scope: 'workflow', request }, { signal });
      if (legacy) { delete response.decision; delete response.reason; }
      return response;
    } }, HookEventName.BeforeModel);
    await hooks.initialize();
    const chat = new GeminiChat({ config, promptId: name });
    const events = [], turn = new Turn(chat, name);
    let error = null;
    try { for await (const event of turn.run({ model }, 'Complete this bounded fixture.', signal)) events.push(event); }
    catch (failure) { error = failure.message; }
    error ??= events.find(event => event.type === 'error')?.value?.error?.message ?? null;
    error ??= events.find(event => event.type === 'agent_execution_stopped')?.value?.reason ?? null;
    const texts = events.filter(event => event.type === 'content').map(event => event.value).join('');
    const history = chat.getHistory();
    const row = { name, hookCalls, providerInvocations, error, eventTypes: events.map(event => event.type), visibleText: texts,
      historyRoles: history.map(item => item.role), syntheticAssistantInHistory: history.some(item => item.role === 'model' && item.parts?.some(part => part.text === JSON.stringify(packet))) };
    observed.push(row);
    return row;
  }
  const complete = await runCase('complete_on', 'on');
  assert.equal(complete.error, null, JSON.stringify(complete));
  assert.equal(complete.providerInvocations, 0);
  assert.equal(complete.visibleText, JSON.stringify(packet));
  const legacy = await runCase('legacy_response_only', 'on', packet, undefined, true);
  assert.equal(legacy.providerInvocations, 1); assert.match(legacy.error, /CONTENT_GENERATOR_SENTINEL/);
  const off = await runCase('off', 'off');
  assert.equal(off.providerInvocations, 1); assert.match(off.error, /CONTENT_GENERATOR_SENTINEL/);
  const incomplete = await runCase('incomplete_on', 'on', { ...packet, status: 'needs_parent', needsParent: 'fixture unresolved' });
  assert.equal(incomplete.providerInvocations, 1); assert.match(incomplete.error, /CONTENT_GENERATOR_SENTINEL/);
  const aborted = new AbortController(); aborted.abort();
  const cancelled = await runCase('already_aborted', 'on', packet, aborted.signal);
  assert.equal(cancelled.providerInvocations, 0);
  assert.equal(cancelled.visibleText, '');
  const evidence = { observedAt: new Date().toISOString(), host: 'gemini', version, grade: 'installed_native_component_with_injected_provider',
    bundleHash: createHash('sha256').update(await fs.readFile(bundle)).digest('hex'), packet,
    actualExternalModelRequests: 0, actualProviderCreated: false, globalsChanged: false, observations: observed,
    streamBypass: 'PASS', fallThroughSentinel: 'PASS', cancellationBeforeRequest: 'PASS',
    history: complete.syntheticAssistantInHistory ? 'PASS' : 'FAIL_INSTALLED_COMPONENT_SYNTHETIC_ASSISTANT_NOT_RETAINED',
    adoption: complete.syntheticAssistantInHistory ? 'UNQUALIFIED_FULL_CLI' : 'NO_GO_FULL_NATIVE_HISTORY',
    adoption: complete.syntheticAssistantInHistory ? 'UNQUALIFIED_NATIVE_CLI' : 'NO_GO_FULL_NATIVE_HISTORY',
    authenticatedCli: 'UNKNOWN', userInterface: 'UNKNOWN', inFlightCancellation: 'UNKNOWN' };
  await fs.writeFile(new URL('gemini-component-evidence.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify({ ...evidence, packet: undefined, observations: observed.map(({ visibleText, ...row }) => ({ ...row, visibleTextBytes: Buffer.byteLength(visibleText) })) }));
} finally { await fs.rm(root, { recursive: true, force: true }); }
