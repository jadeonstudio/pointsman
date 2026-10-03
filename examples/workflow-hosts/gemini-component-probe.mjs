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
import { setMode } from '../../src/storage.mjs';
import { setFeatureMode, setWorkflowNativeMode } from '../../src/feature-policy.mjs';

const installation = '/opt/homebrew/lib/node_modules/@google/gemini-cli';
const bundle = `${installation}/bundle/chunk-7VVHSNDQ.js`;
const version = JSON.parse(await fs.readFile(`${installation}/package.json`, 'utf8')).version;
assert.equal(version, '0.42.0', 'Use the verified installed build; never fake a host version.');
const { GeminiChat, HookSystem, HookType, HookEventName, Turn, AgentExecutionStoppedError } = await import(pathToFileURL(bundle));
const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-gemini-native-')));
const observed = [];
const homes = [];
const priorText = await fs.readFile(new URL('gemini-component-evidence.json', import.meta.url), 'utf8');
const prior = JSON.parse(priorText);
try {
  await fs.writeFile(path.join(root, 'source.mjs'), 'export function sum(a,b) { return a+b; }\n');
  await fs.writeFile(path.join(root, 'source.test.mjs'), 'sum(1,2);\n');
  const request = { workflow: 'repo-evidence', inputs: { symbols: ['sum'] }, acceptance: ['definition', 'direct_callers', 'tests'] };
  const runner = createWorkflowRunner({ root, engine: { status: () => ({ mode: 'on', policyRevision: 'frozen-native-probe' }), decide: () => assert.fail('No model branch is needed') }, getPolicy: () => ({ mode: 'on' }) });
  const packet = await runner.run(request);
  assert.equal(packet.status, 'done');
  assert.deepEqual(Object.keys(packet.acceptance).filter(key => packet.acceptance[key] === 'met').sort(), ['definition', 'direct_callers', 'tests']);
  assert.equal(packet.stats.inferenceCalls, 0);
  async function runCase(name, mode, result = packet, signal = new AbortController().signal, legacy = false, commandHook = false) {
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
      sanitizationConfig: { enableEnvironmentVariableRedaction: true }, storage: { getPlansDir: () => root },
      getContentGenerator: () => ({ generateContentStream() { providerInvocations++; throw new AgentExecutionStoppedError('CONTENT_GENERATOR_SENTINEL'); } }),
    };
    config.config = config;
    const hooks = new HookSystem(config);
    config.getHookSystem = () => hooks;
    const prompt = commandHook ? `pointsman-workflow ${JSON.stringify(request)}` : 'Complete this bounded fixture.';
    const commandOutputs = [];
    let commandHome;
    if (commandHook) {
      const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'pointsman-gemini-home-'))); homes.push(home);
      commandHome = home;
      setMode(home, mode, {}); setFeatureMode(home, 'workflow', 'on'); setWorkflowNativeMode(home, 'on');
      const quote = value => `'${value.replaceAll("'", "'\\''")}'`;
      const cli = path.resolve(new URL('../../bin/pointsman.mjs', import.meta.url).pathname);
      for (const [event, argument] of [[HookEventName.BeforeAgent, 'before-agent'], [HookEventName.BeforeModel, 'before-model']]) {
        const command = `/usr/bin/env -i PATH=/opt/homebrew/bin:/usr/bin:/bin HOME=${quote(home)} ${quote(process.execPath)} ${quote(cli)} workflow-native --host gemini --event ${argument} --root ${quote(root)} --home ${quote(home)}`;
        hooks.getRegistry().registerHook({ type: HookType.Command, name: `pointsman-command-${argument}`, timeout: 5000, command }, event);
      }
      const execute = hooks.hookRunner.executeCommandHook.bind(hooks.hookRunner);
      hooks.hookRunner.executeCommandHook = async (...args) => {
        hookCalls++; const outcome = await execute(...args);
        commandOutputs.push({ event: args[1], success: outcome.success, error: outcome.error?.message ?? null, output: outcome.output });
        return outcome;
      };
    } else hooks.getRegistry().registerHook({ type: HookType.Runtime, name: 'pointsman-component-workflow', timeout: 2000, action: async () => {
      hookCalls++;
      const response = await geminiWorkflowResponse({ runtime: { version, contractRevision: WORKFLOW_HOST_CONTRACTS.gemini.revision }, getMode: () => ({ globalMode: mode, mode }), runner: { run: async () => result } }, { scope: 'workflow', request }, { signal });
      if (legacy) { delete response.decision; delete response.reason; }
      return response;
    } }, HookEventName.BeforeModel);
    await hooks.initialize();
    if (commandHook && !signal.aborted) await hooks.fireBeforeAgentEvent(prompt);
    const chat = new GeminiChat({ config, promptId: name });
    const events = [], turn = new Turn(chat, name);
    let error = null;
    try { for await (const event of turn.run({ model }, prompt, signal)) events.push(event); }
    catch (failure) { error = failure.message; }
    error ??= events.find(event => event.type === 'error')?.value?.error?.message ?? null;
    error ??= events.find(event => event.type === 'agent_execution_stopped')?.value?.reason ?? null;
    const texts = events.filter(event => event.type === 'content').map(event => event.value).join('');
    const history = chat.getHistory();
    const row = { name, hookCalls, providerInvocations, error, eventTypes: events.map(event => event.type), visibleText: texts,
      historyRoles: history.map(item => item.role), syntheticAssistantInHistory: Boolean(texts) && history.some(item => item.role === 'model' && item.parts?.some(part => part.text === texts)) };
    if (commandHook) {
      // Non-text model/tool contents disappear from public stable input. A spent token must still fall through.
      const stale = { model, contents: [{ role: 'user', parts: [{ text: prompt }] }, { role: 'model', parts: [{ functionCall: { name: 'read_file', args: {} } }] }, { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: {} } }] }] };
      row.replayBlocked = (await hooks.fireBeforeModelEvent(stale)).blocked === false;
      assert.equal(row.replayBlocked, true);
      const continuation = [];
      for await (const event of new Turn(chat, `${name}-continuation`).run({ model }, 'ordinary continuation', signal)) continuation.push(event);
      row.continuationProviderInvocations = providerInvocations - row.providerInvocations;
      row.continuationEventTypes = continuation.map(event => event.type);
      if (mode === 'on') {
        await hooks.fireBeforeAgentEvent(prompt);
        const busy = path.join(commandHome, 'run/gemini-workflows/lock'); await fs.mkdir(busy);
        const busyFallback = (await hooks.fireBeforeModelEvent(stale)).blocked === false;
        await fs.rmdir(busy);
        row.busyLockReplayBlocked = busyFallback && (await hooks.fireBeforeModelEvent(stale)).blocked === false;
        assert.equal(row.busyLockReplayBlocked, true);
        await hooks.fireBeforeAgentEvent(prompt);
        row.identicalNewPromptApplied = (await hooks.fireBeforeModelEvent(stale)).blocked === true;
        assert.equal(row.identicalNewPromptApplied, true);
        await hooks.fireBeforeAgentEvent(prompt); await hooks.fireBeforeAgentEvent('ordinary prompt');
        row.ordinaryInvalidated = (await hooks.fireBeforeModelEvent(stale)).blocked === false;
        assert.equal(row.ordinaryInvalidated, true);
      }
      assert.equal(commandOutputs.every(output => output.success), true, JSON.stringify(commandOutputs));
      row.commandHookCalls = commandOutputs.length;
      row.commandResults = commandOutputs.map(({ output, ...entry }) => ({ ...entry, synthetic: Boolean(output?.hookSpecificOutput?.llm_response) }));
    }
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
  const commandComplete = await runCase('command_complete_on', 'on', packet, undefined, false, true);
  assert.equal(commandComplete.providerInvocations, 0); assert.equal(commandComplete.error, null);
  assert.equal(commandComplete.continuationProviderInvocations, 1);
  const commandPacket = JSON.parse(commandComplete.visibleText);
  assert.equal(commandPacket.status, 'done'); assert.equal(commandPacket.stats.inferenceCalls, 0); assert.equal(commandPacket.stats.networkCalls, 0);
  assert.deepEqual(Object.keys(commandPacket.acceptance).filter(key => commandPacket.acceptance[key] === 'met').sort(), ['definition', 'direct_callers', 'tests']);
  for (const evidence of commandPacket.evidence) assert.equal(evidence.hash, createHash('sha256').update(await fs.readFile(path.join(root, evidence.path))).digest('hex'));
  const commandOff = await runCase('command_global_off', 'off', packet, undefined, false, true);
  assert.equal(commandOff.providerInvocations, 1); assert.equal(commandOff.visibleText, '');
  const evidence = { observedAt: new Date().toISOString(), host: 'gemini', version, grade: 'installed_native_component_with_injected_provider',
    bundleHash: createHash('sha256').update(await fs.readFile(bundle)).digest('hex'), packet,
    actualExternalModelRequests: 0, actualProviderCreated: false, globalsChanged: false, observations: observed,
    streamBypass: 'PASS', fallThroughSentinel: 'PASS', cancellationBeforeRequest: 'PASS',
    history: complete.syntheticAssistantInHistory ? 'PASS' : 'FAIL_INSTALLED_COMPONENT_SYNTHETIC_ASSISTANT_NOT_RETAINED',
    adoption: complete.syntheticAssistantInHistory ? 'UNQUALIFIED_NATIVE_CLI' : 'NO_GO_FULL_NATIVE_HISTORY',
    publicCommandConsumer: { grade: 'installed_0.42.0_command_hook_production_cli_and_fresh_fixed_recipe', result: 'PASS', commandPacket,
      replay: 'PASS_ATOMIC_ONE_SHOT', busyLockFallbackReplay: 'PASS_INVALIDATES_PENDING_BEFORE_FALLBACK', identicalHumanPrompt: 'PASS_NEW_BEFORE_AGENT_ENTRY', ordinaryPrompt: 'PASS_INVALIDATES_PENDING', continuation: 'PASS_THROW_ONLY_PROVIDER_SENTINEL',
      history: commandComplete.syntheticAssistantInHistory ? 'PASS' : 'NO_GO_SYNTHETIC_ASSISTANT_NOT_RETAINED',
      sourceHashes: Object.fromEntries(await Promise.all(['src/workflow-hosts.mjs', 'src/features-cli.mjs', 'examples/workflow-hosts/gemini-component-probe.mjs'].map(async file => [file, createHash('sha256').update(await fs.readFile(new URL(`../../${file}`, import.meta.url))).digest('hex')]))) },
    sourceAssessment: prior.sourceAssessment, priorRevision: prior.priorRevision ?? { observedAt: prior.observedAt, receiptHash: createHash('sha256').update(priorText).digest('hex'), grade: prior.grade },
    authenticatedCli: 'UNKNOWN', userInterface: 'UNKNOWN', inFlightCancellation: 'UNKNOWN' };
  await fs.writeFile(new URL('gemini-component-evidence.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify({ ...evidence, packet: undefined, observations: observed.map(({ visibleText, ...row }) => ({ ...row, visibleTextBytes: Buffer.byteLength(visibleText) })) }));
} finally { await fs.rm(root, { recursive: true, force: true }); for (const home of homes) await fs.rm(home, { recursive: true, force: true }); }
