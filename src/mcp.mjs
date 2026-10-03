import { MAX_FRAME_BYTES, VERSION, errorCode, fail, isObject } from './constants.mjs';
import { traceSchema } from './training/schema.mjs';
import { decisionSchema } from './contracts.mjs';
import { createControlLayer, observeSchema } from './control-layer.mjs';
import { routeSchema } from './routing.mjs';
import { filterSchema } from './filtering.mjs';
import { createWorkflowRunner, workflowSchema } from './workflows.mjs';
import { workflowPolicy } from './feature-policy.mjs';

const withTrace = schema => ({ ...schema, properties: { ...schema.properties, trace: traceSchema } });
const protocolVersions = ['2024-11-05', '2025-03-26', '2025-06-18'];
export const TOOLS = [
  { name: 'run', description: 'Run one bounded repo-evidence, test-diagnose or log-triage recipe within this server\'s configured root. Collect source-linked evidence behind one call. OFF/SHADOW retain the host path; requests cannot add roots, commands or capabilities. Completion applies only to the delegated recipe.',
    inputSchema: workflowSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true } },
  { name: 'decide', description: 'One bounded batch of Choice/Noul/Score decisions. OFF never invokes a provider; SHADOW hides suggestions; only apply=true permits consuming an advisory result. Never grants execution permission.', inputSchema: withTrace(decisionSchema),
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true } },
  { name: 'status', description: 'Report the global mode, the router and bulk feature modes, the selected provider (jev or laya) and whether it is ready, training-capture state and the local Laya worker status. Use it to answer questions about Jev configuration or before reporting modes to the user; decision tools already return without a provider call while OFF, so a status check is not needed before calling them. Never returns a key or credential value and makes no network call.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false } },
  { name: 'feedback', description: 'Compare an independently obtained baseline with a recent decision in this process. Record measured usage only; agreement is not accuracy. Expires after five minutes. No raw state or baseline values are logged.',
    inputSchema: { type: 'object', additionalProperties: false, required: ['id', 'baseline'], properties: {
      id: { type: 'string' }, baseline: { type: 'object', additionalProperties: { anyOf: [{ type: 'string' }, { type: 'number' }, { type: 'boolean' }] } },
      baselineUsage: { type: 'object', additionalProperties: false, properties: { inputTokens: { type: ['integer', 'null'], minimum: 0 }, outputTokens: { type: ['integer', 'null'], minimum: 0 } } },
      baselineElapsedMs: { type: 'number', minimum: 0 }, taskSucceeded: { type: 'boolean' },
    } }, annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false } },
  { name: 'route', description: 'Classify intent, ordered difficulty and risk with one selected-provider inference; local policy selects a configured, available target. Advisory only: cannot switch the host model or authorize execution. Feature OFF by default.', inputSchema: withTrace(routeSchema),
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true } },
  { name: 'filter', description: 'Prefilter bounded snippets before reading them; return IDs, never text. Keep required, uncertain, failed and exhaustive-audit items. Never deletes source or context. Feature OFF by default; use the CLI pipeline for inputs not yet in model context.', inputSchema: withTrace(filterSchema),
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: false, openWorldHint: true } },
  { name: 'observe', description: 'After an independent baseline, compare a recent SHADOW route tier or independently labeled relevant IDs. Agreement is not task success; never fabricates cheaper-model outcomes. Five-minute, same-process lifetime.', inputSchema: observeSchema,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false } },
  { name: 'record', description: 'Append a minimal host baseline or outcome only when training capture is explicitly enabled. All MCP outcomes are weak host-review evidence; cannot assert runner/human authority, read datasets, enable capture, train, or promote a model.', inputSchema: {
    type: 'object', additionalProperties: false, required: ['kind', 'data'], properties: { kind: { type: 'string', enum: ['host', 'outcome'] }, data: { type: 'object' } },
  }, annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false } },
];

/** Minimal, version-negotiated MCP stdio tools server; no HTTP listener or sampling. */
export function startMcp(engine, { input = process.stdin, output = process.stdout,
  layer = createControlLayer({ engine, home: engine.status().home }), root = process.cwd(),
  workflowRunner = createWorkflowRunner({ engine, root,
    getPolicy: () => workflowPolicy(engine.status().home, engine.status().mode) }) } = {}) {
  let buffer = Buffer.alloc(0), initialized = false, ready = false, closed = false;
  const pending = new Map();
  const write = object => { if (!closed && !output.destroyed) output.write(JSON.stringify(object) + '\n'); };
  const error = (id, code, message) => write({ jsonrpc: '2.0', id, error: { code, message } });
  const result = (id, value) => write({ jsonrpc: '2.0', id, result: value });
  const close = () => { closed = true; for (const c of pending.values()) c.abort(); pending.clear(); engine.close?.(); input.pause(); };
  async function receive(message) {
    const hasId = isObject(message) && Object.hasOwn(message, 'id');
    const id = hasId ? message.id : null;
    if (!isObject(message) || message.jsonrpc !== '2.0' || typeof message.method !== 'string' ||
        (hasId && !(typeof id === 'string' && id.length <= 128 || Number.isSafeInteger(id)))) {
      error(null, -32600, 'Invalid Request'); return;
    }
    if (!hasId) {
      if (message.method === 'notifications/initialized' && initialized) ready = true;
      if (message.method === 'notifications/cancelled') pending.get(message.params?.requestId)?.abort();
      return;
    }
    if (message.method === 'initialize') {
      if (initialized || !isObject(message.params) || typeof message.params.protocolVersion !== 'string' || !isObject(message.params.clientInfo)) {
        error(id, -32602, 'Invalid initialize parameters'); return;
      }
      initialized = true;
      result(id, { protocolVersion: protocolVersions.includes(message.params.protocolVersion) ? message.params.protocolVersion : protocolVersions.at(-1),
        capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'pointsman', version: VERSION },
        instructions: 'Optional advisory decision layer. Respect apply=false and existing host approvals. Use the local CLI for mode changes; do not send credentials to any tool.' });
      return;
    }
    if (message.method === 'ping') { result(id, {}); return; }
    if (!ready) { error(id, -32002, 'Initialization required'); return; }
    if (message.method === 'tools/list') { result(id, { tools: TOOLS }); return; }
    if (message.method !== 'tools/call') { error(id, -32601, 'Method not found'); return; }
    if (!isObject(message.params) || typeof message.params.name !== 'string') { error(id, -32602, 'Invalid tool parameters'); return; }
    if (!TOOLS.some(t => t.name === message.params.name)) { error(id, -32602, 'Unknown tool'); return; }
    if (pending.has(id) || pending.size >= 8) { error(id, -32000, 'Request capacity exceeded'); return; }
    const controller = new AbortController(); pending.set(id, controller);
    try {
      const args = message.params.arguments ?? {};
      let value;
      if (message.params.name === 'run') value = await workflowRunner.run(args, { signal: controller.signal });
      else if (message.params.name === 'decide') value = await engine.decide(args, { signal: controller.signal });
      else if (message.params.name === 'route') { if (!isObject(args)) fail('INVALID_REQUEST'); const { trace, ...request } = args; value = await layer.route(request, { signal: controller.signal, trace }); }
      else if (message.params.name === 'filter') { if (!isObject(args)) fail('INVALID_REQUEST'); const { trace, ...request } = args; value = await layer.filter(request, { signal: controller.signal, trace }); }
      else if (message.params.name === 'record') {
        if (!isObject(args) || Object.keys(args).some(k => !['kind', 'data'].includes(k)) || !isObject(args.data)) fail('INVALID_REQUEST');
        if (args.kind === 'host') value = engine.recordHost(args.data);
        else if (args.kind === 'outcome') value = engine.recordOutcome(args.data);
        else fail('INVALID_REQUEST');
      }
      else if (message.params.name === 'observe') value = layer.observe(args);
      else if (message.params.name === 'status') {
        if (!isObject(args) || Object.keys(args).length) fail('INVALID_REQUEST');
        value = layer.status();
      } else value = engine.feedback(args);
      result(id, { content: [{ type: 'text', text: JSON.stringify(value) }], isError: false });
    } catch (e) { result(id, { content: [{ type: 'text', text: JSON.stringify({ error: errorCode(e) }) }], isError: true }); }
    finally { pending.delete(id); }
  }
  function onData(chunk) {
    if (closed) return;
    buffer = Buffer.concat([buffer, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)]);
    let end;
    while ((end = buffer.indexOf(10)) !== -1) {
      const line = buffer.subarray(0, end); buffer = buffer.subarray(end + 1);
      if (line.length > MAX_FRAME_BYTES) { error(null, -32600, 'Frame too large'); close(); return; }
      if (!line.toString('utf8').trim()) continue;
      let parsed;
      try { parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(line)); }
      catch { error(null, -32700, 'Parse error'); continue; }
      void receive(parsed).catch(() => error(null, -32603, 'Internal error'));
    }
    if (buffer.length > MAX_FRAME_BYTES) { error(null, -32600, 'Frame too large'); close(); }
  }
  input.on('data', onData);
  input.once('end', close);
  input.once('error', close);
  output.once('error', close);
  return { close };
}
