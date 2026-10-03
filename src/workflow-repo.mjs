import path from 'node:path';

const SOURCE = /\.(?:[cm]?js|jsx|tsx?|py|go|rs|java|kt|swift|rb|[ch](?:pp)?|cs|vue|svelte|json|md|ya?ml|sh|sql|graphql)$/i;
const CODE = /\.(?:[cm]?js|jsx|tsx?|py|go|rs|java|kt|swift|rb|[ch](?:pp)?|cs|vue|svelte|sh)$/i;
const TEST = /(?:^|\/)(?:tests?|__tests__|spec)(?:\/|\.)|\.(?:test|spec)\./i;
const CONTRACT = /(?:contract|interface|schema|types|readme|agents)/i;
const escape = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function invalid() { throw Object.assign(new Error('INVALID_REPO_INPUT'), { code: 'INVALID_REPO_INPUT' }); }
function array(value, max = 64) { if (!Array.isArray(value) || value.length > max || value.some(p => typeof p !== 'string' || !p || p.length > 512)) invalid(); return [...new Set(value)]; }
function validate(inputs) {
  if (Object.keys(inputs).some(k => !['symbols', 'paths', 'requiredPaths', 'uncertainPaths', 'counterevidencePaths', 'semantic'].includes(k))) invalid();
  const symbols = array(inputs.symbols, 16);
  if (!symbols.length || symbols.some(s => !/^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$/.test(s))) invalid();
  const normalized = { symbols };
  for (const key of ['paths', 'requiredPaths', 'uncertainPaths', 'counterevidencePaths']) normalized[key] = inputs[key] === undefined ? [] : array(inputs[key]);
  if (inputs.semantic !== undefined && typeof inputs.semantic !== 'boolean') invalid();
  normalized.semantic = inputs.semantic === true;
  return normalized;
}
function definition(line, symbol) {
  const s = escape(symbol);
  return new RegExp(`^\\s*(?:(?:export|default|public|private|protected|static|async|pub)\\s+)*(?:(?:function|class|def|fn|func|interface|type|enum)\\s+${s}(?![\\w$])|(?:const|let|var)\\s+${s}\\s*(?:=|:)|${s}\\s*\\([^;]*\\)\\s*(?:\\{|:|=>))`).test(line);
}

/** Lexical source evidence, not a fabricated complete call graph. Unknown dynamic edges stay explicit. */
export async function runRepoEvidence(request, ctx) {
  const input = validate(request.inputs), all = await ctx.listFiles(), omissions = [], evidence = [], seen = new Map(), sources = new Map();
  const explicit = new Set([...input.requiredPaths, ...input.uncertainPaths, ...input.counterevidencePaths]);
  const scoped = name => !input.paths.length || input.paths.some(p => name === p || name.startsWith(p.endsWith('/') ? p : `${p}/`));
  const candidates = [...new Set([...all.filter(name => scoped(name) && SOURCE.test(name)), ...explicit])].sort();
  const symbols = Object.fromEntries(input.symbols.map(s => [s, { definitions: [], directCallers: [], tests: [], contracts: [], references: [] }]));
  const add = (file, role, start, end, symbol = null) => {
    const identity = `${file.path}:${start}:${end}:${role}:${symbol}`;
    const ref = `s${evidence.length}`;
    if (seen.has(identity)) return seen.get(identity);
    seen.set(identity, ref); sources.set(file.path, file);
    evidence.push({ ref, path: file.path, hash: file.hash, text: file.text.split('\n').slice(start - 1, end).join('\n'), role, startLine: start, endLine: end, ...(symbol ? { symbol } : {}) });
    return ref;
  };
  let readCount = 0;
  for (let offset = 0; offset < candidates.length; offset += 8) {
    ctx.check();
    const batch = await Promise.all(candidates.slice(offset, offset + 8).map(async name => ({ name, file: await ctx.read(name, explicit.has(name) ? 'required' : 'search') })));
    for (const { name, file } of batch) {
      if (!file.text && file.text !== '') { omissions.push({ path: name, reason: file.reason }); continue; }
      readCount++; const lines = file.text.split('\n');
      for (const [key, role] of [['requiredPaths', 'required'], ['uncertainPaths', 'uncertain'], ['counterevidencePaths', 'counterevidence']]) {
        if (input[key].includes(name)) add(file, role, 1, lines.length);
      }
      for (const symbol of input.symbols) {
        const token = new RegExp(`(?<![\\w$])${escape(symbol)}(?![\\w$])`), call = new RegExp(`(?<![\\w$])${escape(symbol)}\\s*\\(`);
        for (let i = 0; i < lines.length; i++) {
          ctx.check(); const line = lines[i]; if (!token.test(line)) continue;
          const roles = [];
          // Declarations are candidates; comments/strings/import aliases are not proven runtime edges.
          if (CODE.test(name) && !/^\s*(?:\/\/|#|\*|<!--)/.test(line) && definition(line, symbol)) roles.push(['definition', 'definitions']);
          else if (CODE.test(name) && !/^\s*(?:\/\/|#|\*|<!--|import\b|from\b)/.test(line) && call.test(line)) roles.push(['direct_caller', 'directCallers']);
          if (TEST.test(name)) roles.push(['test', 'tests']);
          if (CONTRACT.test(path.basename(name))) roles.push(['contract', 'contracts']);
          if (!roles.length) roles.push(['uncertain', 'references']);
          for (const [role, group] of roles) symbols[symbol][group].push(add(file, role, Math.max(1, i - 2), Math.min(lines.length, i + 4), symbol));
        }
      }
    }
  }
  const known = { definition: input.symbols.every(s => symbols[s].definitions.length > 0), direct_callers: omissions.length === 0, tests: input.symbols.every(s => symbols[s].tests.length > 0), contracts: input.symbols.every(s => symbols[s].contracts.length > 0), counterevidence: input.counterevidencePaths.every(p => evidence.some(e => e.path === p && e.role === 'counterevidence')) };
  const requested = request.acceptance.length ? request.acceptance : ['definition', 'direct_callers', 'tests'];
  const acceptance = Object.fromEntries(requested.map(item => [item, known[item] === true ? 'met' : 'unresolved']));
  const ambiguous = input.symbols.filter(s => symbols[s].definitions.length > 1);
  // No current candidate choice opens another bounded action. Preserve ambiguity without a redundant model call.
  const missing = Object.entries(acceptance).filter(([, status]) => status !== 'met').map(([item]) => item);
  const needsParent = missing.length ? `Resolve missing acceptance: ${missing.join(', ')}.` : omissions.length ? 'Resolve unreadable or refused sources before claiming complete evidence.' : ambiguous.length ? `Resolve multiple definition candidates for ${ambiguous.join(', ')}.` : null;
  const merged = [], refs = new Map();
  for (const span of [...evidence].sort((a, b) => a.path.localeCompare(b.path) || a.startLine - b.startLine || a.endLine - b.endLine)) {
    let target = merged.at(-1);
    if (!target || target.path !== span.path || span.startLine > target.endLine + 1) {
      target = { ...span, ref: `e${merged.length}`, roles: [], symbols: [] }; delete target.symbol; merged.push(target);
    }
    target.endLine = Math.max(target.endLine, span.endLine);
    if (!target.roles.includes(span.role)) target.roles.push(span.role);
    if (span.symbol && !target.symbols.includes(span.symbol)) target.symbols.push(span.symbol);
    refs.set(span.ref, target.ref);
  }
  for (const span of merged) span.text = sources.get(span.path).text.split('\n').slice(span.startLine - 1, span.endLine).join('\n');
  for (const groups of Object.values(symbols)) for (const [key, values] of Object.entries(groups)) groups[key] = values.map(ref => refs.get(ref));
  return { status: needsParent ? 'needs_parent' : 'done', reason: needsParent ? 'UNRESOLVED_EVIDENCE' : 'EVIDENCE_COLLECTED', acceptance, evidence: merged,
    coverage: { requested: request.coverage, complete: omissions.length === 0, filesConsidered: candidates.length, filesRead: readCount, scope: input.paths.length ? input.paths : ['.'], omissions,
      limits: ['Lexical direct callers only; dynamic registries, generated and cross-language edges are UNKNOWN.', 'Comments, string references and aliases require parent interpretation.'] },
    details: { symbols, selectedBranch: null, semanticSelection: 'NOT_IMPLEMENTED', dynamicEdges: 'UNKNOWN', requiredPaths: input.requiredPaths, uncertainPaths: input.uncertainPaths, counterevidencePaths: input.counterevidencePaths }, needsParent };
}
