---
name: pointsman-control
description: Install or inspect pointsman for Codex/Claude, and explicitly control router/bulk/workflow/evidence modes and native adapter gates.
---
# pointsman operator controls
Resolve the bundled `scripts/run.mjs` relative to this skill's own directory, not the project working directory:

```sh
node <this-skill-directory>/scripts/run.mjs status
node <this-skill-directory>/scripts/run.mjs off
node <this-skill-directory>/scripts/run.mjs shadow
node <this-skill-directory>/scripts/run.mjs on
node <this-skill-directory>/scripts/run.mjs policy init
node <this-skill-directory>/scripts/run.mjs policy check
node <this-skill-directory>/scripts/run.mjs policy roles --host codex|claude [--dry-run] [--replace]
node <this-skill-directory>/scripts/run.mjs router off
node <this-skill-directory>/scripts/run.mjs router shadow
node <this-skill-directory>/scripts/run.mjs router on
node <this-skill-directory>/scripts/run.mjs bulk off
node <this-skill-directory>/scripts/run.mjs bulk shadow
node <this-skill-directory>/scripts/run.mjs bulk on
node <this-skill-directory>/scripts/run.mjs evidence status
node <this-skill-directory>/scripts/run.mjs evidence off|shadow|on
node <this-skill-directory>/scripts/run.mjs evidence auto off|local|semantic
node <this-skill-directory>/scripts/run.mjs workflow status
node <this-skill-directory>/scripts/run.mjs workflow off|shadow|on
node <this-skill-directory>/scripts/run.mjs workflow native off|shadow|on
node <this-skill-directory>/scripts/run.mjs run --root /absolute/project < request.json
node <this-skill-directory>/scripts/run.mjs doctor
node <this-skill-directory>/scripts/run.mjs smoke
node <this-skill-directory>/scripts/run.mjs metrics
```

Global mode and per-feature mode combine conservatively: either OFF disables; either SHADOW prevents application. Updating code does not enable new features. Do not toggle without the user's request. `policy init` creates safe defaults without overwriting an existing policy. Router profiles start empty; each target is `{role (required), model?, reasoning?, skills?}` — map only role names (and, for Claude, `haiku|sonnet|opus`) the host actually has agent definitions for. `policy roles --host <codex|claude>` reads `~/.codex/agents/*.toml` or `~/.claude/agents/*.md` file names (never their content) and writes a preset from that; it refuses to overwrite a non-empty profile without `--replace`, and `--dry-run` writes nothing. No invented role/model names or silently lowered thresholds. `policy check` validates configuration, NOT role/model quality or availability.

Reuse the existing TYPESAFE_API_KEY. If missing, ask the user to run `key set` in a normal terminal. Never request a key in chat, pass it as an argument or display/read credentials.env, process environments, host auth files, shell history or backups. The managed key file is private plaintext, not an encrypted vault.

`smoke` and examples are offline. `smoke --live` incurs TypeSafe usage and requires explicit authorization. Feature SHADOW with the `jev` provider also incurs TypeSafe usage; the local `laya` provider makes no API calls. No classifier.dev account/key is used. In-flight requests cannot be unsent; responses observed after OFF/policy changes are not applied. Installation preserves normal host trust/permissions and does not switch the active host model.

For updates: OFF, review Git changes, fast-forward pull, full tests, reinstall the two existing skills/MCP entries, reconnect hosts. Check `src/feature-policy.mjs` and `src/evaluation.mjs` for configuration, limitations and paired evaluation. Never claim savings or native/live success from a synthetic test.

## Evidence controls

`evidence.mode` and `evidence.automatic` are independent private policy settings. Automatic preference is `off|local|semantic` and defaults OFF when absent, including an update from existing evidence ON. Global/evidence OFF prevents collection; SHADOW does not apply it. Enable only within the user's requested scope and read back `evidence status`. Automatic OFF preserves manually requested tools. Automatic local defaults omitted `semantic` false; automatic semantic defaults it true and may incur selected-provider API usage. Explicit `semantic: false` remains local. No provider fallback occurs.

Install both MCP entries and skills with `install --target both` for authorized Codex/Claude use; evidence selection needs no hooks. For evidence-only activation keep router, effort and training capture OFF. Reconnect hosts to receive updated skills and MCP initialization instructions. Verify the root reported by initialization/status matches the current project; `mcp --root` binds it at startup and requests cannot change it.

Enabled automatic preference tells the agent to derive query, terms and scope from ordinary multi-file debugging, investigation or refactoring requests and collect once; users do not name tools or paths. Cheap single-file actions stay direct, and `read_evidence` is only for needed recovery. This is host guidance, not guaranteed tool interception or per-tool classification. Existing source exclusions and exhaustive-coverage rules still apply.

## Workflow controls and root binding

`workflow.mode` and `workflow.nativeMode` are separate OFF defaults in private `features.json`. Global OFF dominates both; workflow OFF or SHADOW returns without filesystem work or provider inference. Enable a gate only within the user's requested activation scope, preserve other settings, and use `workflow status`/`policy check` for readback. Native ON also requires effective workflow ON; installation never enables it or installs native adapter bindings.

`run` accepts one fixed `repo-evidence`, `test-diagnose` or `log-triage` JSON request from stdin. CLI `--root`, MCP `mcp --root`, or JS `createWorkflowRunner({root,...})` binds the permitted project; otherwise the process working directory is used. Request JSON cannot add roots, arbitrary commands or test permissions. Ordinary CLI/MCP read existing test reports; an authorized JS integrator may inject registered test names.

Requests may lower `maxActions`, `maxMs`, `maxDecisionCalls`, `maxOutputBytes` within the private policy and bind `snapshot: {revision, files}` to Git HEAD and source SHA-256 hashes. Sources and modes are rechecked before use. JS cancellation uses `AbortSignal`, CLI SIGINT/SIGTERM, MCP `notifications/cancelled`. For deterministic execution use `maxDecisionCalls: 0`; the current recipes make no provider calls, and no qualified semantic consumer is connected inside `run`. Exhaustive coverage disables semantic selection and retains mandatory/contrary evidence.

Report workflow status, acceptance, source refs/hashes, omissions and measured counters; `done` is segment completion, not task quality. Reducing internal parent round trips is the design benefit, while measured end-to-end cost/latency and quality remain separate gates. Codex owned-client, Claude `turn.step` and Gemini `BeforeModel` adapters require their supported runtime contract and explicit native activation; contract tests are not live bypass evidence. Unsupported/incomplete packets continue through the normal host. Use the repository README's “Bounded workflows” examples rather than adding automatic host setup.

## Provider and opt-in capture controls

When the user explicitly requests these actions, use the same scripts/run.mjs with `provider status|jev|laya`, `training capture status|on|off`, `dataset stats|validate|build`, or `dataset export --version HASH --format laya|canonical`. Never turn capture ON yourself, dump raw samples into chat, run training, or publish datasets. Provider selection requires prepared local configuration; do not download Python/models during a decision. Default capture is OFF.
