# Pointsman

Pointsman is an optional decision and workflow layer for agents, with a paid **Jev/TypeSafe** provider and a local **Laya/pointsman-router** provider behind the same engine. MCP, CLI and JavaScript expose bounded Choice, Noul and Score judgments plus fixed repository-evidence, test-diagnosis and log-triage recipes. The host retains its own permissions and execution controls. Everything ships OFF by default.

The existing automatic integration routes Claude Code subagents to a lighter or stronger role/model, or keeps the host's choice. The local **pointsman-router** checkpoint was fine-tuned from Laya multilingual on Claude Opus reference labels; its measured scope is coding-task routing, not general Jev replacement. Codex has explicit decision tools, with a more limited hook integration described below.

## Cross-agent design and learning roadmap

The [integrated plan](PLAN.md) prioritizes bounded work completed behind one call. The shared executor and native adapter contracts are implemented. A matched Claude 2.1.288 source-evidence segment passed the same independent checks in 2.22 seconds with native completion versus 28.29 seconds through the model and a prepared code batch (one pair, 2→0 corresponding main requests). This is a deterministic execution gain; whole-task efficiency and broad model superiority remain unestablished. [Primary-source research](PLAN.md#high-leverage-design-and-research-findings) maps the Codex, Claude and Gemini paths, and [work packages](PLAN.md#detailed-work-packages) keep their evidence separate.

The [local Clef-Flash experiment](training/clef-local/README.md) ran 4-bit and 8-bit backbones with the official BF16 decision head on a 24 GiB Mac. Both processed all 125 development inputs but failed the changed-rule and Korean gates; neither is a production provider or promoted checkpoint. The [eight-candidate evidence](training/clef-local/evidence.json) preserves the one-family and eight ambiguous-oracle-row limitations. The current recommendation is to retain existing behavior, not scale or adopt these rejected candidates.

Follow-up [policy placement](training/clef-local/placement-evidence.json) also failed the core rule gate; [native FP32 component parity](training/clef-local/parity-evidence.json) passed without certifying the complete converted backbone. The completed [public invoice comparison](training/public-invoice/README.md) found the same 144/150 correct answers (96%) for published Jev 1.13.0 and local Clef-Flash 8-bit on source-derived discount-term gold. Local warm median inference was 61.887 seconds for these long inputs; this is not a matched hosted-speed comparison or broad superiority result. No new paid API calls were made. [Two real regression seeds](training/task-utility/README.md) were already localized by a bounded code batch, so adding a semantic chooser to those segments is not justified.

The independent [tool-selection DEV study](training/tool-selection/README.md) found **80/80 correct primary tool selections for Clef-Flash 4-bit**, but only 24/47 no-call cases and inadequate raw rule compliance. Applying allowed-tool rules in code preserved the semantic choices and removed policy violations. Direct categorical choice retained 80/80 while scoring 23/47 no-call cases: representation alone did not fix that gap. Laya typed improved from 0/80 to 51/80 primary choices with a categorical representation, while still failing all 47 no-call cases. Neither passes its full research screen, and task savings remain unestablished. The interrupted 8-bit arm remains separate, with its original missing-case denominator preserved.

| Surface | Current repository support | Application boundary |
|---|---|---|
| MCP / CLI / JavaScript | Shared typed decision engine and fixed workflow runner | Decisions need `apply=true`; workflows return segment status and source evidence |
| Claude Code | Owned spawn hooks; separate opt-in effort mod | Host/version and policy dependent; native behavior needs verification |
| Codex | MCP decisions/workflows and spawn-recording hooks; direct app-server MCP execution verified on 0.154.0 | The owned probe passed with no `turn/start`; total provider requests and UI/history continuation remain UNKNOWN. Normal MCP keeps outer parent turns |
| Claude / Gemini native adapters | Version-bound synthetic-response contracts | OFF by default; Claude 2.1.288 source-packet/history, continuation and recipe-read cancellation/recovery verified; complete-task acceptance open; Gemini 0.42 assistant-history retention fails |
| Other MCP clients / owned SDK harnesses | Common engine/runner are reusable; no host-specific installer | Root and capabilities are bound by the trusted caller |

Jev is selected explicitly; a local failure never silently calls the paid provider. Likewise, an unqualified local checkpoint does not acquire general-purpose capabilities just because the request uses the same schema.

> An agent asked to install this should read [AGENTS.md](AGENTS.md) first. It preserves existing branches, uncommitted changes and configuration, and never requests or prints a key in chat.

## How it works

A Claude Code `PreToolUse` hook watches every `Agent` (subagent spawn) call. Each spawn's prompt is expected to start with a line like:

```
[route scope=local complete=yes failures=0]
```

The hook applies guard rules first (sensitive content, a model the caller already locked, unknown/incomplete scope, prior failures) that keep the host's own choice untouched. Otherwise the model answers three questions about the task — intent, difficulty (1–5) and risk — and a decision-level gate (tier probability ≥ 0.80 **and** keep-host probability ≤ 0.20, both fitted on held-out data and shipped with the checkpoint) decides whether to rewrite the spawn's `subagent_type`/`model` or leave it as the host chose.

Codex support is more limited: the repository records opaque spawn messages on `codex-cli` 0.154.0, so that integration cannot read task text and only records spawns (role distribution, no routing decision). Current official hook documentation describes broader capabilities; this repository has not reverified task visibility and rewriting on a newer host. The MCP tools (`route`, `decide`, `status`, …) remain available. See the [capability matrix](PLAN.md#host-capability-matrix) before assuming automatic routing.

## Measured results

From the [pointsman-router model card](https://huggingface.co/jadeonstudio/pointsman-router): a 308-task held-out test set, scored as **agreement with Claude Opus reference labels, not task success**.

| Metric | Value |
|---|---|
| Intent agreement | 0.880 |
| Difficulty agreement (exact / within ±1) | 0.718 / 0.994 |
| Risk agreement | 0.692 |
| Routed (the gate made a tier decision instead of keeping the host's choice) | 32.8% |
| Unsafe routes (cheaper tier than warranted, or routed where the host's choice should have been kept) | 11.9% |
| Warm inference latency (p50) | 101 ms |

These numbers describe agreement with a labeling model, not measured savings or task success in your own workflow — verify both before relying on them.

## Requirements

- macOS, Linux or WSL for the Node engine and Jev provider. The optional resident-server LaunchAgent installer is macOS only; other local deployments need an explicitly managed worker lifecycle.
- Node.js version from [package.json](package.json) `engines` (currently `>=22`).
- Python ≥3.8 in a dedicated virtual environment, with the official `laya` package installed:

  ```sh
  python3 -m venv ~/.local/share/laya/.venv
  ~/.local/share/laya/.venv/bin/pip install laya==0.3.4
  ```

  Only needed to run the local Laya provider (routing with the pointsman-router checkpoint). Using the remote Jev/TypeSafe provider instead needs no Python.
- Claude Code or Codex for the existing host installers. Other MCP clients can use the stdio server, but are not verified native integrations.

## Install

```sh
git clone https://github.com/jadeonstudio/pointsman.git ~/.local/share/pointsman/repository
cd ~/.local/share/pointsman/repository
node bin/pointsman.mjs install --target claude --hooks --laya-agent --dry-run   # review the diff first
node bin/pointsman.mjs install --target claude --hooks --laya-agent
```

Clone to a stable location — the installed shim keeps pointing at that path, so don't delete the folder afterward. The installer puts a `pointsman` shim in `~/.local/bin`; make sure that directory is on your `PATH` (or keep calling `node bin/pointsman.mjs`).

- Add `--no-skills` if `~/.claude/skills` is a symlink; the installer refuses to write through a symlinked skills root and this flag leaves it untouched (MCP entry, hooks and the instruction block still install).
- For Codex, install with `--target codex --hooks`, then approve the new hooks yourself in Codex's `/hooks` — the installer never writes that approval state.
- `--target both` installs both hosts at once.
- `--laya-agent` (macOS only) writes the `launchd` plist for the resident server that keeps the local model warm (see [Everyday use](#everyday-use)); it never runs `launchctl` itself.

The install writes the plist but does not load it — run the command the installer prints, or:

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pointsman.laya.plist
```

## Get the model

The [pointsman-router](https://huggingface.co/jadeonstudio/pointsman-router) repository is currently private while licensing questions are being confirmed. Set `HF_TOKEN` in your environment to pull it.

```sh
pointsman laya pull --repo jadeonstudio/pointsman-router \
  --revision <40-character commit sha> \
  --python <absolute path to the venv's python> \
  [--device mps]
pointsman laya adopt --candidate <candidate hash printed by pull>
pointsman provider laya
pointsman policy init
pointsman policy roles --host claude
pointsman router on
pointsman on
```

Notes:

- `--revision` must be a full 40-character commit hash — branch and tag names are refused.
- The data home (`POINTSMAN_HOME`, default `~/.local/share/pointsman`) must not live under a symlinked path such as macOS's `/tmp`; the installer refuses unsafe symlinks anywhere along that path.
- The resident server (previous section) is effectively required: loading the model cold takes about 5 seconds, which exceeds the 2-second decision timeout the hook uses before it gives up and keeps the host's choice.

## Everyday use

```sh
pointsman status     # mode, provider, readiness — never prints a key
pointsman doctor      # offline diagnostics
pointsman off         # stop making decisions
pointsman shadow      # decide and log, but never rewrite a spawn
pointsman on          # decide and apply
pointsman laya rollback   # restore the checkpoint active before the last promote/rollback/adopt
pointsman metrics --days 7
```

The `[route scope=... complete=... failures=...]` line at the top of a routed prompt is how the hook decides whether it has enough context to route at all; leave it out or mark `scope=unknown`/`complete=no` when you're not sure, and the hook keeps the host's own choice.

## Bounded workflows

Use one `run` call to collect evidence, then let the parent interpret the packet and make the change. Internal search/read/parse/group steps reduce parent round trips. Compare against a competent host that batches tools; cost and latency savings remain unmeasured.

| Recipe | `inputs` | Result details |
|---|---|---|
| `repo-evidence` | `symbols`, optional `paths`, `requiredPaths`, `uncertainPaths`, `counterevidencePaths` | Definitions, lexical direct callers, tests/contracts, contradictory sources; dynamic edges stay UNKNOWN |
| `test-diagnose` | `resultPaths`; optionally `registeredTest` in a trusted JS integration | Deduplicated failure groups, passed/skipped evidence, possible flakes, source spans and affected rerun set; cause may remain unresolved |
| `log-triage` | `paths`, optional `timeWindow: {from,to}` | Counts, first/last and singleton failures, correlations, success evidence, source offsets, parse failures and clock uncertainty |

CLI and ordinary MCP consume existing test reports; they cannot accept a test command from request JSON. A JS caller can inject an already authorized `capabilities.registeredTests` map. Requests never add commands, permissions, filesystem roots or capabilities.

Global and workflow modes must both be ON to execute. OFF/SHADOW reads no sources and calls no provider. Installation and updates leave workflow/native gates OFF. When activation is requested:

```sh
pointsman workflow on
pointsman on
pointsman workflow status
pointsman run --root /absolute/path/to/project < request.json
pointsman workflow off
pointsman workflow native off
```

Example `request.json`:

```json
{
  "workflow": "repo-evidence",
  "goal": "Collect definition, callers and tests in the supplied scope",
  "inputs": {"symbols": ["createWorkflowRunner"], "paths": ["src/workflows.mjs", "src/mcp.mjs", "tests/workflows.test.mjs"]},
  "acceptance": ["definition", "direct_callers", "tests"],
  "coverage": "selective",
  "budget": {"maxActions": 24, "maxMs": 10000, "maxDecisionCalls": 0, "maxOutputBytes": 24000}
}
```

Budgets are capped by private policy; a request cannot raise them. Supply `snapshot: {revision, files}` when binding to a prior source snapshot: `revision` is Git HEAD (or `null` without Git), and `files` maps relative paths to SHA-256 hashes (`null` means absent). The runner returns the actual snapshot and rechecks it before returning evidence. Changed sources, policy, cancellation or exhausted budgets prevent a completion claim; narrow the segment or resolve the reported boundary before retrying.

For MCP, call tool **`run`** with the same JSON. Start the server with `pointsman mcp --root /absolute/path/to/project` or bind `root` in `startMcp`; an existing server's working directory is otherwise its root. A request cannot change it. For JavaScript:

```js
import { createWorkflowRunner } from 'pointsman/workflows';
const runner = createWorkflowRunner({ engine, root: projectRoot, getPolicy });
const result = await runner.run(request, { signal: abortController.signal });
```

Use the existing `engine`, trusted `projectRoot`, and effective `getPolicy()` (CLI/MCP use `workflowPolicy(home, globalMode)`). Cancellation uses JS `AbortSignal`, CLI SIGINT/SIGTERM or MCP `notifications/cancelled`.

Consume `status`, `acceptance`, `coverage`, `needsParent`, source-linked `evidence` (refs, hashes, excerpts) and `stats` together. `done` completes only the delegated segment; it does not establish a fix, deployment, test quality or full audit. Preserve original sources and contrary evidence. For exhaustive work set `coverage: "exhaustive"`, retain the full required scope, and treat refused/unreadable paths, unsupported formats and unknown edges as coverage limits. Exhaustive mode disables semantic decisions and never permits classifier omission.

Evidence refs such as `e0` identify source-linked entries within one response; each entry retains its `path`, SHA-256 `hash`, `role` and `text`. Repository spans may merge overlapping excerpts: `roles` and `symbols` retain their associations, and `details.symbols` indexes those spans. Diagnostic packets use `details.recordTableVersion: 1` and `details.records`: group `first`/`last`, timelines, contrary evidence and result lists contain record IDs such as `r0`. Resolve the record, then its `ref.evidenceRef` to an evidence span; line/byte offsets remain in `record.ref`, and the span's `sourceRef` identifies its input evidence. Input evidence retains the original source reference. A missing referenced ID means incomplete evidence.

NDJSON/plain logs are decoded, hashed and aggregated incrementally. JSON containers and a single long line still use bounded buffers. The timeline retains first/new signatures and changes in incident kind; repeated identical events are represented by group count plus first/last observations. Read `details.timelineSemantics` with those groups. Mandatory evidence that exceeds the output budget produces a coverage failure rather than silent omission.

The three recipes currently collect evidence deterministically with zero provider calls. A proposed ambiguous-definition classifier was removed because its answer did not eliminate any downstream action. Existing `decide`, `route` and `filter` remain available; no qualified semantic decision consumer is yet connected inside `run`.

`node scripts/workflow-benchmark.mjs 5 17` compares purpose-built code batches and this executor on controlled fixtures. The [measurement report](examples/workflow-hosts/measurement-evidence.json) preserves all 30 independent checks and packet reductions of 52.4%/51.3%/27.7% for repository/log/test evidence. Purpose-built batches remain faster and smaller than the executor alone.

Later Claude 2.1.288 trials measured native completion against the model using a prepared whole-segment batch. Both sides passed independent source-fact checks and retained their results in native history:

| Source-evidence segment | Through model + batch | Native completion | Observed main requests |
|---|---:|---:|---:|
| Host-contract definition/callers/tests | 28.29 s | 2.22 s | 2 → 0 |
| Representative global-OFF precedence | 26.53 s | 2.52 s | 3 → 0 |

Times include process startup; each row is one pair. The second baseline added a no-op after its single batch, so its 10.54× observation includes that extra work. Native packets were larger in that case. Neither recipe called Jev/Laya: the measured gain is removal of model requests. Actual cache/token usage and invalid preparation overhead are included; existing Max subscription only, usage credits OFF. Whole-task p50/p95, success non-inferiority and semantic-model contribution remain unestablished. Use direct batching when it already finishes the segment.

`pointsman workflow native off|shadow|on` controls a separate adapter gate; ON also requires global/workflow ON and a supported host contract. The Codex owned-client, Claude `turn.step` and Gemini `BeforeModel` adapters retain normal continuation when a packet is incomplete or the host is unsupported. They are not installed automatically by the existing hooks or effort mod. The [Gemini 0.42 installed-component probe](examples/workflow-hosts/gemini-component-evidence.json) verifies synthetic text and zero provider invocations using a throw-only local provider sentinel. That version requires a `BeforeModel` blocking decision to consume the synthetic response, but fails to retain it in assistant history. Full Gemini native adoption is therefore NO-GO; authenticated CLI/UI and in-flight cancellation remain unverified. The response is specific to `BeforeModel` and never answers a tool-permission request.

The Gemini command bridge requires **both** `BeforeAgent` and `BeforeModel` hooks, invoking `pointsman workflow-native --host gemini --event before-agent` and `--event before-model` with the host's JSON stdin, the same private home and trusted working directory. An exact `pointsman-workflow <JSON>` entry creates a short-lived prompt hash; the model hook consumes it once before executing the recipe. This prevents tool-only continuation from replaying the earlier user request. Hook-provided `cwd` never changes the execution root. The installed-component probe exercises real command hooks and a fresh recipe, while the history limitation above still prevents full native adoption. No Gemini hooks are registered automatically.

The separate `mods/pointsman-workflows` Claude mod accepts ordinary prompt text beginning exactly `pointsman-workflow ` followed by request JSON. Use the installed executable `pointsman` shim for its `pointsmanPath`; the source `.mjs` is not executable. In the [authenticated 2.1.288 session](examples/workflow-hosts/claude-session-preflight.json), a two-file recipe completed in 66.502 ms and the next ordinary model turn correctly read its four evidence facts. Native usage showed no main-model usage for the recipe and one main request for continuation; a Haiku helper still used tokens. This verifies a bounded bypass, not zero total provider traffic, cancellation while the bridge runs, or whole-task savings. Production modes remain OFF.

A later [actual cancellation check](examples/workflow-hosts/claude-cancellation-evidence.json) verified termination while the real CLI awaited input, using an explicit 20-second EOF hold, followed by successful native recovery. The CLI now aborts pending stdin and host-version children; the mod drops late results after `next.signal` aborts. A later streaming SDK `Query.interrupt()` trial aborted an actual recipe source read, terminated its bridge and completed an independently checked recovery in the same Claude process, with zero reported model usage. That sandboxed SDK trial has no persistent-history receipt; authenticated pair histories are recorded separately. Model-inference cancellation and complete task economics remain open.

## Measuring cost and time

`pointsman router ab 0.2` randomly holds back 20% of otherwise-rewritten spawns as an unmodified `control` arm (the host's own role/model choice, proposal only logged) instead of always applying the route (`treatment`); `pointsman router ab off` (or `0`) goes back to always applying. Share is capped at 0.5 and is off (0) by default, so nothing changes until you opt in.

```sh
pointsman router ab 0.2                       # start a 20% control / 80% treatment split
pointsman metrics ab --days 7                  # cost/time by arm, from the local hook logs
pointsman metrics ab --days 7 --prices FILE    # also estimate USD cost from a price table
pointsman router ab off                        # back to always applying the route
```

The report groups content-free outcome events (duration, tool-use count, per-model token usage, all read from the host's own `PostToolUse`/`SubagentStop` payload and the agent's transcript usage lines — never the task or transcript content) by arm, and separately counts each observed original-role → final-role transition. It measures **cost and time only, not task quality or correctness**, and always shows the raw counts so a small sample isn't over-read (arms under ~30 outcomes are flagged `insufficientSample`).

A price table (`--prices FILE`, JSON) maps a model id to USD-per-million-token rates: `{"<model id>": {"input": ..., "output": ..., "cache_write": ..., "cache_read": ...}}`. See [`examples/prices.example.json`](examples/prices.example.json) for the shape — fill it in yourself from the model provider's own current pricing page; nothing here hard-codes a price. Without `--prices`, `costUsd` is always `null`.

## Reasoning-effort mod (phase 1, shadow)

`mods/pointsman-effort/` is a separate, opt-in Claude Code mod (not installed by `pointsman install`) that attaches pointsman's existing route judgment to Claude Code's reasoning effort — no new training, no new checkpoint, the same laya `route` questions `router` already uses. It ships OFF and needs `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1`, which you set yourself:

```sh
claude plugin marketplace add jadeonstudio/pointsman/mods   # or a local path: /path/to/pointsman/mods
claude plugin install pointsman-effort@pointsman-mods
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1                  # add to your shell profile to keep it on
pointsman effort shadow    # record recommendations + per-turn outcomes; never applies
pointsman effort on        # may apply — see the two gates below
pointsman effort off       # back to doing nothing
```

Two independent, OFF-by-default gates decide what ON is allowed to touch, because changing effort mid-conversation rewrites the model's cached context and that rewrite can cost more than the thinking-token savings it buys:

- `mainLoop: 'cold-only'` — the main loop's effort may change only when the previous main-loop request of the session was already more than `coldAfterSeconds` (default 3600) ago, or there was none yet. A change on an already-warm turn is refused (`CACHE_WARM`) even in ON mode.
- `subagents: 'on'` — a subagent's effort may be fixed once, from its own first step; a subagent always starts with a fresh context, so this never pays a cache-rewrite cost.

SHADOW always records both the main-loop recommendation (and whether that turn's cache would have been cold) and the subagent recommendation, regardless of these two settings, so you can review before turning anything on:

```sh
pointsman effort ab 0.2   # once ON: hold back 20% of otherwise-applied changes as an unmodified control arm
pointsman effort ab off   # back to always applying
```

It never changes the model, never changes a subagent's role, and never rewrites the prompt. See [AGENTS.md](AGENTS.md) ("Owned Claude Code mod") for the full contract.

These instructions describe the existing effort mod and its older host path. Current Claude documentation describes broader mods on 2.1.287+ with different enablement; the [plan](PLAN.md#host-specific-acceleration-beyond-the-common-workflow-tool) records the migration and native-bypass work separately. That newer workflow integration is not implemented by the existing effort mod.

## Training your own checkpoint

The training kit is included at [training/laya-kit/](training/laya-kit/README.md), with an explicit local training path. The [integrated learning plan](PLAN.md#what-the-model-must-learn) describes the data, evaluation and qualification changes needed for broader decisions. The lifecycle commands are `laya register` → `laya holdout freeze` → `laya qualify` (includes decision-gate flags) → `laya compare` → `laya promote`, and `laya package` builds a folder (weights plus a `pointsman.json` manifest) ready to upload with `hf upload`. All are explicit operator calls — nothing here starts training or promotes a checkpoint automatically.

## Security and privacy

Summary; full detail in [SECURITY.md](SECURITY.md):

- The local Laya provider makes no network calls; it starts only a locally configured Python runtime and checkpoint with an allowlisted environment (no inherited API/HF secrets).
- Operational logs are content-free: only timing, mode, purpose and bounded reason codes. Separately enabled training capture uses a private evidence store; do not publish its raw records.
- Keys are never written into arguments, generated configs or Git; the managed credentials file is plaintext with restrictive permissions, not an encrypted vault.
- The optional Jev/TypeSafe provider sends the caller-supplied state, question instructions/criteria and model name over HTTPS to a fixed endpoint; it is a remote call and incurs API usage.

## Uninstall

```sh
pointsman uninstall --target both --dry-run
pointsman uninstall --target both
```

This removes only what the installer owns (MCP entry, hooks, instruction block, skills, shim) and preserves keys, logs, training data, backups and the cloned repository. Use `uninstall --hooks-only` to remove just the hooks and instruction block while keeping everything else installed.

## Migrating from jev-agent-control

If you previously installed this project under its old name (`jev-agent-control`), uninstall that version first, then move its data home:

```sh
pointsman migrate --from ~/.local/share/jev-agent-control
```

## License

Code is MIT (see [LICENSE](LICENSE)). Model weights for pointsman-router are Apache-2.0 — see the [model card](https://huggingface.co/jadeonstudio/pointsman-router).
