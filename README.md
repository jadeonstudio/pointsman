# Pointsman

Pointsman is an optional typed decision layer for agents, with a paid **Jev/TypeSafe** provider and a local **Laya/pointsman-router** provider behind the same engine. It exposes bounded Choice, Noul and Score judgments through MCP, CLI and JavaScript. The host consumes eligible advice and retains its own permissions and execution controls. Everything ships OFF by default.

The existing automatic integration routes Claude Code subagents to a lighter or stronger role/model, or keeps the host's choice. The local **pointsman-router** checkpoint was fine-tuned from Laya multilingual on Claude Opus reference labels; its measured scope is coding-task routing, not general Jev replacement. Codex has explicit decision tools, with a more limited hook integration described below.

## Cross-agent design and learning roadmap

The [architecture](ARCHITECTURE.md) defines how to reuse the shared engine across Codex, Claude Code, other MCP clients and owned agent harnesses. The [training plan](TRAINING_PLAN.md) and [evaluation gates](EVALUATION.md) separate three goals: better completed-task efficiency, higher decision quality than Jev, and generalization to unseen tasks and schemas. These are plans, not shipped integrations or demonstrated superiority.

| Surface | Current repository support | Application boundary |
|---|---|---|
| MCP / CLI / JavaScript | Shared typed decision engine | Caller must consume `apply=true`; no automatic host control |
| Claude Code | Owned spawn hooks; separate opt-in effort mod | Host/version and policy dependent; native behavior needs verification |
| Codex | MCP decisions and spawn-recording hooks | Previously observed opaque spawn input prevents task-based hook routing |
| Other MCP clients / owned SDK harnesses | Common engine is reusable; no host-specific installer | Proposed capability checks and adapters; native integration unverified |

Jev is selected explicitly; a local failure never silently calls the paid provider. Likewise, an unqualified local checkpoint does not acquire general-purpose capabilities just because the request uses the same schema.

> An agent asked to install this should read [AGENTS.md](AGENTS.md) first. It preserves existing branches, uncommitted changes and configuration, and never requests or prints a key in chat.

## How it works

A Claude Code `PreToolUse` hook watches every `Agent` (subagent spawn) call. Each spawn's prompt is expected to start with a line like:

```
[route scope=local complete=yes failures=0]
```

The hook applies guard rules first (sensitive content, a model the caller already locked, unknown/incomplete scope, prior failures) that keep the host's own choice untouched. Otherwise the model answers three questions about the task — intent, difficulty (1–5) and risk — and a decision-level gate (tier probability ≥ 0.80 **and** keep-host probability ≤ 0.20, both fitted on held-out data and shipped with the checkpoint) decides whether to rewrite the spawn's `subagent_type`/`model` or leave it as the host chose.

Codex support is more limited: the repository records opaque spawn messages on `codex-cli` 0.154.0, so that integration cannot read task text and only records spawns (role distribution, no routing decision). Current official hook documentation describes broader capabilities; this repository has not reverified task visibility and rewriting on a newer host. The MCP tools (`route`, `decide`, `status`, …) remain available. See the [capability matrix](ARCHITECTURE.md#host-capability-matrix) before assuming automatic routing.

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

## Training your own checkpoint

The training kit is included at [training/laya-kit/](training/laya-kit/README.md), with an explicit local training path. The [learning plan](TRAINING_PLAN.md) describes the data, evaluation and qualification changes needed for broader decisions. The lifecycle commands are `laya register` → `laya holdout freeze` → `laya qualify` (includes decision-gate flags) → `laya compare` → `laya promote`, and `laya package` builds a folder (weights plus a `pointsman.json` manifest) ready to upload with `hf upload`. All are explicit operator calls — nothing here starts training or promotes a checkpoint automatically.

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
