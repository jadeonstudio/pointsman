---
name: pointsman-control
description: Install, inspect, test or explicitly toggle the optional shared pointsman layer and its router/bulk features for Codex and Claude Code.
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
node <this-skill-directory>/scripts/run.mjs doctor
node <this-skill-directory>/scripts/run.mjs smoke
node <this-skill-directory>/scripts/run.mjs metrics
```

Global mode and per-feature mode combine conservatively: either OFF disables; either SHADOW prevents application. Updating code does not enable new features. Do not toggle without the user's request. `policy init` creates safe defaults without overwriting an existing policy. Router profiles start empty; each target is `{role (required), model?, reasoning?, skills?}` — map only role names (and, for Claude, `haiku|sonnet|opus`) the host actually has agent definitions for. `policy roles --host <codex|claude>` reads `~/.codex/agents/*.toml` or `~/.claude/agents/*.md` file names (never their content) and writes a preset from that; it refuses to overwrite a non-empty profile without `--replace`, and `--dry-run` writes nothing. No invented role/model names or silently lowered thresholds. `policy check` validates configuration, NOT role/model quality or availability.

Reuse the existing TYPESAFE_API_KEY. If missing, ask the user to run `key set` in a normal terminal. Never request a key in chat, pass it as an argument or display/read credentials.env, process environments, host auth files, shell history or backups. The managed key file is private plaintext, not an encrypted vault.

`smoke` and examples are offline. `smoke --live` incurs TypeSafe usage and requires explicit authorization. Feature SHADOW with the `jev` provider also incurs TypeSafe usage; the local `laya` provider makes no API calls. No classifier.dev account/key is used. In-flight requests cannot be unsent; responses observed after OFF/policy changes are not applied. Installation preserves normal host trust/permissions and does not switch the active host model.

For updates: OFF, review Git changes, fast-forward pull, full tests, reinstall the two existing skills/MCP entries, reconnect hosts. Check `src/feature-policy.mjs` and `src/evaluation.mjs` for configuration, limitations and paired evaluation. Never claim savings or native/live success from a synthetic test.

## Provider and opt-in capture controls

When the user explicitly requests these actions, use the same scripts/run.mjs with `provider status|jev|laya`, `training capture status|on|off`, `dataset stats|validate|build`, or `dataset export --version HASH --format laya|canonical`. Never turn capture ON yourself, dump raw samples into chat, run training, or publish datasets. Provider selection requires prepared local configuration; do not download Python/models during a decision. Default capture is OFF.
