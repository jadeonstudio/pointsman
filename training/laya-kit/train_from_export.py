#!/usr/bin/env python3
# Copyright 2024 Convai Innovations (upstream `laya` package and the official
# fine-tuning notebook this file adapts).
# Copyright 2026 the pointsman project (pointsman-specific adaptation).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local/offline Laya training from pointsman exports.

Ship check_export.py beside this script. Version 3 exports isolate train/dev/
calibration/sealed test. Dev alone selects epochs and methods; calibration fits
final temperatures. Test inference requires --evaluate-sealed-test after the
saved candidate is frozen. Legacy v2 exports remain readable with fixed epochs
and --no-select-best-epoch, without claiming independent dev evaluation.

The default --method supervised trains the full encoder and dynamic scoring
parameters with masked distributional CE for Choice, Noul and Score. The
explicit rlcd-grpo method retains the pinned official notebook experiment.
--input-fit must match checkpoint registration and inference. No command here
promotes a checkpoint. Imports/model loading occur only after export validation.

Pinned upstream: NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
"""
import argparse
import importlib.util
import platform
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Upstream contract constants. These MUST match src/training/dataset.mjs
# (LAYA_UPSTREAM, LAYA_EXPORT_VERSION) exactly. If the export folder's
# manifest.json disagrees, this is either a stale export or a contract that
# has moved since this kit was written -- abort rather than guess.
# ---------------------------------------------------------------------------
EXPECTED_EXPORTER_VERSION = "laya-typed-decisions-json-v3"
LEGACY_EXPORTER_VERSION = "laya-typed-decisions-json-v2"
EXPECTED_UPSTREAM_CONTRACT = (
    "NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527:"
    "notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
)
EXPECTED_LOADER = (
    'datasets.load_dataset("json", data_files={"train": "train.jsonl", '
    '"validation": "calibration.jsonl", "test": "test.jsonl"})'
)
NOTEBOOK_COMMIT = "42626c348753fbb17572a813127df2278a1ec527"
LAYA_VERSION_PINNED = "0.3.4"  # matches training/laya-kit/requirements.lock


class KitError(RuntimeError):
    """Raised for any condition that should abort before training starts."""


def die(message):
    raise KitError(message)


# ---------------------------------------------------------------------------
# Step 0: verify the export contract before touching any data.
# pointsman-change: the official notebook has no such check because it always
# reads the public LocalLLaMA/typed-decisions dataset directly. Exporting a
# local pointsman dataset makes this check load-bearing.
# ---------------------------------------------------------------------------
def verify_export_manifest(export_dir: Path) -> dict:
    if not (export_dir / "manifest.json").is_file():
        die(f"{export_dir}/manifest.json not found -- run `pointsman dataset export --format laya` first")
    # One offline validator owns the exporter contract; ship check_export.py with this kit.
    spec = importlib.util.spec_from_file_location("pointsman_check_export", Path(__file__).with_name("check_export.py"))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    try:
        manifest = checker.verify_export(export_dir)
    except ValueError as exc:
        die(f"invalid export: {exc}")
    expected_loader = EXPECTED_LOADER if manifest["exporter_version"] == LEGACY_EXPORTER_VERSION else (
        'datasets.load_dataset("json", data_files={"train": "train.jsonl", "validation": "dev.jsonl", "calibration": "calibration.jsonl", "test": "test.jsonl"})')
    if manifest.get("loader") != expected_loader:
        die("export loader contract mismatch")
    return manifest


def selection_split(manifest, select_best_epoch):
    if manifest.get("exporter_version") == EXPECTED_EXPORTER_VERSION:
        return "dev"
    if select_best_epoch:
        die("legacy three-way exports have no independent dev split; re-export four-way data or use --no-select-best-epoch with fixed hyperparameters")
    return None


# ---------------------------------------------------------------------------
# pointsman-change: copied verbatim from workers/laya_worker.py (this kit must stay
# standalone for Kaggle, so it cannot import that module). `fit_task_head()`
# is the same opt-in "task-head fit" the inference worker applies when
# `providers.json` sets `laya.inputFit: 'task-head'` -- see
# workers/laya_worker.py. Applying the
# IDENTICAL fit here, before building each training item, keeps train-time
# preprocessing consistent with what inference actually sends when a
# checkpoint is registered with that opt-in; tests/test_laya_kit_input_fit.py
# asserts both copies produce identical output on synthetic cases.
# ---------------------------------------------------------------------------
def assert_lossless(agent, state, questions):
    from laya.common import render_options, serialize_state, build_sequence
    tok = agent.tok
    max_len, head_max = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    for question in questions.values():
        q = agent._to_internal(question)
        fields = [q["ins"], serialize_state(state), *render_options(q)]
        if any(tok.mask_token in value for value in fields):
            raise ValueError("INPUT_REWRITE_REFUSED")
        head = tok("%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)["input_ids"]
        options = [tok(" " + option, add_special_tokens=False)["input_ids"] for option in render_options(q)]
        if any(len(option) > 48 for option in options):
            raise ValueError("INPUT_TRUNCATED")
        budget = head_max - sum(len(option) + 1 for option in options)
        state_tokens = tok(serialize_state(state), add_special_tokens=False)["input_ids"]
        expected = len(head) + sum(len(option) + 1 for option in options) + len(state_tokens) + 4
        if budget < 16 or len(head) > max(8, budget) or expected > max_len:
            raise ValueError("INPUT_TRUNCATED")
        sequence, markers = build_sequence(tok, state, q, max_len, head_max)
        if len(sequence) != expected or len(markers) != len(options):
            raise ValueError("INPUT_TRUNCATED")

# ---------------------------------------------------------------------------
# pointsman-change: --model-subdir support and model_name derivation. The upstream
# HF repo (convaiinnovations/laya) keeps some checkpoint variants at the repo
# root (the english checkpoint, historically) and others under a per-variant
# subfolder (e.g. `multilingual/`, `typed-decisions/` -- confirmed locally via
# the huggingface_hub download cache layout under each variant's
# ~/.local/share/laya/models/<variant>/.cache/huggingface/download/ tree,
# which shows the subfolder name repeated as a path prefix for multilingual
# and typed-decisions but not for english). Neither this kit nor this
# environment can browse the HF repo's file tree directly to double-check
# that pattern against the live repo, so --model-subdir is left for the
# operator to pass explicitly rather than guessed/auto-detected.
# ---------------------------------------------------------------------------
def resolve_model_dir(model_dir, model_subdir):
    """Resolve --model-dir against an optional --model-subdir. Omitting
    --model-subdir (None or "") preserves the pre-existing behavior of using
    model_dir as-is."""
    model_dir = Path(model_dir)
    if not model_subdir:
        return model_dir
    return model_dir / model_subdir


def validate_resolved_model_dir(resolved_dir):
    """Fail clearly (KitError) if the resolved checkpoint dir is missing any of the
    files this kit and the official notebook both require."""
    resolved_dir = Path(resolved_dir)
    missing = []
    if not (resolved_dir / "rl_agent_config.json").is_file():
        missing.append("rl_agent_config.json")
    if not (resolved_dir / "tokenizer").is_dir():
        missing.append("tokenizer/")
    if not (resolved_dir / "encoder").is_dir():
        missing.append("encoder/")
    if missing:
        die(
            f"{resolved_dir} is missing {', '.join(missing)} -- check --model-dir/"
            "--model-subdir (the HF repo may keep this checkpoint under a per-variant "
            "subfolder, e.g. `multilingual/`)"
        )
    return resolved_dir


def derive_model_name(base_cfg, base_model_dir_name):
    """Derive the fine-tuned checkpoint's model_name from the base checkpoint's own
    rl_agent_config.json (falling back to the resolved model dir's name) instead of
    hard-coding 'laya-typed-decisions', which is specific to one checkpoint variant
    and wrong for any other (e.g. multilingual)."""
    base_name = (base_cfg or {}).get("model_name") or base_model_dir_name
    if not base_name:
        die(
            "cannot derive model_name: base rl_agent_config.json has no model_name and "
            "the resolved model dir has no usable name"
        )
    return f"{base_name}-pointsman-ft"


# pointsman-change: official-notebook-cell 4 (inside TRAIN_DDP_SCRIPT below) overrides
# cfg['max_len']/cfg['head_max_len'] to 1024/256 before building the model. These
# constants mirror that literal override so fit_task_head()/build_training_item() in
# THIS process admit sequences against the SAME final budget the trained model actually
# uses -- not the base checkpoint's own (possibly smaller) max_len/head_max_len, which
# would otherwise let preprocessing admit sequences the model's real budget disagrees
# with. If TRAIN_DDP_SCRIPT's override values ever change, these must change with them.
DEFAULT_FINAL_MAX_LEN = 1024
DEFAULT_FINAL_HEAD_MAX_LEN = 256


def resolve_effective_cfg(base_cfg, max_len=DEFAULT_FINAL_MAX_LEN, head_max_len=DEFAULT_FINAL_HEAD_MAX_LEN):
    """Return a NEW cfg dict (base_cfg is not mutated) with max_len/head_max_len set to
    the final training-time values, matching TRAIN_DDP_SCRIPT's override."""
    effective = dict(base_cfg or {})
    effective["max_len"] = max_len
    effective["head_max_len"] = head_max_len
    return effective


# ---------------------------------------------------------------------------
# pointsman-change: --local (single-process, no torchrun/NCCL/DDP) mode support.
# The official recipe's effective global batch is MICRO_BATCH(8) * GRAD_ACCUM(4)
# * DDP_WORLD_SIZE(2) = 64. In --local mode world_size is always 1 (no DDP rank
# split), so to keep the SAME effective global batch (and therefore the same
# optimizer update granularity / LR schedule shape) the default --grad-accum is
# scaled up by the DDP world size the recipe assumes, unless the operator
# explicitly overrides --grad-accum.
# ---------------------------------------------------------------------------
DEFAULT_MICRO_BATCH = 8
DEFAULT_DDP_GRAD_ACCUM = 4
DEFAULT_DDP_WORLD_SIZE = 2
EFFECTIVE_GLOBAL_BATCH = DEFAULT_MICRO_BATCH * DEFAULT_DDP_GRAD_ACCUM * DEFAULT_DDP_WORLD_SIZE  # 64


def resolve_local_batch_and_grad_accum(batch_size_arg, grad_accum_arg):
    """Resolve --batch-size/--grad-accum for --local mode.

    Default --batch-size is DEFAULT_MICRO_BATCH (8, same as the DDP MICRO_BATCH).
    Default --grad-accum (when not explicitly given) is computed so that
    batch_size * grad_accum == EFFECTIVE_GLOBAL_BATCH (64) -- i.e. losing DDP's
    world_size=2 is compensated by doubling grad-accum, and shrinking
    --batch-size (e.g. to dodge an OOM) proportionally grows grad-accum to keep
    the same effective global batch. An explicit --grad-accum always wins.
    """
    batch_size = batch_size_arg if batch_size_arg is not None else DEFAULT_MICRO_BATCH
    if batch_size < 1:
        die(f"--batch-size must be >= 1, got {batch_size}")
    if grad_accum_arg is not None:
        if grad_accum_arg < 1:
            die(f"--grad-accum must be >= 1, got {grad_accum_arg}")
        return batch_size, grad_accum_arg
    grad_accum = max(1, round(EFFECTIVE_GLOBAL_BATCH / batch_size))
    return batch_size, grad_accum


def validate_regularization_args(dropout, rdrop_alpha, local):
    """pointsman-change: --dropout / --rdrop-alpha (2026-09-24, both default OFF).
    Validated before any export/model work so a bad value fails fast, and
    rejected outside --local so a DDP run can never silently ignore them."""
    if dropout is not None and not (math.isfinite(dropout) and 0.0 < dropout <= 0.5):
        die(f"--dropout must satisfy 0 < P <= 0.5, got {dropout}")
    if not (math.isfinite(rdrop_alpha) and rdrop_alpha >= 0.0):
        die(f"--rdrop-alpha must be a finite value >= 0, got {rdrop_alpha}")
    if rdrop_alpha > 0 and dropout is None:
        die("--rdrop-alpha > 0 requires --dropout P (0 < P <= 0.5): R-Drop regularizes the gap between two "
            "dropout-perturbed forward passes, and the base encoder config has encoder dropout 0.0")
    if (dropout is not None or rdrop_alpha > 0) and not local:
        die("--dropout/--rdrop-alpha are --local only (the DDP torchrun path does not pass them through)")


DEFAULT_EPOCHS = 4  # TRAIN_DDP_SCRIPT's EPOCHS (the official notebook's value)
MAX_EPOCHS = 20


def validate_epochs_and_resume_args(epochs, resume, keep_resume, local):
    """pointsman-change: --epochs / --resume / --keep-resume (2026-09-25, --local only).
    Returns the resolved epoch count (DEFAULT_EPOCHS when --epochs is omitted).
    Rejected outside --local like --dropout, so a DDP run never silently ignores them."""
    if epochs is not None and not (1 <= epochs <= MAX_EPOCHS):
        die(f"--epochs must satisfy 1 <= N <= {MAX_EPOCHS}, got {epochs}")
    if (epochs is not None or resume or keep_resume) and not local:
        die("--epochs/--resume/--keep-resume are --local only (the DDP torchrun path does not pass them through)")
    return DEFAULT_EPOCHS if epochs is None else epochs


# ---------------------------------------------------------------------------
# pointsman-change: --local resume support (2026-09-25). TRAIN_DDP_SCRIPT's
# write_resume_checkpoint() writes <output-dir>/resume/{LATEST,epoch-000N/...};
# this side reads it back, refuses a resume whose saved configuration differs
# from the current arguments, and decides whether the cached train/calib items
# can be reused. Resume granularity is one completed epoch.
# ---------------------------------------------------------------------------
RESUME_FORMAT = "laya-kit-resume-v1"  # must equal TRAIN_DDP_SCRIPT's RESUME_FORMAT
RESUME_DIRNAME = "resume"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def items_digest(items):
    """Content digest of preprocessed training items (independent of torch.save's
    byte layout), recorded with the resume state so a resumed run provably
    trains on the same items the interrupted run did."""
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def directory_sha256(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return hashlib.sha256(json.dumps([[p.relative_to(root).as_posix(), file_sha256(p)] for p in files], separators=(",", ":")).encode()).hexdigest()


def build_resume_match_config(*, export_dir, manifest, resolved_model_dir, derived_model_name, epochs,
                              micro_batch, grad_accum, device, mps_autocast, dropout, rdrop_alpha,
                              select_best_epoch, cfg, method="rlcd-grpo", input_fit="task-head"):
    """Everything that must be identical for a resumed run to continue the SAME run.
    --max-steps, --keep-resume and the admission-check flags are deliberately
    excluded (they bound or gate a run without changing its math)."""
    export_dir = Path(export_dir)
    resolved_model_dir = Path(resolved_model_dir)
    return {
        "format": RESUME_FORMAT,
        "train_script_sha256": hashlib.sha256(TRAIN_DDP_SCRIPT.encode("utf-8")).hexdigest(),
        "export_manifest_sha256": file_sha256(export_dir / "manifest.json"),
        "export_train_sha256": file_sha256(export_dir / "train.jsonl"),
        "export_calibration_sha256": file_sha256(export_dir / "calibration.jsonl"),
        "export_dev_sha256": file_sha256(export_dir / "dev.jsonl") if (export_dir / "dev.jsonl").is_file() else None,
        "export_split_hashes": manifest.get("split_hashes"),
        "export_provenance_sha256": manifest.get("provenance_sha256"),
        "tokenizer_sha256": directory_sha256(resolved_model_dir / "tokenizer"),
        "model_config_sha256": file_sha256(resolved_model_dir / "rl_agent_config.json") if (resolved_model_dir / "rl_agent_config.json").is_file() else None,
        "runtime": {"python": platform.python_version(), "torch": getattr(sys.modules.get("torch"), "__version__", None), "laya": LAYA_VERSION_PINNED},
        "exporter_version": manifest.get("exporter_version"),
        "export_dataset_version": manifest.get("dataset_version"),
        "export_source_data_sha256": manifest.get("source_data_sha256"),
        "model_dir": os.path.realpath(resolved_model_dir),
        "model_weights_sha256": file_sha256(resolved_model_dir / "model.safetensors"),
        "model_name": derived_model_name,
        "epochs": epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "device": device,
        "mps_autocast": mps_autocast,
        "dropout": dropout,
        "rdrop_alpha": rdrop_alpha,
        "select_best_epoch": select_best_epoch,
        "method": method, "input_fit": input_fit,
        "max_len": cfg.get("max_len"),
        "head_max_len": cfg.get("head_max_len"),
    }


def diff_resume_config(saved, current):
    """List of human-readable mismatches between a saved and the current config."""
    return [f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
            for key in sorted(set(saved) | set(current)) if saved.get(key) != current.get(key)]


def read_resume_state(output_dir):
    """The latest completed-epoch resume state under <output-dir>/resume/, or
    None when there is none (no LATEST pointer). A LATEST that points at a
    missing/incomplete epoch dir is an error, never "no state"."""
    root = Path(output_dir) / RESUME_DIRNAME
    latest = root / "LATEST"
    if not latest.is_file():
        return None
    epoch_dir = root / latest.read_text(encoding="utf-8").strip()
    state_path = epoch_dir / "state.json"
    if not state_path.is_file() or not (epoch_dir / "training_state.pt").is_file():
        die(f"{latest} points at {epoch_dir}, which is missing state.json/training_state.pt -- "
            "the resume state is damaged; delete the resume directory to start over")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["epoch_dir"] = str(epoch_dir)
    return state


def check_resume_request(output_dir, current_match, resume):
    """--resume: return the saved state after verifying its config matches exactly
    (dies listing every mismatch). Without --resume: die if a resume state exists,
    so a fresh run can never overwrite an interrupted one by accident."""
    state = read_resume_state(output_dir)
    root = Path(output_dir) / RESUME_DIRNAME
    if not resume:
        if state is not None:
            die(f"{root} holds an interrupted run (epochs_completed={state.get('epochs_completed')}). "
                "Pass --resume to continue it, or delete that directory to start a new run.")
        return None
    if state is None:
        die(f"--resume given but no resume state exists under {root} (nothing to resume; "
            "an interruption before the first completed epoch leaves no checkpoint). "
            "Re-run without --resume to start over.")
    mismatches = diff_resume_config(state.get("config") or {}, current_match)
    if state.get("format") != RESUME_FORMAT:
        mismatches.insert(0, f"format: saved={state.get('format')!r} current={RESUME_FORMAT!r}")
    if mismatches:
        die("refusing to resume: the saved run configuration differs from the current arguments:\n  "
            + "\n  ".join(mismatches))
    print(f"[resume] configuration matches; continuing from {state['epoch_dir']} "
          f"(epochs_completed={state['epochs_completed']}/{current_match['epochs']})")
    return state


def load_cached_items_for_resume(train_items_path, calib_items_path, resume_info, torch_module):
    """Reuse <output-dir>/train_items.pt/calib_items.pt on --resume only when both
    exist and their content digests equal the ones recorded by the interrupted run
    (whose config -- export manifest/split hashes, model, max_len -- was already
    verified to match). Returns (train_items, calib_items) or None (re-preprocess)."""
    if not (Path(train_items_path).is_file() and Path(calib_items_path).is_file()):
        return None
    train_items = torch_module.load(train_items_path, weights_only=False)
    calib_items = torch_module.load(calib_items_path, weights_only=False)
    if (items_digest(train_items) != resume_info.get("train_items_digest")
            or items_digest(calib_items) != resume_info.get("calib_items_digest")):
        print("[resume] cached train_items.pt/calib_items.pt do not match the interrupted run; re-preprocessing")
        return None
    return train_items, calib_items


def format_oom_message(batch_size, grad_accum):
    """Build the clear-error suggestion for a local-mode OOM. Kept as a pure,
    independently testable function; TRAIN_DDP_SCRIPT's main_local() embeds the
    identical logic inline (it must stay a standalone script for Kaggle/local
    subprocess execution and cannot import this module) -- see
    tests/test_laya_kit_local_mode.py for the parity check between the two."""
    suggested_batch = max(1, batch_size // 2)
    scale = max(1, batch_size // suggested_batch)
    suggested_grad_accum = grad_accum * scale
    return (
        f"out of memory at --batch-size={batch_size} --grad-accum={grad_accum}. "
        f"Retry with --batch-size {suggested_batch} --grad-accum {suggested_grad_accum} "
        "(keeps the same effective global batch)."
    )


def load_jsonl_rows(path):
    """stdlib-only JSONL reader for --local mode (the laya venv does not have the
    `datasets` package installed). Returns the SAME row shape `datasets.load_dataset`
    would hand back per-row: a dict with string-valued "state"/"questions"/"gold"
    keys (see check_export.py's check_row_columns), which preprocess_split() already
    consumes via row["state"]/row["questions"]/row["gold"] dict indexing regardless
    of whether `rows` is a datasets.Dataset or a plain list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


TRUNCATION_MARK = " …[truncated]"

def fit_task_head(agent, state, questions):
    """Default lossless refusal stays the behavior of assert_lossless(); this is the opt-in path
    (see providers.json `laya.inputFit`) that instead finds the longest character prefix of
    state["task"] that still fits ALL given questions losslessly, leaving every other state key
    (context) untouched. Never truncates questions/instructions/options. Correctness is always
    verified by assert_lossless before returning; token-based estimation below only narrows the
    search window, it never decides the final answer.
    """
    try:
        assert_lossless(agent, state, questions)
        return state, {"truncated": False}
    except ValueError as e:
        original_error = e
    if str(original_error) == "INPUT_REWRITE_REFUSED" or not (isinstance(state, dict) and isinstance(state.get("task"), str)):
        raise original_error
    task = state["task"]
    original_chars = len(task)

    def candidate(p):
        return {**state, "task": task[:p].rstrip() + TRUNCATION_MARK}

    def fits(p):
        try:
            assert_lossless(agent, candidate(p), questions)
            return True
        except ValueError:
            return False

    # Narrow with a cheap analytic token-budget estimate before a short gallop+binary search. Each
    # fits() probe below re-runs assert_lossless() in full (head/options/state tokenize +
    # build_sequence), which is by far the most expensive step, so the estimate's job is to make
    # that search cover only a handful of characters instead of the whole task -- it is never
    # itself the answer, and fits(0) is always reached as a real probe below (never skipped), so
    # "no prefix passes" is still detected and raises the original error, exactly as before.
    try:
        from laya.common import render_options, serialize_state
        tok, max_len = agent.tok, agent.cfg.get("max_len", 512)
        min_state_budget = None
        for question in questions.values():
            q = agent._to_internal(question)
            head = tok("%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)["input_ids"]
            options = [tok(" " + option, add_special_tokens=False)["input_ids"] for option in render_options(q)]
            state_budget = max_len - len(head) - sum(len(o) + 1 for o in options) - 4
            min_state_budget = state_budget if min_state_budget is None else min(min_state_budget, state_budget)
        base_tokens = len(tok(serialize_state(candidate(0)), add_special_tokens=False)["input_ids"])
        task_tokens = max(1, len(tok(task, add_special_tokens=False)["input_ids"]))
        chars_per_token = original_chars / task_tokens
        remaining_tokens = max(0, min_state_budget - base_tokens)
        guess = min(original_chars, max(0, int(remaining_tokens * chars_per_token)))
    except Exception:
        guess = 0

    # Gallop out from the guess (doubling step) to bracket the true boundary in a few probes, then
    # binary search only that small bracket -- not the whole [0, original_chars] range, which is
    # what made an unbracketed binary search from a lone guess cost extra probes in practice.
    if fits(guess):
        lo, hi, probe, step = guess, original_chars, guess, 1
        while probe < hi:
            probe = min(hi, probe + step)
            if fits(probe):
                lo = probe
                step *= 2
            else:
                hi = probe
                break
    else:
        hi, probe, step, lo = guess, guess, 1, None
        while True:
            probe = max(0, probe - step)
            if fits(probe):
                lo = probe
                break
            hi = probe
            if probe == 0:
                break
            step *= 2
        if lo is None:
            raise original_error
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return candidate(lo), {"truncated": True, "original_chars": original_chars, "kept_chars": lo}


# ---------------------------------------------------------------------------
# official-notebook-cell: 3 ("Download & Preprocess Data for DDP")
# pointsman-change: `ds_train = load_dataset("LocalLLaMA/typed-decisions", ...)` is
# replaced by reading the local export split files via the manifest's own
# recorded loader string. `build_training_item()` is otherwise byte-for-byte
# identical to the notebook because the export's gold[qid]["probabilities"]
# shape already matches what the notebook expects.
# ---------------------------------------------------------------------------
def build_training_item(tok, cfg, state, q, gold_q, render_options, build_sequence, QTYPES):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        die(f"unknown question type {t!r}")

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))

    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None, True  # pointsman-change: also return a truncation flag instead of silently discarding
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
    }, False


def preprocess_split(rows, tok, cfg, render_options, build_sequence, QTYPES, agent, input_fit="task-head"):
    """official-notebook-cell: 3 inner loop, generalized to any split's rows.

    pointsman-change: before building any question's training item for a row, the row's state is passed
    through fit_task_head() using ALL of that row's own questions (the same question set the
    export groups gold by, per state) -- the identical fit workers/laya_worker.py applies at
    inference when `providers.json` sets `laya.inputFit: 'task-head'`. Only state["task"] is ever
    shortened; every other state key, and every question/instructions/option, is untouched. The
    existing marker-loss drop (`build_training_item`'s own truncation check) still applies
    afterward, on the fitted state.
    """
    items = []
    truncated = 0
    total_questions = 0
    input_fit_truncated_states = 0
    for row in rows:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        try:
            if input_fit == "lossless":
                assert_lossless(agent, state, questions)
                fitted_state, fit_info = state, {"truncated": False}
            else:
                fitted_state, fit_info = fit_task_head(agent, state, questions)
        except ValueError:
            # Neither the raw state nor any task prefix fits (e.g. context/questions alone
            # overflow): every question of this row is dropped the same way a marker-loss would be.
            truncated += len(questions)
            total_questions += len(questions)
            continue
        if fit_info["truncated"]:
            input_fit_truncated_states += 1
        for qid, q in questions.items():
            if qid not in gold:
                continue
            total_questions += 1
            item, was_truncated = build_training_item(tok, cfg, fitted_state, q, gold[qid], render_options, build_sequence, QTYPES)
            if was_truncated:
                truncated += 1
                continue
            if item is not None:
                items.append(item)
    print(f"[input-fit] {input_fit_truncated_states} states task-head truncated")
    return items, truncated, total_questions


# pointsman-change: tokenizer admission check the official notebook does not
# perform. The notebook silently drops any row whose sequence does not fit
# max_len/head_max_len (`if len(markers) != k: return None`). Silent
# truncation loss is exactly the failure mode the pointsman training-data spec forbids
# ("do not apply this if the input would be truncated, or would need a mask
# token substituted, to fit the official tokenizer's head/options/state budget"),
# so this kit reports it and can abort.
def check_tokenizer_admission(truncated, total_questions, max_truncated_fraction, allow_truncation):
    fraction = (truncated / total_questions) if total_questions else 0.0
    print(f"[tokenizer-admission] {truncated}/{total_questions} question-rows "
          f"({fraction:.2%}) were truncated by the official tokenizer budget and dropped")
    if truncated and not allow_truncation and fraction > max_truncated_fraction:
        die(
            f"tokenizer admission failure: {fraction:.2%} of rows truncated, exceeds "
            f"--max-truncated-fraction={max_truncated_fraction:.2%}. Re-run with "
            "--allow-truncation to proceed anyway, or shorten state/instructions upstream."
        )


# ---------------------------------------------------------------------------
# official-notebook-cell: 4 ("DDP Training Script (train_ddp.py)")
# pointsman-change: (1) the calibration set is no longer `all_items[::15][:400]`
# sampled out of the training data -- it is the export's own held-out
# calibration.jsonl, preprocessed the same way as train, and passed in as
# CALIB_ITEMS_PATH. (2) EPOCHS/MICRO_BATCH/etc. are left as the notebook's
# literal values; nothing else in the training math (RLCD/GRPO loss, DDP
# wiring, AdamW param groups, cosine schedule) is modified.
# ---------------------------------------------------------------------------
TRAIN_DDP_SCRIPT = r'''
# official-notebook-cell: 4 ("DDP Training Script (train_ddp.py)") -- copied
# from the pinned notebook with only the pointsman-change noted at the top of
# train_from_export.py (calibration items come from CALIB_ITEMS_PATH, not an
# in-training-set slice).
#
# pointsman-change: the per-batch loss/optimizer/schedule loop (official-notebook-cell
# 4's inner training loop) and the post-training calibration+save step are
# extracted into run_training_loop()/finalize_and_save() so --local mode
# (single process, no torchrun/NCCL/DDP -- see main_local()) can reuse the
# IDENTICAL math with world_size=1/rank=0. main_ddp() below is byte-for-byte
# the same DDP wiring and hyperparameters as before this refactor; only the
# loop body moved into a shared function.
import os, sys, time, json, random, math
import copy, shutil, tempfile  # pointsman-change: opt-in --dropout (build_model_with_encoder_dropout), --local resume checkpoints
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES

EPOCHS = 4
MICRO_BATCH = 8       # 8 sequences per forward pass per GPU/process
GROUP_SIZE = 4         # GRPO baseline samples
TRAIN_METHOD = "rlcd-grpo"  # outer CLI chooses supervised by default; legacy embedded callers preserved
LR_ENCODER = 2.5e-5    # Encoder adaptation rate
LR_HEAD = 1.0e-4       # Head adaptation rate
SIGMA_START = 0.4      # Exploration noise
SIGMA_END = 0.1


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items])
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def build_optimizer_and_scheduler(named_params, n_items, micro_batch, grad_accum, epochs):
    enc_params = [p for n, p in named_params if "encoder." in n]
    head_params = [p for n, p in named_params if "encoder." not in n]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)
    total_updates = (n_items // (micro_batch * grad_accum)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    return optimizer, scheduler


def oom_suggestion(batch_size, grad_accum):
    # pointsman-change: --local memory safety. Kept textually identical to
    # train_from_export.py's format_oom_message() (this script must stay a
    # standalone file runnable by torchrun/python3 on its own, so it cannot
    # import that module) -- tests/test_laya_kit_local_mode.py checks parity.
    suggested_batch = max(1, batch_size // 2)
    scale = max(1, batch_size // suggested_batch)
    suggested_grad_accum = grad_accum * scale
    return (
        f"out of memory at --batch-size={batch_size} --grad-accum={grad_accum}. "
        f"Retry with --batch-size {suggested_batch} --grad-accum {suggested_grad_accum} "
        "(keeps the same effective global batch)."
    )


# pointsman-change: opt-in regularization (--dropout / --rdrop-alpha, --local only,
# 2026-09-24). Round-2 local training overfit badly (train-subset agreement
# 0.98/0.92/0.94 vs held-out 0.81/0.64/0.59) and the multilingual encoder's
# config.json ships attention/embedding/mlp dropout = 0.0, so the encoder
# trained with no dropout at all (DecisionModel's 2-layer head keeps its own
# hard-coded nn.TransformerEncoderLayer dropout=0.1 either way). Both flags
# default OFF, and with them off every function below reduces to the exact
# pre-change code path.
ENCODER_DROPOUT_KEYS = ("attention_dropout", "embedding_dropout", "mlp_dropout")


def build_model_with_encoder_dropout(cfg, encoder_dir, dropout):
    """dropout=None: exactly build_model(cfg, encoder_dir=encoder_dir); returns (model, None).

    Otherwise builds the model from a temporary copy of encoder_dir whose
    config.json sets ENCODER_DROPOUT_KEYS to `dropout`, and returns
    (model, original) where `original` maps those keys to the checkpoint's own
    values -- finalize_and_save() writes them back so the saved encoder/config.json
    (inference config and identity) is unchanged. The values must be in place at
    construction time: ModernBERT copies them into nn.Dropout(p) and
    ModernBertAttention.attention_dropout in __init__, and builds out_drop as
    nn.Identity when attention_dropout == 0, so mutating the config (or the
    Dropout modules) after build_model() would silently miss part of it. The
    laya package's build_model() only takes an encoder directory, hence the
    temporary copy instead of a config override."""
    if dropout is None:
        return build_model(cfg, encoder_dir=encoder_dir), None
    with open(os.path.join(encoder_dir, "config.json")) as f:
        enc_cfg = json.load(f)
    original = {k: enc_cfg.get(k) for k in ENCODER_DROPOUT_KEYS}
    tmp_root = tempfile.mkdtemp(prefix="laya-encoder-dropout-")
    try:
        tmp_encoder_dir = os.path.join(tmp_root, "encoder")
        shutil.copytree(encoder_dir, tmp_encoder_dir)
        enc_cfg.update({k: float(dropout) for k in ENCODER_DROPOUT_KEYS})
        with open(os.path.join(tmp_encoder_dir, "config.json"), "w") as f:
            json.dump(enc_cfg, f, indent=2)
        model = build_model(cfg, encoder_dir=tmp_encoder_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return model, original


def encoder_dropout_report(encoder):
    """What the built encoder modules will actually apply in train mode (not
    what its config claims): nn.Dropout p values and attention-probability
    dropout (ModernBertAttention.attention_dropout)."""
    dropout_ps = [m.p for m in encoder.modules() if isinstance(m, torch.nn.Dropout)]
    attn_ps = [m.attention_dropout for m in encoder.modules()
               if isinstance(getattr(m, "attention_dropout", None), float)]
    return {
        "dropout_modules": len(dropout_ps),
        "active_dropout_modules": sum(1 for p in dropout_ps if p > 0),
        "dropout_module_ps": sorted(set(dropout_ps)),
        "attention_modules": len(attn_ps),
        "attention_dropout_values": sorted(set(attn_ps)),
    }


def assert_encoder_dropout_applied(encoder, dropout):
    report = encoder_dropout_report(encoder)
    ok = (report["active_dropout_modules"] > 0
          and report["active_dropout_modules"] == report["dropout_modules"]
          and report["dropout_module_ps"] == [float(dropout)]
          and report["attention_modules"] > 0
          and report["attention_dropout_values"] == [float(dropout)])
    if not ok:
        raise RuntimeError(f"--dropout {dropout} did not take effect in the encoder modules: {report}")
    return report


def forward_micro_batch(forward_fn, batch, device, autocast_device, autocast_dtype, autocast_enabled):
    if autocast_enabled:
        with torch.autocast(autocast_device, dtype=autocast_dtype):
            return forward_fn(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device)
            )
    return forward_fn(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device)
    )


def rl_ce_loss_terms(logits, batch, device, group_size, sigma):
    """official-notebook-cell 4's per-micro-batch RLCD/GRPO + CE terms, moved
    here unchanged from run_training_loop(). Returns (loss_rl + 1.0 * loss_ce, r)."""
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    k = mask.sum(-1, keepdim=True).float()
    target = batch["target"].to(device)

    if TRAIN_METHOD == "supervised":
        log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
        loss_ce = -(target * log_probs).sum(-1).mean()
        with torch.no_grad():
            reward = proper_reward(log_probs.exp(), target, batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
        return loss_ce, reward

    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + 1.0 * loss_ce, r


def micro_batch_loss(logits, act, batch, device, group_size, sigma, grad_accum):
    """The default (no R-Drop) per-micro-batch loss -- identical to the
    pre-extraction `(loss_rl + 1.0 * loss_ce) / grad_accum + 0.0 * act.sum()`."""
    base, r = rl_ce_loss_terms(logits, batch, device, group_size, sigma)
    return base / grad_accum + 0.0 * act.sum(), r


def masked_symmetric_kl(logits1, logits2, mask):
    """R-Drop's symmetric KL, 0.5 * (KL(p1||p2) + KL(p2||p1)), between the two
    passes' per-question distributions over VALID options only (the same
    marker_mask / masked_fill(-1e4) handling loss_ce uses; choice, score and
    noul rows are all one row of marker logits here). Summed over options,
    averaged over questions like loss_ce."""
    mask_f = mask.float()
    lp1 = torch.log_softmax(logits1.float().masked_fill(~mask, -1e4), -1)
    lp2 = torch.log_softmax(logits2.float().masked_fill(~mask, -1e4), -1)
    kl12 = (lp1.exp() * (lp1 - lp2) * mask_f).sum(-1)
    kl21 = (lp2.exp() * (lp2 - lp1) * mask_f).sum(-1)
    return (0.5 * (kl12 + kl21)).mean()


def rdrop_micro_batch_loss(out1, out2, batch, device, group_size, sigma, grad_accum, alpha):
    """R-Drop (Liang et al., NeurIPS 2021): mean of the two train-mode passes'
    existing losses + alpha * symmetric KL between them, scaled by 1/grad_accum
    exactly like micro_batch_loss(). Returns (loss, mean reward, kl)."""
    (logits1, act1), (logits2, act2) = out1, out2
    base1, r1 = rl_ce_loss_terms(logits1, batch, device, group_size, sigma)
    base2, r2 = rl_ce_loss_terms(logits2, batch, device, group_size, sigma)
    kl = masked_symmetric_kl(logits1, logits2, batch["marker_mask"].to(device))
    loss = (0.5 * (base1 + base2) + alpha * kl) / grad_accum + 0.0 * (act1.sum() + act2.sum())
    return loss, 0.5 * (r1 + r2), kl


def compute_calib_agreement(calib_preds):
    # pointsman-change: per-epoch best-checkpoint selection (2026-09-23: a real
    # local run overfit badly -- train argmax agreement ~0.99/0.92/0.95 vs
    # held-out test ~0.72/0.58/0.60). Kept textually identical to
    # train_from_export.py's compute_calib_agreement() (this script must stay
    # standalone -- tests/test_laya_kit_eval_and_epoch_select.py checks
    # parity). QTYPES: choice=0, score=1, noul=2 (laya.common.QTYPES).
    names = {0: "choice", 1: "score", 2: "noul"}
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    for qtype, logits, target in calib_preds:
        if not logits or not target:
            continue
        pred_idx = max(range(len(logits)), key=lambda i: logits[i])
        gold_idx = max(range(len(target)), key=lambda i: target[i])
        total[qtype] += 1
        if pred_idx == gold_idx:
            correct[qtype] += 1
    agreement = {names[qt]: (correct[qt] / total[qt] if total[qt] else None) for qt in names}
    scored = [v for v in agreement.values() if v is not None]
    agreement["mean"] = (sum(scored) / len(scored)) if scored else None
    return agreement


def collect_calib_logits(model, calib_items, tok, device, autocast_device, autocast_dtype, autocast_enabled):
    """Run calib_items through model in eval mode; return (qtype, logits, target)
    tuples in the shape compute_calib_agreement() consumes. Shared by
    epoch_end_fn (best-epoch selection) and finalize_and_save (temperature
    fitting) so both read calibration items through the identical forward
    pass."""
    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 16):
            c_chunk = calib_items[c_idx:c_idx + 16]
            cb = collate_train_batch(c_chunk, tok.pad_token_id)
            if autocast_enabled:
                with torch.autocast(autocast_device, dtype=autocast_dtype):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device)
                    )
            else:
                l_sub, _ = model(
                    cb["input_ids"].to(device),
                    cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device),
                    cb["marker_mask"].to(device),
                    cb["qtype"].to(device)
                )
            l_np = l_sub.float().cpu().numpy()
            for r_idx, it in enumerate(c_chunk):
                k = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r_idx, :k].tolist(), it["target"]))
    return calib_preds


def make_epoch_end_fn(model, dev_items, tok, device, *, autocast_device, autocast_dtype, autocast_enabled,
                       rank, world_size, dist_module, select_best_epoch, best_state, epoch_agreements, log_prefix,
                       keep_best_in_memory=True):
    """Build the epoch_end_fn callback run_training_loop calls after every
    epoch (DDP: only rank 0 evaluates, then all ranks barrier so training
    stays in lockstep). Logs "[select] epoch N calib_agreement ..." and, when
    the mean development agreement improves, keeps a CPU copy of the model's
    state_dict in best_state (freed/loaded back in main_ddp()/main_local()
    after training finishes).

    pointsman-change: keep_best_in_memory=False (--local, 2026-09-25) records only the
    best score/epoch; the weights go to disk as the resume checkpoint's
    best_model.pt (write_resume_checkpoint()) instead of a ~1.3GB CPU copy."""
    def epoch_end_fn(epoch):
        if select_best_epoch and rank == 0:
            calib_preds = collect_calib_logits(model, dev_items, tok, device,
                                                autocast_device, autocast_dtype, autocast_enabled)
            agreement = compute_calib_agreement(calib_preds)
            epoch_agreements.append({"epoch": epoch + 1, **agreement})

            def fmt(v):
                return "n/a" if v is None else f"{v:.4f}"

            print(f"{log_prefix} [select] epoch {epoch + 1} dev_agreement "
                  f"choice={fmt(agreement['choice'])} score={fmt(agreement['score'])} "
                  f"noul={fmt(agreement['noul'])} mean={fmt(agreement['mean'])}")
            mean_score = agreement["mean"]
            if mean_score is not None and (best_state["score"] is None or mean_score > best_state["score"]):
                best_state["score"] = mean_score
                best_state["epoch"] = epoch + 1
                if keep_best_in_memory:
                    best_state["state_dict"] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            model.train()
        if world_size > 1:
            dist_module.barrier()
    return epoch_end_fn


def run_training_loop(forward_fn, items, params_for_clip, optimizer, scheduler, tok, *,
                       device, epochs, micro_batch, grad_accum, group_size, sigma_start, sigma_end,
                       rank, world_size, use_scaler, scaler, autocast_device, autocast_dtype,
                       autocast_enabled, max_steps, mem_log_fn, log_prefix, epoch_end_fn=None,
                       rdrop_alpha=0.0, start_epoch=0, start_global_step=0, checkpoint_fn=None):
    """official-notebook-cell 4's inner training loop, generalized over
    (forward_fn, world_size, autocast/scaler settings) so it is IDENTICAL for
    DDP (world_size=2, forward_fn=ddp_model, cuda fp16 autocast+GradScaler) and
    --local (world_size=1, forward_fn=model, fp32 by default / optional MPS
    bf16 autocast, no GradScaler).

    pointsman-change: rdrop_alpha > 0 (--rdrop-alpha, --local only) runs each
    micro-batch's forward twice in train mode and uses rdrop_micro_batch_loss();
    the default 0.0 keeps the single-forward micro_batch_loss() path.

    pointsman-change: --resume (--local only, 2026-09-25). start_epoch/start_global_step
    continue a run restored from a resume checkpoint: the completed epochs'
    in-place shuffles are replayed (random.shuffle mutates `items`, so epoch N's
    order depends on every earlier shuffle) and their training is skipped.
    checkpoint_fn(epoch, global_step, elapsed_s) runs after each COMPLETED
    epoch, after epoch_end_fn; never for a --max-steps partial epoch. The
    defaults (0, 0, None) are the unchanged pre-resume loop."""
    t0 = time.time()
    global_step = start_global_step
    epochs_completed = start_epoch
    for epoch in range(epochs):
        random.seed(42 + epoch + rank)
        random.shuffle(items)
        if epoch < start_epoch:
            continue
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, epochs - 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress

        for b_idx in range(0, len(items), micro_batch):
            chunk = items[b_idx:b_idx + micro_batch]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            try:
                if rdrop_alpha > 0:
                    # pointsman-change: R-Drop -- two train-mode forwards (independent dropout masks).
                    out1 = forward_micro_batch(forward_fn, batch, device, autocast_device, autocast_dtype, autocast_enabled)
                    out2 = forward_micro_batch(forward_fn, batch, device, autocast_device, autocast_dtype, autocast_enabled)
                    loss, r, rdrop_kl = rdrop_micro_batch_loss(out1, out2, batch, device, group_size, sigma,
                                                                grad_accum, rdrop_alpha)
                else:
                    logits, act = forward_micro_batch(forward_fn, batch, device, autocast_device, autocast_dtype, autocast_enabled)
                    loss, r = micro_batch_loss(logits, act, batch, device, group_size, sigma, grad_accum)

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[FATAL] {oom_suggestion(micro_batch, grad_accum)}", file=sys.stderr)
                    sys.exit(1)
                raise

            accum_step += 1

            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(items):
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    # pointsman-change: padded batches have many distinct shapes; without releasing the MPS
                    # caching allocator here, its working set grows until the machine swaps
                    # (observed 2026-09-23, M4 Pro 24GB). CUDA/DDP path unchanged.
                    torch.mps.empty_cache()

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            global_step += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                mem = f" | MPS: {torch.mps.driver_allocated_memory() / 2**20:.0f}MiB" if device.type == "mps" else ""
                rdrop = f" | RDropKL: {rdrop_kl.item():.4f}" if rdrop_alpha > 0 else ""
                print(f"  {log_prefix} Epoch {epoch+1}/{epochs} | Step {n_batches} | Loss: {loss.item()*grad_accum:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}{mem}{rdrop}")

            if max_steps and global_step >= max_steps:
                if rank == 0:
                    print(f"{log_prefix} [smoke] reached --max-steps={max_steps}, stopping early")
                if mem_log_fn:
                    mem_log_fn(epoch)
                if epoch_end_fn:
                    epoch_end_fn(epoch)
                return {"epochs_completed": epochs_completed, "global_step": global_step,
                        "elapsed_s": time.time() - t0, "stopped_early": True}

        epochs_completed = epoch + 1
        if mem_log_fn:
            mem_log_fn(epoch)
        if epoch_end_fn:
            epoch_end_fn(epoch)
        if rank == 0:
            print(f"=== {log_prefix} Epoch {epoch+1}/{epochs} Completed in {time.time()-t0:.1f}s | Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===")
        if checkpoint_fn:
            checkpoint_fn(epoch, global_step, time.time() - t0)

    return {"epochs_completed": epochs_completed, "global_step": global_step,
            "elapsed_s": time.time() - t0, "stopped_early": False}


def finalize_and_save(model, tok, calib_items, output_dir, model_name, base_model_dir_name,
                       exporter_version, cfg, *, device, autocast_device, autocast_dtype, autocast_enabled,
                       encoder_config_restore=None):
    """official-notebook-cell 4's post-training calibration-temperature-fit + save
    step, extracted so both main_ddp() (rank 0 only) and main_local() (always,
    world_size=1) call the IDENTICAL save path -- same checkpoint layout
    (model.safetensors fp16, encoder/, tokenizer/, rl_agent_config.json).

    pointsman-change: calib_items is now the already-loaded list (not a path) so the
    caller fits temperatures on calibration only after selecting weights on dev.

    pointsman-change: encoder_config_restore (from build_model_with_encoder_dropout()
    under --dropout) holds the base checkpoint's own encoder config values; they
    are written into a copy of the encoder config before saving, so the saved
    encoder/config.json keeps the checkpoint's dropout values. None (default)
    saves model.encoder.config as before."""
    print("\nFitting post-training calibration temperatures...")
    calib_preds = collect_calib_logits(model, calib_items, tok, device, autocast_device, autocast_dtype, autocast_enabled)

    fitted_temps = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
            if sel:
                fitted_temps[qt] = fit_one_temp(sel)
        print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
    except Exception as e:
        print("Temperature fitting fallback:", e)

    os.makedirs(output_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))
    if encoder_config_restore:
        saved_encoder_config = copy.deepcopy(model.encoder.config)
        for key, value in encoder_config_restore.items():
            setattr(saved_encoder_config, key, value)
        saved_encoder_config.save_pretrained(os.path.join(output_dir, "encoder"))
    else:
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

    cfg["fine_tuned"] = True
    cfg["model_name"] = model_name
    cfg["base_model_dir_name"] = base_model_dir_name
    cfg["exporter_version"] = exporter_version
    cfg["temperature"] = fitted_temps
    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Model successfully saved to {output_dir}!")


def write_epoch_selection_metadata(output_dir, select_best_epoch, selected_epoch, epoch_agreements):
    # pointsman-change: per-epoch best-checkpoint selection metadata, folded into
    # training_metadata.json by train_from_export.py's outer main() (the
    # promotion flow -- qualify -> compare -> promote -- needs to know which
    # epoch a saved checkpoint actually came from).
    path = os.path.join(output_dir, "epoch_selection.json")
    with open(path, "w") as f:
        json.dump({
            "select_best_epoch": select_best_epoch,
            "selected_epoch": selected_epoch,
            "epoch_agreements": epoch_agreements,
            "selection_split": "dev" if select_best_epoch else None,
        }, f, indent=2)
    print(f"wrote {path}")
    return path


# pointsman-change: per-epoch resumable checkpoints (--local only, 2026-09-25). A full
# local run is ~1.85 h/epoch and the owner does not run overnight, so a 6-epoch
# run spans days. Layout (kept textually in sync with train_from_export.py's
# read_resume_state(), which reads it back before relaunching this script):
#   <output-dir>/resume/LATEST                    -> "epoch-000N" (replaced atomically)
#   <output-dir>/resume/epoch-000N/state.json     config, counters, epoch_agreements, best epoch
#   <output-dir>/resume/epoch-000N/training_state.pt  model + AdamW + scheduler + RNG states
#   <output-dir>/resume/epoch-000N/best_model.pt  best-epoch weights (fp32), hard-linked forward
# Each epoch dir is written under a temporary name and renamed into place before
# LATEST is switched, so an interruption never leaves LATEST pointing at a
# partial checkpoint. Only the latest completed epoch is kept.
RESUME_FORMAT = "laya-kit-resume-v1"
RESUME_DIRNAME = "resume"
TRAIN_SEED = 42


def capture_rng_states(device):
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if device.type == "mps":
        rng["torch_mps"] = torch.mps.get_rng_state()
    return rng


def restore_rng_states(rng, device):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if device.type == "mps" and "torch_mps" in rng:
        torch.mps.set_rng_state(rng["torch_mps"])


def dir_disk_bytes(path):
    """Bytes on disk under path, counting each inode once (best_model.pt is
    hard-linked between epoch dirs)."""
    seen, total = set(), 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            st = os.lstat(os.path.join(root, name))
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total


def write_resume_checkpoint(resume_root, epochs_completed, *, model, optimizer, scheduler, device, global_step,
                            epoch_agreements, best_state, prev_epoch_dir, run_config, train_elapsed_s):
    """Atomically write resume_root/epoch-<N>/ and point resume_root/LATEST at it,
    then delete every other entry. best_model.pt is written fresh when epoch N is
    the new best (best_state["epoch"] == N), otherwise hard-linked (copied if
    linking fails) from prev_epoch_dir. Returns (epoch_dir, bytes_on_disk)."""
    os.makedirs(resume_root, exist_ok=True)
    name = f"epoch-{epochs_completed:04d}"
    final_dir = os.path.join(resume_root, name)
    tmp_dir = os.path.join(resume_root, f".{name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        os.makedirs(tmp_dir)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": capture_rng_states(device)},
                   os.path.join(tmp_dir, "training_state.pt"))
        best_epoch = best_state.get("epoch")
        best_path = os.path.join(tmp_dir, "best_model.pt")
        if best_epoch is not None and best_epoch == epochs_completed:
            torch.save(model.state_dict(), best_path)
        elif best_epoch is not None:
            prev_best = os.path.join(prev_epoch_dir, "best_model.pt") if prev_epoch_dir else None
            if not prev_best or not os.path.isfile(prev_best):
                raise RuntimeError(f"best epoch {best_epoch} snapshot missing from {prev_epoch_dir}")
            try:
                os.link(prev_best, best_path)
            except OSError:
                shutil.copy2(prev_best, best_path)
        state = {
            "format": RESUME_FORMAT,
            "config": run_config["match"],
            "info": run_config["info"],
            "epochs_completed": epochs_completed,
            "global_step": global_step,
            "epoch_agreements": epoch_agreements,
            "best_score": best_state.get("score"),
            "best_epoch": best_epoch,
            "train_elapsed_s": train_elapsed_s,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(tmp_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)
        if os.path.exists(final_dir):  # unreferenced leftover of an earlier crash (LATEST never pointed here)
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    latest_tmp = os.path.join(resume_root, f".LATEST.tmp-{os.getpid()}")
    with open(latest_tmp, "w") as f:
        f.write(name + "\n")
    os.replace(latest_tmp, os.path.join(resume_root, "LATEST"))
    for entry in os.listdir(resume_root):
        if entry in (name, "LATEST"):
            continue
        path = os.path.join(resume_root, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    return final_dir, dir_disk_bytes(resume_root)


def load_resume_checkpoint(epoch_dir, run_config, model, optimizer, scheduler):
    """Restore model/optimizer/scheduler from epoch_dir (written by
    write_resume_checkpoint()). Returns (state, rng) -- the caller restores rng
    with restore_rng_states() immediately before the training loop."""
    with open(os.path.join(epoch_dir, "state.json")) as f:
        state = json.load(f)
    if state.get("format") != RESUME_FORMAT or state.get("config") != run_config["match"]:
        print(f"[FATAL] resume state {epoch_dir} does not match this run's configuration", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(os.path.join(epoch_dir, "training_state.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    rng = ckpt["rng"]
    del ckpt
    return state, rng


def load_common_argv():
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    train_items_path = sys.argv[3]
    # pointsman-change: fourth argv is the preprocessed calibration.jsonl items
    # (the export's real held-out split), replacing the notebook's
    # all_items[::15][:400] slice of the *training* data.
    calib_items_path = sys.argv[4]
    # pointsman-change: fifth/sixth/seventh argv let the saved checkpoint record which base
    # checkpoint it was fine-tuned from, instead of hard-coding model_name to
    # 'laya-typed-decisions' (see derive_model_name() in train_from_export.py's main()).
    model_name = sys.argv[5]
    base_model_dir_name = sys.argv[6]
    exporter_version = sys.argv[7]
    # pointsman-change: argv[8] selects "ddp" or "local".
    mode = sys.argv[8] if len(sys.argv) > 8 else "ddp"
    # pointsman-change: argv[9] is "1"/"0" for --select-best-epoch, shared by
    # main_ddp() and main_local() (see make_epoch_end_fn()). Defaults to
    # enabled when omitted, matching train_from_export.py's own default.
    select_best_epoch = (sys.argv[9] != "0") if len(sys.argv) > 9 else True
    return (model_dir, output_dir, train_items_path, calib_items_path, model_name,
            base_model_dir_name, exporter_version, mode, select_best_epoch)


def main_ddp():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    (model_dir, output_dir, train_items_path, calib_items_path, model_name,
     base_model_dir_name, exporter_version, _mode, select_best_epoch) = load_common_argv()

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(train_items_path, weights_only=False)
    my_items = all_items[rank::world_size]
    # pointsman-change: independent dev items choose epochs; calibration items fit final temperatures.
    calib_items = torch.load(calib_items_path, weights_only=False)
    dev_items = torch.load(os.path.join(output_dir, "dev_items.pt"), weights_only=False) if select_best_epoch else []

    GRAD_ACCUM = 4  # Effective batch across 2 GPUs = 64 sequences (8 * 2 * 4)

    optimizer, scheduler = build_optimizer_and_scheduler(
        list(ddp_model.named_parameters()), len(my_items), MICRO_BATCH, GRAD_ACCUM, EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(f"Starting 2xT4 DDP training: {len(all_items)} total items | {len(my_items)} per rank | {EPOCHS} epochs")

    # pointsman-change: per-epoch development-agreement checkpoint selection. Only
    # rank 0 evaluates/keeps the CPU state_dict copy; epoch_end_fn barriers
    # all ranks afterward so training stays in lockstep.
    best_state = {"score": None, "epoch": None, "state_dict": None}
    epoch_agreements = []
    epoch_end_fn = make_epoch_end_fn(
        model, dev_items, tok, device, autocast_device="cuda", autocast_dtype=torch.float16,
        autocast_enabled=True, rank=rank, world_size=world_size, dist_module=dist,
        select_best_epoch=select_best_epoch, best_state=best_state, epoch_agreements=epoch_agreements,
        log_prefix="[ddp]")

    run_training_loop(
        ddp_model, my_items, list(ddp_model.parameters()), optimizer, scheduler, tok,
        device=device, epochs=EPOCHS, micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM, group_size=GROUP_SIZE,
        sigma_start=SIGMA_START, sigma_end=SIGMA_END, rank=rank, world_size=world_size,
        use_scaler=True, scaler=scaler, autocast_device="cuda", autocast_dtype=torch.float16,
        autocast_enabled=True, max_steps=0, mem_log_fn=None, log_prefix="[ddp]", epoch_end_fn=epoch_end_fn)

    dist.barrier()

    if rank == 0:
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        selected_epoch = None
        if select_best_epoch and best_state["state_dict"] is not None:
            selected_epoch = best_state["epoch"]
            model.load_state_dict(best_state["state_dict"], strict=True)
            del best_state["state_dict"]  # pointsman-change: free the ~1.3GB CPU copy once loaded.
        finalize_and_save(model, tok, calib_items, output_dir, model_name, base_model_dir_name,
                           exporter_version, cfg, device=device, autocast_device="cuda",
                           autocast_dtype=torch.float16, autocast_enabled=True)
        write_epoch_selection_metadata(output_dir, select_best_epoch, selected_epoch, epoch_agreements)

    dist.destroy_process_group()


def main_local():
    # pointsman-change: --local mode (Apple Silicon MPS, or CPU) -- single process,
    # no torchrun/NCCL/DDP. Reuses run_training_loop()/finalize_and_save() with
    # world_size=1/rank=0 and forward_fn=model (no DDP wrapper) so the loss,
    # optimizer, LR schedule, epochs, per-device batch size and seed are the
    # SAME as main_ddp(); only process/device wiring and precision differ.
    # (--local-only opt-ins: --epochs, --dropout/--rdrop-alpha, a fixed torch
    # seed plus per-epoch resume checkpoints -- see write_resume_checkpoint().)
    (model_dir, output_dir, train_items_path, calib_items_path, model_name,
     base_model_dir_name, exporter_version, _mode, select_best_epoch) = load_common_argv()
    device_name = sys.argv[10] if len(sys.argv) > 10 else "mps"
    grad_accum = int(sys.argv[11]) if len(sys.argv) > 11 else 8
    micro_batch = int(sys.argv[12]) if len(sys.argv) > 12 else MICRO_BATCH
    mps_autocast = sys.argv[13] if len(sys.argv) > 13 else "off"
    max_steps = int(sys.argv[14]) if len(sys.argv) > 14 else 0
    # pointsman-change: argv[15]/[16] are --dropout ("none" = keep the checkpoint's
    # encoder config) and --rdrop-alpha (0 = off); see build_model_with_encoder_dropout().
    dropout_arg = sys.argv[15] if len(sys.argv) > 15 else "none"
    encoder_dropout = None if dropout_arg == "none" else float(dropout_arg)
    rdrop_alpha = float(sys.argv[16]) if len(sys.argv) > 16 else 0.0
    # pointsman-change: argv[17] is --epochs (default EPOCHS -- drives the loop AND the
    # cosine schedule length); argv[18] is the run configuration JSON
    # ({"match": ..., "info": ...}, built by train_from_export.py) that enables
    # per-epoch resume checkpoints; argv[19] is "fresh" or the resume epoch dir;
    # argv[20] is "keep" (--keep-resume) or "clean". Without argv[18] no resume
    # checkpoint is written (direct invocations keep the pre-resume behavior).
    epochs = int(sys.argv[17]) if len(sys.argv) > 17 else EPOCHS
    run_config = json.loads(sys.argv[18]) if len(sys.argv) > 18 else None
    resume_from = sys.argv[19] if len(sys.argv) > 19 and sys.argv[19] != "fresh" else None
    keep_resume = len(sys.argv) > 20 and sys.argv[20] == "keep"
    resume_root = os.path.join(output_dir, RESUME_DIRNAME)
    if resume_from is not None and run_config is None:
        print("[FATAL] resuming requires the run configuration (argv[18])", file=sys.stderr)
        sys.exit(1)
    if rdrop_alpha > 0 and not encoder_dropout:
        print("[FATAL] --rdrop-alpha > 0 requires --dropout > 0", file=sys.stderr)
        sys.exit(1)

    if device_name == "mps" and not torch.backends.mps.is_available():
        print("[FATAL] --device mps requested but torch.backends.mps.is_available() is False", file=sys.stderr)
        sys.exit(1)
    device = torch.device(device_name)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model, encoder_config_restore = build_model_with_encoder_dropout(
        cfg, os.path.join(model_dir, "encoder"), encoder_dropout)

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()
    encoder_dropout_check = None
    if encoder_dropout is not None:
        encoder_dropout_check = assert_encoder_dropout_applied(model.encoder, encoder_dropout)
        print(f"[local] encoder dropout applied: {encoder_dropout_check}")

    all_items = torch.load(train_items_path, weights_only=False)
    my_items = list(all_items)  # world_size=1: no DDP rank split, every item is "mine"
    # pointsman-change: independent dev items choose epochs; calibration items fit final temperatures.
    calib_items = torch.load(calib_items_path, weights_only=False)
    dev_items = torch.load(os.path.join(output_dir, "dev_items.pt"), weights_only=False) if select_best_epoch else []

    optimizer, scheduler = build_optimizer_and_scheduler(
        list(model.named_parameters()), len(my_items), micro_batch, grad_accum, epochs)

    # pointsman-change: MPS fp16 autocast is unreliable for training (mixed-dtype
    # matmul asserts observed on this hardware during inference spikes).
    # Default is plain fp32 training; --mps-autocast bf16 is opt-in. No GradScaler is used
    # (bf16/fp32 do not need loss scaling the way fp16 does).
    autocast_enabled = mps_autocast == "bf16"
    autocast_dtype = torch.bfloat16 if autocast_enabled else None
    autocast_device = "mps" if device_name == "mps" else "cpu"

    print(f"[local] device={device_name} items={len(my_items)} epochs={epochs} micro_batch={micro_batch} "
          f"grad_accum={grad_accum} effective_global_batch={micro_batch*grad_accum} "
          f"mps_autocast={mps_autocast} max_steps={max_steps or 'unlimited'} "
          f"dropout={'off' if encoder_dropout is None else encoder_dropout} "
          f"rdrop_alpha={rdrop_alpha if rdrop_alpha > 0 else 'off'}")

    def mem_log_fn(epoch):
        # pointsman-change: memory-safety logging (spec: log peak MPS memory per
        # epoch). torch.mps has no peak-tracking counter as of this torch
        # version -- driver_allocated_memory() is the best-effort proxy
        # logged here, not a true high-water mark.
        if device_name == "mps" and hasattr(torch, "mps"):
            try:
                mib = torch.mps.driver_allocated_memory() / (1024 * 1024)
                print(f"[mem] epoch {epoch+1} mps_driver_allocated_mib={mib:.1f}")
            except Exception as e:
                print(f"[mem] epoch {epoch+1} mps memory read failed: {e}")

    # pointsman-change: per-epoch development-agreement checkpoint selection --
    # world_size=1 here, so epoch_end_fn never barriers. With resume
    # checkpoints on (run_config given) the best weights live on disk in the
    # checkpoint's best_model.pt instead of a ~1.3GB CPU copy.
    checkpointing = run_config is not None
    best_state = {"score": None, "epoch": None, "state_dict": None}
    epoch_agreements = []
    start_epoch, start_global_step, prior_elapsed_s = 0, 0, 0.0
    saved = {"epoch_dir": None, "epochs": 0}
    rng = None
    if resume_from is not None:
        state, rng = load_resume_checkpoint(resume_from, run_config, model, optimizer, scheduler)
        start_epoch, start_global_step = state["epochs_completed"], state["global_step"]
        prior_elapsed_s = state["train_elapsed_s"]
        epoch_agreements.extend(state["epoch_agreements"])
        best_state["score"], best_state["epoch"] = state["best_score"], state["best_epoch"]
        saved["epoch_dir"], saved["epochs"] = resume_from, start_epoch
        print(f"[resume] restored {resume_from}: epochs_completed={start_epoch}/{epochs} "
              f"global_step={start_global_step} best_epoch={best_state['epoch']}")
    epoch_end_fn = make_epoch_end_fn(
        model, dev_items, tok, device, autocast_device=autocast_device, autocast_dtype=autocast_dtype,
        autocast_enabled=autocast_enabled, rank=0, world_size=1, dist_module=None,
        select_best_epoch=select_best_epoch, best_state=best_state, epoch_agreements=epoch_agreements,
        log_prefix="[local]", keep_best_in_memory=not checkpointing)

    def checkpoint_fn(epoch, global_step, elapsed_s):
        epoch_dir, size = write_resume_checkpoint(
            resume_root, epoch + 1, model=model, optimizer=optimizer, scheduler=scheduler, device=device,
            global_step=global_step, epoch_agreements=epoch_agreements, best_state=best_state,
            prev_epoch_dir=saved["epoch_dir"], run_config=run_config, train_elapsed_s=prior_elapsed_s + elapsed_s)
        saved["epoch_dir"], saved["epochs"] = epoch_dir, epoch + 1
        print(f"[resume] saved epoch {epoch + 1}/{epochs} checkpoint to {epoch_dir} "
              f"({size / 2**30:.2f} GiB on disk under {resume_root})")

    # pointsman-change: fixed torch seed for a fresh local run (RLCD noise, dropout masks) so
    # an interrupted+resumed run can be checked against an uninterrupted one; a resumed
    # run restores the saved RNG states instead.
    if rng is not None:
        restore_rng_states(rng, device)
        del rng
    else:
        torch.manual_seed(TRAIN_SEED)
    result = run_training_loop(
        model, my_items, list(model.parameters()), optimizer, scheduler, tok,
        device=device, epochs=epochs, micro_batch=micro_batch, grad_accum=grad_accum, group_size=GROUP_SIZE,
        sigma_start=SIGMA_START, sigma_end=SIGMA_END, rank=0, world_size=1,
        use_scaler=False, scaler=None, autocast_device=autocast_device, autocast_dtype=autocast_dtype,
        autocast_enabled=autocast_enabled, max_steps=max_steps, mem_log_fn=mem_log_fn, log_prefix="[local]",
        epoch_end_fn=epoch_end_fn, rdrop_alpha=rdrop_alpha, start_epoch=start_epoch,
        start_global_step=start_global_step, checkpoint_fn=checkpoint_fn if checkpointing else None)

    del optimizer, scheduler
    selected_epoch = None
    if select_best_epoch and best_state["state_dict"] is not None:
        selected_epoch = best_state["epoch"]
        model.load_state_dict(best_state["state_dict"], strict=True)
        del best_state["state_dict"]  # pointsman-change: free the ~1.3GB CPU copy once loaded.
    elif select_best_epoch and best_state["epoch"] is not None:
        selected_epoch = best_state["epoch"]
        # The model already holds the best weights when the best epoch is the last one
        # evaluated (always true for a --max-steps partial epoch, which is never
        # checkpointed); otherwise the best epoch is <= the last saved checkpoint.
        if selected_epoch != epoch_agreements[-1]["epoch"]:
            best_path = os.path.join(saved["epoch_dir"], "best_model.pt")
            model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
    finalize_and_save(model, tok, calib_items, output_dir, model_name, base_model_dir_name,
                       exporter_version, cfg, device=device, autocast_device=autocast_device,
                       autocast_dtype=autocast_dtype, autocast_enabled=autocast_enabled,
                       encoder_config_restore=encoder_config_restore)
    write_epoch_selection_metadata(output_dir, select_best_epoch, selected_epoch, epoch_agreements)

    # pointsman-change: structured local-run metadata the outer train_from_export.py
    # main() reads back and folds into training_metadata.json (device/local
    # mode/grad_accum/wall time -- the official notebook has no such metadata).
    metadata_path = os.path.join(output_dir, "local_training_run.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "device": device_name, "mode": "local", "grad_accum": grad_accum, "micro_batch": micro_batch,
            "effective_global_batch": micro_batch * grad_accum, "mps_autocast": mps_autocast,
            "epochs": epochs, "epochs_completed": result["epochs_completed"], "global_step": result["global_step"],
            "stopped_early": result["stopped_early"],
            # pointsman-change: training-loop time summed over every resumed leg.
            "wall_time_s": round(prior_elapsed_s + result["elapsed_s"], 3),
            "resumed_from_epoch": start_epoch if resume_from is not None else None,
            "dropout": encoder_dropout, "rdrop_alpha": rdrop_alpha if rdrop_alpha > 0 else None,
            "encoder_dropout_check": encoder_dropout_check,
        }, f, indent=2)
    print(f"[local] wrote {metadata_path}")

    # pointsman-change: the saved checkpoint above is the product; the resume state is
    # only for an interrupted run, so it is removed unless --keep-resume.
    if os.path.isdir(resume_root):
        if keep_resume:
            print(f"[resume] kept {resume_root} ({dir_disk_bytes(resume_root) / 2**30:.2f} GiB, --keep-resume)")
        else:
            shutil.rmtree(resume_root)
            print(f"[resume] removed {resume_root} after the final save")


def main():
    _mode_probe = load_common_argv()
    mode = _mode_probe[7]
    if mode == "local":
        main_local()
    else:
        main_ddp()

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# pointsman-change: the official notebook's evaluation cell assumes the HF
# `LocalLLaMA/typed-decisions` test split's gold answers carry an explicit
# "label"/"score" key. The pointsman export never writes one -- exportDataset() in
# src/training/dataset.mjs writes `gold[qid] = {"probabilities": {...}}`
# only (confirmed against check_export.py's check_gold_probabilities(), which
# enforces exactly that shape). A real local run's post-training eval step
# failed with "[eval] evaluation step failed or was skipped: 'label'" because
# evaluate_checkpoint() read `g_ans["label"]` directly. resolve_gold_label()
# derives the gold label the same way build_training_item() derives the
# training target's own label (`target.index(max(target))`, i.e. argmax over
# probabilities), so evaluation compares like with like. An explicit "label"
# key still wins when present (e.g. a hand-authored fixture), for backward
# compatibility with the official notebook's own dataset shape.
# ---------------------------------------------------------------------------
def resolve_gold_label(q_type, g_ans, keys=None, n_levels=None):
    if q_type == "choice":
        if "label" in g_ans:
            return str(g_ans["label"])
        if not keys:
            die("resolve_gold_label: q_type='choice' requires keys")
        probs = g_ans.get("probabilities") or {}
        values = [probs.get(k, 0.0) for k in keys]
        return keys[values.index(max(values))]
    if q_type == "noul":
        if "label" in g_ans:
            return str(g_ans["label"]).lower()
        probs = g_ans.get("probabilities") or {}
        values = [probs.get("false", 0.5), probs.get("true", 0.5)]
        return "false" if values.index(max(values)) == 0 else "true"
    if q_type == "score":
        if "label" in g_ans:
            return int(g_ans["label"])
        if not n_levels:
            die("resolve_gold_label: q_type='score' requires n_levels")
        probs = g_ans.get("probabilities") or {}
        values = [probs.get(str(i), 0.0) for i in range(n_levels)]
        return values.index(max(values))
    die(f"resolve_gold_label: unknown question type {q_type!r}")


# ---------------------------------------------------------------------------
# pointsman-change: pure aggregation helper for per-epoch best-checkpoint selection
# (see TRAIN_DDP_SCRIPT's epoch_end_fn below). Computes the argmax(logits) ==
# argmax(target) agreement fraction per question type (QTYPES: choice=0,
# score=1, noul=2, matching laya.common.QTYPES) plus the unweighted mean over
# qtypes that had at least one calibration item. Duplicated verbatim inside
# TRAIN_DDP_SCRIPT (that script must stay standalone for Kaggle/local
# subprocess execution and cannot import this module) --
# tests/test_laya_kit_eval_and_epoch_select.py checks parity between the two,
# the same way tests/test_laya_kit_local_mode.py checks
# oom_suggestion()/format_oom_message() parity.
# ---------------------------------------------------------------------------
def compute_calib_agreement(calib_preds):
    names = {0: "choice", 1: "score", 2: "noul"}
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    for qtype, logits, target in calib_preds:
        if not logits or not target:
            continue
        pred_idx = max(range(len(logits)), key=lambda i: logits[i])
        gold_idx = max(range(len(target)), key=lambda i: target[i])
        total[qtype] += 1
        if pred_idx == gold_idx:
            correct[qtype] += 1
    agreement = {names[qt]: (correct[qt] / total[qt] if total[qt] else None) for qt in names}
    scored = [v for v in agreement.values() if v is not None]
    agreement["mean"] = (sum(scored) / len(scored)) if scored else None
    return agreement


# ---------------------------------------------------------------------------
# official-notebook-cell: 6/7 ("Run Benchmark Evaluation" / "Compute Official
# Metrics"). pointsman-change: reads test.jsonl from --export-dir instead of the
# HF `test` split, via the same row-shape (state/questions/gold JSON-string
# columns) the export writes.
# ---------------------------------------------------------------------------
def evaluate_checkpoint(output_dir, export_dir, laya_module, device="cuda"):
    # pointsman-change: device is now a parameter (was hard-coded "cuda") so --local
    # mode's evaluation step loads the fine-tuned checkpoint on the SAME device
    # it was trained on (mps/cpu), instead of unconditionally requiring CUDA.
    import numpy as np

    ece_score = laya_module.common.ece_score
    agent_ft = laya_module.Agent(output_dir, device=device)

    test_path = Path(export_dir) / "test.jsonl"
    predictions = []
    latencies_ms = []
    with test_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        t0 = time.perf_counter()
        res = agent_ft.predict(state, questions)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt_ms)
        predictions.append({"pred": res["answers"], "gold": gold, "questions": questions, "latency_ms": dt_ms})

    accuracies, soft_accuracies, brier_scores, kl_divs, tv_distances = [], [], [], [], []
    score_maes, within_one, all_confs, all_corrects = [], [], [], []

    for item in predictions:
        pred_answers, gold_answers, questions = item["pred"], item["gold"], item["questions"]
        for qid, qdef in questions.items():
            p_ans, g_ans = pred_answers[qid], gold_answers[qid]
            q_type = qdef["type"]
            if q_type == "choice":
                keys = list(qdef["criteria"].keys())
                pred_choice = p_ans["choice"]
                gold_label = resolve_gold_label(q_type, g_ans, keys=keys)
                is_corr = float(pred_choice == gold_label)
                accuracies.append(is_corr); all_corrects.append(is_corr)
                # pointsman-change: dtype=float64 is required, not just convenient --
                # the export's objective/human-labeled gold probabilities are a
                # one-hot {key: 1 or 0} dict (targetDistribution() in
                # src/training/schema.mjs); JSON round-trips a whole number
                # like `1`/`0` as a Python int, so an untyped np.array() here
                # can infer int64 and the in-place `/=` below then raises
                # "Cannot cast ufunc 'divide' output from dtype('float64') to
                # dtype('int64')" (observed on a real smoke run, 2026-09-23).
                p_probs = np.array([p_ans["probabilities"].get(k, 1e-6) for k in keys], dtype=np.float64)
                g_probs = np.array([g_ans["probabilities"].get(k, 1e-6) for k in keys], dtype=np.float64)
                p_probs /= p_probs.sum(); g_probs /= g_probs.sum()
                all_confs.append(float(p_probs.max()))
                soft_accuracies.append(float((p_probs * g_probs).sum()))
                brier_scores.append(float(((p_probs - g_probs) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_probs - g_probs).sum()))
                kl_divs.append(float((g_probs * np.log(np.clip(g_probs / p_probs, 1e-12, 1e4))).sum()))
            elif q_type == "noul":
                p_val = p_ans["noul"]
                g_val = g_ans.get("noul", g_ans.get("probabilities", {}).get("true", 0.5))
                gold_label = resolve_gold_label(q_type, g_ans)
                pred_label = "true" if p_val >= 0.5 else "false"
                is_corr = float(pred_label == gold_label)
                accuracies.append(is_corr); all_corrects.append(is_corr)
                all_confs.append(float(max(p_val, 1.0 - p_val)))
                p_dist = np.array([1.0 - p_val, p_val])
                g_dist = np.array([1.0 - g_val, g_val])
                soft_accuracies.append(float((p_dist * g_dist).sum()))
                brier_scores.append(float(((p_dist - g_dist) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_dist - g_dist).sum()))
                kl_divs.append(float((g_dist * np.log(np.clip(g_dist / p_dist, 1e-12, 1e4))).sum()))
            elif q_type == "score":
                p_score = p_ans["score"]
                n_levels = len(qdef.get("criteria", []))
                # pointsman-change: the real export's gold answer has no "score" key
                # (only "probabilities"), so the gold level is the argmax over
                # probabilities (resolve_gold_label), and the gold scalar used
                # for score_mae/within_1_level falls back to that level instead
                # of the notebook's own gold "score" float, which this export
                # never provides.
                g_lvl = resolve_gold_label(q_type, g_ans, n_levels=n_levels)
                g_score = g_ans.get("score", float(g_lvl))
                score_maes.append(abs(p_score - g_score))
                within_one.append(float(abs(p_score - g_score) <= 1.0))
                p_probs_score = np.array([p_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)], dtype=np.float64)
                if p_probs_score.sum() > 0:
                    p_probs_score /= p_probs_score.sum()
                    p_lvl = int(np.argmax(p_probs_score))
                    all_confs.append(float(p_probs_score.max()))
                else:
                    p_lvl = int(round(p_score))
                    all_confs.append(0.5)
                is_corr = float(p_lvl == g_lvl)
                accuracies.append(is_corr); all_corrects.append(is_corr)

    metrics = {
        "accuracy": round(float(np.mean(accuracies)), 4) if accuracies else None,
        "soft_accuracy": round(float(np.mean(soft_accuracies)), 4) if soft_accuracies else None,
        "brier_score": round(float(np.mean(brier_scores)), 4) if brier_scores else None,
        "ece": round(float(ece_score(np.array(all_confs), np.array(all_corrects))), 4) if all_confs else None,
        "score_mae": round(float(np.mean(score_maes)), 4) if score_maes else 0.0,
        "within_1_level": round(float(np.mean(within_one)), 4) if within_one else 0.0,
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 1) if latencies_ms else None,
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1) if latencies_ms else None,
        "kl_divergence": round(float(np.mean(kl_divs)), 4) if kl_divs else None,
        "total_variation": round(float(np.mean(tv_distances)), 4) if tv_distances else None,
        "n_cases": len(rows),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, help="exports/<version>/laya folder with train/calibration/test.jsonl + manifest.json")
    parser.add_argument("--model-dir", default=None, help="Path to a pre-downloaded convaiinnovations/laya snapshot. If omitted, uses huggingface_hub.snapshot_download (network + model download -- do this on Kaggle, not locally).")
    parser.add_argument("--model-subdir", default=None, help="Per-checkpoint subfolder under --model-dir/the downloaded snapshot root (e.g. 'multilingual' for the multilingual checkpoint). Omit for a checkpoint that lives at the snapshot root (e.g. the english checkpoint's historical layout).")
    parser.add_argument("--output-dir", required=True, help="Where to write the fine-tuned checkpoint + training_metadata.json")
    parser.add_argument("--max-truncated-fraction", type=float, default=0.02, help="Abort if more than this fraction of question-rows are dropped by tokenizer admission (default 2%%)")
    parser.add_argument("--allow-truncation", action="store_true", help="Proceed even if the truncated fraction exceeds --max-truncated-fraction")
    parser.add_argument("--dry-run", action="store_true", help="Verify export + tokenizer admission and stop before invoking torchrun (no GPU/model needed for the manifest check; still needs laya+tokenizer installed for the admission check)")
    parser.add_argument("--skip-admission-check", action="store_true", help="Skip the tokenizer admission pass entirely (NOT recommended; only for debugging the export-manifest check in isolation)")
    parser.add_argument("--local", action="store_true", help="Single-process training (no torchrun/NCCL/DDP), for Apple Silicon MPS or CPU. Does not import `datasets` -- reads the export's JSONL splits with the stdlib json module instead.")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps", help="--local only: torch device to train on (default mps)")
    parser.add_argument("--grad-accum", type=int, default=None, help="--local only: gradient accumulation steps. Default keeps the same effective global batch as the official 2xT4 DDP recipe (micro_batch * grad_accum * 2 == 64) given --batch-size; an explicit value here overrides that.")
    parser.add_argument("--batch-size", type=int, default=None, help="--local only: per-step micro batch size (default 8, same as the DDP recipe's MICRO_BATCH). If --grad-accum is not also given, it is scaled to keep the same effective global batch -- use a smaller --batch-size if training OOMs.")
    parser.add_argument("--mps-autocast", choices=["bf16", "off"], default="off", help="--local only: MPS autocast dtype during the forward pass. Default off (fp32) -- MPS fp16 autocast is unreliable for training; bf16 is opt-in.")
    parser.add_argument("--max-steps", type=int, default=None, help="--local only, for smoke validation: stop after this many optimizer micro-steps total instead of running full EPOCHS. Omit for a real training run.")
    parser.add_argument("--select-best-epoch", action=argparse.BooleanOptionalAction, default=True,
                         help="After each epoch, evaluate dev.jsonl argmax agreement and keep the "
                              "best epoch's checkpoint instead of always the last one (default: on). A real "
                              "local run (2026-09-23) overfit badly by the final epoch -- train agreement "
                              "~0.99/0.92/0.95 (intent/difficulty/risk) vs held-out test ~0.72/0.58/0.60. "
                              "Pass --no-select-best-epoch to keep the old always-last-epoch behavior.")
    parser.add_argument("--dropout", type=float, default=None,
                         help="--local only: train with encoder attention/embedding/mlp dropout = P (0 < P <= 0.5). "
                              "Default: not set -- keep the base checkpoint's encoder config (0.0 for the "
                              "multilingual checkpoint). The saved encoder/config.json keeps the checkpoint's own "
                              "values; inference always runs in eval mode.")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0,
                         help="--local only: R-Drop weight A >= 0 (default 0 = off). When > 0, each micro-batch "
                              "runs two train-mode forward passes and adds A * symmetric KL between their "
                              "per-question option distributions to the mean of the two losses. Requires "
                              "--dropout. Roughly doubles forward/backward compute per step.")
    parser.add_argument("--epochs", type=int, default=None,
                         help=f"--local only: number of epochs N (1 <= N <= {MAX_EPOCHS}, default {DEFAULT_EPOCHS}). "
                              "Sets both the loop count and the cosine LR schedule length.")
    parser.add_argument("--resume", action="store_true",
                         help="--local only: continue the interrupted run whose per-epoch checkpoint is in "
                              "<output-dir>/resume/ (from the next epoch after the last COMPLETED one). Refuses "
                              "unless every saved setting matches the current arguments; dies if there is no "
                              "resume state.")
    parser.add_argument("--keep-resume", action="store_true",
                         help="--local only: keep <output-dir>/resume/ after the final save (default: removed).")
    parser.add_argument("--method", choices=["supervised", "rlcd-grpo"], default="supervised", help="Supervised CE baseline by default; RLCD/GRPO is an explicit experiment")
    parser.add_argument("--input-fit", choices=["lossless", "task-head"], default="lossless", help="Must match checkpoint registration/inference inputFit")
    parser.add_argument("--evaluate-sealed-test", action="store_true", help="Evaluate the frozen saved candidate once after training/calibration; test is otherwise not loaded")
    args = parser.parse_args()
    validate_regularization_args(args.dropout, args.rdrop_alpha, args.local)
    resolved_epochs = validate_epochs_and_resume_args(args.epochs, args.resume, args.keep_resume, args.local)

    if args.local:
        resolved_batch_size, resolved_grad_accum = resolve_local_batch_and_grad_accum(args.batch_size, args.grad_accum)
    else:
        resolved_batch_size, resolved_grad_accum = DEFAULT_MICRO_BATCH, DEFAULT_DDP_GRAD_ACCUM

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest = verify_export_manifest(export_dir)
    dev_split = selection_split(manifest, args.select_best_epoch)

    if args.skip_admission_check and not args.dry_run:
        print("[WARN] --skip-admission-check set: proceeding without verifying tokenizer truncation")

    # official-notebook-cell: 3 imports (moved here so --dry-run's manifest-only
    # path above can run without laya/transformers installed).
    try:
        import torch
        from huggingface_hub import snapshot_download
        from laya.agent import _fix_tokenizer_config
        from laya.common import build_sequence, render_options, QTYPES
        import laya as laya_module
    except ImportError as exc:
        die(f"missing dependency ({exc}). Install training/laya-kit/requirements.lock on the training machine (Kaggle), not locally.")

    # pointsman-change: `datasets` is only required for the non-local (Kaggle DDP) path.
    # The local laya venv does not have `datasets` installed (inference-only
    # environment) -- --local reads the export's JSONL splits directly via
    # load_jsonl_rows() (stdlib json) instead.
    if not args.local:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            die(f"missing dependency ({exc}). Install training/laya-kit/requirements.lock on the training machine (Kaggle), not locally, or pass --local.")

    model_dir = args.model_dir
    if model_dir is None:
        print("[download] no --model-dir given: fetching convaiinnovations/laya via snapshot_download (network)")
        model_dir = snapshot_download("convaiinnovations/laya")
    resolved_model_dir = resolve_model_dir(model_dir, args.model_subdir)
    validate_resolved_model_dir(resolved_model_dir)
    _fix_tokenizer_config(str(resolved_model_dir))

    tok = AutoTokenizer = __import__("transformers").AutoTokenizer.from_pretrained(os.path.join(str(resolved_model_dir), "tokenizer"))
    with open(os.path.join(str(resolved_model_dir), "rl_agent_config.json")) as f:
        base_cfg = json.load(f)
    # pointsman-change: fit_task_head()/build_training_item() in this process must admit
    # sequences against the SAME max_len/head_max_len the model is actually trained with
    # (TRAIN_DDP_SCRIPT overrides these to DEFAULT_FINAL_MAX_LEN/DEFAULT_FINAL_HEAD_MAX_LEN
    # regardless of the base checkpoint's own values) -- see resolve_effective_cfg().
    cfg = resolve_effective_cfg(base_cfg)
    base_model_dir_name = args.model_subdir or Path(resolved_model_dir).name
    derived_model_name = derive_model_name(base_cfg, base_model_dir_name)
    print(f"[model] resolved_model_dir={resolved_model_dir} base_model_dir_name={base_model_dir_name!r} "
          f"derived_model_name={derived_model_name!r}")
    # pointsman-change: a minimal agent-shaped object exposing only what fit_task_head()/assert_lossless()
    # read (tok, cfg, ._to_internal) -- the same pattern scripts/laya-budget.py's Shim uses, since
    # the notebook's training path never constructs a real laya.agent.Agent.
    import types as _types
    agent_shim = _types.SimpleNamespace(tok=tok, cfg=cfg, _to_internal=laya_module.agent.Agent._to_internal)

    # official-notebook-cell: 3, pointsman-change described at module top: read the
    # export split files via the manifest's own recorded loader instead of
    # the public HF dataset. pointsman-change: --local reads the selected JSONL
    # files with the stdlib json module (load_jsonl_rows) instead of
    # datasets.load_dataset("json", ...), since `datasets` is not installed
    # in the local laya venv -- the resulting row shape (dict with
    # state/questions/gold string keys) is identical either way.
    # Test labels are never loaded during candidate selection or calibration.
    data_files = {"train": str(export_dir / "train.jsonl"), "validation": str(export_dir / "calibration.jsonl")}
    if dev_split:
        data_files["dev"] = str(export_dir / "dev.jsonl")
    dataset = {key: load_jsonl_rows(Path(value)) for key, value in data_files.items()} if args.local else load_dataset("json", data_files=data_files)

    # pointsman-change: --local resume (2026-09-25). The run configuration is checked
    # BEFORE preprocessing so a mismatched --resume fails fast; a fresh --local
    # run refuses to start over an existing resume state.
    train_items_path = output_dir / "train_items.pt"
    calib_items_path = output_dir / "calib_items.pt"
    dev_items_path = output_dir / "dev_items.pt"
    run_match, resume_state, cached_items = None, None, None
    if args.local:
        run_match = build_resume_match_config(
            export_dir=export_dir, manifest=manifest, resolved_model_dir=resolved_model_dir,
            derived_model_name=derived_model_name, epochs=resolved_epochs, micro_batch=resolved_batch_size,
            grad_accum=resolved_grad_accum, device=args.device, mps_autocast=args.mps_autocast,
            dropout=args.dropout, rdrop_alpha=args.rdrop_alpha, select_best_epoch=args.select_best_epoch, cfg=cfg, method=args.method, input_fit=args.input_fit)
        resume_state = check_resume_request(output_dir, run_match, args.resume)
    if resume_state is not None:
        cached_items = load_cached_items_for_resume(train_items_path, calib_items_path, resume_state["info"], torch)
    dev_items, dev_truncated, dev_total = [], 0, 0
    if cached_items is not None and dev_split:
        if dev_items_path.is_file():
            dev_items = torch.load(dev_items_path, weights_only=False)
        if not dev_items or items_digest(dev_items) != resume_state["info"].get("dev_items_digest"):
            cached_items = None
        else:
            dev_truncated, dev_total = resume_state["info"]["dev_truncated"], resume_state["info"]["dev_total"]
    if cached_items is not None:
        train_items, calib_items = cached_items
        info = resume_state["info"]
        train_truncated, train_total = info["train_truncated"], info["train_total"]
        calib_truncated, calib_total = info["calib_truncated"], info["calib_total"]
        print("[resume] reusing cached train_items.pt/calib_items.pt (export and item digests match)")
    else:
        train_items, train_truncated, train_total = preprocess_split(dataset["train"], tok, cfg, render_options, build_sequence, QTYPES, agent_shim, input_fit=args.input_fit)
        calib_items, calib_truncated, calib_total = preprocess_split(dataset["validation"], tok, cfg, render_options, build_sequence, QTYPES, agent_shim, input_fit=args.input_fit)
        if dev_split:
            dev_items, dev_truncated, dev_total = preprocess_split(dataset["dev"], tok, cfg, render_options, build_sequence, QTYPES, agent_shim, input_fit=args.input_fit)
    run_info = None
    if args.local:
        run_info = {
            "train_items_digest": items_digest(train_items), "calib_items_digest": items_digest(calib_items),
            "train_truncated": train_truncated, "train_total": train_total,
            "calib_truncated": calib_truncated, "calib_total": calib_total,
            "dev_items_digest": items_digest(dev_items), "dev_truncated": dev_truncated, "dev_total": dev_total,
            "started_at": resume_state["info"]["started_at"] if resume_state else started_at,
        }
        if resume_state is not None and (run_info["train_items_digest"] != resume_state["info"]["train_items_digest"]
                                         or run_info["calib_items_digest"] != resume_state["info"]["calib_items_digest"]
                                         or run_info["dev_items_digest"] != resume_state["info"].get("dev_items_digest", items_digest([]))):
            die("refusing to resume: re-preprocessed train/calibration items differ from the ones the interrupted "
                "run trained on (preprocessing code or tokenizer changed)")

    if not args.skip_admission_check:
        check_tokenizer_admission(train_truncated + calib_truncated + dev_truncated, train_total + calib_total + dev_total, args.max_truncated_fraction, args.allow_truncation)

    print(f"[preprocess] train: {len(train_items)} sequences ({train_truncated} truncated of {train_total})")
    print(f"[preprocess] calibration: {len(calib_items)} sequences ({calib_truncated} truncated of {calib_total})")

    if not train_items:
        die("no trainable sequences after preprocessing (all truncated or empty train split)")
    if not calib_items:
        die("no usable calibration sequences after preprocessing")

    if dev_split and not dev_items:
        die("no usable dev sequences after preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    if cached_items is None:
        torch.save(train_items, train_items_path)
        torch.save(calib_items, calib_items_path)
        if dev_split:
            torch.save(dev_items, dev_items_path)

    if args.dry_run:
        print("[dry-run] export + tokenizer admission verified. Stopping before torchrun/local training.")
        return 0

    ddp_script_path = output_dir / "train_ddp.py"
    ddp_script_path.write_text(TRAIN_DDP_SCRIPT.replace('TRAIN_METHOD = "rlcd-grpo"', f'TRAIN_METHOD = "{args.method}"'), encoding="utf-8")

    select_best_epoch_flag = "1" if args.select_best_epoch else "0"
    if args.local:
        # pointsman-change: --local runs the SAME embedded script with plain python3
        # (no torchrun/NCCL/DDP) -- argv[8] selects main_local(), argv[9] is
        # --select-best-epoch, argv[10:] are device/grad-accum/batch-size/
        # autocast/max-steps/dropout ("none" = unset)/rdrop-alpha. argv[1:8] are
        # UNCHANGED from the DDP cmd below.
        cmd = [
            sys.executable, str(ddp_script_path),
            str(resolved_model_dir), str(output_dir), str(train_items_path), str(calib_items_path),
            derived_model_name, base_model_dir_name, str(manifest.get("exporter_version") or ""),
            "local", select_best_epoch_flag, args.device, str(resolved_grad_accum), str(resolved_batch_size),
            args.mps_autocast, str(args.max_steps or 0),
            "none" if args.dropout is None else str(args.dropout), str(args.rdrop_alpha),
            # pointsman-change: argv[17:21] = --epochs, run configuration JSON (enables per-epoch resume
            # checkpoints), "fresh" or the resume epoch dir, "keep"/"clean" (--keep-resume).
            str(resolved_epochs), json.dumps({"match": run_match, "info": run_info}),
            resume_state["epoch_dir"] if resume_state else "fresh", "keep" if args.keep_resume else "clean",
        ]
    else:
        # official-notebook-cell: 5 ("Launch Multi-GPU Fine-Tuning with torchrun")
        # pointsman-change: argv[5:] (derived_model_name, base_model_dir_name, exporter_version) let
        # TRAIN_DDP_SCRIPT's save step record model_name/base_model_dir_name/exporter_version
        # instead of hard-coding model_name -- see derive_model_name() above. argv[8]="ddp" and
        # argv[9]=--select-best-epoch are explicit so load_common_argv() can parse the latter.
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=2", str(ddp_script_path),
            str(resolved_model_dir), str(output_dir), str(train_items_path), str(calib_items_path),
            derived_model_name, base_model_dir_name, str(manifest.get("exporter_version") or ""),
            "ddp", select_best_epoch_flag,
        ]
    print("[train] executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    finished_training_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    metrics = None
    if args.evaluate_sealed_test:
        metrics = evaluate_checkpoint(str(output_dir), str(export_dir), laya_module,
                                      device=args.device if args.local else "cuda")

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # pointsman-change: the official notebook never writes structured run metadata;
    # the promotion flow (qualify -> compare -> promote) needs it to attribute
    # a checkpoint to a specific export + commit.
    training_metadata = {
        "notebook_commit": NOTEBOOK_COMMIT,
        "notebook_upstream_contract": EXPECTED_UPSTREAM_CONTRACT,
        "laya_version_pinned": LAYA_VERSION_PINNED,
        "export_dataset_version": manifest.get("dataset_version"),
        "export_source_data_sha256": manifest.get("source_data_sha256"),
        "export_sample_count": manifest.get("sample_count"),
        "exporter_version": manifest.get("exporter_version"),
        "split_hashes": manifest.get("split_hashes"),
        "provenance_sha256": manifest.get("provenance_sha256"),
        "group_sha256": manifest.get("group_sha256"),
        "selection_split": dev_split if args.select_best_epoch else None,
        "method": args.method, "input_fit": args.input_fit,
        "calibration_split": "calibration",
        "sealed_test_evaluated": args.evaluate_sealed_test,
        "output_weights_sha256": file_sha256(output_dir / "model.safetensors"),
        "tokenizer_sha256": directory_sha256(output_dir / "tokenizer"),
        "model_config_sha256": file_sha256(output_dir / "rl_agent_config.json"),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "laya": getattr(laya_module, "__version__", LAYA_VERSION_PINNED)},
        "precision": "bf16" if args.local and args.mps_autocast == "bf16" else "fp32" if args.local else "fp16",
        "base_model_dir_name": base_model_dir_name,
        "model_subdir": args.model_subdir,
        "derived_model_name": derived_model_name,
        # pointsman-change: mode/device/local-run fields record whether this checkpoint
        # came from the official 2xT4 DDP path or --local (Apple Silicon MPS/CPU).
        "mode": "local" if args.local else "ddp",
        "device": args.device if args.local else "cuda",
        "hyperparameters": {
            "epochs": resolved_epochs, "micro_batch": resolved_batch_size, "grad_accum": resolved_grad_accum, "group_size": 4,
            "lr_encoder": 2.5e-5, "lr_head": 1.0e-4, "sigma_start": 0.4, "sigma_end": 0.1,
            "weight_decay": 0.01, "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256),
            "select_best_epoch": args.select_best_epoch,
            "dropout": args.dropout, "rdrop_alpha": args.rdrop_alpha if args.rdrop_alpha > 0 else None,
        },
        "dev_sequences": len(dev_items),
        "dev_truncated": dev_truncated,
        "train_sequences": len(train_items),
        "train_truncated": train_truncated,
        "calibration_sequences": len(calib_items),
        "calibration_truncated": calib_truncated,
        # pointsman-change: a resumed run keeps the interrupted run's start time and records when it resumed.
        "started_at": run_info["started_at"] if run_info else started_at,
        "resumed_at": started_at if resume_state else None,
        "resumed_from_epoch": resume_state["epochs_completed"] if resume_state else None,
        "finished_training_at": finished_training_at,
        "finished_at": finished_at,
        "metrics": metrics,
        "trained": True,
    }
    if args.local:
        local_run_path = output_dir / "local_training_run.json"
        if local_run_path.exists():
            training_metadata["local_run"] = json.loads(local_run_path.read_text(encoding="utf-8"))
    # pointsman-change: fold epoch_selection.json (written by TRAIN_DDP_SCRIPT's
    # main_ddp()/main_local() -- see write_epoch_selection_metadata()) into
    # training_metadata.json so the promotion flow can see which epoch a
    # saved checkpoint actually came from, for both --local and DDP runs.
    selection_path = output_dir / "epoch_selection.json"
    if selection_path.exists():
        epoch_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        training_metadata["epoch_selection"] = epoch_selection
        training_metadata["selected_epoch"] = epoch_selection.get("selected_epoch")
    (output_dir / "training_metadata.json").write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
    print(f"[done] wrote {output_dir / 'training_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
