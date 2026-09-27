#!/usr/bin/env python3
"""check_export.py -- offline sanity check for a pointsman Laya export folder.

Standard library only. No network, no model loading, no training.

Verifies (per the pointsman training-data spec's Laya export section and
src/training/dataset.mjs `exportDataset`):
  - train.jsonl / calibration.jsonl / test.jsonl exist and are non-empty
  - each row has exactly the three JSON-string columns: state, questions, gold
    (each column's *value* must itself be a JSON string that further
    json.loads()-parses to an object, matching the official notebook's
    `json.loads(row["state"])` etc.)
  - each row's gold[question_id].probabilities sums to 1 +/- 0.02
  - manifest.json is present, declares the expected exporter_version /
    upstream_contract, and its declared sample_count matches the actual
    number of rows summed across the three splits

What this script CANNOT verify offline (report as "cannot confirm", never guess):
  - manifest.json's source_data_sha256 against the original canonical
    dataset, unless the sibling dataset manifest
    (manifests/<dataset_version>.json) is also supplied with --dataset-manifest
  - whether the export corresponds to the *current* repository state

Usage:
  python3 check_export.py --export-dir /path/to/exports/<version>/laya \
      [--dataset-manifest /path/to/manifests/<version>.json]

Exit code 0 = all checks passed. Non-zero = at least one FAIL.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_EXPORTER_VERSION = "laya-typed-decisions-json-v2"
EXPECTED_UPSTREAM_CONTRACT = (
    "NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527:"
    "notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
)
SPLITS = ("train", "calibration", "test")
PROB_SUM_TOLERANCE = 0.02


class Reporter:
    def __init__(self):
        self.failures = []
        self.warnings = []
        self.unknown = []

    def fail(self, msg):
        self.failures.append(msg)
        print(f"[FAIL] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"[WARN] {msg}")

    def unk(self, msg):
        self.unknown.append(msg)
        print(f"[cannot confirm] {msg}")

    def ok(self, msg):
        print(f"[OK]   {msg}")


def load_jsonl_rows(path, reporter):
    rows = []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        reporter.fail(f"{path.name} is empty")
        return rows
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            reporter.fail(f"{path.name}:{line_no} is not valid JSON ({exc})")
            continue
        rows.append((line_no, row))
    return rows


def check_row_columns(path, line_no, row, reporter):
    """Each row must have exactly state/questions/gold, each a JSON *string*
    column (per the export contract: the official notebook calls
    json.loads() on each of the three columns)."""
    if not isinstance(row, dict):
        reporter.fail(f"{path.name}:{line_no} row is not an object")
        return None, None, None
    expected_keys = {"state", "questions", "gold"}
    if set(row.keys()) != expected_keys:
        reporter.fail(
            f"{path.name}:{line_no} expected exactly {sorted(expected_keys)}, "
            f"got {sorted(row.keys())}"
        )
    state_raw = row.get("state")
    questions_raw = row.get("questions")
    gold_raw = row.get("gold")
    for name, raw in (("state", state_raw), ("questions", questions_raw), ("gold", gold_raw)):
        if not isinstance(raw, str):
            reporter.fail(f"{path.name}:{line_no} column '{name}' is not a JSON string")
    state = questions = gold = None
    try:
        state = json.loads(state_raw) if isinstance(state_raw, str) else None
    except json.JSONDecodeError as exc:
        reporter.fail(f"{path.name}:{line_no} column 'state' does not parse as JSON ({exc})")
    try:
        questions = json.loads(questions_raw) if isinstance(questions_raw, str) else None
    except json.JSONDecodeError as exc:
        reporter.fail(f"{path.name}:{line_no} column 'questions' does not parse as JSON ({exc})")
    try:
        gold = json.loads(gold_raw) if isinstance(gold_raw, str) else None
    except json.JSONDecodeError as exc:
        reporter.fail(f"{path.name}:{line_no} column 'gold' does not parse as JSON ({exc})")
    return state, questions, gold


def check_gold_probabilities(path, line_no, gold, reporter):
    if not isinstance(gold, dict):
        return
    for question_id, entry in gold.items():
        if not isinstance(entry, dict) or "probabilities" not in entry:
            reporter.fail(f"{path.name}:{line_no} gold[{question_id!r}] missing 'probabilities'")
            continue
        probs = entry["probabilities"]
        if not isinstance(probs, dict) or not probs:
            reporter.fail(f"{path.name}:{line_no} gold[{question_id!r}].probabilities is not a non-empty object")
            continue
        try:
            values = [float(v) for v in probs.values()]
        except (TypeError, ValueError):
            reporter.fail(f"{path.name}:{line_no} gold[{question_id!r}].probabilities has a non-numeric value")
            continue
        total = sum(values)
        if abs(total - 1.0) > PROB_SUM_TOLERANCE:
            reporter.fail(
                f"{path.name}:{line_no} gold[{question_id!r}].probabilities sums to "
                f"{total:.4f}, outside 1 +/- {PROB_SUM_TOLERANCE}"
            )
        if any(v < 0 for v in values):
            reporter.fail(f"{path.name}:{line_no} gold[{question_id!r}].probabilities has a negative value")


def check_manifest(export_dir, split_counts, reporter, dataset_manifest_path=None):
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        reporter.fail(f"{manifest_path.name} not found in {export_dir}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reporter.fail(f"manifest.json is not valid JSON ({exc})")
        return

    exporter_version = manifest.get("exporter_version")
    if exporter_version != EXPECTED_EXPORTER_VERSION:
        reporter.fail(
            f"manifest.exporter_version={exporter_version!r}, "
            f"expected {EXPECTED_EXPORTER_VERSION!r}"
        )
    else:
        reporter.ok(f"exporter_version matches ({exporter_version})")

    upstream_contract = manifest.get("upstream_contract")
    if upstream_contract != EXPECTED_UPSTREAM_CONTRACT:
        reporter.fail(
            f"manifest.upstream_contract={upstream_contract!r}, "
            f"expected {EXPECTED_UPSTREAM_CONTRACT!r}"
        )
    else:
        reporter.ok("upstream_contract matches the pinned official notebook commit")

    declared_count = manifest.get("sample_count")
    actual_count = sum(split_counts.values())
    if not isinstance(declared_count, int):
        reporter.fail("manifest.sample_count missing or not an integer")
    elif declared_count != actual_count:
        reporter.fail(
            f"manifest.sample_count={declared_count} but train+calibration+test "
            f"rows actually total {actual_count} ({split_counts})"
        )
    else:
        reporter.ok(f"manifest.sample_count matches actual row count ({actual_count})")

    source_hash = manifest.get("source_data_sha256")
    if not source_hash:
        reporter.fail("manifest.source_data_sha256 missing")
    elif dataset_manifest_path is None:
        reporter.unk(
            "source_data_sha256 present in export manifest, but no --dataset-manifest "
            "was supplied to cross-check it against the canonical dataset manifest"
        )
    else:
        try:
            dataset_manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reporter.fail(f"could not read --dataset-manifest {dataset_manifest_path}: {exc}")
        else:
            dataset_hash = dataset_manifest.get("data_sha256")
            if dataset_hash != source_hash:
                reporter.fail(
                    f"export manifest.source_data_sha256={source_hash!r} does not match "
                    f"dataset manifest.data_sha256={dataset_hash!r}"
                )
            else:
                reporter.ok("source_data_sha256 matches the supplied dataset manifest")

    if manifest.get("training_executed") is not False:
        reporter.warn("manifest.training_executed is not explicitly false")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True, help="Path to exports/<version>/laya")
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Optional path to manifests/<version>.json to cross-check source_data_sha256",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"[FAIL] export dir does not exist: {export_dir}")
        return 1

    reporter = Reporter()
    split_counts = {}

    for split in SPLITS:
        path = export_dir / f"{split}.jsonl"
        if not path.exists():
            reporter.fail(f"{path.name} not found in {export_dir}")
            split_counts[split] = 0
            continue
        rows = load_jsonl_rows(path, reporter)
        split_counts[split] = len(rows)
        if not rows:
            reporter.fail(f"{path.name} has no data rows")
            continue
        for line_no, row in rows:
            state, questions, gold = check_row_columns(path, line_no, row, reporter)
            check_gold_probabilities(path, line_no, gold, reporter)
        reporter.ok(f"{path.name}: {len(rows)} rows checked")

    check_manifest(export_dir, split_counts, reporter, args.dataset_manifest)

    print()
    print(f"Summary: {len(reporter.failures)} FAIL, {len(reporter.warnings)} WARN, {len(reporter.unknown)} cannot-confirm")
    if reporter.failures:
        print("check_export.py: FAILED")
        return 1
    print("check_export.py: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
