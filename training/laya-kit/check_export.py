#!/usr/bin/env python3
"""Offline export validation, shared with the trainer. Standard library only.

Accept legacy v2 train/calibration/test and v3 train/dev/calibration/test JSON
string rows. V3 binds per-file counts/hashes, row associations, split/group/
source rights/oracle/provenance fingerprints and family isolation. A canonical
source hash is cross-checked when --dataset-manifest is supplied. Unannotated
semantic paraphrases remain UNKNOWN; this validator does not prove model quality.

Usage: check_export.py --export-dir DIR [--dataset-manifest FILE]
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_EXPORTER_VERSION = "laya-typed-decisions-json-v3"
LEGACY_EXPORTER_VERSION = "laya-typed-decisions-json-v2"
SPLIT_ROLES = {"train": "parameter_training", "dev": "epoch_method_hyperparameter_selection", "calibration": "temperature_and_gate_fitting", "test": "sealed_frozen_candidate_evaluation"}
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
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
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
    if exporter_version not in (EXPECTED_EXPORTER_VERSION, LEGACY_EXPORTER_VERSION):
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


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_export(export_dir, dataset_manifest=None):
    """Shared offline trust boundary used by this CLI and the trainer before model loading."""
    export_dir = Path(export_dir)
    reporter = Reporter()
    try:
        manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid export manifest: {exc}") from exc
    four_way = manifest.get("exporter_version") == EXPECTED_EXPORTER_VERSION
    splits = tuple(SPLIT_ROLES) if four_way else SPLITS
    split_counts, rows_by_split = {}, {}
    for split in splits:
        path = export_dir / f"{split}.jsonl"
        if not path.is_file():
            reporter.fail(f"{path.name} missing")
            split_counts[split] = 0
            continue
        rows = load_jsonl_rows(path, reporter)
        rows_by_split[split] = rows
        split_counts[split] = len(rows)
        for line_no, row in rows:
            state, questions, gold = check_row_columns(path, line_no, row, reporter)
            if not all(isinstance(x, dict) for x in (state, questions, gold)) or set(questions or {}) != set(gold or {}):
                reporter.fail(f"{path.name}:{line_no} invalid input/question/gold association")
            check_gold_probabilities(path, line_no, gold, reporter)
        if four_way and manifest.get("split_files", {}).get(split) != {"sha256": digest(path.read_text(encoding="utf-8")), "count": len(rows)}:
            reporter.fail(f"{split} file identity mismatch")
    check_manifest(export_dir, split_counts, reporter, dataset_manifest)
    if four_way:
        if manifest.get("split_version") != 2 or manifest.get("split_roles") != SPLIT_ROLES:
            reporter.fail("four-way split role/version mismatch")
        try:
            metadata_text = (export_dir / "metadata.jsonl").read_text(encoding="utf-8")
            metadata = [json.loads(x) for x in metadata_text.splitlines() if x.strip()]
            if digest(metadata_text) != manifest.get("metadata_sha256") or len(metadata) != sum(split_counts.values()):
                reporter.fail("metadata hash/count mismatch")
            positions, identities, ids = set(), {}, set()
            for row in metadata:
                split, index = row["split"], row["row_index"]
                position = (split, index)
                if split not in splits or not isinstance(index, int) or index < 0 or index >= split_counts[split] or position in positions:
                    raise ValueError("metadata row association mismatch")
                positions.add(position)
                if row["sample_id"] in ids:
                    raise ValueError("duplicate sample identity")
                ids.add(row["sample_id"])
                for field in ("sample_id", "group_id", "request_hash", "state_sha256"):
                    if not isinstance(row[field], str) or len(row[field]) != 64 or any(c not in "0123456789abcdef" for c in row[field]):
                        raise ValueError(f"invalid {field}")
                state = json.loads(rows_by_split[split][index][1]["state"])
                if digest(state) != row["state_sha256"]:
                    raise ValueError("state identity mismatch")
                keys = [f"{key}:{row[key]}" for key in ("group_id", "request_hash", "state_sha256")]
                lineage = row.get("lineage") or {}
                keys += [f"{key}:{lineage[key]}" for key in ("source_id", "template_id", "semantic_family_id") if key in lineage]
                keys += [f"sibling:{key}" for key in lineage.get("sibling_ids", [])]
                for key in keys:
                    if key in identities and identities[key] != split:
                        raise ValueError("cross-split family/input leakage")
                    identities[key] = split
                if row.get("oracle"):
                    rights = row.get("data_rights") or {}
                    if row.get("label_source") != "objective" or not rights.get("source") or not rights.get("license") or not rights.get("revision") or ("learning" if split == "train" else "evaluation") not in rights.get("permitted_use", []):
                        raise ValueError("oracle label basis/data rights mismatch")
            split_hashes = {split: digest(sorted(row["sample_id"] for row in metadata if row["split"] == split)) for split in splits}
            group_hash = digest(sorted([row["sample_id"], row["group_id"], row["split"]] for row in metadata))
            provenance_hash = digest(sorted([{key: row.get(key) for key in ("sample_id", "lineage", "data_rights", "oracle", "provenance", "label_source")} for row in metadata], key=lambda row: row["sample_id"]))
            if (manifest.get("split_hashes") != split_hashes or manifest.get("group_sha256") != group_hash or manifest.get("provenance_sha256") != provenance_hash):
                reporter.fail("split/group/provenance fingerprint mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reporter.fail(f"invalid metadata: {exc}")
    if reporter.failures:
        raise ValueError("; ".join(reporter.failures))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--dataset-manifest", default=None)
    args = parser.parse_args()
    try:
        verify_export(args.export_dir, args.dataset_manifest)
    except ValueError as exc:
        print(f"check_export.py: FAILED ({exc})")
        return 1
    print("check_export.py: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
