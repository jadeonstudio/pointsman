#!/usr/bin/env python3
"""Archive one pinned public snapshot; export allowlisted inputs without labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

REPO = "typesafe/evalsafe-invoice-processing"
REVISION = "6beeb2d2acd65c086c835022f5f4d7434114cafc"
FILES = ("LICENSE", "README.md", "dataset.json", "data/cases.parquet",
         "data/questions.parquet", "data/run_results.parquet")
PINNED_HASHES = dict(zip(FILES, (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "1919a1a8e9471cc6e329ae2d2fbc6c30213ea01af3c27ccd64737aff04561d15",
    "72e0cea51735ae3d24dba0dad53536aca72b5698262d57d2506db1b900730e37",
    "6a62a51f4b6cda738e0be6a376284ec8c864d580259dd9a6aa30e77685404854",
    "416db987cf173b5916c79f7f1d537d2a237e32301aeab4a6d3813bc61d1080bb",
    "d4cc742758998aaaf4e4b21cad9055642ed57184e07ed124499051b645414a35",
)))  # Fixed official revision downloaded on 2026-10-03; not local receipt authority.
INPUT_COLUMNS = {
    "cases": ("case_id", "input_json"),
    "questions": ("question_instance_id", "case_id", "node_id", "question_id",
                  "kind", "question_json", "state_json"),
}
JEV_RUN = "code-typesafe-jev-1.13.0-off-160259a2ba80"
TARGET_QUESTION = "discount_days"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)


def source_url(name):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"


def verify_source(root, expected_hashes=PINNED_HASHES):
    receipt = json.loads((root / "source-hashes.json").read_text())
    if (receipt.get("repository") != REPO or receipt.get("revision") != REVISION
            or set(receipt.get("files", {})) != set(FILES)):
        raise ValueError("Source receipt identity/file list differs from pinned snapshot")
    for name, identity in receipt["files"].items():
        path = root / "source" / name
        if (identity.get("url") != source_url(name)
                or identity.get("sha256") != expected_hashes[name]
                or path.stat().st_size != identity.get("bytes")
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[name]):
            raise ValueError(f"Source receipt/bytes differ from pinned snapshot: {name}")
    return receipt


def acquire(root):
    if (root / "source-hashes.json").exists():
        receipt = verify_source(root)
        print(json.dumps({"revision": REVISION, "verified_files": len(receipt["files"])}))
        return
    if (root / "source").exists() and any((root / "source").rglob("*")):
        raise FileExistsError("Nonempty source has no receipt; use a fresh directory")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    source = root / "source"
    hashes = {}
    for name in FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        url = source_url(name)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            request = urllib.request.Request(url, headers={"User-Agent": "pointsman-public-invoice"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.chmod(0o600)
            temporary.replace(path)
        hashes[name] = {"url": url, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if hashes[name]["sha256"] != PINNED_HASHES[name]:
            raise ValueError(f"Downloaded bytes differ from pinned snapshot: {name}")
    manifest = json.loads((source / "dataset.json").read_text())
    assert manifest["license"] == "apache-2.0"
    write_json(root / "source-hashes.json", {"repository": REPO, "revision": REVISION, "files": hashes})
    print(json.dumps({"revision": REVISION, "files": len(hashes), "bytes": sum(x["bytes"] for x in hashes.values())}))


def inspect(root):
    verify_source(root)
    import pyarrow.parquet as pq
    summary = {}
    for name in FILES:
        if name.endswith(".parquet"):
            parquet = pq.ParquetFile(root / "source" / name)
            summary[name] = {"rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow)}
    write_json(root / "schema.json", summary)
    print(json.dumps(summary, indent=2))


def input_record(row, columns):
    return {name: row[name] for name in columns}


def canonical_input_hash(question_json, state_json):
    value = {"question": json.loads(question_json), "state": json.loads(state_json)}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def jev_inputs(root):
    """Project recorded inputs only; do not decode any prediction leaf."""
    verify_source(root)
    destination = root / "blindjev"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Recorded-input export exists; preserve it and use a fresh directory")
    import pyarrow
    import pyarrow.parquet as pq
    blind_manifest = json.loads((root / "blinded" / "manifest.json").read_text())
    assert blind_manifest["revision"] == REVISION and blind_manifest["repository"] == REPO
    for name, identity in blind_manifest["outputs"].items():
        assert hashlib.sha256((root / "blinded" / name).read_bytes()).hexdigest() == identity["sha256"]
    blind = [json.loads(line) for line in (root / "blinded" / "blinded-inputs.jsonl").read_text().splitlines()]
    target = [row for row in blind if row["question_id"] == TARGET_QUESTION]
    expected = {}
    for row in target:
        key = (row["case_id"], row["node_id"], row["question_id"],
               canonical_input_hash(row["question_json"], row["state_json"]))
        expected.setdefault(key, []).append(row)
    leaves = ["run_id", "case_id", "decisions.list.element.policy_id"] + [
        f"questions.list.element.{name}" for name in
        ("node_id", "question_id", "kind", "question_json", "state_json")]
    parquet = pq.ParquetFile(root / "source" / "data" / "run_results.parquet")
    assert set(leaves) <= {parquet.schema.column(i).path for i in range(len(parquet.schema))}
    rows = parquet.read(columns=leaves).to_pylist()
    records = []
    selected_cases = []
    exact_text_matches = 0
    canonical_matches = 0
    for source_row, row in enumerate(rows):
        if row["run_id"] != JEV_RUN:
            continue
        selected_cases.append(row["case_id"])
        for source_question, question in enumerate(row["questions"]):
            if question["question_id"] != TARGET_QUESTION:
                continue
            digest = canonical_input_hash(question["question_json"], question["state_json"])
            key = (row["case_id"], question["node_id"], question["question_id"], digest)
            matches = expected.get(key, [])
            canonical_matches += bool(matches)
            exact_text_matches += any((m["question_json"], m["state_json"]) ==
                                      (question["question_json"], question["state_json"]) for m in matches)
            records.append({"run_id": JEV_RUN, "case_id": row["case_id"], **question,
                            "policy_ids_for_case": [d["policy_id"] for d in row["decisions"]],
                            "source_row_index": source_row, "source_question_index": source_question,
                            "canonical_input_sha256": digest,
                            "matching_blind_question_instance_ids": [m["question_instance_id"] for m in matches]})
    counts = Counter(record["case_id"] for record in records)
    input_counts = Counter((r["case_id"], r["canonical_input_sha256"]) for r in records)
    target_cases = {row["case_id"] for row in target}
    summary = {
        "blind_target_instances": len(target), "blind_target_cases": len(target_cases),
        "recorded_run_rows": len(selected_cases), "duplicate_run_case_rows": len(selected_cases) - len(set(selected_cases)),
        "recorded_target_instances": len(records), "recorded_target_cases": len(counts),
        "canonical_matches": canonical_matches, "exact_json_text_matches": exact_text_matches,
        "unmatched_recorded_inputs": len(records) - canonical_matches,
        "absent_target_cases": len(target_cases - set(counts)),
        "duplicate_same_case_input_instances": sum(n - 1 for n in input_counts.values()),
        "cases_with_multiple_different_inputs": sum(len({r["canonical_input_sha256"] for r in records if r["case_id"] == case}) > 1 for case in counts),
        "policy_context_counts_per_input": dict(Counter(len(r["policy_ids_for_case"]) for r in records)),
        "answer_status_inspected": False,
    }
    destination.mkdir(exist_ok=True, mode=0o700)
    path = destination / "discount-days-recorded-inputs.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))
    path.chmod(0o600)
    write_json(destination / "receipt.json", {
        "repository": REPO, "revision": REVISION, "run_id": JEV_RUN, "question_id": TARGET_QUESTION,
        "selection": "Every recorded target input, without reading answer or status leaves; absent is not fabricated",
        "policy_association": "Case-level policy IDs only; source stores no question-level policy ID; these are not independent input instances",
        "source_columns": leaves, "source_sha256": PINNED_HASHES["data/run_results.parquet"],
        "blinded_manifest_sha256": hashlib.sha256((root / "blinded" / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version.split()[0], "pyarrow": pyarrow.__version__},
        "output": {"file": path.name, "rows": len(records), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "counts": summary,
    })
    print(json.dumps(summary, indent=2))


def blind(root):
    verify_source(root)
    destination = root / "blinded"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Blinded output already exists; preserve it and use a fresh directory")
    import pyarrow
    import pyarrow.parquet as pq
    destination.mkdir(exist_ok=True, mode=0o700)
    output = {}
    case_ids = set()
    for name, columns in INPUT_COLUMNS.items():
        # Read only allowlisted Parquet columns. Never read run_results values.
        rows = pq.read_table(root / "source" / "data" / f"{name}.parquet", columns=columns).to_pylist()
        path = destination / ("blinded-cases.jsonl" if name == "cases" else "blinded-inputs.jsonl")
        with path.open("w") as stream:
            for row in rows:
                record = input_record(row, columns)
                assert set(record) == set(columns)
                for field in columns:
                    if field.endswith("_json"):
                        json.loads(record[field])  # Preserve the original JSON text exactly.
                if name == "cases":
                    assert row["case_id"] not in case_ids
                    case_ids.add(row["case_id"])
                else:
                    assert row["case_id"] in case_ids
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        path.chmod(0o600)
        recovered = [json.loads(line) for line in path.read_text().splitlines()]
        assert recovered == [input_record(row, columns) for row in rows]
        if name == "questions":
            assert len({row["question_instance_id"] for row in rows}) == len(rows)
        output[path.name] = {"rows": len(rows), "columns": columns,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = json.loads((root / "source" / "dataset.json").read_text())
    write_json(destination / "manifest.json", {
        "repository": REPO, "revision": REVISION, "license": metadata["license"],
        "source_hashes_sha256": hashlib.sha256((root / "source-hashes.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version.split()[0], "pyarrow": pyarrow.__version__},
        "selection": "All source cases and question instances, no label/status/run/outcome filter",
        "policy_metadata": metadata["policies"],
        "policy_rules": "Not present in dataset snapshot; executable workflow rules require separate provenance",
        "upstream_input_caveat": "Question instances originate from published reference workflow traces; not a newly preregistered unseen cohort",
        "outputs": output,
    })
    print(json.dumps(output, indent=2))


def self_test():
    row = {"case_id": "synthetic", "input_json": "{}", "consensus": "MUST_NOT_LEAK",
           "answer_json": "MUST_NOT_LEAK", "confidence": "MUST_NOT_LEAK",
           "decisions": "MUST_NOT_LEAK", "wall_time_s": "MUST_NOT_LEAK"}
    assert input_record(row, INPUT_COLUMNS["cases"]) == {"case_id": "synthetic", "input_json": "{}"}
    assert all(not set(columns) & {"answer_json", "confidence", "decisions", "wall_time_s", "status"}
               for columns in INPUT_COLUMNS.values())
    assert canonical_input_hash('{"b":2,"a":1}', '{}') == canonical_input_hash('{"a":1, "b":2}', '{}')
    assert canonical_input_hash('{}', '{"amount":1}') != canonical_input_hash('{}', '{"amount":2}')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        files = {}
        expected = {}
        for name in FILES:
            path = root / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            files[name] = {"url": source_url(name), "bytes": path.stat().st_size, "sha256": expected[name]}
        receipt = {"repository": REPO, "revision": REVISION, "files": files}
        receipt_path = root / "source-hashes.json"
        write_json(receipt_path, receipt)
        before = receipt_path.read_bytes()
        assert verify_source(root, expected) == verify_source(root, expected) == receipt
        assert receipt_path.read_bytes() == before

        def rejected():
            try:
                verify_source(root, expected)
            except ValueError:
                return
            raise AssertionError("Tampered source/receipt was accepted")

        (root / "source" / "LICENSE").write_bytes(b"tampered")
        rejected()
        (root / "source" / "LICENSE").write_bytes(b"LICENSE")
        receipt["revision"] = "wrong"
        write_json(receipt_path, receipt)
        rejected()
        receipt["revision"] = REVISION
        receipt["files"]["LICENSE"]["sha256"] = "0" * 64
        write_json(receipt_path, receipt)
        rejected()
        del receipt["files"]["LICENSE"]
        write_json(receipt_path, receipt)
        rejected()
        receipt_path.unlink()
        try:
            acquire(root)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Unreceipted local bytes were accepted")
    print("Input projection and archive immutability self-checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "inspect", "blind", "jev-inputs", "self-test"))
    parser.add_argument("--root", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "self-test":
        self_test()
    elif arguments.root is None:
        parser.error("--root is required")
    else:
        globals()[arguments.command.replace("-", "_")](arguments.root)
