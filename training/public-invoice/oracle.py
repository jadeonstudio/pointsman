#!/usr/bin/env python3
"""Blind, source-only oracle for canonical invoice discount-term questions."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

QUESTION_ID = "discount_days"
DAYS = (7, 10, 15, 20, 30)
GRAMMAR = {
    "normalization": "strip outer whitespace and casefold; full-string matching only",
    "no_discount": "due on receipt | net <positive integer>",
    "discount": "<positive decimal percent <= 100, optional %>/<positive integer> net <positive integer>",
    "classification": "discount window in {7,10,15,20,30} => its string; otherwise none",
    "unsupported": "missing, non-string, conflicting, unknown or nonpositive terms; never guessed",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def classify(terms):
    if not isinstance(terms, str):
        raise ValueError("unsupported_payment_terms")
    text = terms.strip().casefold()
    if text == "due on receipt" or re.fullmatch(r"net\s+[1-9][0-9]*", text):
        return "none"
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:%)?/([1-9][0-9]*)\s+net\s+([1-9][0-9]*)", text)
    if not match or not 0 < Decimal(match[1]) <= 100:
        raise ValueError("unsupported_payment_terms")
    return match[2] if int(match[2]) in DAYS else "none"


def question(terms):
    return {
        "type": "choice",
        "instructions": f'Do the invoice\'s payment terms ("{terms}") offer an EARLY-PAYMENT DISCOUNT, and if so within how many days of the invoice date? Do not judge whether the window is still open; code dates it.',
        "criteria": {
            **{str(day): f"A discount if paid within {day} days of the invoice date ('2/{day} net 30' means {day})." for day in DAYS},
            "none": "No early-payment discount is offered, or the window is some other number of days.",
        },
    }


def validated_input(row, packet):
    state = json.loads(row["state_json"])
    terms = packet["invoice"]["fields"].get("payment_terms")
    if state.get("case_id") != row["case_id"] or packet.get("case_id") != row["case_id"]:
        raise ValueError("case_identity_mismatch")
    # The packet, not exact_facts, supplies the oracle's source value.
    if state["invoice"]["fields"].get("payment_terms") != terms or state["exact_facts"].get("payment_terms") != terms:
        raise ValueError("payment_terms_mismatch")
    if row["question_id"] != QUESTION_ID or row["kind"] != "choice" or json.loads(row["question_json"]) != question(terms):
        raise ValueError("question_identity_mismatch")
    label = classify(terms)
    return {
        "question_instance_id": row["question_instance_id"],
        "case_id": row["case_id"],
        "node_id": row["node_id"],
        "question_id": QUESTION_ID,
        "kind": "choice",
        "payment_terms": terms,
        "gold": label,
        "question_identity": identity(json.loads(row["question_json"])),
        "state_identity": identity(state),
        "question_json_sha256": digest(row["question_json"].encode()),
        "state_json_sha256": digest(row["state_json"].encode()),
    }


def freeze(cases_path, inputs_path, output, source_dir, recorded_path):
    cases = read_jsonl(cases_path)
    packets = {row["case_id"]: json.loads(row["input_json"]) for row in cases}
    inputs = read_jsonl(inputs_path)
    if len(packets) != len(cases):
        raise ValueError("duplicate_case")
    gold, excluded, seen = [], Counter(), set()
    for row in inputs:
        if row["question_id"] != QUESTION_ID:
            excluded["outside_frozen_question_scope"] += 1
            continue
        if row["case_id"] in seen:
            raise ValueError("duplicate_discount_question")
        seen.add(row["case_id"])
        try:
            gold.append(validated_input(row, packets[row["case_id"]]))
        except ValueError as error:
            if str(error) != "unsupported_payment_terms":
                raise
            excluded[str(error)] += 1
    gold.sort(key=lambda row: row["case_id"])
    recorded = read_jsonl(recorded_path)
    matched = match_inputs(gold, recorded)
    run_ids = sorted({row["run_id"] for row in recorded})
    if len(run_ids) != 1:
        raise ValueError("multiple_recorded_runs")
    prereg = {
        "schema_version": 1,
        "scope": "narrow canonical discount-term classification only; no invoice approval or fraud gold",
        "question_id": QUESTION_ID,
        "grammar": GRAMMAR,
        "candidate_identity": list(question("x")["criteria"]),
        "gold_authority": "original case input invoice.fields.payment_terms plus explicit question criteria; exact_facts corroborates only",
        "selection": "all rows of discount_days that satisfy the frozen grammar, regardless of case outcome or model answer",
        "input_identity": "logical JSON identities of entire question and entire state; raw JSON hashes retained",
        "match_scope": "all eligible case/question pairs; no missing, duplicate, altered question or altered state accepted",
        "metrics": ["exact classification", "always-none baseline", "per-class recall", "observed-class macro precision/recall/F1", "case-clustered paired comparison"],
        "limitations": ["published reference-trace questions, not an unseen prospective cohort", "observed labels only none and 10; imbalance requires macro metrics", "no broad Jev superiority claim"],
        "source": {
            "dataset": "typesafe/evalsafe-invoice-processing",
            "dataset_revision": "6beeb2d2acd65c086c835022f5f4d7434114cafc",
            "code": "typesafe-ai/WorkflowEvals",
            "code_revision": "0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb",
            "dataset_license": "Apache-2.0; LICENSE independently checked",
            "code_license": "Apache-2.0",
            "oracle_license": "MIT (pointsman LICENSE)",
            "oracle_license_sha256": digest((Path(__file__).parents[2] / "LICENSE").read_bytes()),
            "code_sha256": {path.name: digest(path.read_bytes()) for path in sorted(Path(source_dir).iterdir()) if path.is_file()},
            "cases_sha256": digest(Path(cases_path).read_bytes()),
            "inputs_sha256": digest(Path(inputs_path).read_bytes()),
            "oracle_sha256": digest(Path(__file__).read_bytes()),
            "recorded_inputs_sha256": digest(Path(recorded_path).read_bytes()),
            "recorded_run_id": run_ids[0],
        },
        "recorded_input_match": matched,
        "coverage": {"cases": len(cases), "question_rows": len(inputs), "eligible": len(gold), "eligible_cases": len(gold), "excluded": dict(excluded)},
        "gold_distribution": dict(Counter(row["gold"] for row in gold)),
        "terms_distribution": dict(Counter(row["payment_terms"] for row in gold)),
        "row_ids": [row["question_instance_id"] for row in gold],
    }
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any((target / name).exists() for name in ("preregistration.json", "gold.jsonl", "manifest.json")):
        raise ValueError("freeze_already_exists")
    # Exclusive writes prevent accidental revision of the pre-unblind contract.
    with (target / "preregistration.json").open("x") as stream:
        json.dump(prereg, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    with (target / "gold.jsonl").open("x") as stream:
        for row in gold:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {path.name: digest(path.read_bytes()) for path in (target / "preregistration.json", target / "gold.jsonl")}
    with (target / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    for path in target.iterdir():
        path.chmod(0o600)
    return {"files": manifest, "coverage": prereg["coverage"], "gold_distribution": prereg["gold_distribution"]}


def read_frozen(gold_path):
    path = Path(gold_path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    for name in ("gold.jsonl", "preregistration.json"):
        if digest((path.parent / name).read_bytes()) != manifest[name]:
            raise ValueError("frozen_artifact_hash_mismatch")
    return read_jsonl(path)


def match_inputs(gold, recorded):
    expected = {(row["case_id"], row["question_id"]): row for row in gold}
    seen = set()
    for row in recorded:
        if row["question_id"] != QUESTION_ID:
            continue
        key = (row["case_id"], row["question_id"])
        if key not in expected or key in seen:
            raise ValueError("unexpected_or_duplicate_recorded_input")
        entry = expected[key]
        if row["kind"] != entry["kind"] or row["node_id"] != entry["node_id"]:
            raise ValueError("recorded_question_metadata_mismatch")
        if identity(json.loads(row["question_json"])) != entry["question_identity"]:
            raise ValueError("recorded_question_mismatch")
        if identity(json.loads(row["state_json"])) != entry["state_identity"]:
            raise ValueError("recorded_state_mismatch")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("missing_recorded_inputs")
    return {"matched": len(seen), "input_identity": "complete question and state JSON"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    for name in ("cases", "inputs", "output", "source-dir", "recorded"):
        freeze_parser.add_argument("--" + name, required=True)
    match_parser = sub.add_parser("match-inputs")
    match_parser.add_argument("--gold", required=True)
    match_parser.add_argument("--recorded", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.cases, args.inputs, args.output, args.source_dir, args.recorded)
    else:
        result = match_inputs(read_frozen(args.gold), read_jsonl(args.recorded))
    print(json.dumps(result, sort_keys=True))
