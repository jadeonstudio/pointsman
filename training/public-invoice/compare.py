#!/usr/bin/env python3
"""Frozen, case-paired comparison for one invoice question; no inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

GOLD_HASHES = {"preregistration.json": "010c1aa3df38294833c6cd2e93992f5011af410402bafce91838b384e4f77555",
               "gold.jsonl": "8b0b2883a48612dc0945f9d474e0fcffa259d804cc8ced1e62225175aeec73f2"}
SOURCE_HASH = "d4cc742758998aaaf4e4b21cad9055642ed57184e07ed124499051b645414a35"
INPUT_HASH = "f3e8826843fbca0f00ad93bcd63ccc272d1d19584be990d8e4622d2a9ddc2a94"
PRE_UNBLIND_HASH = "8b53b218389e8a4d305249c84403be938106fcc3cd9f148d9d384b02239b56a4"
JEV_RUN = "code-typesafe-jev-1.13.0-off-160259a2ba80"
CLEF_CHECKPOINT = "2d78acadca4a2d3865b6c9efd8402d1b2483c1dc6c43a802f81ecc72dcb45e63"
OPTIONS = ("7", "10", "15", "20", "30", "none")
TOLERANCE = .02  # Same rounded probability-mass contract as general-decisions/evaluate.mjs.
SEED, BOOTSTRAPS = 42, 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def frozen_gold(root):
    directory = root / "private-gold"
    for name, expected in GOLD_HASHES.items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen gold identity mismatch")
    rows = read_rows(directory / "gold.jsonl")
    if len(rows) != 150 or len({row["case_id"] for row in rows}) != 150:
        raise ValueError("Expected exactly 150 frozen cases")
    return rows


def freeze(root):
    gold = frozen_gold(root)  # Required before touching any prediction values.
    if sha(root / "blindjev" / "discount-days-recorded-inputs.jsonl") != INPUT_HASH:
        raise ValueError("Recorded input identity mismatch")
    directory = root / "comparison"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Comparison already frozen")
    directory.mkdir(mode=0o700)
    prereg = {
        "gold_hashes": GOLD_HASHES, "recorded_inputs_sha256": INPUT_HASH,
        "source_predictions_sha256": SOURCE_HASH, "source_run_id": JEV_RUN,
        "script_sha256": sha(__file__), "options": OPTIONS,
        "models": {"jev": "jev-1.13.0", "clef_model_id": "clef_flash_mlx8_torch_bf16",
                   "clef_weights_checkpoint_reference": CLEF_CHECKPOINT,
                   "clef_inference_contract": "invoice-recorded-payload-v1", "control": "always-none"},
        "selection": "All 150 frozen case/question/state pairs, no model quality/status filtering",
        "row_identity": "case_id,question_id,node_id,complete question_identity and state_identity",
        "missing_invalid": "Full-envelope failure; expose coverage; exclude from distribution metrics only",
        "distribution": {"exact_option_keys": OPTIONS, "finite": True, "range": [0, 1],
                         "positive_mass": True, "mass_tolerance": TOLERANCE,
                         "normalization": "Normalize total rounded mass only; preserve actual choice and original probabilities",
                         "nll_floor": 1e-15, "brier": "Sum of squared class errors, no division by class count"},
        "confidence": "ECE uses normalized probability of the actual selected choice; provider confidence is separate and unused",
        "metrics": ["six-class confusion plus invalid column", "full-envelope/served accuracy", "per-class recall",
                    "observed-gold-class macro precision/recall/F1", "six-class macro F1 with zero-support F1=0",
                    "NLL", "Brier", "10 equal-width-bin ECE", "coverage", "two-sided exact McNemar",
                    "paired case bootstrap accuracy difference", "Clopper-Pearson error bounds"],
        "paired": {"unit": "case", "bootstrap_draws": BOOTSTRAPS, "seed": SEED,
                   "interval": "percentile 2.5/97.5 using sorted indices floor(B*p)",
                   "mcnemar": "min(1,2*BinomialCDF(min(improved,regressed),discordant,0.5))"},
        "error_bounds": "Two-sided 95% exact interval and one-sided 95% exact upper bound; iid-case assumption",
        "gold_distribution": dict(Counter(row["gold"] for row in gold)),
        "scope": "One narrow public reference-trace family; not unseen-domain C evidence or whole-task A",
        "timing": "Published Jev run timing is not a matched local Clef latency comparison",
        "parser_control": "Frozen grammar oracle is 100% correct by construction; not learned-task or efficiency evidence",
    }
    previous = root / "comparison-pre-unblind-v1" / "preregistration.json"
    if sha(previous) != PRE_UNBLIND_HASH:
        raise ValueError("Original pre-unblind contract changed")
    original = json.loads(previous.read_text())
    keys = set(original) - {"script_sha256"}
    if identity({key: prereg[key] for key in keys}) != identity({key: original[key] for key in keys}):
        raise ValueError("Numerical/eligibility pre-unblind rules changed")
    prereg["original_pre_unblind_preregistration_sha256"] = PRE_UNBLIND_HASH
    prereg["source_status_protocol_clarification"] = {
        "answered": "internal ok", "ok": "internal ok", "other_status": "coverage failure",
        "retention": "Original source_status retained; actual choice never replaced by probability argmax",
        "reason": "Published question status uses answered; protocol preparation correction, not a quality/eligibility change",
    }
    write_json(directory / "preregistration.json", prereg)
    write_json(directory / "preregistration-manifest.json", {"sha256": sha(directory / "preregistration.json")})
    print(json.dumps({"comparison_preregistration_sha256": sha(directory / "preregistration.json")}))


def verify_contract(root):
    frozen_gold(root)
    directory = root / "comparison"
    receipt = json.loads((directory / "preregistration-manifest.json").read_text())
    path = directory / "preregistration.json"
    if sha(path) != receipt["sha256"]:
        raise ValueError("Comparison preregistration changed")
    spec = json.loads(path.read_text())
    if spec["gold_hashes"] != GOLD_HASHES or spec["script_sha256"] != sha(__file__):
        raise ValueError("Comparison implementation/gold changed")
    return spec


def validated(prediction):
    if prediction is None:
        return "missing", None
    if prediction.get("status") != "ok":
        return str(prediction.get("status", "invalid_status")), None
    raw = prediction.get("probabilities")
    if (prediction.get("choice") not in OPTIONS or not isinstance(raw, dict) or set(raw) != set(OPTIONS)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or v < 0 or v > 1 for v in raw.values())):
        return "malformed", None
    mass = sum(raw.values())
    if mass <= 0 or abs(mass - 1) > TOLERANCE:
        return "malformed", None
    return "ok", {key: raw[key] / mass for key in OPTIONS}


def extract(root):
    verify_contract(root)  # The answer parser starts only after this frozen check.
    gold = {row["case_id"]: row for row in frozen_gold(root)}
    source = root / "source" / "data" / "run_results.parquet"
    if sha(source) != SOURCE_HASH:
        raise ValueError("Published Jev snapshot changed")
    output = root / "comparison" / "jev-predictions.jsonl"
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Jev prediction artifact already frozen")
    # Reuse the already frozen extraction; do not read unchanged full Parquet again.
    previous = root / "comparison-pre-unblind-v1" / "jev-predictions.jsonl"
    if previous.exists():
        receipt = json.loads(previous.with_suffix(".manifest.json").read_text())
        if sha(previous) != receipt["sha256"] or receipt["preregistration_sha256"] != PRE_UNBLIND_HASH:
            raise ValueError("Original frozen Jev extraction changed")
        predictions = read_rows(previous)
        for row in predictions:
            row["source_status"] = row["status"]
            row["status"] = "ok" if row["source_status"] in ("answered", "ok") else row["source_status"]
        return freeze_predictions(root, output, predictions, gold)
    import pyarrow.parquet as pq
    fields = ("node_id", "question_id", "question_json", "state_json", "status", "answer_json",
              "probabilities", "confidence", "confidence_method")
    parquet = pq.ParquetFile(source)
    rows = parquet.read(columns=["run_id", "case_id"] + [f"questions.list.element.{f}" for f in fields]).to_pylist()
    predictions = []
    seen = set()
    for source_row, row in enumerate(rows):
        if row["run_id"] != JEV_RUN:
            continue
        for source_question, question in enumerate(row["questions"]):
            if question["question_id"] != "discount_days":
                continue
            case = row["case_id"]
            if case not in gold or case in seen:
                raise ValueError("Unexpected/duplicate Jev input")
            expected = gold[case]
            qid, sid = identity(json.loads(question["question_json"])), identity(json.loads(question["state_json"]))
            if (qid, sid, question["node_id"]) != (expected["question_identity"], expected["state_identity"], expected["node_id"]):
                raise ValueError("Jev input differs from frozen gold")
            seen.add(case)
            try:
                answer = json.loads(question["answer_json"]) if question["answer_json"] is not None else None
            except json.JSONDecodeError:
                answer = None
            choice = answer if isinstance(answer, str) else str(answer) if type(answer) is int else answer.get("choice") if isinstance(answer, dict) else None
            pairs = question["probabilities"]
            probabilities = {p["option"]: p["probability"] for p in pairs} if pairs is not None else None
            if pairs is not None and len(probabilities) != len(pairs):
                probabilities = None  # Preserve raw duplicate entries, mark the map invalid.
            predictions.append({"case_id": case, "question_id": "discount_days", "node_id": question["node_id"],
                                "question_identity": qid, "state_identity": sid, "model_id": "jev-1.13.0",
                                "checkpoint": JEV_RUN, "source_status": question["status"],
                                "status": "ok" if question["status"] in ("answered", "ok") else question["status"], "choice": choice,
                                "probabilities": probabilities, "answer_json_raw": question["answer_json"],
                                "probabilities_raw": pairs, "provider_confidence": question["confidence"],
                                "confidence_method": question["confidence_method"],
                                "source_row_index": source_row, "source_question_index": source_question})
    freeze_predictions(root, output, predictions, gold)


def freeze_predictions(root, output, predictions, gold):
    seen = set()
    for row in predictions:
        if row["case_id"] not in gold or row["case_id"] in seen:
            raise ValueError("Unexpected/duplicate frozen prediction")
        expected = gold[row["case_id"]]
        if any(row[key] != expected[key] for key in ("question_id", "node_id", "question_identity", "state_identity")):
            raise ValueError("Reused extraction input differs from frozen gold")
        seen.add(row["case_id"])
    predictions.sort(key=lambda row: row["case_id"])
    with output.open("x") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    output.chmod(0o600)
    statuses = Counter(validated(row)[0] for row in predictions)
    statuses["missing"] += len(gold) - len(seen)
    write_json(output.with_suffix(".manifest.json"), {
        "sha256": sha(output), "source_sha256": SOURCE_HASH, "run_id": JEV_RUN,
        "preregistration_sha256": sha(root / "comparison" / "preregistration.json"),
        "recorded_rows": len(predictions), "expected_rows": len(gold), "status": dict(statuses),
        "precision": "Original Parquet doubles and raw JSON retained; no confidence substitution",
    })
    print(json.dumps({"prediction_sha256": sha(output), "recorded_rows": len(predictions), "status": dict(statuses)}))


def binomial_cdf(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def cdf_inverse(k, n, target):
    low, high = 0., 1.
    for _ in range(70):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def error_bounds(errors, n):
    return {"two_sided_95": [0. if errors == 0 else cdf_inverse(errors - 1, n, .975),
                              1. if errors == n else cdf_inverse(errors, n, .025)],
            "one_sided_95_upper": 1. if errors == n else cdf_inverse(errors, n, .05)}


def metrics(gold, predictions):
    by_case = {row["case_id"]: row for row in predictions}
    if len(by_case) != len(predictions) or set(by_case) - {row["case_id"] for row in gold}:
        raise ValueError("Prediction join has duplicate/extra cases")
    confusion = {label: {prediction: 0 for prediction in (*OPTIONS, "invalid")} for label in OPTIONS}
    status, correct, nll, brier, calibration = Counter(), [], [], [], []
    argmax_disagreements = 0
    for row in gold:
        prediction = by_case.get(row["case_id"])
        if prediction is not None and (prediction.get("question_id"), prediction.get("node_id"), prediction.get("question_identity"), prediction.get("state_identity")) != (row["question_id"], row["node_id"], row["question_identity"], row["state_identity"]):
            raise ValueError("Prediction input identity mismatch")
        state, probabilities = validated(prediction)
        status[state] += 1
        choice = prediction["choice"] if state == "ok" else "invalid"
        confusion[row["gold"]][choice] += 1
        correct.append(choice == row["gold"])
        if state == "ok":
            argmax_disagreements += probabilities[choice] != max(probabilities.values())
            nll.append(-math.log(max(1e-15, probabilities[row["gold"]])))
            brier.append(sum((probabilities[k] - (k == row["gold"])) ** 2 for k in OPTIONS))
            calibration.append((probabilities[choice], choice == row["gold"]))
    classes = {}
    for label in OPTIONS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[k][label] for k in OPTIONS)
        precision, recall = tp / predicted if predicted else 0., tp / support if support else None
        classes[label] = {"support": support, "precision": precision, "recall": recall,
                          "f1": 2 * tp / (support + predicted) if support + predicted else 0.}
    observed = [value for value in classes.values() if value["support"]]
    mean = lambda values: sum(values) / len(values) if values else None
    bins = []
    for index in range(10):
        subset = [(p, c) for p, c in calibration if min(9, int(p * 10)) == index]
        bins.append({"count": len(subset), "mean_probability": mean([p for p, c in subset]),
                     "accuracy": mean([c for p, c in subset])})
    count, served = len(gold), status["ok"]
    result = {"count": count, "served": served, "coverage": served / count, "status": dict(status),
              "correct": sum(correct), "errors": count - sum(correct), "accuracy": mean(correct),
              "served_accuracy": sum(correct) / served if served else None, "confusion": confusion,
              "per_class": classes, "observed_class_macro_precision": mean([v["precision"] for v in observed]),
              "observed_class_macro_recall": mean([v["recall"] for v in observed]),
              "observed_class_macro_f1": mean([v["f1"] for v in observed]),
              "six_class_macro_f1_zero_support_zero": mean([v["f1"] for v in classes.values()]),
              "nll": mean(nll), "brier": mean(brier), "reliability": bins,
              "ece": sum(b["count"] * abs(b["accuracy"] - b["mean_probability"]) for b in bins if b["count"]) / served if served else None,
              "choice_argmax_disagreements": argmax_disagreements,
              "error_bounds": error_bounds(count - sum(correct), count)}
    return result, correct


def paired(left, right, draws=BOOTSTRAPS, seed=SEED):
    if len(left) != len(right) or not left:
        raise ValueError("Invalid paired inputs")
    differences = [int(a) - int(b) for a, b in zip(left, right)]
    improved, regressed = differences.count(1), differences.count(-1)
    discordant = improved + regressed
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(draws))
    return {"improved": improved, "regressed": regressed, "both_correct": sum(a and b for a, b in zip(left, right)),
            "both_wrong": sum(not a and not b for a, b in zip(left, right)),
            "accuracy_difference": sum(differences) / len(differences),
            "mcnemar_exact_two_sided_p": min(1., 2 * binomial_cdf(min(improved, regressed), discordant, .5)) if discordant else 1.,
            "case_bootstrap_95": [estimates[int(draws * .025)], estimates[min(draws - 1, int(draws * .975))]],
            "draws": draws, "seed": seed, "scope": "Within one narrow task family; not unseen-domain C"}


def score(root, clef_path):
    verify_contract(root)
    gold = frozen_gold(root)
    directory = root / "comparison"
    if (directory / "report.json").exists():
        raise FileExistsError("Comparison report already frozen")
    jev_path = directory / "jev-predictions.jsonl"
    if sha(jev_path) != json.loads(jev_path.with_suffix(".manifest.json").read_text())["sha256"]:
        raise ValueError("Frozen Jev predictions changed")
    clef = read_rows(clef_path)
    if any(row.get("model_id") != "clef_flash_mlx8_torch_bf16" or row.get("checkpoint") != CLEF_CHECKPOINT
           or row.get("inference_contract") != "invoice-recorded-payload-v1" for row in clef):
        raise ValueError("Clef weights/inference identity mismatch")
    source_ids = {row.get("predictor_source_sha256") for row in clef}
    if len(source_ids) != 1 or any(not isinstance(value, str) or len(value) != 64 for value in source_ids):
        raise ValueError("Missing/mixed Clef predictor identity")
    control = [{**{key: row[key] for key in ("case_id", "question_id", "node_id", "question_identity", "state_identity")},
                "status": "ok", "choice": "none", "probabilities": {key: float(key == "none") for key in OPTIONS}} for row in gold]
    results = {name: metrics(gold, values) for name, values in
               (("jev", read_rows(jev_path)), ("clef", clef), ("always_none", control))}
    report = {"preregistration_sha256": sha(directory / "preregistration.json"), "gold_sha256": GOLD_HASHES["gold.jsonl"],
              "jev_predictions_sha256": sha(jev_path), "clef_predictions_sha256": sha(clef_path),
              "clef_weights_checkpoint_reference": CLEF_CHECKPOINT, "clef_predictor_source_sha256": next(iter(source_ids)),
              "metrics": {name: values[0] for name, values in results.items()},
              "paired": {"clef_minus_jev": paired(results["clef"][1], results["jev"][1]),
                         "jev_minus_always_none": paired(results["jev"][1], results["always_none"][1]),
                         "clef_minus_always_none": paired(results["clef"][1], results["always_none"][1])},
              "scope": "150 cases, one canonical discount-term family; no A/B/C acceptance or latency superiority claim"}
    write_json(directory / "report.json", report)
    print(json.dumps({"report_sha256": sha(directory / "report.json"), "scope": report["scope"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "extract", "score"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clef", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        if args.clef is None:
            parser.error("--clef is required")
        score(args.root, args.clef)
    else:
        globals()[args.command](args.root)
