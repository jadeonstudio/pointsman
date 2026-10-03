import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("invoice_compare", Path(__file__).parents[1] / "training/public-invoice/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def case(identifier, label="none"):
    return {"case_id": identifier, "question_id": "discount_days", "node_id": "semantic",
            "question_identity": "question", "state_identity": "state", "gold": label}


def prediction(row, choice="none", probabilities=None):
    return {**{key: row[key] for key in ("case_id", "question_id", "node_id", "question_identity", "state_identity")},
            "status": "ok", "choice": choice,
            "probabilities": probabilities or {key: float(key == choice) for key in compare.OPTIONS}}


class InvoiceComparisonTests(unittest.TestCase):
    def test_distribution_rejects_wrong_keys_nonfinite_and_excess_mass(self):
        row = case("a")
        for probabilities in ({"none": 1.}, {**prediction(row)["probabilities"], "none": math.nan},
                              {key: 0. for key in compare.OPTIONS},
                              {key: .2 for key in compare.OPTIONS}):
            self.assertEqual(compare.validated(prediction(row, probabilities=probabilities))[0], "malformed")
        rounded = {key: .01 for key in compare.OPTIONS}
        rounded["none"] = .94
        status, normalized = compare.validated(prediction(row, probabilities=rounded))
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(sum(normalized.values()), 1.)

    def test_actual_choice_is_not_replaced_with_argmax_or_provider_confidence(self):
        row = case("a")
        raw = {key: 0. for key in compare.OPTIONS}
        raw.update({"10": .8, "none": .2})
        value = prediction(row, probabilities=raw)
        value["provider_confidence"] = .99
        result, correct = compare.metrics([row], [value])
        self.assertEqual(correct, [True])
        self.assertEqual(result["choice_argmax_disagreements"], 1)
        self.assertAlmostEqual(result["ece"], .8)
        self.assertEqual(value["probabilities"], raw)

    def test_missing_invalid_are_full_envelope_failures(self):
        a, b, c = case("a"), case("b", "10"), case("c")
        bad = prediction(b)
        bad["status"] = "not_answered"
        result, correct = compare.metrics([a, b, c], [prediction(a), bad])
        self.assertEqual(correct, [True, False, False])
        self.assertEqual(result["served"], 1)
        self.assertEqual(result["status"], {"ok": 1, "not_answered": 1, "missing": 1})
        self.assertEqual(result["confusion"]["none"]["invalid"], 1)

    def test_duplicate_and_changed_input_rejected(self):
        row = case("a")
        with self.assertRaises(ValueError):
            compare.metrics([row], [prediction(row), prediction(row)])
        changed = prediction(row)
        changed["state_identity"] = "changed"
        with self.assertRaises(ValueError):
            compare.metrics([row], [changed])

    def test_exact_error_bounds_include_zero_error_uncertainty(self):
        bounds = compare.error_bounds(0, 150)
        self.assertEqual(bounds["two_sided_95"][0], 0.)
        self.assertAlmostEqual(bounds["one_sided_95_upper"], 1 - .05 ** (1 / 150), places=12)
        self.assertGreater(bounds["one_sided_95_upper"], .01)

    def test_paired_exact_test_and_bootstrap_reproducibility(self):
        result = compare.paired([True] * 6, [False] * 6, draws=100)
        self.assertEqual(result["improved"], 6)
        self.assertEqual(result["regressed"], 0)
        self.assertAlmostEqual(result["mcnemar_exact_two_sided_p"], 2 / 64)
        self.assertEqual(result["case_bootstrap_95"], [1., 1.])
        self.assertEqual(result, compare.paired([True] * 6, [False] * 6, draws=100))


if __name__ == "__main__":
    unittest.main()
