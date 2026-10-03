"""Behavior checks for independent discount-term gold and full-input matching."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("invoice_oracle", Path(__file__).parents[1] / "training/public-invoice/oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class OracleTests(unittest.TestCase):
    def test_canonical_terms_and_nonstandard_window(self):
        for terms in ["Due on receipt", "Net 15", "Net 45", "Net 60", "Net 90", "Net 30", "  NET 1  "]:
            self.assertEqual(oracle.classify(terms), "none")
        self.assertEqual(oracle.classify("2/10 Net 30"), "10")
        self.assertEqual(oracle.classify("0.25%/7 Net 30"), "7")
        self.assertEqual(oracle.classify("2/12 Net 30"), "none")

    def test_unsupported_conflicts_missing_and_invalid_numbers(self):
        for terms in [None, "", "Net 0", "2/0 Net 30", "0/10 Net 30", "101/10 Net 30", "2/10 Net 0", "Net 30 or 2/10 Net 30", "2/10 Net 30; 3/15 Net 45", "pay promptly"]:
            with self.subTest(terms=terms), self.assertRaises(ValueError):
                oracle.classify(terms)

    def fixture(self):
        packet = {"case_id": "case", "invoice": {"fields": {"payment_terms": "2/10 Net 30"}}}
        state = {**packet, "exact_facts": {"payment_terms": "2/10 Net 30"}, "vendor": {"id": "v"}}
        row = {"case_id": "case", "question_instance_id": "id", "node_id": "semantic", "question_id": "discount_days", "kind": "choice", "question_json": json.dumps(oracle.question("2/10 Net 30")), "state_json": json.dumps(state)}
        return row, packet

    def test_candidates_and_source_corroboration_are_required(self):
        row, packet = self.fixture()
        altered = copy.deepcopy(row)
        q = json.loads(altered["question_json"])
        q["criteria"]["10"] = "other rule"
        altered["question_json"] = json.dumps(q)
        with self.assertRaisesRegex(ValueError, "question_identity"):
            oracle.validated_input(altered, packet)
        state = json.loads(row["state_json"])
        state["exact_facts"]["payment_terms"] = "Net 30"
        row["state_json"] = json.dumps(state)
        with self.assertRaisesRegex(ValueError, "payment_terms_mismatch"):
            oracle.validated_input(row, packet)

    def test_same_case_is_insufficient_and_full_coverage_required(self):
        row, packet = self.fixture()
        gold = [oracle.validated_input(row, packet)]
        self.assertEqual(oracle.match_inputs(gold, [row])["matched"], 1)
        for field, error in [("state_json", "recorded_state_mismatch"), ("question_json", "recorded_question_mismatch")]:
            changed = copy.deepcopy(row)
            data = json.loads(changed[field])
            data["unexpected"] = True
            changed[field] = json.dumps(data)
            with self.assertRaisesRegex(ValueError, error):
                oracle.match_inputs(gold, [changed])
        with self.assertRaisesRegex(ValueError, "missing_recorded_inputs"):
            oracle.match_inputs(gold, [])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            oracle.match_inputs(gold, [row, row])


if __name__ == "__main__":
    unittest.main()
