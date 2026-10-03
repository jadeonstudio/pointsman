import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("clef_local", Path(__file__).parents[1] / "training/clef-local/evaluate.py")
clef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clef)


class ClefAdmissionTest(unittest.TestCase):
    def test_precision_artifacts_are_separate(self):
        root = Path("/private/experiment")
        self.assertEqual(clef.paths(root, 4), (root / "upstream", root / "mlx4", root / "artifacts"))
        self.assertEqual(clef.paths(root, 8), (root / "upstream", root / "mlx8", root / "artifacts8"))

    def test_full_record_boundary(self):
        class Encoder:
            @staticmethod
            def encode_record(tokenizer, request, max_length):
                self.assertGreater(max_length, clef.MAX_TOKENS)
                return SimpleNamespace(input_ids=tuple(range(request["length"])))
        self.assertEqual(len(clef.encode_complete(None, {"length": 2048}, Encoder).input_ids), 2048)
        with self.assertRaisesRegex(ValueError, "INPUT_TRUNCATED"):
            clef.encode_complete(None, {"length": 2049}, Encoder)

    def test_wire_preserves_state_question_and_option_order(self):
        sample = {"state": {"language": "ko", "text": "fixture"}, "question_id": "team", "question": {
            "type": "choice", "instructions": "Choose the owner", "criteria": {"z": "last", "a": "first"}}}
        wire = clef.wire(sample)
        self.assertIs(wire["state"], sample["state"])
        self.assertEqual(list(wire["questions"]["team"]["criteria"]), ["z", "a"])
        self.assertEqual(wire["questions"]["team"]["instructions"], [clef.WRAPPER, "Choose the owner"])
        self.assertEqual(sample["question"]["instructions"], "Choose the owner")


if __name__ == "__main__":
    unittest.main()
