"""Focused source-reference, admission and frozen split checks; no model/data download."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("tool_selection_builder", Path(__file__).parents[1] / "training/tool-selection/build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def item(identity, names):
    return {"id": identity, "question": [[{"role": "user", "content": "fixture"}]],
            "function": [{"name": name, "description": name, "parameters": {}} for name in names]}


class ToolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = {"multiple": [item("m1", ["a.tool", "b.tool"]), item("m2", ["a.tool", "c.tool"])],
                     "irrelevance": [item("i1", ["d.tool"])]}

    def test_shared_schema_and_variants_preserve_split(self):
        origins = builder.origins(self.data)
        self.assertEqual(origins[0]["family_id"], origins[1]["family_id"])
        self.assertEqual(origins[0]["split"], origins[1]["split"])
        cases, labels = builder.prepare(self.data, origins, {"m1": [{"a.tool": {}}], "m2": [{"c.tool": {}}]}, "fixture rule")
        self.assertEqual(len(cases), 7)
        self.assertEqual({r["split"] for r in cases if r["source_id"] == "m1"}, {origins[0]["split"]})
        by_id = {r["sample_id"]: r for r in labels}
        self.assertEqual(by_id["m1/all_allowed"]["target_tool_names"], ["a.tool"])
        self.assertEqual(by_id["m1/deny_first_lexicographic_candidate"]["target_tool_names"], [])
        self.assertEqual(by_id["m2/deny_first_lexicographic_candidate"]["target_tool_names"], ["c.tool"])
        self.assertEqual(by_id["i1/all_allowed"]["target_tool_names"], [])
        self.assertEqual(by_id["m1/all_allowed"]["label_source"], "benchmark_reference")
        self.assertEqual(by_id["m1/deny_all"]["label_source"], "derived_policy_oracle")
        self.assertNotIn("target_tool_names", cases[0])
        self.assertNotIn("source_category", cases[0]["request"]["state"])

    def test_invalid_reference_is_not_silent_no_call(self):
        for bad in (None, [], [{}], [{"a": {}, "b": {}}]):
            with self.assertRaises(ValueError):
                builder.reference_names(bad)
        with self.assertRaises(KeyError):
            builder.prepare(self.data, builder.origins(self.data), {}, "fixture rule")
        with self.assertRaises(ValueError):
            builder.prepare(self.data, builder.origins(self.data), {"m1": [{"outside": {}}], "m2": [{"c.tool": {}}]}, "fixture rule")

    def test_sanitized_collision_and_duplicate_origin_rejected(self):
        for rows in ([item("x", ["a.b", "a_b"])], [item("x", ["a"]), item("x", ["b"])]):
            with self.assertRaises(ValueError):
                builder.origins({"multiple": rows, "irrelevance": []})

    def test_runtime_wire_and_isolation_check(self):
        cases, _ = builder.prepare(self.data, builder.origins(self.data), {"m1": [{"a.tool": {}}], "m2": [{"c.tool": {}}]}, "fixture rule")
        wires = builder.common_wires(cases)
        self.assertEqual(wires[0]["wire"]["questions"]["tool_a_tool"]["type"], "choice")
        self.assertEqual(set(wires[0]["wire"]["questions"]["tool_a_tool"]["criteria"]), {"yes", "no"})
        self.assertEqual(len(wires[0]["wire"]["questions"]["tool_a_tool"]["instructions"]), 2)
        cases[1]["split"] = "test" if cases[0]["split"] != "test" else "train"
        with self.assertRaises(builder.subprocess.CalledProcessError):
            builder.common_wires(cases)

    def test_existing_output_refused_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "OUTPUT_EXISTS"):
                builder.build(Path("absent"), Path(directory))


if __name__ == "__main__":
    unittest.main()
