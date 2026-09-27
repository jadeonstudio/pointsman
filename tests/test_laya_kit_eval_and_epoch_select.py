"""Unit tests for two train_from_export.py fixes/features (2026-09-23):

1. evaluate_checkpoint()'s gold-label lookup used bracket access on a "label"
   key (`g_ans["label"]`) that the real pointsman export never writes -- the export
   only ever writes `gold[qid] = {"probabilities": {...}}` (see
   src/training/dataset.mjs exportDataset() and
   training/laya-kit/check_export.py check_gold_probabilities()). A real
   local run's post-training eval step failed with
   "[eval] evaluation step failed or was skipped: 'label'" because of this.
   resolve_gold_label() derives the gold label by argmax over probabilities
   (the same way build_training_item() derives `label = target.index(max(target))`),
   and evaluate_checkpoint() is fixed to use it.

2. Per-epoch calibration-agreement checkpoint selection: compute_calib_agreement()
   is the pure aggregation helper (also duplicated inside the embedded
   TRAIN_DDP_SCRIPT, parity-tested here the same way
   test_laya_kit_local_mode.py checks oom_suggestion()/format_oom_message()
   parity) that epoch_end_fn (inside TRAIN_DDP_SCRIPT) uses to log
   "[select] epoch N calib_agreement ..." and decide whether to keep this
   epoch's state_dict as the new best.

Standard library only -- no torch/laya import required to run this file
(matching tests/test_laya_kit_local_mode.py's pattern).
"""
import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kit = load_module('train_from_export_for_eval_and_select_tests', 'training/laya-kit/train_from_export.py')


# ---------------------------------------------------------------------------
# 1. resolve_gold_label() -- fixes the KeyError('label') eval crash.
# ---------------------------------------------------------------------------
class ResolveGoldLabel(unittest.TestCase):
    def test_choice_argmaxes_probabilities_real_export_shape(self):
        # Real export gold shape: only "probabilities", never "label".
        g_ans = {"probabilities": {"a": 0.1, "b": 0.7, "c": 0.2}}
        self.assertEqual(kit.resolve_gold_label("choice", g_ans, keys=["a", "b", "c"]), "b")

    def test_choice_tie_picks_first_key_matching_build_training_item(self):
        g_ans = {"probabilities": {"a": 0.5, "b": 0.5}}
        self.assertEqual(kit.resolve_gold_label("choice", g_ans, keys=["a", "b"]), "a")

    def test_choice_missing_key_defaults_to_zero_probability(self):
        g_ans = {"probabilities": {"a": 0.9}}
        self.assertEqual(kit.resolve_gold_label("choice", g_ans, keys=["a", "b"]), "a")

    def test_noul_true_wins(self):
        g_ans = {"probabilities": {"false": 0.3, "true": 0.7}}
        self.assertEqual(kit.resolve_gold_label("noul", g_ans), "true")

    def test_noul_false_wins(self):
        g_ans = {"probabilities": {"false": 0.6, "true": 0.4}}
        self.assertEqual(kit.resolve_gold_label("noul", g_ans), "false")

    def test_noul_tie_picks_false_matching_index_zero(self):
        g_ans = {"probabilities": {"false": 0.5, "true": 0.5}}
        self.assertEqual(kit.resolve_gold_label("noul", g_ans), "false")

    def test_score_argmaxes_probabilities_over_levels(self):
        g_ans = {"probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
        self.assertEqual(kit.resolve_gold_label("score", g_ans, n_levels=3), 1)

    def test_explicit_label_key_wins_for_choice_backward_compat(self):
        g_ans = {"label": "z", "probabilities": {"a": 0.9, "b": 0.1}}
        self.assertEqual(kit.resolve_gold_label("choice", g_ans, keys=["a", "b"]), "z")

    def test_explicit_label_key_wins_for_noul_backward_compat(self):
        g_ans = {"label": "TRUE", "probabilities": {"false": 0.9, "true": 0.1}}
        self.assertEqual(kit.resolve_gold_label("noul", g_ans), "true")

    def test_explicit_label_key_wins_for_score_backward_compat(self):
        g_ans = {"label": 2, "probabilities": {"0": 0.9, "1": 0.05, "2": 0.05}}
        self.assertEqual(kit.resolve_gold_label("score", g_ans, n_levels=3), 2)

    def test_unknown_qtype_raises_kit_error(self):
        with self.assertRaises(kit.KitError):
            kit.resolve_gold_label("bogus", {"probabilities": {}})


# ---------------------------------------------------------------------------
# 1b. evaluate_checkpoint() end-to-end on a synthetic export.test.jsonl row in
# the REAL export row shape (three JSON-string columns; gold entries only
# ever carry "probabilities" -- see check_export.py check_row_columns() /
# check_gold_probabilities()). Before the fix this raised KeyError('label')
# on the choice/noul rows.
# ---------------------------------------------------------------------------
class FakeAgent:
    """Stands in for laya.agent.Agent -- predict() returns the exact answer
    shape evaluate_checkpoint() indexes (p_ans["choice"/"noul"/"score"] plus
    "probabilities"), always agreeing with the row's gold argmax so a passing
    test proves "no crash + correct wiring", not "the real model is accurate"."""

    def __init__(self, output_dir, device="cuda"):
        self.output_dir = output_dir
        self.device = device

    def predict(self, state, questions):
        answers = {}
        for qid, qdef in questions.items():
            t = qdef["type"]
            if t == "choice":
                keys = list(qdef["criteria"].keys())
                answers[qid] = {"choice": keys[0], "probabilities": {k: (1.0 if k == keys[0] else 0.0) for k in keys}}
            elif t == "noul":
                answers[qid] = {"noul": 0.9}
            elif t == "score":
                answers[qid] = {"score": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}
        return {"answers": answers}


class FakeLayaModule:
    Agent = FakeAgent

    class common:
        @staticmethod
        def ece_score(conf, correct, bins=15):
            return 0.0


def write_jsonl(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')


class EvaluateCheckpointRealExportShape(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.export_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_integer_valued_gold_probabilities_do_not_crash(self):
        # Real export gold: objective/human-labeled targets come from
        # targetDistribution() in src/training/schema.mjs, a one-hot
        # {key: 1 or 0} dict. JSON round-trips a whole number like `1` or `0`
        # (no decimal point) as a Python int, not a float, so
        # np.array([...]) over those values can infer an int64 dtype;
        # evaluate_checkpoint's in-place `/=` on such an array raised
        # "Cannot cast ufunc 'divide' output from dtype('float64') to
        # dtype('int64')" on a real smoke run (2026-09-23).
        rows = [
            {
                "state": json.dumps({"task": "t"}),
                "questions": json.dumps({"q1": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}),
                "gold": json.dumps({"q1": {"probabilities": {"a": 1, "b": 0}}}),  # ints, not floats
            },
            {
                "state": json.dumps({"task": "t"}),
                "questions": json.dumps({"q3": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]}}),
                "gold": json.dumps({"q3": {"probabilities": {"0": 0, "1": 1, "2": 0}}}),  # ints, not floats
            },
        ]
        write_jsonl(self.export_dir / "test.jsonl", rows)
        metrics = kit.evaluate_checkpoint(str(self.export_dir / "ckpt"), str(self.export_dir), FakeLayaModule(), device="cpu")
        self.assertEqual(metrics["n_cases"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_choice_noul_score_rows_do_not_raise_and_produce_metrics(self):
        rows = [
            {
                "state": json.dumps({"task": "t"}),
                "questions": json.dumps({"q1": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}),
                # Real export gold shape: ONLY "probabilities", never "label".
                "gold": json.dumps({"q1": {"probabilities": {"a": 1.0, "b": 0.0}}}),
            },
            {
                "state": json.dumps({"task": "t"}),
                "questions": json.dumps({"q2": {"type": "noul", "instructions": "yes/no"}}),
                "gold": json.dumps({"q2": {"probabilities": {"false": 0.05, "true": 0.95}}}),
            },
            {
                "state": json.dumps({"task": "t"}),
                "questions": json.dumps({"q3": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]}}),
                "gold": json.dumps({"q3": {"probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}}),
            },
        ]
        write_jsonl(self.export_dir / "test.jsonl", rows)

        metrics = kit.evaluate_checkpoint(str(self.export_dir / "ckpt"), str(self.export_dir), FakeLayaModule(), device="cpu")

        self.assertEqual(metrics["n_cases"], 3)
        self.assertIsNotNone(metrics["accuracy"])
        # FakeAgent always predicts the gold argmax -> perfect agreement.
        self.assertEqual(metrics["accuracy"], 1.0)


# ---------------------------------------------------------------------------
# 2. compute_calib_agreement() -- per-qtype argmax agreement used for
# best-epoch selection.
# ---------------------------------------------------------------------------
class ComputeCalibAgreement(unittest.TestCase):
    def test_all_correct_single_qtype(self):
        # QTYPES: choice=0, score=1, noul=2 (laya.common.QTYPES).
        preds = [(0, [0.1, 0.9], [0.0, 1.0]), (0, [0.8, 0.2], [1.0, 0.0])]
        out = kit.compute_calib_agreement(preds)
        self.assertEqual(out["choice"], 1.0)
        self.assertIsNone(out["score"])
        self.assertIsNone(out["noul"])
        self.assertEqual(out["mean"], 1.0)

    def test_mixed_qtypes_partial_agreement(self):
        preds = [
            (0, [0.1, 0.9], [0.0, 1.0]),   # choice correct
            (0, [0.9, 0.1], [0.0, 1.0]),   # choice wrong
            (1, [0.1, 0.2, 0.7], [0.0, 0.0, 1.0]),  # score correct
            (2, [0.6, 0.4], [1.0, 0.0]),   # noul correct
        ]
        out = kit.compute_calib_agreement(preds)
        self.assertAlmostEqual(out["choice"], 0.5)
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(out["noul"], 1.0)
        self.assertAlmostEqual(out["mean"], (0.5 + 1.0 + 1.0) / 3)

    def test_empty_input_returns_all_none(self):
        out = kit.compute_calib_agreement([])
        self.assertIsNone(out["choice"])
        self.assertIsNone(out["score"])
        self.assertIsNone(out["noul"])
        self.assertIsNone(out["mean"])

    def test_tie_picks_first_index_matching_argmax_convention(self):
        preds = [(0, [0.5, 0.5], [1.0, 0.0])]  # pred argmax=0 (first max), gold argmax=0 -> correct
        out = kit.compute_calib_agreement(preds)
        self.assertEqual(out["choice"], 1.0)


# ---------------------------------------------------------------------------
# 2b. Structural checks on the embedded TRAIN_DDP_SCRIPT for epoch selection
# wiring (mirrors test_laya_kit_local_mode.py's TrainDdpScriptLocalModeStructure
# static-check pattern -- running the real thing needs torch+laya+a real
# checkpoint, which this test suite intentionally avoids).
# ---------------------------------------------------------------------------
class TrainDdpScriptEpochSelectionStructure(unittest.TestCase):
    @staticmethod
    def _extract_function_body(source, func_name):
        marker = f'\ndef {func_name}('
        start = source.index(marker)
        rest = source[start + 1:]
        end_rel = rest.find('\ndef ')
        return rest if end_rel == -1 else rest[:end_rel]

    def test_compute_calib_agreement_duplicated_in_script_matches_outer_module(self):
        # Parity requirement (TRAIN_DDP_SCRIPT must stay a standalone file, so
        # it cannot import train_from_export.py): both copies must agree on
        # the same synthetic input.
        self.assertIn('def compute_calib_agreement(', kit.TRAIN_DDP_SCRIPT)
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'compute_calib_agreement')
        ns = {}
        exec(compile(body, '<train_ddp_script_fragment:compute_calib_agreement>', 'exec'), ns)
        script_fn = ns['compute_calib_agreement']
        preds = [(0, [0.1, 0.9], [0.0, 1.0]), (1, [0.7, 0.2, 0.1], [1.0, 0.0, 0.0])]
        self.assertEqual(script_fn(preds), kit.compute_calib_agreement(preds))

    def test_epoch_end_fn_logs_select_line_with_expected_fields(self):
        self.assertIn('[select] epoch', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('calib_agreement', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('choice=', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('score=', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('noul=', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('mean=', kit.TRAIN_DDP_SCRIPT)

    def test_both_mains_pass_epoch_end_fn_to_run_training_loop(self):
        for func in ('main_ddp', 'main_local'):
            marker = f'\ndef {func}('
            start = kit.TRAIN_DDP_SCRIPT.index(marker)
            rest = kit.TRAIN_DDP_SCRIPT[start + 1:]
            end_rel = rest.find('\ndef ')
            body = rest if end_rel == -1 else rest[:end_rel]
            self.assertIn('epoch_end_fn=', body, f'{func} does not pass epoch_end_fn to run_training_loop')

    def test_run_training_loop_signature_accepts_epoch_end_fn(self):
        sig_line = re.search(r'def run_training_loop\(.*?\):', kit.TRAIN_DDP_SCRIPT, re.S).group(0)
        self.assertIn('epoch_end_fn', sig_line)

    def test_ddp_epoch_end_fn_gates_eval_on_rank_zero_and_barriers(self):
        start = kit.TRAIN_DDP_SCRIPT.index('\ndef main_ddp(')
        rest = kit.TRAIN_DDP_SCRIPT[start + 1:]
        end_rel = rest.find('\ndef ')
        body = rest if end_rel == -1 else rest[:end_rel]
        self.assertIn('rank == 0', body)
        self.assertIn('dist.barrier()', body)

    def test_best_state_dict_kept_on_cpu(self):
        self.assertRegex(kit.TRAIN_DDP_SCRIPT, r'\.to\(\s*["\']cpu["\']')

    def test_best_state_dict_loaded_before_finalize_and_save(self):
        for func in ('main_ddp', 'main_local'):
            marker = f'\ndef {func}('
            start = kit.TRAIN_DDP_SCRIPT.index(marker)
            rest = kit.TRAIN_DDP_SCRIPT[start + 1:]
            end_rel = rest.find('\ndef ')
            body = rest if end_rel == -1 else rest[:end_rel]
            # rfind, not find: the FIRST load_state_dict in each body is the base
            # checkpoint weight load; the best-epoch reload (the one this test cares
            # about) is the LAST one, right before finalize_and_save().
            load_idx = body.rfind('load_state_dict')
            # Search for the actual call (finalize_and_save(model, ...)), not a prose
            # mention in a comment/docstring like "...finalize_and_save() with...".
            finalize_idx = body.find('finalize_and_save(model')
            self.assertGreater(load_idx, -1, f'{func} never reloads a state_dict before finalize_and_save')
            self.assertGreater(finalize_idx, -1)
            self.assertLess(load_idx, finalize_idx, f'{func} loads the best state_dict AFTER finalize_and_save')

    def test_epoch_selection_metadata_written(self):
        self.assertIn('epoch_selection.json', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('selected_epoch', kit.TRAIN_DDP_SCRIPT)


# ---------------------------------------------------------------------------
# 2c. Outer CLI/metadata wiring for --select-best-epoch.
# ---------------------------------------------------------------------------
class SelectBestEpochCli(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'training/laya-kit/train_from_export.py').read_text(encoding='utf-8')

    def test_select_best_epoch_flag_present_and_defaults_on(self):
        self.assertIn('"--select-best-epoch"', self.source)
        self.assertIn('BooleanOptionalAction', self.source)
        self.assertIn('default=True', self.source)

    def test_outer_main_folds_epoch_selection_metadata_into_training_metadata(self):
        self.assertIn('epoch_selection.json', self.source)
        self.assertIn('epoch_selection', self.source)


if __name__ == '__main__':
    unittest.main()
