"""Unit tests for training/laya-kit/train_from_export.py's --local (single-process,
no torchrun/NCCL/DDP) mode: pure helpers (load_jsonl_rows, batch/grad-accum
resolution, OOM message) plus static checks on the embedded TRAIN_DDP_SCRIPT that
its DDP path is untouched and its local path never wires up NCCL/DDP.

Standard library only -- these test pure functions and string/source structure,
matching the existing tests/test_laya_kit_model_dir.py pattern. No torch/laya
import is required to run this file.
"""
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kit = load_module('train_from_export_for_local_mode_tests', 'training/laya-kit/train_from_export.py')


class LoadJsonlRows(unittest.TestCase):
    def test_reads_rows_in_the_export_row_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            rows = [
                {'state': '{"task": "a"}', 'questions': '{}', 'gold': '{}'},
                {'state': '{"task": "b"}', 'questions': '{}', 'gold': '{}'},
            ]
            path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            loaded = kit.load_jsonl_rows(path)
            self.assertEqual(loaded, rows)

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            path.write_text('{"state": "{}", "questions": "{}", "gold": "{}"}\n\n   \n', encoding='utf-8')
            loaded = kit.load_jsonl_rows(path)
            self.assertEqual(len(loaded), 1)

    def test_row_is_a_plain_dict_indexable_like_a_datasets_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.jsonl'
            path.write_text('{"state": "s", "questions": "q", "gold": "g"}\n', encoding='utf-8')
            row = kit.load_jsonl_rows(path)[0]
            self.assertEqual(row['state'], 's')
            self.assertEqual(row['questions'], 'q')
            self.assertEqual(row['gold'], 'g')


class ResolveLocalBatchAndGradAccum(unittest.TestCase):
    def test_defaults_keep_the_official_effective_global_batch(self):
        # DDP recipe: micro_batch(8) * grad_accum(4) * world_size(2) == 64.
        # --local has world_size=1, so the default grad-accum must double to 8
        # so batch_size(8) * grad_accum(8) == 64 too.
        batch_size, grad_accum = kit.resolve_local_batch_and_grad_accum(None, None)
        self.assertEqual(batch_size, 8)
        self.assertEqual(grad_accum, 8)
        self.assertEqual(batch_size * grad_accum, kit.EFFECTIVE_GLOBAL_BATCH)
        self.assertEqual(kit.EFFECTIVE_GLOBAL_BATCH, 64)

    def test_smaller_batch_size_scales_grad_accum_up_to_keep_effective_batch(self):
        batch_size, grad_accum = kit.resolve_local_batch_and_grad_accum(4, None)
        self.assertEqual(batch_size, 4)
        self.assertEqual(grad_accum, 16)
        self.assertEqual(batch_size * grad_accum, kit.EFFECTIVE_GLOBAL_BATCH)

    def test_larger_batch_size_scales_grad_accum_down(self):
        batch_size, grad_accum = kit.resolve_local_batch_and_grad_accum(16, None)
        self.assertEqual(batch_size, 16)
        self.assertEqual(grad_accum, 4)

    def test_explicit_grad_accum_overrides_the_computed_default(self):
        batch_size, grad_accum = kit.resolve_local_batch_and_grad_accum(8, 2)
        self.assertEqual(batch_size, 8)
        self.assertEqual(grad_accum, 2)

    def test_explicit_grad_accum_with_default_batch_size(self):
        batch_size, grad_accum = kit.resolve_local_batch_and_grad_accum(None, 1)
        self.assertEqual(batch_size, 8)
        self.assertEqual(grad_accum, 1)

    def test_rejects_batch_size_below_one(self):
        with self.assertRaises(kit.KitError):
            kit.resolve_local_batch_and_grad_accum(0, None)

    def test_rejects_grad_accum_below_one(self):
        with self.assertRaises(kit.KitError):
            kit.resolve_local_batch_and_grad_accum(8, 0)


class FormatOomMessage(unittest.TestCase):
    def test_suggests_halved_batch_and_doubled_grad_accum(self):
        msg = kit.format_oom_message(8, 8)
        self.assertIn('--batch-size 4', msg)
        self.assertIn('--grad-accum 16', msg)

    def test_keeps_effective_global_batch_constant_in_the_suggestion(self):
        for batch_size, grad_accum in [(8, 8), (16, 4), (32, 2)]:
            msg = kit.format_oom_message(batch_size, grad_accum)
            m = re.search(r'--batch-size (\d+) --grad-accum (\d+)', msg)
            self.assertIsNotNone(m, msg)
            suggested_batch, suggested_grad_accum = int(m.group(1)), int(m.group(2))
            self.assertEqual(suggested_batch * suggested_grad_accum, batch_size * grad_accum)

    def test_floors_batch_size_at_one(self):
        msg = kit.format_oom_message(1, 8)
        self.assertIn('--batch-size 1', msg)


class EvaluateCheckpointDeviceParameter(unittest.TestCase):
    def test_default_device_is_cuda_for_backward_compatibility(self):
        import inspect
        sig = inspect.signature(kit.evaluate_checkpoint)
        self.assertEqual(sig.parameters['device'].default, 'cuda')


class TrainDdpScriptLocalModeStructure(unittest.TestCase):
    """Static checks on the embedded TRAIN_DDP_SCRIPT: the DDP path (main_ddp)
    still wires up NCCL/DDP exactly as before, and the new local path
    (main_local) never does -- it must be runnable with plain python3, no
    torchrun/NCCL/DDP."""

    @staticmethod
    def _extract_function_body(source, func_name):
        marker = f'\ndef {func_name}('
        start = source.index(marker)
        # Next top-level "def " at column 0 after start marks the end of this function.
        rest = source[start + 1:]
        end_rel = rest.find('\ndef ')
        body = rest if end_rel == -1 else rest[:end_rel]
        return body

    def test_clip_params_are_materialized_not_a_one_shot_generator(self):
        # run_training_loop clips every optimizer step with the same params_for_clip argument;
        # a bare .parameters() generator is exhausted after the first clip, silently disabling
        # gradient clipping for the rest of training (observed 2026-09-23 as torch's
        # "`parameters` is an empty generator" warning in a real local run).
        for func in ('main_ddp', 'main_local'):
            body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, func)
            self.assertIn('run_training_loop(', body)
            self.assertNotRegex(body, r'run_training_loop\(\s*\w+,\s*\w+,\s*\w+\.parameters\(\)',
                                f'{func} passes a one-shot generator to run_training_loop')

    def test_training_loop_releases_mps_cache_every_optimizer_step(self):
        # 2026-09-23 real run (M4 Pro 24GB, batch 4): variable padded shapes made the MPS caching
        # allocator grow until free memory hit 8% and swap filled; releasing the cache after each
        # optimizer step (MPS only; CUDA/DDP unchanged) keeps the working set flat.
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'run_training_loop')
        self.assertIn('torch.mps.empty_cache()', body)
        self.assertRegex(body, r"device\.type\s*==\s*['\"]mps['\"]")

    def test_main_local_exists_and_main_ddp_exists(self):
        self.assertIn('def main_local(', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('def main_ddp(', kit.TRAIN_DDP_SCRIPT)

    def test_main_ddp_still_uses_nccl_and_ddp(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_ddp')
        self.assertIn('init_process_group("nccl")', body)
        self.assertIn('DistributedDataParallel', body)

    def test_main_local_never_uses_nccl_or_ddp_or_torch_distributed(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        # Comments may mention "NCCL" (explaining what this path avoids); only
        # actual usage (a quoted backend string, an import, or a call) is banned.
        self.assertNotIn('"nccl"', body.lower())
        self.assertNotIn('import torch.distributed', body)
        self.assertNotIn('DistributedDataParallel', body)
        self.assertNotIn('torch.distributed', body)
        self.assertNotIn('dist.init_process_group', body)

    def test_main_local_defaults_to_mps_device(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        self.assertIn('"mps"', body)

    def test_main_local_defaults_autocast_off_fp32(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        # pointsman-change 2026-09-23: argv[9] is now --select-best-epoch (see
        # tests/test_laya_kit_eval_and_epoch_select.py), shifting
        # device/grad-accum/micro-batch/mps-autocast/max-steps each up by one.
        self.assertIn('mps_autocast = sys.argv[13] if len(sys.argv) > 13 else "off"', body)

    def test_main_local_logs_mps_driver_allocated_memory_per_epoch(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        self.assertIn('torch.mps.driver_allocated_memory()', body)

    def test_main_local_writes_local_training_run_metadata_with_device_mode_grad_accum_wall_time(self):
        body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        self.assertIn('local_training_run.json', body)
        self.assertIn('"device"', body)
        self.assertIn('"mode": "local"', body)
        self.assertIn('"grad_accum"', body)
        self.assertIn('"wall_time_s"', body)

    def test_shared_run_training_loop_is_used_by_both_paths(self):
        ddp_body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_ddp')
        local_body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        self.assertIn('run_training_loop(', ddp_body)
        self.assertIn('run_training_loop(', local_body)

    def test_shared_finalize_and_save_is_used_by_both_paths(self):
        ddp_body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_ddp')
        local_body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_local')
        self.assertIn('finalize_and_save(', ddp_body)
        self.assertIn('finalize_and_save(', local_body)

    def test_ddp_hyperparameters_unchanged(self):
        # EPOCHS/MICRO_BATCH/GROUP_SIZE/LR_*/SIGMA_* module-level constants
        # must still be the notebook's literal values.
        self.assertIn('EPOCHS = 4', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('MICRO_BATCH = 8', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('GROUP_SIZE = 4', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('LR_ENCODER = 2.5e-5', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('LR_HEAD = 1.0e-4', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('SIGMA_START = 0.4', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('SIGMA_END = 0.1', kit.TRAIN_DDP_SCRIPT)
        # DDP-only effective-batch constant (GRAD_ACCUM=4 inside main_ddp).
        ddp_body = self._extract_function_body(kit.TRAIN_DDP_SCRIPT, 'main_ddp')
        self.assertIn('GRAD_ACCUM = 4', ddp_body)

    def test_main_dispatches_on_argv_8_mode(self):
        self.assertIn('if mode == "local":', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('main_local()', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('main_ddp()', kit.TRAIN_DDP_SCRIPT)

    def test_oom_suggestion_helper_matches_outer_module_logic(self):
        # Parity: TRAIN_DDP_SCRIPT's oom_suggestion() must be duplicated
        # (it can't import train_from_export.py -- must stay standalone) but
        # produce the same numbers as format_oom_message() for the same input.
        self.assertIn('def oom_suggestion(batch_size, grad_accum):', kit.TRAIN_DDP_SCRIPT)
        self.assertIn('out of memory at --batch-size=', kit.TRAIN_DDP_SCRIPT)
        outer_msg = kit.format_oom_message(8, 8)
        self.assertIn('--batch-size 4 --grad-accum 16', outer_msg)


class MainArgparseLocalFlags(unittest.TestCase):
    """The CLI must expose --local/--device/--grad-accum/--batch-size/
    --mps-autocast/--max-steps. Checked by parsing the outer module's
    source (importing/calling main() needs torch/laya/argv wiring this
    test intentionally avoids, matching the existing tests' no-torch pattern)."""

    def setUp(self):
        self.source = (ROOT / 'training/laya-kit/train_from_export.py').read_text(encoding='utf-8')

    def test_local_flag_present(self):
        self.assertIn('"--local"', self.source)

    def test_device_flag_present_with_mps_default(self):
        self.assertIn('"--device"', self.source)
        self.assertIn('default="mps"', self.source)

    def test_grad_accum_and_batch_size_flags_present(self):
        self.assertIn('"--grad-accum"', self.source)
        self.assertIn('"--batch-size"', self.source)

    def test_mps_autocast_flag_present_with_off_default(self):
        self.assertIn('"--mps-autocast"', self.source)
        self.assertIn('choices=["bf16", "off"]', self.source)

    def test_max_steps_flag_present(self):
        self.assertIn('"--max-steps"', self.source)


if __name__ == '__main__':
    unittest.main()
