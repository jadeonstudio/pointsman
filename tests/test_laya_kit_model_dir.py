"""Unit tests for training/laya-kit/train_from_export.py's pure helpers around
--model-subdir resolution, model_name derivation, and the max_len/head_max_len
admission-vs-training consistency fix.

Standard library only -- these test pure functions that train_from_export.py's module
top-level does not require torch/laya/transformers for (those are imported lazily inside
main()/TRAIN_DDP_SCRIPT), matching the existing tests/test_laya_kit_input_fit.py pattern.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kit = load_module('train_from_export_for_model_dir_tests', 'training/laya-kit/train_from_export.py')


class ResolveModelDir(unittest.TestCase):
    def test_no_subdir_returns_model_dir_unchanged(self):
        # Default behavior (no --model-subdir) must keep using the repo root, matching the
        # english checkpoint's historical layout.
        resolved = kit.resolve_model_dir('/snapshot/root', None)
        self.assertEqual(Path(resolved), Path('/snapshot/root'))

    def test_empty_string_subdir_returns_model_dir_unchanged(self):
        resolved = kit.resolve_model_dir('/snapshot/root', '')
        self.assertEqual(Path(resolved), Path('/snapshot/root'))

    def test_subdir_is_joined_onto_model_dir(self):
        resolved = kit.resolve_model_dir('/snapshot/root', 'multilingual')
        self.assertEqual(Path(resolved), Path('/snapshot/root/multilingual'))


class ValidateResolvedModelDir(unittest.TestCase):
    def _make_valid_dir(self, base):
        (base / 'tokenizer').mkdir()
        (base / 'encoder').mkdir()
        (base / 'rl_agent_config.json').write_text('{}', encoding='utf-8')

    def test_passes_when_all_required_entries_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'multilingual'
            base.mkdir()
            self._make_valid_dir(base)
            # Must not raise.
            kit.validate_resolved_model_dir(base)

    def test_fails_clearly_when_rl_agent_config_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'root'
            base.mkdir()
            (base / 'tokenizer').mkdir()
            (base / 'encoder').mkdir()
            with self.assertRaises(kit.KitError) as cm:
                kit.validate_resolved_model_dir(base)
            self.assertIn('rl_agent_config.json', str(cm.exception))

    def test_fails_clearly_when_tokenizer_and_encoder_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'root'
            base.mkdir()
            (base / 'rl_agent_config.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(kit.KitError) as cm:
                kit.validate_resolved_model_dir(base)
            msg = str(cm.exception)
            self.assertIn('tokenizer', msg)
            self.assertIn('encoder', msg)


class DeriveModelName(unittest.TestCase):
    def test_uses_base_cfg_model_name_when_present(self):
        name = kit.derive_model_name({'model_name': 'rl-agent'}, 'multilingual')
        self.assertEqual(name, 'rl-agent-pointsman-ft')

    def test_falls_back_to_base_model_dir_name_when_cfg_has_no_model_name(self):
        name = kit.derive_model_name({}, 'multilingual')
        self.assertEqual(name, 'multilingual-pointsman-ft')

    def test_never_returns_the_old_hard_coded_typed_decisions_name_for_a_different_base(self):
        name = kit.derive_model_name({'model_name': 'rl-agent'}, 'multilingual')
        self.assertNotEqual(name, 'laya-typed-decisions')

    def test_dies_when_neither_cfg_nor_dir_name_available(self):
        with self.assertRaises(kit.KitError):
            kit.derive_model_name({}, None)


class ResolveEffectiveCfg(unittest.TestCase):
    def test_overrides_max_len_and_head_max_len_to_final_training_values(self):
        base_cfg = {'max_len': 512, 'head_max_len': 192, 'encoder': 'answerdotai/ModernBERT-large'}
        effective = kit.resolve_effective_cfg(base_cfg)
        self.assertEqual(effective['max_len'], kit.DEFAULT_FINAL_MAX_LEN)
        self.assertEqual(effective['head_max_len'], kit.DEFAULT_FINAL_HEAD_MAX_LEN)
        # Matches the literal override hard-coded in TRAIN_DDP_SCRIPT (official-notebook-cell 4).
        self.assertEqual(kit.DEFAULT_FINAL_MAX_LEN, 1024)
        self.assertEqual(kit.DEFAULT_FINAL_HEAD_MAX_LEN, 256)

    def test_does_not_mutate_base_cfg(self):
        base_cfg = {'max_len': 512, 'head_max_len': 192}
        kit.resolve_effective_cfg(base_cfg)
        self.assertEqual(base_cfg['max_len'], 512)
        self.assertEqual(base_cfg['head_max_len'], 192)

    def test_preserves_other_cfg_keys(self):
        base_cfg = {'max_len': 512, 'head_max_len': 192, 'encoder': 'jhu-clsp/mmBERT-base'}
        effective = kit.resolve_effective_cfg(base_cfg)
        self.assertEqual(effective['encoder'], 'jhu-clsp/mmBERT-base')

    def test_already_matching_base_cfg_is_unchanged_in_effect(self):
        # multilingual/typed-decisions checkpoints already ship max_len=1024/head_max_len=256.
        base_cfg = {'max_len': 1024, 'head_max_len': 256}
        effective = kit.resolve_effective_cfg(base_cfg)
        self.assertEqual(effective['max_len'], 1024)
        self.assertEqual(effective['head_max_len'], 256)


class TrainDdpScriptUsesArgvModelName(unittest.TestCase):
    """The saved-checkpoint model_name must come from argv (computed by
    derive_model_name in main()), not remain hard-coded to 'laya-typed-decisions'."""

    def test_hard_coded_typed_decisions_literal_is_gone_from_the_save_path(self):
        self.assertNotIn('cfg["model_name"] = "laya-typed-decisions"', kit.TRAIN_DDP_SCRIPT)

    def test_train_ddp_script_reads_model_name_from_argv(self):
        self.assertIn('sys.argv[5]', kit.TRAIN_DDP_SCRIPT)

    def test_train_ddp_script_records_base_model_dir_name(self):
        self.assertIn('base_model_dir_name', kit.TRAIN_DDP_SCRIPT)

    def test_train_ddp_script_records_exporter_version(self):
        self.assertIn('exporter_version', kit.TRAIN_DDP_SCRIPT)


if __name__ == '__main__':
    unittest.main()
