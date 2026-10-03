"""Saved-tokenizer compatibility uses the pinned official fix before asset hashing."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kit_tokenizer_export',ROOT/'training/laya-kit/train_from_export.py')
kit=importlib.util.module_from_spec(spec);spec.loader.exec_module(kit)
try:
    import laya.agent
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    HAVE_STACK=True
except ImportError:
    HAVE_STACK=False

@unittest.skipUnless(HAVE_STACK,'local tokenizer/Laya stack unavailable')
class TokenizerExport(unittest.TestCase):
    def test_transformers_export_becomes_worker_compatible_without_token_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'tokenizer').mkdir()
            backend=Tokenizer(models.WordLevel({'<unk>':0,'<mask>':1,'hello':2,'안녕':3},unk_token='<unk>'))
            backend.pre_tokenizer=pre_tokenizers.Whitespace()
            tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',mask_token='<mask>')
            tok.save_pretrained(root/'tokenizer')
            path=root/'tokenizer/tokenizer_config.json'
            config=json.loads(path.read_text());config.update(tokenizer_class='TokenizersBackend',backend='tokenizers',is_local=True)
            path.write_text(json.dumps(config))
            payload_before=kit.file_sha256(root/'tokenizer/tokenizer.json')
            actual=kit.prepare_exported_tokenizer(root)
            after=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
            config=json.loads(path.read_text())
            self.assertEqual(config['tokenizer_class'],'PreTrainedTokenizerFast')
            self.assertNotIn('backend',config);self.assertNotIn('is_local',config)
            self.assertEqual(payload_before,kit.file_sha256(root/'tokenizer/tokenizer.json'))
            self.assertEqual(tok.get_vocab(),after.get_vocab())
            for text in ('hello 안녕','<mask> hello','unknown text','안녕 hello 안녕'):
                self.assertEqual(tok(text)['input_ids'],after(text)['input_ids'])
            self.assertEqual(actual['tokenizer_sha256'],kit.directory_sha256(root/'tokenizer'))
            self.assertEqual(actual,kit.prepare_exported_tokenizer(root))

    def test_official_fix_failure_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'tokenizer').mkdir();(root/'tokenizer/tokenizer_config.json').write_text('{"tokenizer_class":"TokenizersBackend"}')
            with mock.patch.object(laya.agent,'_fix_tokenizer_config',lambda path:None):
                with self.assertRaises(kit.KitError):kit.prepare_exported_tokenizer(root)

class FinalizeBoundary(unittest.TestCase):
    def test_runtime_preparation_follows_save_in_the_embedded_finalizer(self):
        section=kit.TRAIN_DDP_SCRIPT.split('def finalize_and_save(',1)[1].split('def write_epoch_selection_metadata(',1)[0]
        self.assertLess(section.index('tok.save_pretrained('),section.index('prepare_exported_tokenizer(output_dir, with_hash=False)'))
        self.assertIn('TOKENIZER_PREPARATION_REQUIRED',kit.TRAIN_DDP_SCRIPT)

if __name__=='__main__':unittest.main()
