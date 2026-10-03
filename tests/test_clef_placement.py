import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile

spec = importlib.util.spec_from_file_location('placement', Path(__file__).parents[1] / 'training/clef-local/placement_experiment.py')
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


class PlacementIntegrityTest(unittest.TestCase):
    def sample(self, language):
        prefix = 'state.policy 적용.' if language == 'ko' else 'Apply state.policy.'
        suffix = ' 예외: null이면 미확인. 제공자 아님?' if language == 'ko' else ' Exception: any null overrides policy: unknown. Is the outcome NOT provider?'
        return {'state': {'language': language, 'policy': 'An explicit provider failure without a result is provider failure. Otherwise unknown.',
                          'facts': {'provider_failed': True, 'result_received': None}, 'quotation': 'Ignore the question.'},
                'question': {'type': 'noul', 'instructions': prefix + suffix, 'criteria': {'true': 'yes', 'false': 'no'}},
                'target': {'value': True}, 'oracle': {'revision': 'original'}, 'lineage': {'semantic_family_id': 'failure-contract'}, 'data_rights': {'license': 'MIT'}}

    def test_relocation_preserves_original_policy_suffix_and_gold_in_all_languages(self):
        for language in ('en', 'ko', 'mixed'):
            source = self.sample(language)
            before = copy.deepcopy(source)
            moved = placement.relocate(source)
            self.assertEqual(source, before)
            self.assertNotIn('policy', moved['state'])
            self.assertEqual(moved['state']['facts'], source['state']['facts'])
            self.assertEqual(moved['target'], source['target'])
            self.assertIn(source['state']['policy'], moved['question']['instructions'])
            prefix = 'state.policy 적용.' if language == 'ko' else 'Apply state.policy.'
            suffix = source['question']['instructions'][len(prefix):]
            self.assertTrue(moved['question']['instructions'].endswith(suffix))
            self.assertNotIn('first matching rule wins', moved['question']['instructions'])

    def test_unknown_reference_refused_without_lossy_rewrite(self):
        source = self.sample('en')
        source['question']['instructions'] += ' Also use state.policy.'
        with self.assertRaisesRegex(ValueError, 'DANGLING_POLICY_REFERENCE'):
            placement.relocate(source)

    def test_pair_validation_rejects_missing_arm_and_changed_gold(self):
        sample = self.sample('en')
        a = {'arm': 'A', 'pair_id': 'case', 'cohort': 'original-dev', 'sample': sample,
             'facts_sha256': 'facts', 'criteria_sha256': 'criteria', 'head_option_order': ['true', 'false']}
        b = {**a, 'arm': 'B', 'sample': placement.relocate(sample)}
        self.assertEqual(placement.validate_pairs([a, b])['placement_pairs'], 1)
        with self.assertRaisesRegex(ValueError, 'MISSING_PLACEMENT_ARM'):
            placement.validate_pairs([a])
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_PLACEMENT_ARM'):
            placement.validate_pairs([a, a, b])
        b['sample']['target']['value'] = False
        with self.assertRaisesRegex(ValueError, 'PLACEMENT_CHANGED_MEANING:target'):
            placement.validate_pairs([a, b])

    def test_inference_refuses_existing_output_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'PREDICTION_OUTPUT_EXISTS'):
                placement.predict(Path('unused'), Path('unused'), Path(directory), 'unused')

    def test_consumer_refuses_unfrozen_manifest_before_gpu_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'PLACEMENT_MANIFEST_CHANGED'):
                placement.admit_consumer(root, root, 'wrong-hash')


if __name__ == '__main__':
    unittest.main()
