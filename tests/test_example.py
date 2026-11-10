import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from mesh_refinement import evaluate_mesh, refine_mesh


class BundledExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / 'examples/axl2_fish'
        with np.load(cls.root / 'input.npz', allow_pickle=False) as archive:
            cls.arrays = {key: archive[key] for key in archive.files}

    def test_numeric_source_arrays_and_provenance(self):
        a = self.arrays
        self.assertEqual(a['vertices'].shape, (1098, 3))
        self.assertEqual(a['faces'].shape, (544, 3))
        self.assertEqual(a['skin'].shape, (1098, 8))
        self.assertTrue(all(x.dtype.kind != 'O' for x in a.values()))
        self.assertFalse(np.array_equal(a['skin'], a['gt_skin']))
        source = json.loads((self.root / 'provenance.json').read_text())
        self.assertEqual(source['sample_id'], 'c76bfca7-6e53-50d1-a0e9-ff7cd6690866')
        self.assertEqual(source['asset_license'], 'CC0-1.0')
        self.assertEqual(source['dataset_license'], 'CC-BY-4.0')
        for key in ('vertices', 'faces', 'skin', 'gt_skin', 'matrix_local', 'parents'):
            self.assertEqual(hashlib.sha256(a[key].tobytes()).hexdigest(), source[key + '_sha256'])
        self.assertNotIn('/workspace/', str(a['metadata_json']))

    def test_real_example_refinement_and_evaluation(self):
        a = self.arrays
        result = refine_mesh(a['vertices'], a['faces'], a['skin'])['evaluation']
        self.assertEqual(result['vertices'].shape, (274, 3))
        self.assertEqual(result['faces'].shape, (544, 3))
        np.testing.assert_array_equal(result['vertices'], a['vertices'][result['keep_indices']])
        scores = evaluate_mesh(a['vertices'], a['faces'], a['gt_skin'], result,
            skeleton={key: a[key] for key in ('matrix_local', 'parents')})
        self.assertEqual(scores['cd'], 0)
        self.assertEqual(scores['fpr'], 1)
        self.assertEqual(scores['nme'], 0)
        self.assertEqual(scores['pose_count'], 64)
        self.assertTrue(np.isfinite(scores['de']))


if __name__ == '__main__':
    unittest.main()
