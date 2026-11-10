import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mesh_refinement import evaluate_mesh, evaluate_seam_opening, refine_mesh
from mesh_refinement import _review_metrics as R


def fixture(copies=2, opposite=True):
    v = np.tile([[0., 0, 0], [1., 0, 0], [0., 1, 0]], (copies, 1))
    f = np.arange(3 * copies).reshape(-1, 3)
    if opposite:
        f[1::2] = f[1::2, ::-1]
    w = np.tile([.4, .6], (len(v), 1))
    return v, f, w


def original(v, f, w):
    return dict(vertices=v.copy(), faces=f.copy(), skin=w.copy(),
                keep_indices=np.arange(len(v)), merged_old_to_new=np.arange(len(v)),
                kept_source_face_indices=np.arange(len(f)))


def skeleton():
    bind = np.tile(np.eye(4), (2, 1, 1))
    bind[1, 0, 3] = 1
    return dict(matrix_local=bind, parents=np.array([-1, 0]))


class ReviewMetricsTest(unittest.TestCase):
    def test_opposite_pair_preserved_removed_and_collapsed(self):
        v, f, w = fixture()
        state = refine_mesh(v, f, w)['evaluation']
        self.assertEqual(evaluate_mesh(v, f, w, state)['ofpr'], 1)
        # An already shared-index opposite pair is not a source-layer pair.
        shared = original(v[:3], np.array([[0, 1, 2], [0, 2, 1]]), w[:3])
        self.assertIsNone(evaluate_mesh(v[:3], shared['faces'], w[:3], shared)['ofpr'])
        state = original(v, f, w)
        state['faces'] = f[:1]
        state['kept_source_face_indices'] = np.array([0])
        self.assertEqual(evaluate_mesh(v, f, w, state)['ofpr'], 0)

    def test_multiple_opposite_copies_count_all_pairs(self):
        v, f, w = fixture(3)
        result = evaluate_mesh(v, f, w, original(v, f, w))
        self.assertEqual(result['ofpr_source_pair_count'], 2)
        self.assertEqual(result['ofpr_preserved_pair_count'], 2)
        self.assertEqual(result['ofpr'], 1)

    def test_local_uses_original_targets_not_just_representatives(self):
        v, f, gt = fixture(3)
        pred = gt.copy()
        pred[:3] = [1, 0]
        state = refine_mesh(v, f, pred)['evaluation']
        scores = evaluate_mesh(v, f, gt, state, skeleton=skeleton(), pose_count=3)
        self.assertEqual(scores['local_source_vertex_count'], 9)
        self.assertEqual(scores['local_mapped_vertex_count'], 9)
        self.assertEqual(scores['local_skin_valid_vertex_count'], 9)
        mapped = state['skin'][state['merged_old_to_new']]
        expected = np.abs(mapped / mapped.sum(axis=1, keepdims=True) - gt).mean()
        self.assertAlmostEqual(scores['local_skin_l1'], expected)
        self.assertGreater(scores['local_de'], 0)

    def test_no_targets_no_skeleton_and_legacy_output(self):
        v, f, w = fixture(opposite=False)
        state = refine_mesh(v, f, w)['evaluation']
        scores = evaluate_mesh(v, f, w, state)
        self.assertIsNone(scores['local_skin_l1'])
        self.assertEqual(scores['local_source_vertex_count'], 0)
        self.assertIsNone(scores['local_de'])
        self.assertIsNone(scores['so'])
        state = {k: state[k] for k in ('vertices', 'faces', 'skin', 'keep_indices')}
        scores = evaluate_mesh(v, f, w, state)
        self.assertIsNone(scores['ofpr'])
        self.assertIsNone(scores['so'])

    def test_local_invalid_rows_and_missing_targets(self):
        v, f, w = fixture()
        state = original(v, f, w)
        state['skin'][0] = 0
        result = evaluate_mesh(v, f, w, state)
        self.assertEqual(result['local_skin_valid_vertex_count'], 5)
        self.assertEqual(result['local_skin_l1'], 0)
        state = original(v[:3], f[:1], w[:3])
        state['merged_old_to_new'] = np.r_[np.arange(3), [-1, -1, -1]]
        result = evaluate_mesh(v, f, w, state)
        self.assertEqual(result['local_mapped_vertex_count'], 3)
        self.assertIsNone(result['local_skin_l1'])

    def test_seam_integral_analytic_and_quadrature(self):
        self.assertEqual(R.edge_integral(np.array([0., 2, 0]), np.array([0., 2, 0])), 4)
        rng = np.random.RandomState(123)
        a, b = rng.randn(2, 11, 3)
        t = np.linspace(0, 1, 20001)
        samples = a[:, None] * (1 - t[None, :, None]) + b[:, None] * t[None, :, None]
        expected = np.trapz(np.square(samples).sum(axis=-1), t, axis=1)
        np.testing.assert_allclose(R.edge_integral(a, b), expected, rtol=0, atol=2e-8)

    def test_three_edge_copies_all_pairs_and_gt_filter(self):
        v, f, w = fixture(3)
        pairs, lengths = R.seam_source_pairs(v, np.r_[f, f], w)
        self.assertEqual(len(pairs), 9)
        np.testing.assert_array_equal(v[pairs[:, 0]], v[pairs[:, 1]])
        for i, x in enumerate((.2, .20000075, .2000015)):
            w[3*i:3*i+3] = [x, 1-x]
        pairs, _ = R.seam_source_pairs(v, f, w)
        self.assertEqual(len(pairs), 6)  # pairwise tolerance, not transitive closure
        w[3:] = [0, 1]
        pairs, _ = R.seam_source_pairs(v, f, w)
        self.assertEqual(len(pairs), 3)

    def test_common_pairs_exclude_missing_edge_not_zero_fill(self):
        v, f, w = fixture()
        full = original(v, f, w)
        partial = original(v, f[:1], w)
        scores = evaluate_seam_opening(v, f, w, {'full': full, 'partial': partial},
                                       skeleton=skeleton(), pose_count=2)
        self.assertEqual(scores['source_pair_count'], 3)
        self.assertEqual(scores['common_pair_count'], 0)
        self.assertIsNone(scores['methods']['full']['so'])
        self.assertEqual(scores['methods']['partial']['missing_pair_count'], 3)

    def test_merged_edge_is_valid_zero_and_different_weights_separate(self):
        v, f, w = fixture(opposite=False)
        raw = original(v, f, w)
        raw['skin'][:3] = [1, 0]
        raw['skin'][3:] = [0, 1]
        merged = refine_mesh(v, f, raw['skin'])['evaluation']
        result = evaluate_seam_opening(v, f, w, {'raw': raw, 'merged': merged},
                                       skeleton=skeleton(), pose_count=4)
        self.assertEqual(result['common_pair_count'], 3)
        self.assertGreater(result['methods']['raw']['so'], 0)
        self.assertEqual(result['methods']['merged']['so'], 0)
        single = evaluate_mesh(v, f, w, raw, skeleton=skeleton(), pose_count=4)
        self.assertEqual(single['so'], result['methods']['raw']['so'])

    def test_no_pairs_invalid_map_and_invalid_pose_options(self):
        v, f, w = fixture(1)
        state = original(v, f, w)
        result = evaluate_seam_opening(v, f, w, {'only': state}, skeleton=skeleton(), pose_count=2)
        self.assertIsNone(result['methods']['only']['so'])
        self.assertEqual(result['source_pair_count'], 0)
        state['merged_old_to_new'][0] = 99
        with self.assertRaises(ValueError):
            evaluate_mesh(v, f, w, state)
        with self.assertRaises(ValueError):
            evaluate_seam_opening(v, f, w, {}, pose_count=0)

    def test_cli_loads_correspondence_arrays(self):
        v, f, w = fixture()
        state = refine_mesh(v, f, w)['evaluation']
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            np.savez(base/'input.npz', vertices=v, faces=f, gt_skin=w)
            np.savez(base/'refined.npz', **{k: x for k, x in state.items() if isinstance(x, np.ndarray)})
            subprocess.run([sys.executable, '-m', 'mesh_refinement', 'evaluate', '--input',
                str(base/'input.npz'), '--refined', str(base/'refined.npz'), '--output',
                str(base/'scores.json')], check=True, capture_output=True)
            result = json.loads((base/'scores.json').read_text())
            self.assertEqual(result['ofpr'], 1)
            self.assertEqual(result['local_source_vertex_count'], 6)
            self.assertEqual(result['so_source_pair_count'], 3)


if __name__ == '__main__':
    unittest.main()
