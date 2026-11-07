import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mesh_refinement import evaluate_mesh, fuse_skinning_weights, paper_fusion_config, refine_mesh
from mesh_refinement._topology import _partition_pairwise_compatible_groups
from mesh_refinement.fusion import SkinningFusionConfig


def fixture():
    vertices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 2, 3], [1, 2, 3]])
    skin = np.array([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.float32)
    return vertices, faces, skin


class CoreTest(unittest.TestCase):
    def test_paper_defaults(self):
        c = paper_fusion_config()
        self.assertEqual((c.policy, c.self_included, c.bandwidth_mode, c.adaptive_scale),
                         ("gaussian_l2", True, "group_rms", 0.5))

    def test_two_vertices_mean_and_cleanup(self):
        v, f, w = fixture()
        r = refine_mesh(v, f, w)
        self.assertEqual(r["merge_groups"], [[0, 1]])
        self.assertEqual(len(r["analysis"]["faces"]), 2)
        self.assertEqual(len(r["evaluation"]["faces"]), 1)
        np.testing.assert_array_equal(r["evaluation"]["skin"][0], [0.5, 0.5])
        self.assertEqual(r["skin_fusion_config"]["bandwidth_mode"], "group_rms")

    def test_three_vertex_fusion_is_permutation_invariant(self):
        w = np.array([[0.7, 0.3, 0], [0.68, 0.32, 0], [0.1, 0.1, 0.8]])
        a = fuse_skinning_weights(w, config=paper_fusion_config())
        b = fuse_skinning_weights(w[[2, 0, 1]], config=paper_fusion_config())
        np.testing.assert_allclose(a.fused_skin, b.fused_skin, atol=1e-12)
        self.assertAlmostEqual(a.fused_skin.sum(), 1)

    def test_two_row_kde_equals_mean(self):
        w = np.array([[1, 0], [0, 1]], dtype=float)
        for config in (paper_fusion_config(), SkinningFusionConfig(policy="uniform")):
            np.testing.assert_allclose(fuse_skinning_weights(w, config=config).fused_skin, [0.5, 0.5])

    def test_reverse_orientation_protected(self):
        v, f, w = fixture()
        f[1] = [1, 3, 2]
        r = refine_mesh(v, f, w)
        self.assertEqual(r["merge_groups"], [])
        self.assertEqual(r["protected_pairs_reverse_face"], [[0, 1]])
        self.assertEqual(len(r["evaluation"]["faces"]), 2)

    def test_missing_incident_normal_is_not_merged(self):
        v, f, w = fixture()
        r = refine_mesh(v, f[:1], w)
        self.assertEqual(r["merge_groups"], [])
        self.assertIn([0, 1], r["protected_pairs_invalid_incident"])

    def test_greedy_groups_are_cliques_not_components(self):
        graph = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
        self.assertEqual(_partition_pairwise_compatible_groups([0, 1, 2], graph), [[0, 1], [2]])

    def test_mapping_and_input_preservation(self):
        v, f, w = fixture()
        before = (v.copy(), f.copy(), w.copy())
        r = refine_mesh(v, f, w)["evaluation"]
        np.testing.assert_array_equal(r["merged_old_to_new"], [0, 0, 1, 2])
        np.testing.assert_array_equal(v[r["keep_indices"]], r["vertices"])
        for a, b in zip(before, (v, f, w)):
            np.testing.assert_array_equal(a, b)

    def test_zero_rows_fallback(self):
        v, f, w = fixture()
        r = refine_mesh(v, f, w * 0)
        self.assertTrue(np.isfinite(r["evaluation"]["skin"]).all())
        self.assertEqual(r["skin_fusion_groups"][0]["fallback_reason"], "all_zero_rows")

    def test_empty_mesh(self):
        r = refine_mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty((0, 2)))
        self.assertEqual(r["evaluation"]["vertices"].shape, (0, 3))

    def test_invalid_inputs(self):
        v, f, w = fixture()
        cases = [(v, f.astype(float), w), (v, f - 10, w),
                 (v, f, -w), (v, f, w[:-1]), (v * np.nan, f, w)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                refine_mesh(*args)
        for angle in (-1, 181, float("nan")):
            with self.assertRaises(ValueError):
                refine_mesh(v, f, w, angle_threshold_deg=angle)


class MetricsTest(unittest.TestCase):
    def test_original_gt_not_fused_reference(self):
        v, f, gt = fixture()
        r = refine_mesh(v, f, gt)["evaluation"]
        m = evaluate_mesh(v, f, gt, r)
        # One retained vertex has differences [0.5,0.5]; three retained vertices,
        # two joints. The processed row must NOT also replace the GT row.
        self.assertAlmostEqual(m["skin_l1"], 1 / 6)
        self.assertAlmostEqual(m["skin_l1_joint_sum"], 1 / 3)
        self.assertAlmostEqual(m["skin_l1_joint_sum_raw"], 1 / 3)

    def test_geometry_metrics(self):
        v, f, gt = fixture()
        m = evaluate_mesh(v, f, gt, refine_mesh(v, f, gt)["evaluation"])
        self.assertEqual(m["fpr"], 0.5)
        self.assertEqual(m["nme"], 3)  # Includes the three boundary edges.
        self.assertAlmostEqual(m["cd"], 0)
        self.assertIsNone(m["de"])

    def test_skin_rows_normalized_before_entry_mean(self):
        v, f, gt = fixture()
        r = refine_mesh(v, f, gt)["evaluation"]
        a = evaluate_mesh(v, f, gt, r)
        r["skin"] = r["skin"] * 2
        b = evaluate_mesh(v, f, gt, r)
        self.assertEqual(a["skin_l1"], b["skin_l1"])

    def test_invalid_gt_excluded(self):
        v, f, gt = fixture()
        r = refine_mesh(v, f, gt)["evaluation"]
        gt[:] = 0
        m = evaluate_mesh(v, f, gt, r)
        self.assertEqual(m["skin_valid_vertex_count"], 0)
        self.assertIsNone(m["skin_l1"])

    def test_de_seed_and_reference(self):
        v, f, gt = fixture()
        skeleton = {"matrix_local": np.repeat(np.eye(4)[None], 2, axis=0),
                    "parents": np.array([-1, 0])}
        skeleton["matrix_local"][1, 0, 3] = 1
        r = refine_mesh(v, f, gt)["evaluation"]
        a = evaluate_mesh(v, f, gt, r, skeleton=skeleton, sample_index=3)
        b = evaluate_mesh(v, f, gt, r, skeleton=skeleton, sample_index=3)
        self.assertEqual(a["de"], b["de"])
        self.assertGreater(a["de"], 0)
        self.assertEqual(a["pose_base_seed"], 321 + 3 * 15991)
        self.assertEqual(a["pose_count"], 64)
        self.assertEqual(a["de_valid_vertex_count"], 3)
        r["skin"] = gt[r["keep_indices"]].copy()
        self.assertEqual(evaluate_mesh(v, f, gt, r, skeleton=skeleton)["de"], 0)

    def test_invalid_skeleton(self):
        v, f, gt = fixture()
        r = refine_mesh(v, f, gt)["evaluation"]
        for parents in (np.array([0, 1]), np.array([-1, 2]), np.array([-2, 0])):
            with self.assertRaises(ValueError):
                evaluate_mesh(v, f, gt, r, skeleton={
                    "matrix_local": np.repeat(np.eye(4)[None], 2, axis=0), "parents": parents})

    def test_correspondence_not_implicit(self):
        v, f, gt = fixture()
        r = refine_mesh(v, f, gt)["evaluation"]
        r["vertices"][0, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "retained"):
            evaluate_mesh(v, f, gt, r)

    def test_empty_final_mesh_metrics_explicit(self):
        v, f, gt = fixture()
        r = refine_mesh(v, np.array([[0, 0, 0]]), gt)["evaluation"]
        m = evaluate_mesh(v, np.array([[0, 0, 0]]), gt, r)
        self.assertEqual(m["fpr"], 0)
        self.assertIsNone(m["skin_l1"])
        self.assertIsNone(m["cd"])


class CLITest(unittest.TestCase):
    def run_cli(self, *args, ok=True):
        r = subprocess.run([sys.executable, "-m", "mesh_refinement", *map(str, args)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stderr)
        else:
            self.assertNotEqual(r.returncode, 0)
        return r

    def test_complete_cli_and_no_overwrite(self):
        v, f, gt = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, scores = [Path(tmp) / n for n in ("input.npz", "refined.npz", "scores.json")]
            np.savez(src, vertices=v, faces=f, skin=gt, gt_skin=gt)
            self.run_cli("refine", "--input", src, "--output", dst)
            digest = dst.read_bytes()
            self.run_cli("refine", "--input", src, "--output", dst, ok=False)
            self.assertEqual(dst.read_bytes(), digest)
            self.run_cli("evaluate", "--input", src, "--refined", dst, "--output", scores)
            self.assertAlmostEqual(json.loads(scores.read_text())["skin_l1"], 1 / 6)
            with np.load(dst, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["gt_skin"], gt[data["keep_indices"]])
                self.assertEqual(json.loads(str(data["metadata_json"]))["fusion"]["adaptive_scale"], 0.5)

    def test_pickle_arrays_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.npz"
            np.savez(path, vertices=np.array([{}], dtype=object), faces=np.zeros((0, 3)), skin=np.zeros((1, 1)))
            self.run_cli("refine", "--input", path, "--output", Path(tmp) / "out.npz", ok=False)

    def test_invalid_options_and_partial_skeleton(self):
        v, f, gt = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
            np.savez(src, vertices=v, faces=f, skin=gt, gt_skin=gt, parents=np.array([-1, 0]))
            self.run_cli("refine", "--input", src, "--output", dst, "--bandwidth-scale", -1, ok=False)
            self.run_cli("refine", "--input", src, "--output", dst)
            self.run_cli("evaluate", "--input", src, "--refined", dst,
                         "--output", Path(tmp) / "scores.json", ok=False)


if __name__ == "__main__":
    unittest.main()
