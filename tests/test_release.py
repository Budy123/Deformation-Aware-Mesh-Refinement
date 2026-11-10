import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from mesh_refinement import evaluate_mesh, fuse_skinning_weights, paper_fusion_config, refine_mesh
from mesh_refinement._topology import _partition_pairwise_compatible_groups, _position_groups
from mesh_refinement.fusion import SkinningFusionConfig, format_skinning_fusion_tag


def fixture():
    vertices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 2, 3], [1, 2, 3]])
    skin = np.array([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.float32)
    return vertices, faces, skin


class CoreTest(unittest.TestCase):
    def test_paper_defaults(self):
        c = paper_fusion_config()
        self.assertEqual(c, SkinningFusionConfig(policy="gaussian_l2", adaptive_scale=0.5))

    def test_position_groups_keep_first_seen_order(self):
        vertices = np.array([[2, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]])
        groups, position_ids = _position_groups(vertices)
        self.assertEqual(groups, [[0, 2], [1, 4]])
        np.testing.assert_array_equal(position_ids, [0, 1, 0, 2, 1])

    def test_kde_and_mean_configs(self):
        v, f, w = fixture()
        v = np.concatenate([v, v[:1]])
        f = np.concatenate([f, [[4, 2, 3]]])
        w = np.concatenate([w, np.array([[0.3, 0.7]], dtype=np.float32)])
        configs = [SkinningFusionConfig(policy="uniform"), paper_fusion_config(),
                   SkinningFusionConfig(adaptive_scale=0.8)]
        for config in configs:
            with self.subTest(config=config):
                expected = fuse_skinning_weights(w[[0, 1, 4]], config=config).fused_skin
                actual = refine_mesh(v, f, w, fusion_config=config)["evaluation"]["skin"][0]
                np.testing.assert_array_equal(actual, expected)

    def test_legacy_fusion_options_are_rejected(self):
        for option, value in [("self_included", False), ("bandwidth_mode", "fixed"),
                              ("bandwidth", 0.1)]:
            with self.subTest(option=option), self.assertRaises(TypeError):
                SkinningFusionConfig(**{option: value})
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                refine_mesh(*fixture(), fusion_config=SkinningFusionConfig(adaptive_scale=value))

    def test_zero_dispersion_remains_finite(self):
        rows = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        result = fuse_skinning_weights(rows, config=paper_fusion_config())
        np.testing.assert_allclose(result.fused_skin, rows[0])
        self.assertEqual(result.bandwidth, 1e-12)
        self.assertTrue(np.isfinite(result.coefficients).all())

    def test_complete_config_defaults(self):
        self.assertEqual(asdict(paper_fusion_config()), {
            "policy": "gaussian_l2", "adaptive_scale": 0.5,
            "adaptive_bandwidth_floor": 1e-12, "gaussian_exponent_scale": 0.5})
        result = refine_mesh(*fixture())
        for key, value in asdict(paper_fusion_config()).items():
            self.assertEqual(result["skin_fusion_config"][key], value)

    def test_gaussian_exponent_matches_configured_formula(self):
        rows = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
        results = []
        for exponent in (0.5, 1.0):
            result = fuse_skinning_weights(rows, config=SkinningFusionConfig(
                gaussian_exponent_scale=exponent))
            densities = np.exp(-exponent * np.square(result.pairwise_l2 / result.bandwidth)).sum(axis=1)
            expected = densities / densities.sum()
            np.testing.assert_allclose(result.coefficients, expected, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(result.fused_skin, expected @ rows, rtol=1e-14, atol=1e-14)
            self.assertEqual(result.diagnostics()["gaussian_exponent_scale"], exponent)
            results.append(result.fused_skin)
        self.assertFalse(np.allclose(results[0], results[1]))

    def test_configured_bandwidth_floor(self):
        config = SkinningFusionConfig(adaptive_bandwidth_floor=0.8)
        rows = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
        result = fuse_skinning_weights(rows, config=config)
        self.assertEqual(result.bandwidth, 0.8)
        self.assertEqual(result.diagnostics()["adaptive_bandwidth_floor"], 0.8)
        default = fuse_skinning_weights(rows, config=paper_fusion_config())
        self.assertFalse(np.allclose(result.coefficients, default.coefficients))
        for special in (rows[:1], np.repeat(rows[:1], 3, axis=0),
                        np.vstack([rows[:1], np.zeros((1, 2))])):
            self.assertEqual(fuse_skinning_weights(special, config=config).bandwidth, 0.8)
        zero = fuse_skinning_weights(np.zeros((3, 2)), config=config)
        self.assertEqual(zero.fallback_reason, "all_zero_rows")
        self.assertEqual(zero.diagnostics()["adaptive_bandwidth_floor"], 0.8)

    def test_invalid_numeric_config_fields(self):
        for key in ("adaptive_scale", "adaptive_bandwidth_floor", "gaussian_exponent_scale"):
            for value in (0, -1, float("nan"), float("inf"), "invalid"):
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, key):
                    refine_mesh(*fixture(), fusion_config=SkinningFusionConfig(**{key: value}))

    def test_config_identity_tags(self):
        original = "skin-fusion-gaussian-l2-self-group-rms-c0p5"
        self.assertEqual(format_skinning_fusion_tag(paper_fusion_config()), original)
        tags = [format_skinning_fusion_tag(config) for config in (
            paper_fusion_config(), SkinningFusionConfig(adaptive_bandwidth_floor=0.8),
            SkinningFusionConfig(gaussian_exponent_scale=1.0))]
        self.assertEqual(len(set(tags)), 3)

    def test_mean_ignores_kde_parameters(self):
        config = SkinningFusionConfig(policy="uniform", adaptive_scale=0.9,
            adaptive_bandwidth_floor=0.8, gaussian_exponent_scale=1.0)
        rows = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
        result = fuse_skinning_weights(rows, config=config)
        np.testing.assert_allclose(result.fused_skin, rows.mean(axis=0))
        self.assertIsNone(result.bandwidth)
        self.assertIsNone(result.gaussian_exponent_scale)
        self.assertEqual(format_skinning_fusion_tag(config), "skin-fusion-uniform")

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

    def test_coincident_face_vertex_orders(self):
        base = np.array([[0.1, 0.2, 0.3], [1.3, 0.5, -0.2], [-0.4, 1.7, 0.8]])
        vertices = np.concatenate([base, base])
        skin = np.ones((6, 1), dtype=np.float32)
        same = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        opposite = [(0, 2, 1), (2, 1, 0), (1, 0, 2)]
        for order in same + opposite:
            with self.subTest(order=order):
                faces = np.array([[0, 1, 2], np.array(order) + 3])
                result = refine_mesh(vertices, faces, skin)
                self.assertEqual(bool(result["protected_pairs_reverse_face"]), order in opposite)
                self.assertEqual(len(result["merge_groups"]), 0 if order in opposite else 3)

    def test_noncoincident_faces_do_not_block_by_normal_angle(self):
        vertices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]])
        faces = np.array([[0, 2, 3], [1, 3, 4]])
        result = refine_mesh(vertices, faces, np.ones((5, 1), dtype=np.float32))
        self.assertEqual(result["merge_groups"], [[0, 1]])
        self.assertEqual(result["protected_pairs_reverse_face"], [])

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

    def test_angle_option_removed(self):
        with self.assertRaises(TypeError):
            refine_mesh(*fixture(), angle_threshold_deg=180)


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

    def test_refine_help_has_no_angle_threshold(self):
        self.assertNotIn("--angle-threshold-deg", self.run_cli("refine", "--help").stdout)

    def test_mean_cli(self):
        v, f, w = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input.npz", Path(tmp) / "mean.npz"
            np.savez(src, vertices=v, faces=f, skin=w)
            self.run_cli("refine", "--input", src, "--output", dst, "--fusion", "mean")
            with np.load(dst, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["skin"][0], [0.5, 0.5])
                config = json.loads(str(data["metadata_json"]))["fusion"]
                self.assertEqual(config["policy"], "uniform")
                self.assertIsNone(config["bandwidth_mode"])
                self.assertNotIn("bandwidth", config)

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
                metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(json.loads(scores.read_text())["refinement"], metadata)
            for key, value in asdict(paper_fusion_config()).items():
                self.assertEqual(metadata["fusion"][key], value)

    def test_yaml_config_controls_refinement(self):
        v, f, w = fixture()
        v = np.concatenate([v, v[:1]])
        f = np.concatenate([f, [[4, 2, 3]]])
        w = np.concatenate([w, [[0.2, 0.8]]])
        config = SkinningFusionConfig(adaptive_scale=0.2,
            adaptive_bandwidth_floor=0.75, gaussian_exponent_scale=1.0)
        expected = refine_mesh(v, f, w, fusion_config=config)["evaluation"]
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, path = [Path(tmp) / n for n in ("input.npz", "refined.npz", "config.yaml")]
            np.savez(src, vertices=v, faces=f, skin=w)
            path.write_text("# Configuration comments must be accepted.\n"
                "policy: gaussian_l2\nadaptive_scale: 0.2\n"
                "adaptive_bandwidth_floor: 0.75\n"
                "gaussian_exponent_scale: 1.0  # denominator h^2\n", encoding="utf-8")
            self.run_cli("refine", "--input", src, "--output", dst, "--config", path)
            with np.load(dst, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["skin"], expected["skin"])
                metadata = json.loads(str(data["metadata_json"]))
                for key, value in asdict(config).items():
                    self.assertEqual(metadata["fusion"][key], value)

    def test_yaml_config_defaults_and_cli_precedence(self):
        v, f, w = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, path = Path(tmp) / "input.npz", Path(tmp) / "config.yaml"
            np.savez(src, vertices=v, faces=f, skin=w)
            path.write_text(yaml.safe_dump({"policy": "uniform", "adaptive_scale": 0.2}), encoding="utf-8")
            dst = Path(tmp) / "mean.npz"
            self.run_cli("refine", "--input", src, "--output", dst, "--config", path)
            with np.load(dst, allow_pickle=False) as data:
                self.assertEqual(json.loads(str(data["metadata_json"]))["fusion"]["policy"], "uniform")
            dst = Path(tmp) / "kde.npz"
            self.run_cli("refine", "--input", src, "--output", dst, "--config", path,
                         "--fusion", "kde", "--bandwidth-scale", 0.9)
            with np.load(dst, allow_pickle=False) as data:
                config = json.loads(str(data["metadata_json"]))["fusion"]
                self.assertEqual(config["policy"], "gaussian_l2")
                self.assertEqual(config["adaptive_scale"], 0.9)
                self.assertEqual(config["adaptive_bandwidth_floor"], 1e-12)
                self.assertEqual(config["gaussian_exponent_scale"], 0.5)

    def test_invalid_yaml_configs(self):
        v, f, w = fixture()
        cases = [None, [], {"unexpected": 1}, {"self_included": False}, {"bandwidth_mode": "fixed"},
                 {"adaptive_bandwidth_floor": 0}, {"gaussian_exponent_scale": -1},
                 {"adaptive_bandwidth_floor": float("nan")}, {"gaussian_exponent_scale": float("inf")}]
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, path = [Path(tmp) / n for n in ("input.npz", "out.npz", "config.yaml")]
            np.savez(src, vertices=v, faces=f, skin=w)
            for value in cases:
                with self.subTest(config=value):
                    path.write_text(yaml.safe_dump(value), encoding="utf-8")
                    self.run_cli("refine", "--input", src, "--output", dst, "--config", path, ok=False)
                    self.assertFalse(dst.exists())

    def test_malformed_yaml_and_python_tags_rejected(self):
        v, f, w = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, path = [Path(tmp) / n for n in ("input.npz", "out.npz", "config.yaml")]
            np.savez(src, vertices=v, faces=f, skin=w)
            for contents in ("adaptive_scale: [unclosed", "!!python/object/apply:builtins.str [unsafe]"):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    self.run_cli("refine", "--input", src, "--output", dst, "--config", path, ok=False)
                    self.assertFalse(dst.exists())

    def test_yaml_config_matches_defaults(self):
        path = Path(__file__).resolve().parents[1] / "config/fusion_config.yaml"
        self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8")), asdict(paper_fusion_config()))
        v, f, w = fixture()
        expected = refine_mesh(v, f, w)["evaluation"]
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input.npz", Path(tmp) / "refined.npz"
            np.savez(src, vertices=v, faces=f, skin=w)
            self.run_cli("refine", "--input", src, "--output", dst, "--config", path)
            with np.load(dst, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["skin"], expected["skin"])
                config = json.loads(str(data["metadata_json"]))["fusion"]
                for key, value in asdict(paper_fusion_config()).items():
                    self.assertEqual(config[key], value)

    def test_evaluation_without_refinement_metadata(self):
        v, f, w = fixture()
        phase = refine_mesh(v, f, w)["evaluation"]
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, scores = [Path(tmp) / n for n in ("input.npz", "refined.npz", "scores.json")]
            np.savez(src, vertices=v, faces=f, gt_skin=w)
            np.savez(dst, **phase)
            self.run_cli("evaluate", "--input", src, "--refined", dst, "--output", scores)
            self.assertIsNone(json.loads(scores.read_text())["refinement"])

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
