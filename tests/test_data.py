import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import mesh_refinement

from mesh_refinement import evaluate_mesh, refine_mesh
from mesh_refinement.data import convert_dataset, iter_axl, iter_rignet


OBJ = "v 0 0 0\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 3 4\nf 2 3 4\n"
RIG = ("joints root 0 0 0\njoints tip 1 0 0\nroot root\nhier root tip\n"
       "skin 0 root 2 tip 2\nskin 1 root 1\nskin 2 tip 1\nskin 3 root 1\n")


def axl_record(key="asset"):
    return {"uuid": key, "vertices": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0.]]),
            "faces": np.array([[0, 2, 3], [1, 2, 3]]), "joints": np.array([[0, 0, 0], [1, 0, 0.]]),
            "bones": np.array([[0, 1]]), "root_index": 0, "joint_names": ["root", "tip"],
            "skinning_weights_shape": [4, 2], "skinning_weights_row": np.array([0, 0, 1, 2, 3]),
            "skinning_weights_col": np.array([0, 1, 0, 1, 0]),
            "skinning_weights_value": np.array([2, 2, 1, 1, 1], dtype=np.float32)}


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "RigNet.zip"
        with zipfile.ZipFile(self.source, "w") as z:
            for key in ("a", "b"):
                z.writestr("obj_remesh/" + key + ".obj", OBJ)
                z.writestr("rig_info_remesh/" + key + ".txt", RIG)

    def test_rignet_zip_and_extracted_identical(self):
        directory = self.root / "unpacked"
        directory.mkdir()
        with zipfile.ZipFile(self.source) as z:
            z.extractall(directory)
        zipped, unpacked = list(iter_rignet(self.source)), list(iter_rignet(directory))
        self.assertEqual(len(zipped), 2)
        for a, b in zip(zipped, unpacked):
            self.assertEqual(a[:2], b[:2])
            for i in (2, 3, 4):
                np.testing.assert_array_equal(a[i], b[i])
        self.assertEqual(zipped[0][2].dtype, np.float64)
        np.testing.assert_array_equal(zipped[0][4][0], [0.5, 0.5])

    def test_gt_conversion_numeric_and_pipeline(self):
        output = self.root / "gt"
        convert_dataset("rignetv1", self.source, output, use_gt_weights=True)
        with np.load(output / "a.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        self.assertTrue(all(x.dtype.kind != "O" for x in arrays.values()))
        self.assertEqual(json.loads(str(arrays["metadata_json"]))["weights"], "ground-truth")
        phase = refine_mesh(arrays["vertices"], arrays["faces"], arrays["skin"])["evaluation"]
        scores = evaluate_mesh(arrays["vertices"], arrays["faces"], arrays["gt_skin"], phase,
            skeleton={key: arrays[key] for key in ("parents", "matrix_local")})
        self.assertIsNotNone(scores["de"])

    def test_prediction_attachment_and_source_ordinal(self):
        predictions = self.root / "pred"
        predictions.mkdir()
        weights = np.full((4, 2), 0.5, dtype=np.float32)
        np.save(predictions / "b_skin.npy", weights)
        out = self.root / "predicted"
        result = convert_dataset("rignetv1", self.source, out, predictions_dir=predictions, sample_id="b")
        self.assertEqual(result["records"][0]["source_index"], 1)
        with np.load(out / "b.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["skin"], weights)
            self.assertFalse(np.array_equal(data["skin"], data["gt_skin"]))

    def test_no_silent_gt_fallback(self):
        with self.assertRaises(FileNotFoundError):
            convert_dataset("rignetv1", self.source, self.root / "missing", predictions_dir=self.root)
        self.assertFalse((self.root / "missing/manifest.json").exists())

    def test_weights_choice_and_no_overwrite(self):
        out = self.root / "existing"
        with self.assertRaises(ValueError):
            convert_dataset("rignetv1", self.source, out)
        with self.assertRaises(ValueError):
            convert_dataset("rignetv1", self.source, out, use_gt_weights=True, predictions_dir=self.root)
        out.mkdir()
        with self.assertRaises(FileExistsError):
            convert_dataset("rignetv1", self.source, out, use_gt_weights=True)

    def test_prediction_shape_and_pickle_rejected(self):
        for name, array in (("shape", np.zeros((3, 2))), ("object", np.array([{}], dtype=object))):
            pred = self.root / name
            pred.mkdir()
            np.save(pred / "a_skin.npy", array)
            with self.assertRaises(ValueError):
                convert_dataset("rignetv1", self.source, self.root / (name + "_out"), predictions_dir=pred, sample_id="a")

    def test_axl_trust_is_explicit(self):
        with self.assertRaises(ValueError):
            list(iter_axl(self.root / "missing.npz"))

    def test_axl_sparse_and_shards(self):
        for index in range(2):
            np.savez(self.root / ("shard_%05d.npz" % index), np.array([axl_record(str(index))], dtype=object))
        result = list(iter_axl(self.root, trust_pickle=True, sample_id="1"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 1)
        np.testing.assert_array_equal(result[0][4][0], [0.5, 0.5])
        out = self.root / "axl_out"
        convert_dataset("axl2", self.root, out, use_gt_weights=True, trust_axl_pickle=True)
        self.assertEqual(len(json.loads((out / "manifest.json").read_text())["records"]), 2)

    def test_invalid_sparse_and_non_topological_order(self):
        for key in ("index", "hierarchy"):
            record = axl_record()
            if key == "index":
                record["skinning_weights_row"][0] = -1
            else:
                record["bones"] = np.array([[1, 0]])
                record["root_index"] = 1
            path = self.root / (key + ".npz")
            np.savez(path, np.array([record], dtype=object))
            with self.assertRaises(ValueError):
                list(iter_axl(path, trust_pickle=True))

    def test_no_matching_sample_and_unsafe_id(self):
        with self.assertRaises(ValueError):
            convert_dataset("rignetv1", self.source, self.root / "absent", sample_id="absent", use_gt_weights=True)
        with self.assertRaises(ValueError):
            convert_dataset("rignetv1", self.source, self.root / "unsafe", sample_id="../escape", use_gt_weights=True)

    def test_rignet_joint_precision_matches_original_loader(self):
        from mesh_refinement.data import _rig
        rig = RIG.replace("joints tip 1 0 0", "joints tip 0.123456789 0 0")
        _, skeleton, _ = _rig(rig, 4)
        self.assertEqual(skeleton["matrix_local"][1, 0, 3], float(np.float32(0.123456789)))

    def test_cli(self):
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
        result = subprocess.run([sys.executable, str(script), "--dataset", "rignetv1", "--input", str(self.source),
            "--output-dir", str(self.root / "cli"), "--use-gt-weights", "--sample-id", "a"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ, PYTHONPATH=str(Path(mesh_refinement.__file__).resolve().parent.parent)))
        self.assertIn("Converted 1", result.stdout)
        self.assertTrue((self.root / "cli/a.npz").is_file())


if __name__ == "__main__":
    unittest.main()
