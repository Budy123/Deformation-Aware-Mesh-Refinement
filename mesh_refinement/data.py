"""Convert the public RigNetv1 and AXL 2.0 releases to numeric per-mesh NPZ."""

from pathlib import Path
import sys
import zipfile

import numpy as np

from .core import validate_geometry, validate_skin
from .metrics import _validated_skeleton


def _sample_id(value):
    value = str(value)
    if not value or value in (".", "..") or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in value):
        raise ValueError("Unsafe or empty sample ID: " + repr(value))
    return value


def _normalized_gt(skin, vertex_count, joint_count):
    skin = validate_skin(skin, vertex_count, joint_count).astype(np.float32)
    if not np.isfinite(skin).all():
        raise ValueError("GT weights cannot be represented as float32")
    sums = skin.sum(axis=1, keepdims=True)
    if not np.isfinite(sums).all():
        raise ValueError("GT row sums overflow float32")
    nonzero = sums[:, 0] > 0
    skin[nonzero] /= np.maximum(sums[nonzero], 1e-8)
    return skin


def _skeleton(joints, bones, root_index):
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3 or not len(joints):
        raise ValueError("joints must be a nonempty (J, 3) array")
    bones = np.asarray(bones)
    if bones.size == 0:
        bones = np.empty((0, 2), dtype=np.int64)
    if bones.ndim != 2 or bones.shape[1] != 2 or bones.dtype.kind not in "iu":
        raise ValueError("bones must be an integer (B, 2) array")
    parents = np.full(len(joints), -1, dtype=np.int64)
    for parent, child in bones:
        if not (0 <= parent < len(joints) and 0 <= child < len(joints)) or parent == child:
            raise ValueError("Invalid bone index")
        if parents[child] != -1 and parents[child] != parent:
            raise ValueError("Joint has multiple parents")
        parents[child] = parent
    if not (0 <= root_index < len(joints)) or parents[root_index] != -1:
        raise ValueError("Invalid skeleton root")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(joints), axis=0)
    matrices[:, :3, 3] = joints
    # Do not reorder joints: that would change the seeded per-joint pose bank.
    return _validated_skeleton({"matrix_local": matrices, "parents": parents}, len(joints))


def _obj(text):
    vertices, faces = [], []
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            indices = [int(x.split("/")[0]) for x in parts[1:]]
            if len(indices) < 3 or 0 in indices:
                raise ValueError("Invalid OBJ face")
            indices = [x - 1 if x > 0 else len(vertices) + x for x in indices]
            faces.extend([[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)])
    return validate_geometry(np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
                             np.asarray(faces, dtype=np.int64).reshape(-1, 3))


def _rig(text, vertex_count):
    names, joints, edges, rows, root = [], [], [], {}, None
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "joints":
            if len(parts) != 5 or parts[1] in names:
                raise ValueError("Invalid or repeated joint declaration")
            names.append(parts[1])
            joints.append([float(x) for x in parts[2:]])
        elif parts[0] == "root":
            if len(parts) != 2 or root is not None:
                raise ValueError("Invalid or repeated root declaration")
            root = parts[1]
        elif parts[0] == "hier":
            if len(parts) != 3:
                raise ValueError("Invalid hierarchy declaration")
            edges.append(parts[1:])
        elif parts[0] == "skin":
            if len(parts) < 4 or len(parts) % 2:
                raise ValueError("Invalid skin declaration")
            index = int(parts[1])
            if not 0 <= index < vertex_count or index in rows:
                raise ValueError("Invalid or repeated skin vertex index")
            rows[index] = list(zip(parts[2::2], map(float, parts[3::2])))
    indices = {name: i for i, name in enumerate(names)}
    if root not in indices:
        raise ValueError("Missing or unknown root joint")
    skin = np.zeros((vertex_count, len(names)), dtype=np.float32)
    for row, weights in rows.items():
        for name, weight in weights:
            skin[row, indices[name]] += weight
    # The experiment's RigNet text loader uses float32 joint positions.
    skeleton = _skeleton(np.asarray(joints, dtype=np.float32),
                         np.asarray([[indices[a], indices[b]] for a, b in edges], dtype=np.int64),
                         indices[root])
    return _normalized_gt(skin, vertex_count, len(names)), skeleton, names


def iter_rignet(source, sample_id=None):
    """Read the official preprocessed ZIP or its extracted root, in OBJ-name order."""
    source = Path(source)
    if source.is_dir():
        obj_dir, rig_dir = source / "obj_remesh", source / "rig_info_remesh"
        if not obj_dir.is_dir() or not rig_dir.is_dir():
            raise ValueError("RigNetv1 root must contain obj_remesh/ and rig_info_remesh/")
        paths = sorted(obj_dir.glob("*.obj"), key=lambda p: p.stem)
        for index, path in enumerate(paths):
            if sample_id is not None and path.stem != sample_id:
                continue
            vertices, faces = _obj(path.read_text(encoding="utf-8"))
            gt, skeleton, names = _rig((rig_dir / (path.stem + ".txt")).read_text(encoding="utf-8"), len(vertices))
            yield path.stem, index, vertices, faces, gt, skeleton, names
    else:
        with zipfile.ZipFile(source) as archive:
            paths = sorted((x for x in archive.namelist()
                            if x.startswith("obj_remesh/") and x.endswith(".obj")),
                           key=lambda x: Path(x).stem)
            for index, path in enumerate(paths):
                key = Path(path).stem
                if sample_id is not None and key != sample_id:
                    continue
                vertices, faces = _obj(archive.read(path).decode("utf-8"))
                gt, skeleton, names = _rig(archive.read("rig_info_remesh/" + key + ".txt").decode("utf-8"), len(vertices))
                yield key, index, vertices, faces, gt, skeleton, names


def iter_axl(source, *, trust_pickle=False, sample_id=None):
    """Read an official object-record NPZ, or local shard_*.npz files one at a time."""
    if not trust_pickle:
        raise ValueError("AXL object records require --trust-axl-pickle; use only trusted official files")
    # Match the research loader's NumPy 2-written object-array compatibility.
    import numpy.core as core
    import numpy.core.multiarray as multiarray
    import numpy.core.numeric as numeric
    for name, module in (("numpy._core", core), ("numpy._core.multiarray", multiarray),
                         ("numpy._core.numeric", numeric)):
        sys.modules.setdefault(name, module)
    source = Path(source)
    paths = sorted(source.glob("shard_*.npz")) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No shard_*.npz files found")
    ordinal = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            records = archive["arr_0"]
        if records.dtype != object or records.ndim != 1:
            raise ValueError("AXL arr_0 must be a one-dimensional object array of records")
        for record in records:
            index = ordinal
            ordinal += 1
            key = _sample_id(record["uuid"])
            if sample_id is not None and key != sample_id:
                continue
            vertices, faces = validate_geometry(record["vertices"], record["faces"])
            joint_count = len(record["joints"])
            shape = tuple(record["skinning_weights_shape"])
            if shape != (len(vertices), joint_count):
                raise ValueError("AXL sparse skin shape mismatch")
            rows, cols = (np.asarray(record["skinning_weights_" + k]) for k in ("row", "col"))
            values = np.asarray(record["skinning_weights_value"])
            if rows.dtype.kind not in "iu" or cols.dtype.kind not in "iu" or not (rows.shape == cols.shape == values.shape) or rows.ndim != 1:
                raise ValueError("Invalid AXL sparse skin arrays")
            if rows.size and (rows.min() < 0 or rows.max() >= shape[0] or cols.min() < 0 or cols.max() >= shape[1]):
                raise ValueError("AXL sparse skin index out of range")
            gt = np.zeros(shape, dtype=np.float32)
            gt[rows, cols] = values
            gt = _normalized_gt(gt, len(vertices), joint_count)
            skeleton = _skeleton(record["joints"], record["bones"], int(record["root_index"]))
            names = [str(x) for x in record["joint_names"]]
            if len(names) != joint_count:
                raise ValueError("Joint name count mismatch")
            yield key, index, vertices, faces, gt, skeleton, names
        del records


def convert_dataset(dataset, source, output_dir, *, predictions_dir=None,
                    use_gt_weights=False, trust_axl_pickle=False, sample_id=None):
    """Write fresh numeric NPZ files and a manifest; never silently substitute GT."""
    import json
    if (predictions_dir is not None) == bool(use_gt_weights):
        raise ValueError("Choose exactly one of --predictions-dir or --use-gt-weights")
    if sample_id is not None:
        sample_id = _sample_id(sample_id)
    if dataset == "rignetv1":
        records = iter_rignet(source, sample_id)
    elif dataset == "axl2":
        if not trust_axl_pickle:
            raise ValueError("AXL requires --trust-axl-pickle for trusted official object records")
        records = iter_axl(source, trust_pickle=True, sample_id=sample_id)
    else:
        raise ValueError("Unsupported dataset: " + str(dataset))
    output_dir = Path(output_dir)
    # Fresh directory also prevents accidentally mixing subsets or weight modes.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": dataset, "source": str(source),
                "weights": "ground-truth" if use_gt_weights else "predicted",
                "records": []}
    for key, ordinal, vertices, faces, gt, skeleton, names in records:
        key = _sample_id(key)
        if use_gt_weights:
            skin, prediction_path = gt.copy(), None
        else:
            prediction_path = Path(predictions_dir) / (key + "_skin.npy")
            skin = validate_skin(np.load(prediction_path, allow_pickle=False), len(vertices), gt.shape[1])
        arrays = {"vertices": vertices, "faces": faces, "skin": skin,
                  "gt_skin": gt, **skeleton, "joint_names": np.asarray(names, dtype=str)}
        metadata = {"dataset": dataset, "sample_id": key, "source_index": ordinal,
                    "weights": manifest["weights"], "prediction": str(prediction_path) if prediction_path else None}
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
        filename = key + ".npz"
        with (output_dir / filename).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        manifest["records"].append({"sample_id": key, "source_index": ordinal, "file": filename,
                                     "vertex_count": len(vertices), "joint_count": gt.shape[1]})
    if not manifest["records"]:
        raise ValueError("No matching records; check dataset path and sample ID")
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest
