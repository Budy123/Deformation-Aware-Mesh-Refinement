"""Paper metrics evaluated against unmodified GT at retained original indices."""

import numpy as np

from . import _metric_kernels as K
from .core import validate_geometry, validate_skin


def _validated_skeleton(skeleton, joint_count):
    matrices = np.asarray(skeleton["matrix_local"], dtype=np.float64)
    parents = np.asarray(skeleton["parents"])
    if matrices.shape != (joint_count, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("matrix_local must contain finite (J, 4, 4) global rest transforms")
    if parents.shape != (joint_count,) or parents.dtype.kind not in "iu":
        raise ValueError("parents must be an integer (J,) array")
    if np.any(parents < -1) or np.any(parents >= np.arange(joint_count)):
        raise ValueError("parents must use -1 for roots and precede children in joint order")
    if not np.allclose(matrices[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-10):
        raise ValueError("rest transforms must be affine homogeneous matrices")
    try:
        np.linalg.inv(matrices)
    except np.linalg.LinAlgError as exc:
        raise ValueError("rest transforms must be invertible") from exc
    return {"matrix_local": matrices, "parents": parents.astype(np.int64)}


def evaluate_mesh(vertices, faces, gt_skin, refined, *, skeleton=None,
                  pose_count=64, pose_angle_deg=30.0, seed=321, sample_index=0):
    """Evaluate one analysis/evaluation phase dictionary from ``refine_mesh``.

    Skin L1 is a normalized vertex/joint entry mean, not a joint-summed mean.
    DE is mean Euclidean distance, not coordinate MSE. Return None when a
    metric is unavailable; validity counts are explicit. Macro-average valid
    per-mesh scores to reproduce dataset reporting, not all vertices at once.
    """
    vertices, faces = validate_geometry(vertices, faces)
    gt_skin = validate_skin(gt_skin, len(vertices))
    out_vertices, out_faces = validate_geometry(refined["vertices"], refined["faces"])
    out_skin = validate_skin(refined["skin"], len(out_vertices), gt_skin.shape[1])
    keep = np.asarray(refined["keep_indices"])
    if keep.shape != (len(out_vertices),) or keep.dtype.kind not in "iu":
        raise ValueError("keep_indices must be integer indices aligned with refined vertices")
    if keep.size and (keep.min() < 0 or keep.max() >= len(vertices)):
        raise ValueError("keep_indices out of range")
    if len(np.unique(keep)) != len(keep) or not np.array_equal(vertices[keep], out_vertices):
        raise ValueError("refined vertices must match distinct retained original vertices")
    if isinstance(pose_count, bool) or int(pose_count) != pose_count or pose_count <= 0:
        raise ValueError("pose_count must be a positive integer")
    if int(sample_index) != sample_index or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(pose_angle_deg) or pose_angle_deg < 0:
        raise ValueError("pose_angle_deg must be finite and non-negative")

    gt_retained = gt_skin[keep]
    skin = K._skin_l1_for_skin_pair(gt_retained, out_skin)
    valid_skin = skin["skin_pred_valid_row_count"]
    result = {
        "vertex_count": len(out_vertices), "face_count": len(out_faces),
        "joint_count": gt_skin.shape[1], "skin_valid_vertex_count": valid_skin,
        "skin_l1": skin["skin_pred_unirig_loss"] if valid_skin else None,
        "skin_l1_joint_sum": skin["skin_pred_unirig_loss"] * gt_skin.shape[1] if valid_skin else None,
        "skin_l1_joint_sum_raw": skin["skin_pred_loss"] if valid_skin else None,
        "fpr": len(out_faces) / len(faces) if len(faces) else None,
        "nme": K._non_manifold_edge_count_python(out_faces),
        "cd": None, "cd_squared": None, "de": None, "de_mse": None,
        "cd_backend": "scipy_float64" if K.cKDTree is not None else "numpy_float32_fallback",
        "de_valid_vertex_count": None,
        "pose_count": int(pose_count), "pose_angle_deg": float(pose_angle_deg),
        "seed": int(seed), "sample_index": int(sample_index),
        "pose_base_seed": int(seed) + int(sample_index) * 15991,
    }
    if len(vertices) and len(out_vertices):
        a, _ = K._nearest_neighbor_query(vertices, out_vertices)
        b, _ = K._nearest_neighbor_query(out_vertices, vertices)
        result["cd"] = 0.5 * (float(np.sqrt(a).mean()) + float(np.sqrt(b).mean()))
        result["cd_squared"] = 0.5 * (float(a.mean()) + float(b.mean()))
    if skeleton is not None:
        skeleton = _validated_skeleton(skeleton, gt_skin.shape[1])
        poses = K._sample_pose_matrices(skeleton, pose_count=int(pose_count),
                    pose_angle_deg=float(pose_angle_deg), base_seed=result["pose_base_seed"])
        deformation = K._unirig_vertex_loss_for_skin_pair(
            out_vertices, gt_retained, out_skin, skeleton, poses)
        result["de_valid_vertex_count"] = deformation.get("deformation_pred_valid_vertex_count", 0)
        if deformation["deformation_pred_supported"]:
            result["de"] = deformation["deformation_pred_vertex_loss_l2"]
            result["de_mse"] = deformation["deformation_pred_vertex_loss_mse"]
    return result
