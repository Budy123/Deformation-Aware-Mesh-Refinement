"""Unchanged production metric kernels, without dataset/inference dependencies."""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

EPS = 1e-12
DEFAULT_NN_CHUNK = 1024

def _axis_angle_to_matrix_batch(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(f"[SURFACE-METRICS] axis_angle must be (J, 3), got {axis_angle.shape}")

    angle = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    axis = np.divide(axis_angle, np.maximum(angle, EPS), out=np.zeros_like(axis_angle), where=angle > EPS)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    cos = np.cos(angle[:, 0])
    sin = np.sin(angle[:, 0])
    one_minus_cos = 1.0 - cos

    rot = np.zeros((axis_angle.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = cos + x * x * one_minus_cos
    rot[:, 0, 1] = x * y * one_minus_cos - z * sin
    rot[:, 0, 2] = x * z * one_minus_cos + y * sin
    rot[:, 1, 0] = y * x * one_minus_cos + z * sin
    rot[:, 1, 1] = cos + y * y * one_minus_cos
    rot[:, 1, 2] = y * z * one_minus_cos - x * sin
    rot[:, 2, 0] = z * x * one_minus_cos - y * sin
    rot[:, 2, 1] = z * y * one_minus_cos + x * sin
    rot[:, 2, 2] = cos + z * z * one_minus_cos

    small = (angle[:, 0] <= EPS)
    if np.any(small):
        rot[small] = np.eye(3, dtype=np.float64)
    return rot


def _sample_unirig_style_pose_basis(
    joint_count: int,
    *,
    angle_deg: float,
    rng_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(rng_seed))
    axis_angle = (rng.random((int(joint_count), 3)) - 0.5) * float(angle_deg) / 180.0 * np.pi * 2.0
    rot = _axis_angle_to_matrix_batch(axis_angle)
    basis = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], int(joint_count), axis=0)
    basis[:, :3, :3] = rot
    return basis


def _compose_pose_matrix(
    matrix_local: np.ndarray,
    parents: np.ndarray,
    matrix_basis: np.ndarray,
) -> np.ndarray:
    matrix_local = np.asarray(matrix_local, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    matrix_basis = np.asarray(matrix_basis, dtype=np.float64)
    joint_count = int(matrix_local.shape[0])
    matrix = np.zeros((joint_count, 4, 4), dtype=np.float64)
    for joint_idx in range(joint_count):
        parent_idx = int(parents[joint_idx]) if joint_idx < parents.shape[0] else -1
        if parent_idx < 0:
            matrix[joint_idx] = matrix_local[joint_idx] @ matrix_basis[joint_idx]
        else:
            matrix[joint_idx] = (
                matrix[parent_idx]
                @ (np.linalg.inv(matrix_local[parent_idx]) @ matrix_local[joint_idx])
                @ matrix_basis[joint_idx]
            )
    return matrix


def _linear_blend_skinning_np(
    vertex: np.ndarray,
    matrix_local: np.ndarray,
    matrix: np.ndarray,
    skin: np.ndarray,
    *,
    pad: int = 1,
    value: float = 1.0,
) -> np.ndarray:
    vertex = np.asarray(vertex, dtype=np.float64)
    matrix_local = np.asarray(matrix_local, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    skin = np.asarray(skin, dtype=np.float64)
    if vertex.ndim != 2 or vertex.shape[1] + int(pad) != 4:
        raise ValueError(f"[SURFACE-METRICS] vertex must be (N, {4 - int(pad)}), got {vertex.shape}")
    joint_count = int(matrix_local.shape[0])
    vertex_count = int(vertex.shape[0])
    padded = np.pad(vertex, ((0, 0), (0, int(pad))), mode="constant", constant_values=(0.0, float(value))).T
    trans = matrix @ np.linalg.inv(matrix_local)
    weighted = np.zeros((joint_count, 4, vertex_count), dtype=np.float64)
    mask = (skin > 0).T
    skin_t = skin.T
    for joint_idx in range(joint_count):
        active = mask[joint_idx]
        if not np.any(active):
            continue
        weighted[joint_idx][:, active] = (
            trans[joint_idx] @ padded[:, active]
        ) * skin_t[joint_idx, active]
    g = np.sum(weighted, axis=0)
    denom = np.sum(skin, axis=1) + 1e-8
    final = g[:3, :] / denom[None, :]
    return final.T


def _sample_pose_matrices(
    skeleton: Optional[Dict[str, Any]],
    *,
    pose_count: int,
    pose_angle_deg: float,
    base_seed: int,
) -> Optional[np.ndarray]:
    if skeleton is None or int(pose_count) <= 0:
        return None
    matrix_local = np.asarray(skeleton["matrix_local"], dtype=np.float64)
    parents = np.asarray(skeleton["parents"], dtype=np.int64)
    pose_matrices: List[np.ndarray] = []
    for pose_idx in range(int(pose_count)):
        matrix_basis = _sample_unirig_style_pose_basis(
            matrix_local.shape[0],
            angle_deg=float(pose_angle_deg),
            rng_seed=int(base_seed) + pose_idx * 65537,
        )
        pose_matrices.append(_compose_pose_matrix(matrix_local, parents, matrix_basis))
    if not pose_matrices:
        return None
    return np.stack(pose_matrices, axis=0)


def _unirig_vertex_loss_for_skin_pair(
    vertices: np.ndarray,
    gt_skin: np.ndarray,
    pred_skin: np.ndarray,
    skeleton: Optional[Dict[str, Any]],
    pose_matrices: Optional[np.ndarray],
) -> Dict[str, Any]:
    if skeleton is None or pose_matrices is None:
        return {"deformation_pred_supported": False}
    vertices = np.asarray(vertices, dtype=np.float64)
    gt_skin = np.asarray(gt_skin, dtype=np.float64)
    pred_skin = np.asarray(pred_skin, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return {"deformation_pred_supported": False}
    if gt_skin.shape != pred_skin.shape or gt_skin.shape[0] != vertices.shape[0]:
        return {"deformation_pred_supported": False}
    gt_row_sum = gt_skin.sum(axis=1)
    pred_row_sum = pred_skin.sum(axis=1)
    valid_row_mask = (gt_row_sum > EPS) & (pred_row_sum > EPS)
    invalid_gt_count = int((gt_row_sum <= EPS).sum())
    invalid_pred_count = int((pred_row_sum <= EPS).sum())
    if not np.any(valid_row_mask):
        return {
            "deformation_pred_supported": False,
            "deformation_pred_valid_vertex_count": 0,
            "deformation_pred_invalid_gt_skin_row_count": invalid_gt_count,
            "deformation_pred_invalid_pred_skin_row_count": invalid_pred_count,
        }
    vertices_eval = vertices[valid_row_mask]
    gt_skin_eval = gt_skin[valid_row_mask]
    pred_skin_eval = pred_skin[valid_row_mask]

    matrix_local = np.asarray(skeleton["matrix_local"], dtype=np.float64)
    mse_values: List[float] = []
    l2_values: List[float] = []
    for pose_matrix in np.asarray(pose_matrices, dtype=np.float64):
        vertices_gt = _linear_blend_skinning_np(
            vertices_eval,
            matrix_local,
            pose_matrix,
            gt_skin_eval,
            pad=1,
            value=1.0,
        )
        vertices_pred = _linear_blend_skinning_np(
            vertices_eval,
            matrix_local,
            pose_matrix,
            pred_skin_eval,
            pad=1,
            value=1.0,
        )
        diff = vertices_gt - vertices_pred
        mse_values.append(float(np.mean(np.square(diff))))
        l2_values.append(float(np.linalg.norm(diff, axis=1).mean()))
    return {
        "deformation_pred_supported": True,
        "deformation_pred_vertex_loss_mse": float(np.mean(mse_values)),
        "deformation_pred_vertex_loss_l2": float(np.mean(l2_values)),
        "deformation_pred_valid_vertex_count": int(valid_row_mask.sum()),
        "deformation_pred_valid_vertex_ratio": float(valid_row_mask.sum() / max(int(vertices.shape[0]), 1)),
        "deformation_pred_invalid_gt_skin_row_count": invalid_gt_count,
        "deformation_pred_invalid_pred_skin_row_count": invalid_pred_count,
    }


def _normalize_skin_rows(skin: np.ndarray) -> np.ndarray:
    skin = np.asarray(skin, dtype=np.float64)
    row_sum = skin.sum(axis=1, keepdims=True)
    out = np.zeros_like(skin, dtype=np.float64)
    valid = np.abs(row_sum[:, 0]) > EPS
    out[valid] = skin[valid] / row_sum[valid]
    return out


def _skin_l1_for_skin_pair(gt_skin: np.ndarray, pred_skin: np.ndarray) -> Dict[str, Any]:
    gt_skin = np.asarray(gt_skin, dtype=np.float64)
    pred_skin = np.asarray(pred_skin, dtype=np.float64)
    if gt_skin.ndim != 2 or pred_skin.ndim != 2 or gt_skin.shape != pred_skin.shape:
        return {"skin_pred_supported": False}

    diff = np.abs(pred_skin - gt_skin)
    vertex_l1 = diff.sum(axis=1)
    gt_row_sum = gt_skin.sum(axis=1)
    valid_row_mask = np.abs(gt_row_sum - 1.0) < 1e-2
    if np.any(valid_row_mask):
        skin_l1 = float(vertex_l1[valid_row_mask].mean())
    else:
        skin_l1 = 0.0

    gt_norm = _normalize_skin_rows(gt_skin)
    pred_norm = _normalize_skin_rows(pred_skin)
    unirig_diff = np.abs(pred_norm - gt_norm)
    if np.any(valid_row_mask):
        skin_l1_unirig = float(unirig_diff[valid_row_mask].mean())
    else:
        skin_l1_unirig = 0.0
    return {
        "skin_pred_supported": True,
        "skin_pred_loss": skin_l1,
        "skin_pred_unirig_loss": skin_l1_unirig,
        "skin_pred_valid_row_count": int(valid_row_mask.sum()),
        "skin_pred_vertex_count": int(gt_skin.shape[0]),
        "skin_pred_joint_count": int(gt_skin.shape[1]),
    }


def _non_manifold_edge_count_python(faces: np.ndarray) -> int:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0
    edge_face_count: Dict[Tuple[int, int], int] = defaultdict(int)
    for tri in faces:
        tri_int = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((tri_int[0], tri_int[1]), (tri_int[1], tri_int[2]), (tri_int[2], tri_int[0])):
            if int(u) == int(v):
                # A collapsed edge is non-manifold for our cleanup diagnostics.
                edge = (int(u), int(v))
            else:
                edge = (min(int(u), int(v)), max(int(u), int(v)))
            edge_face_count[edge] += 1
    return int(sum(1 for count in edge_face_count.values() if int(count) != 2))


def _nearest_neighbor_query(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    chunk_size: int = DEFAULT_NN_CHUNK,
) -> Tuple[np.ndarray, np.ndarray]:
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"[SURFACE-METRICS] src_points must be (N,3), got {src.shape}")
    if dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError(f"[SURFACE-METRICS] dst_points must be (M,3), got {dst.shape}")
    if src.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    if dst.shape[0] == 0:
        return (
            np.full((src.shape[0],), np.inf, dtype=np.float64),
            np.full((src.shape[0],), -1, dtype=np.int64),
        )

    if cKDTree is not None:
        tree = cKDTree(dst)
        dist, idx = tree.query(src, k=1, eps=0.0, workers=1)
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
        return dist * dist, idx

    dst32 = dst.astype(np.float32, copy=False)
    dst_sq = np.sum(dst32 * dst32, axis=1, keepdims=True).T
    out_dist_sq = np.empty((src.shape[0],), dtype=np.float64)
    out_index = np.empty((src.shape[0],), dtype=np.int64)
    chunk_size = max(int(chunk_size), 1)

    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        src_chunk32 = src[start:end].astype(np.float32, copy=False)
        src_sq = np.sum(src_chunk32 * src_chunk32, axis=1, keepdims=True)
        dist_sq = src_sq + dst_sq - 2.0 * (src_chunk32 @ dst32.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        best_idx = np.argmin(dist_sq, axis=1)
        best_dist_sq = dist_sq[np.arange(end - start), best_idx]
        out_dist_sq[start:end] = best_dist_sq.astype(np.float64, copy=False)
        out_index[start:end] = best_idx.astype(np.int64, copy=False)

    return out_dist_sq, out_index
