"""Production topology kernels; public callers should use core.refine_mesh."""
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .fusion import (DEFAULT_L2_KDE_BANDWIDTH, SkinningFusionConfig,
    format_skinning_fusion_tag, fuse_skinning_weights,
    validate_l2_kde_bandwidth, validate_skinning_fusion_config)

EPS = 1e-12

def _position_groups(vertices: np.ndarray) -> List[List[int]]:
    groups: Dict[Tuple[float, float, float], List[int]] = {}
    for idx, vertex in enumerate(np.asarray(vertices)):
        key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        groups.setdefault(key, []).append(int(idx))
    return [[int(i) for i in indices] for indices in groups.values() if len(indices) > 1]


def _position_group_ids(vertices: np.ndarray) -> np.ndarray:
    group_ids = np.full(int(np.asarray(vertices).shape[0]), -1, dtype=np.int64)
    groups: Dict[Tuple[float, float, float], List[int]] = {}
    for idx, vertex in enumerate(np.asarray(vertices)):
        key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        groups.setdefault(key, []).append(int(idx))
    for gid, indices in enumerate(groups.values()):
        group_ids[np.asarray(indices, dtype=np.int64)] = int(gid)
    return group_ids


def _canonical_cyclic_face_key(tri: Sequence[int]) -> Tuple[int, int, int]:
    a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
    return min((a, b, c), (b, c, a), (c, a, b))


def _face_geom_key_and_orientation(tri: Sequence[int]) -> Tuple[Tuple[int, int, int], int]:
    same = _canonical_cyclic_face_key(tri)
    reverse = _canonical_cyclic_face_key((int(tri[0]), int(tri[2]), int(tri[1])))
    if same <= reverse:
        return same, 0
    return reverse, 1


def _compute_incident_face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    position_group_ids: np.ndarray,
) -> Tuple[List[np.ndarray], List[Dict[Tuple[int, int, int], set]], int]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    incident = [np.zeros((0, 3), dtype=np.float64) for _ in range(vertices.shape[0])]
    incident_signatures: List[Dict[Tuple[int, int, int], set]] = [{} for _ in range(vertices.shape[0])]
    if faces.size == 0:
        return incident, incident_signatures, 0

    tri0 = vertices[faces[:, 0]]
    tri1 = vertices[faces[:, 1]]
    tri2 = vertices[faces[:, 2]]
    raw = np.cross(tri1 - tri0, tri2 - tri0)
    norm = np.linalg.norm(raw, axis=1)
    valid = norm > EPS

    buckets: List[List[np.ndarray]] = [[] for _ in range(vertices.shape[0])]
    for tri, normal in zip(faces[valid], raw[valid] / norm[valid, None]):
        buckets[int(tri[0])].append(normal)
        buckets[int(tri[1])].append(normal)
        buckets[int(tri[2])].append(normal)

        pos_tri = (
            int(position_group_ids[int(tri[0])]),
            int(position_group_ids[int(tri[1])]),
            int(position_group_ids[int(tri[2])]),
        )
        if len({pos_tri[0], pos_tri[1], pos_tri[2]}) != 3:
            continue
        geom_key, orientation = _face_geom_key_and_orientation(pos_tri)
        for vertex_idx in tri.tolist():
            incident_signatures[int(vertex_idx)].setdefault(geom_key, set()).add(int(orientation))

    return [
        np.stack(items, axis=0).astype(np.float64) if items else np.zeros((0, 3), dtype=np.float64)
        for items in buckets
    ], incident_signatures, int((~valid).sum())


def _min_incident_face_angle_deg(normals_a: np.ndarray, normals_b: np.ndarray) -> Optional[float]:
    if normals_a.shape[0] == 0 or normals_b.shape[0] == 0:
        return None
    max_dot = float(np.max(normals_a @ normals_b.T))
    max_dot = float(np.clip(max_dot, -1.0, 1.0))
    return float(np.degrees(np.arccos(max_dot)))


def _has_reverse_face_conflict(
    signatures_a: Dict[Tuple[int, int, int], set],
    signatures_b: Dict[Tuple[int, int, int], set],
) -> bool:
    shared_keys = set(signatures_a).intersection(signatures_b)
    for geom_key in shared_keys:
        orients_a = signatures_a[geom_key]
        orients_b = signatures_b[geom_key]
        if (0 in orients_a and 1 in orients_b) or (1 in orients_a and 0 in orients_b):
            return True
    return False


def _partition_pairwise_compatible_groups(indices: Sequence[int], compatible: np.ndarray) -> List[List[int]]:
    ordered = [int(idx) for idx in indices]
    index_to_local = {int(idx): local for local, idx in enumerate(ordered)}
    remaining = set(ordered)
    groups: List[List[int]] = []

    while remaining:
        seed = min(remaining)
        group = [int(seed)]
        remaining.remove(seed)
        candidates = sorted(remaining)
        for candidate in candidates:
            local_candidate = index_to_local[int(candidate)]
            can_join = True
            for member in group:
                local_member = index_to_local[int(member)]
                if not bool(compatible[local_candidate, local_member]):
                    can_join = False
                    break
            if can_join:
                group.append(int(candidate))
                remaining.remove(candidate)
        groups.append(group)
    return groups


def _finalize_mesh_output(
    *,
    vertices_after_merge: np.ndarray,
    skin_after_merge: np.ndarray,
    keep_indices_after_merge: np.ndarray,
    remapped_faces: np.ndarray,
    representative_old_index: np.ndarray,
    original_vertex_count: int,
    remove_degenerate: bool,
    remove_same_orientation_duplicates: bool,
    remove_reverse_duplicates: bool,
) -> Dict[str, Any]:
    kept_faces: List[np.ndarray] = []
    kept_source_face_indices: List[int] = []
    seen_oriented_keys = set()
    seen_geom_keys = set()

    encountered_degenerate = 0
    encountered_duplicate = 0
    encountered_reverse_duplicate = 0
    removed_degenerate = 0
    removed_duplicate = 0
    removed_reverse_duplicate = 0

    for source_face_idx, tri in enumerate(np.asarray(remapped_faces, dtype=np.int64)):
        tri_int = np.asarray(tri, dtype=np.int64)
        if len({int(tri_int[0]), int(tri_int[1]), int(tri_int[2])}) != 3:
            encountered_degenerate += 1
            if remove_degenerate:
                removed_degenerate += 1
                continue
            kept_faces.append(tri_int)
            kept_source_face_indices.append(int(source_face_idx))
            continue

        oriented_key = _canonical_cyclic_face_key(tri_int)
        geom_key, _ = _face_geom_key_and_orientation(tri_int)

        if oriented_key in seen_oriented_keys:
            encountered_duplicate += 1
            if remove_same_orientation_duplicates:
                removed_duplicate += 1
                continue
        if geom_key in seen_geom_keys:
            encountered_reverse_duplicate += 1
            if remove_reverse_duplicates:
                removed_reverse_duplicate += 1
                continue

        seen_oriented_keys.add(oriented_key)
        seen_geom_keys.add(geom_key)
        kept_faces.append(tri_int)
        kept_source_face_indices.append(int(source_face_idx))

    faces_out = np.stack(kept_faces, axis=0) if kept_faces else np.zeros((0, 3), dtype=np.int64)
    if faces_out.size > 0:
        used_indices = np.unique(faces_out.reshape(-1))
    else:
        used_indices = np.zeros((0,), dtype=np.int64)

    compact_old_to_new = np.full(vertices_after_merge.shape[0], -1, dtype=np.int64)
    compact_old_to_new[used_indices] = np.arange(len(used_indices), dtype=np.int64)
    compacted_faces = (
        compact_old_to_new[faces_out] if faces_out.size > 0 else np.zeros((0, 3), dtype=np.int64)
    )
    vertices_out = vertices_after_merge[used_indices]
    skin_out = skin_after_merge[used_indices]
    keep_indices_out = keep_indices_after_merge[used_indices]

    final_old_to_new = np.full(int(original_vertex_count), -1, dtype=np.int64)
    final_old_to_new[keep_indices_out] = np.arange(len(keep_indices_out), dtype=np.int64)
    merged_old_to_new = final_old_to_new[np.asarray(representative_old_index, dtype=np.int64)]

    return {
        "vertices": np.asarray(vertices_out),
        "faces": np.asarray(compacted_faces, dtype=np.int64),
        "kept_source_face_indices": np.asarray(kept_source_face_indices, dtype=np.int64),
        "skin": np.asarray(skin_out, dtype=np.float32),
        "keep_indices": np.asarray(keep_indices_out, dtype=np.int64),
        "old_to_new": final_old_to_new.astype(np.int64),
        "merged_old_to_new": merged_old_to_new.astype(np.int64),
        "removed_vertex_count_unused_after_face_cleanup": int(vertices_after_merge.shape[0] - len(used_indices)),
        "encountered_face_count_degenerate": int(encountered_degenerate),
        "encountered_face_count_duplicate": int(encountered_duplicate),
        "encountered_face_count_reverse_duplicate": int(encountered_reverse_duplicate),
        "removed_face_count_degenerate": int(removed_degenerate),
        "removed_face_count_duplicate": int(removed_duplicate),
        "removed_face_count_reverse_duplicate": int(removed_reverse_duplicate),
    }


def dedup_vertices_with_gt_skin_incident_face_threshold(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin: np.ndarray,
    *,
    atol: float,
    angle_threshold_deg: float,
    skin_kde_bandwidth: float = DEFAULT_L2_KDE_BANDWIDTH,
    skin_fusion_config: Optional[SkinningFusionConfig] = None,
) -> Dict[str, Any]:
    """Fuse and merge orientation-compatible exact-position duplicates.

    The original positional/keyword contract is preserved.  The optional KDE
    bandwidth is validated before topology inspection so invalid experiment
    settings fail even when the mesh contains no mergeable duplicate group.
    """

    skin_kde_bandwidth = validate_l2_kde_bandwidth(skin_kde_bandwidth)
    if skin_fusion_config is None:
        fusion_config = SkinningFusionConfig(
            policy="gaussian_l2",
            self_included=True,
            bandwidth_mode="fixed",
            bandwidth=skin_kde_bandwidth,
        )
    else:
        fusion_config = validate_skinning_fusion_config(skin_fusion_config)
    duplicate_items = _position_groups(vertices)
    position_group_ids = _position_group_ids(vertices)
    incident_normals, incident_signatures, degenerate_faces_total = _compute_incident_face_normals(
        vertices,
        faces,
        position_group_ids,
    )

    representative = np.arange(vertices.shape[0], dtype=np.int64)
    drop_mask = np.zeros(vertices.shape[0], dtype=bool)
    skin_after_fusion = np.asarray(skin, dtype=np.float32).copy()
    merge_groups: List[List[int]] = []
    protected_groups: List[List[int]] = []
    groups_blocked_by_normal: List[List[int]] = []
    invalid_normal_groups: List[List[int]] = []
    reverse_face_groups: List[List[int]] = []
    component_merge_groups: List[List[int]] = []
    protected_pairs_angle: List[List[int]] = []
    protected_pairs_invalid_incident: List[List[int]] = []
    protected_pairs_reverse_face: List[List[int]] = []
    skin_fusion_groups: List[Dict[str, Any]] = []
    invalid_pair_count = 0

    for indices in duplicate_items:
        ordered_indices = [int(index) for index in indices]
        compatible = np.eye(len(ordered_indices), dtype=bool)
        group_has_angle_block = False
        group_has_invalid = False
        group_has_reverse = False

        for local_a, local_b in itertools.combinations(range(len(ordered_indices)), 2):
            idx_a = int(ordered_indices[local_a])
            idx_b = int(ordered_indices[local_b])

            angle = _min_incident_face_angle_deg(incident_normals[idx_a], incident_normals[idx_b])
            if angle is None:
                invalid_pair_count += 1
                group_has_invalid = True
                compatible[local_a, local_b] = False
                compatible[local_b, local_a] = False
                protected_pairs_invalid_incident.append([idx_a, idx_b])
                continue

            if _has_reverse_face_conflict(incident_signatures[idx_a], incident_signatures[idx_b]):
                group_has_reverse = True
                compatible[local_a, local_b] = False
                compatible[local_b, local_a] = False
                protected_pairs_reverse_face.append([idx_a, idx_b])
                continue

            if angle > float(angle_threshold_deg):
                group_has_angle_block = True
                compatible[local_a, local_b] = False
                compatible[local_b, local_a] = False
                protected_pairs_angle.append([idx_a, idx_b])
                continue

            compatible[local_a, local_b] = True
            compatible[local_b, local_a] = True

        if group_has_invalid:
            invalid_normal_groups.append(ordered_indices)
        if group_has_angle_block:
            groups_blocked_by_normal.append(ordered_indices)
        if group_has_reverse:
            reverse_face_groups.append(ordered_indices)

        partitions = _partition_pairwise_compatible_groups(ordered_indices, compatible)
        if len(partitions) != 1 or len(partitions[0]) != len(ordered_indices):
            protected_groups.append(ordered_indices)

        for component in partitions:
            if len(component) <= 1:
                continue
            local_indices = [ordered_indices.index(int(idx)) for idx in component]
            for local_a, local_b in itertools.combinations(local_indices, 2):
                if not bool(compatible[local_a, local_b]):
                    raise RuntimeError(
                        f"Greedy clique partition produced incompatible merge pair in group {ordered_indices}: "
                        f"{ordered_indices[local_a]} vs {ordered_indices[local_b]}"
                    )

            fusion = fuse_skinning_weights(
                skin_after_fusion[np.asarray(component, dtype=np.int64)],
                config=fusion_config,
            )
            rep = int(component[0])
            skin_after_fusion[rep] = fusion.fused_skin
            skin_fusion_groups.append(fusion.diagnostics(component))
            merge_groups.append([int(i) for i in component])
            component_merge_groups.append([int(i) for i in component])
            for idx in component[1:]:
                representative[int(idx)] = rep
                drop_mask[int(idx)] = True

    keep_indices_after_merge = np.flatnonzero(~drop_mask)
    old_to_new_after_merge = np.full(vertices.shape[0], -1, dtype=np.int64)
    old_to_new_after_merge[keep_indices_after_merge] = np.arange(len(keep_indices_after_merge), dtype=np.int64)
    remapped_faces = old_to_new_after_merge[representative][np.asarray(faces, dtype=np.int64)]
    merged_vertices = np.asarray(vertices)[keep_indices_after_merge]
    merged_skin = skin_after_fusion[keep_indices_after_merge]

    analysis_output = _finalize_mesh_output(
        vertices_after_merge=merged_vertices,
        skin_after_merge=merged_skin,
        keep_indices_after_merge=keep_indices_after_merge,
        remapped_faces=remapped_faces,
        representative_old_index=representative,
        original_vertex_count=int(vertices.shape[0]),
        remove_degenerate=False,
        remove_same_orientation_duplicates=False,
        remove_reverse_duplicates=False,
    )
    evaluation_output = _finalize_mesh_output(
        vertices_after_merge=merged_vertices,
        skin_after_merge=merged_skin,
        keep_indices_after_merge=keep_indices_after_merge,
        remapped_faces=remapped_faces,
        representative_old_index=representative,
        original_vertex_count=int(vertices.shape[0]),
        remove_degenerate=True,
        remove_same_orientation_duplicates=True,
        remove_reverse_duplicates=True,
    )

    issue_types: List[str] = []
    if protected_pairs_invalid_incident:
        issue_types.append("pairwise_invalid_incident_blocked")
    if protected_pairs_reverse_face:
        issue_types.append("pairwise_reverse_face_conflict_blocked")
    if protected_pairs_angle:
        issue_types.append("pairwise_angle_conflict_blocked")
    if analysis_output["encountered_face_count_degenerate"] > 0:
        issue_types.append("degenerate_after_remap")
    if analysis_output["encountered_face_count_reverse_duplicate"] > 0:
        issue_types.append("reverse_duplicate_after_merge")
    if (
        int(analysis_output["vertices"].shape[0]) != int(evaluation_output["vertices"].shape[0])
        or int(analysis_output["faces"].shape[0]) != int(evaluation_output["faces"].shape[0])
    ):
        issue_types.append("analysis_evaluation_topology_diff")

    dual_output = bool(issue_types)
    if not dual_output:
        evaluation_output = {
            key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in analysis_output.items()
        }

    return {
        "analysis": analysis_output,
        "evaluation": evaluation_output,
        "output_mode": "dual_output" if dual_output else "analysis_only",
        "issue_types": issue_types,
        "keep_indices_after_merge": keep_indices_after_merge.astype(np.int64),
        "representative_old_index": representative.astype(np.int64),
        "old_to_new_after_merge": old_to_new_after_merge.astype(np.int64),
        "merge_groups": merge_groups,
        "protected_groups": protected_groups,
        "removed_vertex_count": int(drop_mask.sum()),
        "groups_blocked_by_normal": groups_blocked_by_normal,
        "invalid_normal_groups": invalid_normal_groups,
        "reverse_face_groups": reverse_face_groups,
        "pairs_without_valid_incident_faces": int(invalid_pair_count),
        "protected_pairs_angle": protected_pairs_angle,
        "protected_pairs_invalid_incident": protected_pairs_invalid_incident,
        "protected_pairs_reverse_face": protected_pairs_reverse_face,
        "component_merge_groups": component_merge_groups,
        "skin_fusion_method": (
            "uniform_mean" if fusion_config.policy == "uniform" else "l2_gaussian_kde"
        ),
        "skin_fusion_config": {
            "policy": fusion_config.policy,
            "self_included": bool(fusion_config.self_included),
            "bandwidth_mode": fusion_config.bandwidth_mode,
            "bandwidth": fusion_config.bandwidth,
            "adaptive_scale": fusion_config.adaptive_scale,
            "adaptive_bandwidth_floor": fusion_config.adaptive_bandwidth_floor,
            "identity_tag": format_skinning_fusion_tag(fusion_config),
        },
        "skin_kde_bandwidth": skin_kde_bandwidth,
        "skin_fusion_leave_one_out": not bool(fusion_config.self_included),
        "skin_fusion_groups": skin_fusion_groups,
        "input_degenerate_face_count": int(degenerate_faces_total),
        "merge_partition_strategy": "greedy_clique",
    }
