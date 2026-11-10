"""Paper/revision metric cohorts, without research-runner dependencies."""

import numpy as np

from . import _metric_kernels as K
from ._topology import (_position_groups, _compute_incident_face_normals,
                        _has_reverse_face_conflict, _partition_pairwise_compatible_groups,
                        _canonical_cyclic_face_key)


def local_target_indices(vertices, faces):
    """All original vertices in groups split by the production orientation rule."""
    groups, ids = _position_groups(vertices)
    valid, signatures, _ = _compute_incident_face_normals(vertices, faces, ids)
    targets = []
    for group in groups:
        compatible = np.eye(len(group), dtype=bool)
        for a, first in enumerate(group):
            for b in range(a + 1, len(group)):
                second = group[b]
                ok = (valid[first] and valid[second] and
                      not _has_reverse_face_conflict(signatures[first], signatures[second]))
                compatible[a, b] = compatible[b, a] = ok
        if len(_partition_pairwise_compatible_groups(group, compatible)) > 1:
            targets.extend(group)
    return np.asarray(sorted(targets), dtype=np.int64)


def source_mapping(refined, source_count):
    value = refined.get("merged_old_to_new")
    if value is None:
        return None
    mapping = np.asarray(value)
    count = len(refined["vertices"])
    if (mapping.shape != (source_count,) or mapping.dtype.kind not in "iu" or
            np.any(mapping < -1) or np.any(mapping >= count)):
        raise ValueError("merged_old_to_new must map every source vertex to -1 or an output index")
    keep = np.asarray(refined["keep_indices"])
    if not np.array_equal(mapping[keep], np.arange(count)):
        raise ValueError("merged_old_to_new disagrees with keep_indices")
    return mapping


def _face_buckets(vertices, faces):
    _, ids = np.unique(vertices, axis=0, return_inverse=True)
    buckets = {}
    for index, face in enumerate(faces):
        tri = tuple(int(x) for x in ids[face])
        if len(set(tri)) != 3:
            continue
        key = tuple(sorted(tri))
        parity = int(_canonical_cyclic_face_key(tri) != key)
        buckets.setdefault(key, ([], []))[parity].append(index)
    return buckets


def opposite_face_preservation(vertices, faces, refined):
    """Fixed original opposite pairs, excluding already shared index sets."""
    source_pairs = []
    for front, back in _face_buckets(vertices, faces).values():
        for a in front:
            for b in back:
                if set(faces[a]) != set(faces[b]):
                    source_pairs.append((a, b))
    result = {"ofpr": None, "ofpr_source_pair_count": len(source_pairs),
              "ofpr_preserved_pair_count": None}
    kept = refined.get("kept_source_face_indices")
    if kept is None:
        return result
    kept = np.asarray(kept)
    out_faces = np.asarray(refined["faces"])
    if (kept.shape != (len(out_faces),) or kept.dtype.kind not in "iu" or
            np.any(kept < 0) or np.any(kept >= len(faces)) or len(np.unique(kept)) != len(kept)):
        raise ValueError("kept_source_face_indices must uniquely identify each output face")
    if "merged_old_to_new" in refined:
        mapping = np.asarray(refined["merged_old_to_new"])
        if not np.array_equal(out_faces, mapping[faces[kept]]):
            raise ValueError("output faces disagree with source-face and vertex mappings")
    source_to_final = {int(source): i for i, source in enumerate(kept)}
    preserved = 0
    for a, b in source_pairs:
        if a not in source_to_final or b not in source_to_final:
            continue
        first, second = out_faces[[source_to_final[a], source_to_final[b]]]
        if len(set(first)) == 3 and len(set(second)) == 3 and set(first) != set(second):
            preserved += 1
    result.update(ofpr=preserved / len(source_pairs) if source_pairs else None,
                  ofpr_preserved_pair_count=preserved)
    return result


def local_errors(vertices, faces, gt_skin, refined, mapping, skeleton, poses):
    targets = local_target_indices(vertices, faces)
    result = {"local_skin_l1": None, "local_de": None,
              "local_source_vertex_count": len(targets), "local_mapped_vertex_count": None,
              "local_skin_valid_vertex_count": None, "local_de_valid_vertex_count": None}
    if mapping is None:
        return result
    full_count = len(targets)
    targets = targets[mapping[targets] >= 0]
    result["local_mapped_vertex_count"] = len(targets)
    if len(targets) != full_count:
        # A partial target average is not the Table II metric.
        return result
    truth = np.asarray(gt_skin[targets], dtype=np.float64)
    pred = np.asarray(refined["skin"][mapping[targets]], dtype=np.float64)
    # Table II uses normalized per-joint L1, valid when both rows are nonzero.
    valid = (truth.sum(axis=1) > K.EPS) & (pred.sum(axis=1) > K.EPS)
    count = int(valid.sum())
    result["local_skin_valid_vertex_count"] = count
    if count:
        result["local_skin_l1"] = float(np.abs(
            K._normalize_skin_rows(pred[valid]) - K._normalize_skin_rows(truth[valid])).mean())
    if skeleton is not None:
        de = K._unirig_vertex_loss_for_skin_pair(vertices[targets], truth, pred, skeleton, poses)
        result["local_de_valid_vertex_count"] = de.get("deformation_pred_valid_vertex_count", 0)
        if de["deformation_pred_supported"]:
            result["local_de"] = de["deformation_pred_vertex_loss_l2"]
    return result


def _normalized_rows(skin):
    skin = np.asarray(skin, dtype=np.float64)
    sums = skin.sum(axis=1)
    valid = np.isfinite(skin).all(axis=1) & (skin >= 0).all(axis=1)
    valid &= np.isfinite(sums) & (sums > K.EPS)
    rows = np.zeros_like(skin)
    rows[valid] = skin[valid] / sums[valid, None]
    return rows, valid


def _index_edges(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def seam_source_pairs(vertices, faces, gt_skin):
    """All unordered coincident edge-copy pairs with matching normalized GT."""
    gt, valid_gt = _normalized_rows(gt_skin)
    _, ids = np.unique(vertices, axis=0, return_inverse=True)
    edges = _index_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    edges = edges[lengths > 0].copy()
    reverse = ids[edges[:, 0]] > ids[edges[:, 1]]
    edges[reverse] = edges[reverse, ::-1]
    buckets = {}
    for edge in edges:
        buckets.setdefault(tuple(ids[edge]), []).append(edge)
    chunks, length_chunks = [], []
    for copies in buckets.values():
        copies = np.asarray(copies, dtype=np.int64)
        for first in range(len(copies) - 1):
            a = copies[first]
            for start in range(first + 1, len(copies), 4096):
                others = copies[start:start + 4096]
                valid = valid_gt[a].all() & valid_gt[others].all(axis=1)
                valid &= np.max(np.abs(gt[a][None] - gt[others]), axis=(1, 2)) <= 1e-6
                selected = others[valid]
                if len(selected):
                    chunks.append(np.stack((np.broadcast_to(a, selected.shape), selected), axis=1))
                    length_chunks.append(np.full(len(selected), np.linalg.norm(vertices[a[1]] - vertices[a[0]])))
    pairs = np.concatenate(chunks) if chunks else np.empty((0, 2, 2), dtype=np.int64)
    lengths = np.concatenate(length_chunks) if chunks else np.empty(0, dtype=np.float64)
    return pairs, lengths


def map_seam_pairs(pairs, state, mapping):
    """Require both mapped copies to survive as actual, noncollapsed edges."""
    mapped = mapping[pairs]
    valid = (mapped >= 0).all(axis=(1, 2))
    chosen = np.flatnonzero(valid)
    edges = _index_edges(np.asarray(state["faces"], dtype=np.int64))
    key_dtype = np.dtype([("a", np.int64), ("b", np.int64)])
    keys = np.ascontiguousarray(edges).view(key_dtype).reshape(-1)
    _, weights_valid = _normalized_rows(state["skin"])
    for start in range(0, len(chosen), 32768):
        indices = chosen[start:start + 32768]
        copies = mapped[indices]
        points = state["vertices"][copies]
        ok = np.linalg.norm(points[:, :, 1] - points[:, :, 0], axis=2).min(axis=1) > 0
        ok &= weights_valid[copies].all(axis=(1, 2))
        query = np.ascontiguousarray(np.sort(copies, axis=2)).view(key_dtype).reshape(-1)
        positions = np.searchsorted(keys, query)
        matches = np.zeros(len(query), dtype=bool)
        inside = positions < len(keys)
        matches[inside] = keys[positions[inside]] == query[inside]
        valid[indices] = ok & matches.reshape(-1, 2).all(axis=1)
    return mapped, valid


def edge_integral(delta0, delta1):
    """Exact integral of squared distance between corresponding straight edges."""
    return ((np.square(delta0 + delta1).sum(axis=-1) / 4) +
            (np.square(delta1 - delta0).sum(axis=-1) / 12))


def seam_scores(pairs, lengths, states, mappings, skeleton, poses):
    mapped, masks = {}, {}
    for name, state in states.items():
        mapped[name], masks[name] = map_seam_pairs(pairs, state, mappings[name])
    common = np.logical_and.reduce(list(masks.values()))
    weights = lengths[common]
    scores = {}
    for name, state in states.items():
        value = None
        if len(weights) and poses is not None:
            unique, inverse = np.unique(mapped[name][common], return_inverse=True)
            inverse = inverse.reshape(-1, 2, 2)
            total = np.zeros(len(weights), dtype=np.float64)
            for pose in poses:
                posed = K._linear_blend_skinning_np(state["vertices"][unique], skeleton["matrix_local"],
                                                   pose, state["skin"][unique])
                for start in range(0, len(weights), 32768):
                    points = posed[inverse[start:start + 32768]]
                    delta = points[:, 0] - points[:, 1]
                    total[start:start + len(delta)] += edge_integral(delta[:, 0], delta[:, 1])
            value = float(weights @ (total / len(poses)) / weights.sum())
        scores[name] = {"so": value, "valid_pair_count": int(masks[name].sum()),
                        "missing_pair_count": int((~masks[name]).sum())}
    return {"source_pair_count": len(pairs), "common_pair_count": int(common.sum()),
            "common_pair_length": float(weights.sum()), "gt_atol": 1e-6,
            "methods": scores}
