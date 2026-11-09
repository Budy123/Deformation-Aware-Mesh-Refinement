"""Validated public interface with the paper-final fusion defaults."""

import numpy as np

from ._topology import _refine_mesh
from .fusion import SkinningFusionConfig


def paper_fusion_config():
    """Self-included L2 Gaussian support, h=max(0.5*group_RMS, 1e-12)."""
    return SkinningFusionConfig()


def validate_geometry(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (F, 3) array")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face index out of range")
    return vertices, faces.astype(np.int64, copy=False)


def validate_skin(skin, vertex_count, joint_count=None):
    skin = np.asarray(skin)
    if skin.dtype.kind not in "fiu" or skin.ndim != 2 or skin.shape[0] != vertex_count:
        raise ValueError("skin must be a numeric (N, J) array aligned with vertices")
    if skin.shape[1] < 1 or (joint_count is not None and skin.shape[1] != joint_count):
        raise ValueError("skin joint count mismatch or zero joints")
    if not np.isfinite(skin).all() or np.any(skin < 0):
        raise ValueError("skin must contain finite non-negative weights")
    return skin


def refine_mesh(vertices, faces, skin, *, fusion_config=None):
    """Return production analysis/evaluation meshes and original-index mappings.

    ``skin`` is the input prediction, not a GT reference. No inference is run.
    Coordinates are not rounded or moved. Output skin uses production float32.
    Invalid (all-zero) skin rows are kept for production fallback/diagnostics.
    """
    vertices, faces = validate_geometry(vertices, faces)
    skin = validate_skin(skin, len(vertices))
    # Match the production evaluator's input dtype before calling its kernel.
    with np.errstate(over="ignore"):
        skin = skin.astype(np.float32)
    if not np.isfinite(skin).all():
        raise ValueError("skin cannot be represented as finite float32")
    return _refine_mesh(
        vertices, faces, skin,
        skin_fusion_config=paper_fusion_config() if fusion_config is None else fusion_config)
