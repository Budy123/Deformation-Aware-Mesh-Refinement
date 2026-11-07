"""Deformation-aware refinement of duplicate-position rigged mesh vertices."""

from .core import paper_fusion_config, refine_mesh
from .fusion import SkinningFusionConfig, fuse_skinning_weights
from .metrics import evaluate_mesh

__version__ = "0.1.0"
__all__ = ["refine_mesh", "evaluate_mesh", "paper_fusion_config",
           "SkinningFusionConfig", "fuse_skinning_weights"]
