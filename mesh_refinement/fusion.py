"""Mesh-independent skinning-weight fusion policies.

This module owns only the question "how should one duplicate group's skin
rows become one representative row?" Geometry grouping, representative
selection, remapping, and face cleanup remain in exporters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_L2_KDE_BANDWIDTH = 0.1
DEFAULT_SIMPLEX_ATOL = 1e-6
DEFAULT_ADAPTIVE_BANDWIDTH_FLOOR = 1e-12
DEFAULT_FUSION_POLICY = "gaussian_l2"
DEFAULT_FUSION_BANDWIDTH_MODE = "fixed"
_ROW_SUM_EPS = 1e-12


def validate_l2_kde_bandwidth(bandwidth: float) -> float:
    """Return ``bandwidth`` as a validated finite positive Python float."""

    try:
        value = float(bandwidth)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bandwidth must be a finite positive scalar") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("bandwidth must be a finite positive value")
    return value


def format_l2_kde_bandwidth_tag(bandwidth: float) -> str:
    """Format a validated bandwidth as a deterministic filesystem-safe tag."""

    value = validate_l2_kde_bandwidth(bandwidth)
    token = repr(value).lower().replace("-", "m").replace("+", "").replace(".", "p")
    return f"bw{token}"


def _format_float_tag(value: float) -> str:
    return repr(float(value)).lower().replace("-", "m").replace("+", "").replace(".", "p")


@dataclass(frozen=True)
class SkinningFusionConfig:
    """Select one mesh-independent skin-row fusion policy."""

    policy: str = DEFAULT_FUSION_POLICY
    self_included: bool = True
    bandwidth_mode: str = DEFAULT_FUSION_BANDWIDTH_MODE
    bandwidth: Optional[float] = DEFAULT_L2_KDE_BANDWIDTH
    adaptive_scale: Optional[float] = None
    adaptive_bandwidth_floor: float = DEFAULT_ADAPTIVE_BANDWIDTH_FLOOR


def validate_skinning_fusion_config(config: SkinningFusionConfig) -> SkinningFusionConfig:
    """Validate and canonicalize a fusion policy configuration."""

    if not isinstance(config, SkinningFusionConfig):
        raise TypeError("config must be a SkinningFusionConfig")
    policy = str(config.policy)
    if policy not in {"uniform", "gaussian_l2"}:
        raise ValueError("policy must be one of: uniform, gaussian_l2")
    if not isinstance(config.self_included, (bool, np.bool_)):
        raise ValueError("self_included must be boolean")
    mode = str(config.bandwidth_mode)
    if mode not in {"fixed", "group_rms"}:
        raise ValueError("bandwidth_mode must be one of: fixed, group_rms")
    floor = validate_l2_kde_bandwidth(config.adaptive_bandwidth_floor)

    if policy == "uniform":
        return SkinningFusionConfig(
            policy=policy,
            self_included=bool(config.self_included),
            bandwidth_mode=mode,
            bandwidth=None,
            adaptive_scale=None,
            adaptive_bandwidth_floor=floor,
        )
    if mode == "fixed":
        if config.bandwidth is None:
            raise ValueError("fixed gaussian_l2 fusion requires bandwidth")
        return SkinningFusionConfig(
            policy=policy,
            self_included=bool(config.self_included),
            bandwidth_mode=mode,
            bandwidth=validate_l2_kde_bandwidth(config.bandwidth),
            adaptive_scale=None,
            adaptive_bandwidth_floor=floor,
        )
    if config.adaptive_scale is None:
        raise ValueError("group_rms gaussian_l2 fusion requires adaptive_scale")
    return SkinningFusionConfig(
        policy=policy,
        self_included=bool(config.self_included),
        bandwidth_mode=mode,
        bandwidth=None,
        adaptive_scale=validate_l2_kde_bandwidth(config.adaptive_scale),
        adaptive_bandwidth_floor=floor,
    )


def format_skinning_fusion_tag(config: SkinningFusionConfig) -> str:
    """Return a deterministic tag that isolates policy-specific outputs."""

    resolved = validate_skinning_fusion_config(config)
    if resolved.policy == "uniform":
        return "skin-fusion-uniform"
    inclusion = "self" if resolved.self_included else "loo"
    if resolved.bandwidth_mode == "fixed":
        assert resolved.bandwidth is not None
        return f"skin-fusion-gaussian-l2-{inclusion}-fixed-{format_l2_kde_bandwidth_tag(resolved.bandwidth)}"
    assert resolved.adaptive_scale is not None
    return f"skin-fusion-gaussian-l2-{inclusion}-group-rms-c{_format_float_tag(resolved.adaptive_scale)}"


@dataclass(frozen=True)
class SkinningFusionResult:
    """Result of fusing one group of skinning-weight observations."""

    fused_skin: np.ndarray
    coefficients: np.ndarray
    log_densities: np.ndarray
    valid_row_mask: np.ndarray
    pairwise_l2: np.ndarray
    bandwidth: Optional[float]
    leave_one_out: bool
    fallback_reason: Optional[str]
    policy: str = DEFAULT_FUSION_POLICY
    self_included: bool = True
    bandwidth_mode: str = DEFAULT_FUSION_BANDWIDTH_MODE
    configured_bandwidth: Optional[float] = None
    adaptive_scale: Optional[float] = None
    group_dispersion: Optional[float] = None

    def diagnostics(self, indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Return a JSON-friendly summary without serializing dense matrices."""

        group_size = int(self.coefficients.shape[0])
        if indices is None:
            index_values = list(range(group_size))
        else:
            index_values = [int(index) for index in indices]
            if len(index_values) != group_size:
                raise ValueError("indices must have the same length as coefficients")
        if self.pairwise_l2.shape[0] > 1:
            upper = self.pairwise_l2[np.triu_indices(self.pairwise_l2.shape[0], k=1)]
        else:
            upper = np.zeros((0,), dtype=np.float64)
        log_density_values = [
            float(value) if np.isfinite(value) else None for value in self.log_densities.tolist()
        ]
        coefficient_square_sum = float(np.sum(np.square(self.coefficients, dtype=np.float64)))
        effective_sample_size = 0.0 if coefficient_square_sum <= 0.0 else 1.0 / coefficient_square_sum
        return {
            "method": self.policy,
            "policy": self.policy,
            "indices": index_values,
            "group_size": group_size,
            "valid_row_count": int(np.count_nonzero(self.valid_row_mask)),
            "zero_row_count": int(group_size - np.count_nonzero(self.valid_row_mask)),
            "bandwidth": None if self.bandwidth is None else float(self.bandwidth),
            "effective_bandwidth": None if self.bandwidth is None else float(self.bandwidth),
            "configured_bandwidth": (
                None if self.configured_bandwidth is None else float(self.configured_bandwidth)
            ),
            "bandwidth_mode": self.bandwidth_mode,
            "adaptive_scale": None if self.adaptive_scale is None else float(self.adaptive_scale),
            "group_dispersion": None if self.group_dispersion is None else float(self.group_dispersion),
            "self_included": bool(self.self_included),
            "leave_one_out": bool(self.leave_one_out),
            "coefficients": [float(value) for value in self.coefficients.tolist()],
            "log_densities": log_density_values,
            "effective_sample_size": float(effective_sample_size),
            "pairwise_l2_min": None if upper.size == 0 else float(np.min(upper)),
            "pairwise_l2_median": None if upper.size == 0 else float(np.median(upper)),
            "pairwise_l2_max": None if upper.size == 0 else float(np.max(upper)),
            "fallback_reason": self.fallback_reason,
        }


def _row_logsumexp(values: np.ndarray) -> np.ndarray:
    maxima = np.max(values, axis=1)
    output = np.full(maxima.shape, -np.inf, dtype=np.float64)
    finite_rows = np.isfinite(maxima)
    if not np.any(finite_rows):
        return output
    shifted = values[finite_rows] - maxima[finite_rows, None]
    sums = np.sum(np.exp(shifted), axis=1)
    positive = sums > 0.0
    finite_indices = np.flatnonzero(finite_rows)
    output[finite_indices[positive]] = maxima[finite_rows][positive] + np.log(sums[positive])
    return output


def _softmax_finite(values: np.ndarray) -> Optional[np.ndarray]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    maximum = float(np.max(values[finite]))
    unnormalized = np.zeros(values.shape, dtype=np.float64)
    unnormalized[finite] = np.exp(values[finite] - maximum)
    denominator = float(np.sum(unnormalized))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return unnormalized / denominator


def _normalize_skin_rows(weights: np.ndarray, *, simplex_atol: float) -> Tuple[np.ndarray, np.ndarray, np.dtype]:
    input_array = np.asarray(weights)
    if input_array.ndim != 2 or input_array.shape[0] == 0 or input_array.shape[1] == 0:
        raise ValueError("weights must be a non-empty 2D array with shape (N, J)")
    if not np.all(np.isfinite(input_array)):
        raise ValueError("weights must contain only finite values")
    if not np.isfinite(simplex_atol) or float(simplex_atol) < 0.0:
        raise ValueError("simplex_atol must be a finite non-negative value")
    work = np.asarray(input_array, dtype=np.float64).copy()
    if np.any(work < -float(simplex_atol)):
        minimum = float(np.min(work))
        raise ValueError(f"weights must be non-negative; minimum value is {minimum}")
    work = np.maximum(work, 0.0)
    row_sums = np.sum(work, axis=1)
    valid_mask = row_sums > _ROW_SUM_EPS
    normalized = np.zeros_like(work, dtype=np.float64)
    normalized[valid_mask] = work[valid_mask] / row_sums[valid_mask, None]
    output_dtype = np.float64 if input_array.dtype == np.float64 else np.float32
    return normalized, valid_mask, output_dtype


def _pairwise_l2(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    squared_norms = np.einsum("ij,ij->i", rows, rows)
    squared_l2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (rows @ rows.T)
    return np.sqrt(np.maximum(squared_l2, 0.0))


def _group_rms_dispersion(valid_rows: np.ndarray) -> float:
    if valid_rows.shape[0] == 0:
        return 0.0
    centered = valid_rows - np.mean(valid_rows, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", centered, centered))))


def _build_result(
    *,
    fused_skin: np.ndarray,
    coefficients: np.ndarray,
    log_densities: np.ndarray,
    valid_row_mask: np.ndarray,
    pairwise_l2: np.ndarray,
    output_dtype: np.dtype,
    config: SkinningFusionConfig,
    effective_bandwidth: Optional[float],
    group_dispersion: Optional[float],
    fallback_reason: Optional[str],
) -> SkinningFusionResult:
    return SkinningFusionResult(
        fused_skin=np.asarray(fused_skin, dtype=output_dtype),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        log_densities=np.asarray(log_densities, dtype=np.float64),
        valid_row_mask=np.asarray(valid_row_mask, dtype=bool),
        pairwise_l2=np.asarray(pairwise_l2, dtype=np.float64),
        bandwidth=effective_bandwidth,
        leave_one_out=not bool(config.self_included),
        fallback_reason=fallback_reason,
        policy=str(config.policy),
        self_included=bool(config.self_included),
        bandwidth_mode=str(config.bandwidth_mode),
        configured_bandwidth=config.bandwidth,
        adaptive_scale=config.adaptive_scale,
        group_dispersion=group_dispersion,
    )


def fuse_skinning_weights(
    weights: np.ndarray,
    *,
    config: SkinningFusionConfig,
    simplex_atol: float = DEFAULT_SIMPLEX_ATOL,
) -> SkinningFusionResult:
    """Fuse one ``(N, J)`` group according to ``config``."""

    resolved = validate_skinning_fusion_config(config)
    normalized, valid_mask, output_dtype = _normalize_skin_rows(weights, simplex_atol=simplex_atol)
    group_size, joint_count = normalized.shape
    coefficients = np.zeros((group_size,), dtype=np.float64)
    log_densities = np.full((group_size,), -np.inf, dtype=np.float64)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        coefficients.fill(1.0 / float(group_size))
        return _build_result(
            fused_skin=np.zeros((joint_count,), dtype=np.float64), coefficients=coefficients,
            log_densities=log_densities, valid_row_mask=valid_mask,
            pairwise_l2=np.zeros((0, 0), dtype=np.float64), output_dtype=output_dtype,
            config=resolved, effective_bandwidth=None, group_dispersion=None,
            fallback_reason="all_zero_rows",
        )

    valid_rows = normalized[valid_mask]
    pairwise = _pairwise_l2(valid_rows)
    dispersion = _group_rms_dispersion(valid_rows)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_count == 1:
        coefficients[int(valid_indices[0])] = 1.0
        log_densities[int(valid_indices[0])] = 0.0
        if resolved.policy == "uniform":
            singleton_bandwidth = None
        elif resolved.bandwidth_mode == "fixed":
            singleton_bandwidth = resolved.bandwidth
        else:
            singleton_bandwidth = resolved.adaptive_bandwidth_floor
        return _build_result(
            fused_skin=valid_rows[0], coefficients=coefficients, log_densities=log_densities,
            valid_row_mask=valid_mask, pairwise_l2=pairwise, output_dtype=output_dtype,
            config=resolved, effective_bandwidth=singleton_bandwidth, group_dispersion=dispersion,
            fallback_reason="single_valid_row",
        )

    effective_bandwidth: Optional[float] = None
    fallback_reason: Optional[str] = None
    if resolved.policy == "uniform":
        valid_coefficients = np.full((valid_count,), 1.0 / float(valid_count), dtype=np.float64)
        valid_log_densities = np.zeros((valid_count,), dtype=np.float64)
    else:
        if resolved.bandwidth_mode == "fixed":
            assert resolved.bandwidth is not None
            effective_bandwidth = float(resolved.bandwidth)
        else:
            assert resolved.adaptive_scale is not None
            effective_bandwidth = max(
                float(resolved.adaptive_scale) * dispersion,
                float(resolved.adaptive_bandwidth_floor),
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_affinity = -0.5 * np.square(pairwise / effective_bandwidth)
        if not resolved.self_included:
            np.fill_diagonal(log_affinity, -np.inf)
        valid_log_densities = _row_logsumexp(log_affinity)
        valid_coefficients = _softmax_finite(valid_log_densities)
        if valid_coefficients is None:
            valid_coefficients = np.full((valid_count,), 1.0 / float(valid_count), dtype=np.float64)
            fallback_reason = "nonfinite_density"

    coefficients[valid_indices] = valid_coefficients
    log_densities[valid_indices] = valid_log_densities
    fused = np.asarray(valid_coefficients @ valid_rows, dtype=np.float64)
    fused = np.maximum(fused, 0.0)
    fused_sum = float(np.sum(fused))
    if not np.isfinite(fused_sum) or fused_sum <= _ROW_SUM_EPS:
        fused = np.mean(valid_rows, axis=0)
        fused_sum = float(np.sum(fused))
        fallback_reason = "invalid_fused_row"
    fused = fused / fused_sum
    return _build_result(
        fused_skin=fused, coefficients=coefficients, log_densities=log_densities,
        valid_row_mask=valid_mask, pairwise_l2=pairwise, output_dtype=output_dtype,
        config=resolved, effective_bandwidth=effective_bandwidth, group_dispersion=dispersion,
        fallback_reason=fallback_reason,
    )


def fuse_skinning_weights_l2_kde(
    weights: np.ndarray,
    *,
    bandwidth: float,
    leave_one_out: bool = True,
    simplex_atol: float = DEFAULT_SIMPLEX_ATOL,
) -> SkinningFusionResult:
    """Compatibility wrapper for the previous fixed-bandwidth L2 KDE API."""

    return fuse_skinning_weights(
        weights,
        config=SkinningFusionConfig(
            policy="gaussian_l2", self_included=not bool(leave_one_out),
            bandwidth_mode="fixed", bandwidth=bandwidth,
        ),
        simplex_atol=simplex_atol,
    )
