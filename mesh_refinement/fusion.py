"""Mesh-independent skinning-weight fusion policies.

This module owns only the question "how should one duplicate group's skin
rows become one representative row?" Geometry grouping, representative
selection, remapping, and face cleanup remain in exporters.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


DEFAULT_SIMPLEX_ATOL = 1e-6
DEFAULT_ADAPTIVE_SCALE = 0.5
DEFAULT_ADAPTIVE_BANDWIDTH_FLOOR = 1e-12
DEFAULT_GAUSSIAN_EXPONENT_SCALE = 0.5
DEFAULT_FUSION_POLICY = "gaussian_l2"
_ROW_SUM_EPS = 1e-12


def _format_float_tag(value: float) -> str:
    return repr(float(value)).lower().replace("-", "m").replace("+", "").replace(".", "p")


@dataclass(frozen=True)
class SkinningFusionConfig:
    """Select paper KDE (self-included, group RMS) or uniform averaging."""

    policy: str = DEFAULT_FUSION_POLICY
    adaptive_scale: float = DEFAULT_ADAPTIVE_SCALE
    adaptive_bandwidth_floor: float = DEFAULT_ADAPTIVE_BANDWIDTH_FLOOR
    gaussian_exponent_scale: float = DEFAULT_GAUSSIAN_EXPONENT_SCALE


def validate_skinning_fusion_config(config: SkinningFusionConfig) -> SkinningFusionConfig:
    """Validate and canonicalize a fusion policy configuration."""

    if not isinstance(config, SkinningFusionConfig):
        raise TypeError("config must be a SkinningFusionConfig")
    policy = str(config.policy)
    if policy not in {"uniform", "gaussian_l2"}:
        raise ValueError("policy must be one of: uniform, gaussian_l2")
    parameters = {}
    for name in ("adaptive_scale", "adaptive_bandwidth_floor", "gaussian_exponent_scale"):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite positive scalar") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive scalar")
        parameters[name] = value
    if policy == "uniform":
        return SkinningFusionConfig(policy=policy)
    return SkinningFusionConfig(policy=policy, **parameters)


def format_skinning_fusion_tag(config: SkinningFusionConfig) -> str:
    """Return a deterministic tag that isolates policy-specific outputs."""

    resolved = validate_skinning_fusion_config(config)
    if resolved.policy == "uniform":
        return "skin-fusion-uniform"
    tag = f"skin-fusion-gaussian-l2-self-group-rms-c{_format_float_tag(resolved.adaptive_scale)}"
    if resolved.adaptive_bandwidth_floor != DEFAULT_ADAPTIVE_BANDWIDTH_FLOOR:
        tag += f"-floor{_format_float_tag(resolved.adaptive_bandwidth_floor)}"
    if resolved.gaussian_exponent_scale != DEFAULT_GAUSSIAN_EXPONENT_SCALE:
        tag += f"-exp{_format_float_tag(resolved.gaussian_exponent_scale)}"
    return tag


@dataclass(frozen=True)
class SkinningFusionResult:
    """Result of fusing one group of skinning-weight observations."""

    fused_skin: np.ndarray
    coefficients: np.ndarray
    log_densities: np.ndarray
    valid_row_mask: np.ndarray
    pairwise_l2: np.ndarray
    bandwidth: Optional[float]
    fallback_reason: Optional[str]
    policy: str = DEFAULT_FUSION_POLICY
    adaptive_scale: Optional[float] = None
    group_dispersion: Optional[float] = None
    adaptive_bandwidth_floor: Optional[float] = None
    gaussian_exponent_scale: Optional[float] = None

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
            "policy": self.policy,
            "indices": index_values,
            "group_size": group_size,
            "valid_row_count": int(np.count_nonzero(self.valid_row_mask)),
            "zero_row_count": int(group_size - np.count_nonzero(self.valid_row_mask)),
            "bandwidth": None if self.bandwidth is None else float(self.bandwidth),
            "adaptive_scale": None if self.adaptive_scale is None else float(self.adaptive_scale),
            "adaptive_bandwidth_floor": self.adaptive_bandwidth_floor,
            "gaussian_exponent_scale": self.gaussian_exponent_scale,
            "group_dispersion": None if self.group_dispersion is None else float(self.group_dispersion),
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


def _normalize_skin_rows(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.dtype]:
    input_array = np.asarray(weights)
    if input_array.ndim != 2 or input_array.shape[0] == 0 or input_array.shape[1] == 0:
        raise ValueError("weights must be a non-empty 2D array with shape (N, J)")
    if not np.all(np.isfinite(input_array)):
        raise ValueError("weights must contain only finite values")
    work = np.asarray(input_array, dtype=np.float64).copy()
    if np.any(work < -DEFAULT_SIMPLEX_ATOL):
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
        fallback_reason=fallback_reason,
        policy=str(config.policy),
        adaptive_scale=config.adaptive_scale if config.policy == "gaussian_l2" else None,
        group_dispersion=group_dispersion,
        adaptive_bandwidth_floor=config.adaptive_bandwidth_floor if config.policy == "gaussian_l2" else None,
        gaussian_exponent_scale=config.gaussian_exponent_scale if config.policy == "gaussian_l2" else None,
    )


def fuse_skinning_weights(
    weights: np.ndarray,
    *,
    config: SkinningFusionConfig,
) -> SkinningFusionResult:
    """Fuse one ``(N, J)`` group according to ``config``."""

    resolved = validate_skinning_fusion_config(config)
    normalized, valid_mask, output_dtype = _normalize_skin_rows(weights)
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
        singleton_bandwidth = resolved.adaptive_bandwidth_floor if resolved.policy == "gaussian_l2" else None
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
        effective_bandwidth = max(
            resolved.adaptive_scale * dispersion,
            resolved.adaptive_bandwidth_floor,
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_affinity = -resolved.gaussian_exponent_scale * np.square(pairwise / effective_bandwidth)
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
