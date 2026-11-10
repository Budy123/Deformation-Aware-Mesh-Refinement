# Evaluation reference

[Back to README](../README.md) · [Input and output formats](usage.md)

## Metrics

The evaluator uses the original GT rows at retained vertex indices. It reports
the following per-mesh metrics; average valid per-mesh scores to obtain dataset
macro averages, rather than pooling every vertex across the dataset.

| JSON field | Definition |
| --- | --- |
| `skin_l1` | Absolute difference of normalized output and GT weights, averaged over valid retained vertices and all joints. GT rows satisfy `abs(sum(row)-1) < 1e-2`. |
| `de` | Mean Euclidean distance between corresponding vertices deformed with GT/output weights under identical skeletons and sampled poses. Zero-sum GT/output rows are excluded. |
| `fpr` | Output face count divided by original face count. |
| `cd` | Half the sum of the two directed mean nearest-vertex Euclidean distances in the rest pose. |
| `nme` | Count of undirected edges incident to a number of faces other than two, including boundary edges. |

Additional diagnostic fields are not interchangeable with the paper columns:

- `skin_l1_joint_sum`: normalized L1 summed over joints, then averaged over
  valid vertices. It equals `J * skin_l1`.
- `skin_l1_joint_sum_raw`: joint-summed L1 without row normalization.
- `de_mse`: coordinate MSE, not mean Euclidean DE.
- `cd_squared`: squared-distance counterpart, not `cd`.

Undefined metrics are `null` with validity counts, not artificial zeros. Without
a skeleton, only DE is unavailable. Use the pinned SciPy environment for float64
nearest-neighbor CD; the legacy float32 fallback is identified by `cd_backend`.

## Deformation poses

The default bank contains 64 poses. Each joint's three rotation-vector components
are sampled uniformly within `[-30, 30]` degrees. These are not Euler angles;
30 degrees is a component bound, not a bound on the rotation vector's norm.
The original LBS row-sum denominator includes `1e-8`.

Pose seeds follow:

```text
seed + sample_index * 15991 + pose_index * 65537
```

Keep the original record ordinal as `sample_index`, not its new position in a
filtered subset. Across methods, preserve the same geometry, original GT,
skeleton, joint ordering, prediction version and pose bank. Defaults are
`seed=321`, `sample_index=0`, `pose_count=64`, `pose_angle_deg=30`.

This uses the local evaluator's UniRig-style pose/LBS convention, not the
unchanged UniRig coordinate-MSE training loss.

## Reproduction scope

The standalone package provides the main refinement, Mean fusion comparison,
and the five metrics above. It does not bundle dataset adapters, predictions,
Ma et al./Blender baselines, shared-weight experiments, SO, OFPR or Local
Skin L1/DE. It also has no no-partition ablation runner. The included synthetic
demo checks the pipeline; it does not reproduce full-dataset averages.
