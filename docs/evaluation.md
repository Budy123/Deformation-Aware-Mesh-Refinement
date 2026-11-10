# Evaluation reference

[Back to README](../README.md) · [Input and output formats](usage.md)

## Metrics

Global Skin L1/DE use the original GT rows at retained vertex indices. The
evaluator reports per-mesh scores and cohort counts. Global metrics and SO use
valid per-mesh macro averages in dataset reporting. For the orientation ablation,
pool OFPR's preserved/source pair counts and weight Local metric means by their
respective valid original-target vertex counts, as in the existing benchmark.

| JSON field | Definition |
| --- | --- |
| `skin_l1` | Absolute difference of normalized output and GT weights, averaged over valid retained vertices and all joints. GT rows satisfy `abs(sum(row)-1) < 1e-2`. |
| `de` | Mean Euclidean distance between corresponding vertices deformed with GT/output weights under identical skeletons and sampled poses. Zero-sum GT/output rows are excluded. |
| `fpr` | Output face count divided by original face count. |
| `cd` | Half the sum of the two directed mean nearest-vertex Euclidean distances in the rest pose. |
| `nme` | Count of undirected edges incident to a number of faces other than two, including boundary edges. |
| `ofpr` | Fraction of original opposite-oriented duplicate face pairs retained as two distinct, noncollapsed faces. Pairs already sharing the same three vertex indices are excluded from the source population. |
| `local_skin_l1` | Normalized per-joint L1 averaged over original vertices in exact-position groups split by surface orientation partitioning. Both GT/output rows must be nonzero. |
| `local_de` | Mean Euclidean DE on the same original local vertices, using the same pose bank as global DE. |
| `so` | Original-edge-length-weighted, pose-averaged integral of squared separation between corresponding edge copies. No square root is applied. |

Additional diagnostic fields are not interchangeable with the paper columns:

- `skin_l1_joint_sum`: normalized L1 summed over joints, then averaged over
  valid vertices. It equals `J * skin_l1`.
- `skin_l1_joint_sum_raw`: joint-summed L1 without row normalization.
- `de_mse`: coordinate MSE, not mean Euclidean DE.
- `cd_squared`: squared-distance counterpart, not `cd`.

Undefined metrics are `null` with validity counts, not artificial zeros. Without
a skeleton, DE, Local DE and SO are unavailable. Use the pinned SciPy environment for float64
nearest-neighbor CD; the legacy float32 fallback is identified by `cd_backend`.

### Local and face correspondences

Local metrics assign each original target vertex the skinning weights at its
`merged_old_to_new` output index, then compare against that vertex's original GT.
Merged source copies remain separate observations; targets are not reduced to
retained representatives. If cleanup removed any target's mapped output, the
local score is `null` rather than an average over a smaller, changed cohort.
`local_source_vertex_count` and `local_mapped_vertex_count` expose this coverage.
OFPR uses `kept_source_face_indices` to track each original pair through cleanup;
either face being removed makes that pair unpreserved.

The CLI loads these mappings from ordinary `refine` outputs. Older/external
outputs missing the required mappings retain the existing global scores, but
the dependent new metrics are `null`. Missing cohorts are not scored as zero.

### Seam Opening (SO)

As in the revision experiment, source edge copies must have exactly matching
rest-pose endpoints and normalized GT weights matching within absolute `1e-6`
at both corresponding endpoints. This excludes intended differences in GT
motion. For three or more copies, all individually compatible unordered pairs
are included; compatibility is not transitively extended.

For endpoint differences `a` and `b` after posing, the exact straight-edge
integral is `(||a||² + a·b + ||b||²)/3`. Scores average over the pose bank and
weight pairs by their original edge length. This adapts the interval discrepancy
in Liu et al., [*Seamless: Seam erasure and seam-aware decoupling of shape from
mesh resolution*](https://cragl.cs.gmu.edu/seamless/), 2017; GT filtering and the 3D edge/pose population are the
experiment's additional choices, not claims about that reference's protocol.

For method comparisons, use the **intersection of valid source pairs across all
methods**. A pair is valid only when both mapped edges survive in the final mesh
and their endpoint weights are valid. Two copies mapped to the same surviving
edge are valid and score zero. A missing/collapsed edge is excluded, not assigned
zero; preservation must be assessed separately with OFPR/qualitative results.

`evaluate_mesh` and the CLI report a single output's eligible-pair score and
coverage (`so_pair_scope="single_output_valid_pairs"`). Do not compare these
scores directly if their eligible populations differ. To enforce a shared
population, pass all already-refined outputs to the comparison API:

```python
from mesh_refinement import evaluate_seam_opening

seam = evaluate_seam_opening(vertices, faces, gt_weights,
    {"method_a": output_a, "method_b": output_b}, skeleton=skeleton,
    seed=321, sample_index=original_record_ordinal)
```

This returns `common_pair_count`, `common_pair_length` and per-method SO/coverage.
It evaluates supplied meshes; it does not run either baseline or change weights.

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
and the metrics above. A [converter](data_preparation.md) reads the public
RigNetv1/AXL 2.0 data. Apart from the single licensed Fish1 example, it does not
bundle datasets or predictions. It also does not include
Ma et al./Blender baselines, shared-weight experiments or a no-partition
ablation runner. The included Fish1 example checks
the pipeline; it does not reproduce full-dataset averages.

[Dataset evaluation](batch_evaluation.md) provides batch execution and aggregation,
including the recorded paper sample IDs and pose ordinals. The renderer in
[visualization](visualization.md) previews supplied weights without inference.
