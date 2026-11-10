# Usage reference

[Back to README](../README.md)

## Input data

For RigNetv1 and AXL 2.0, use the supplied [dataset converter](data_preparation.md)
instead of constructing these arrays manually.

The package operates on numeric NPZ files containing a triangle mesh and
precomputed skinning weights. It does not load FBX files or run an auto-rigging
model. Prepare predictions separately, with the same vertex and joint ordering
as the input geometry and skeleton.

| Field | Shape | Description |
| --- | --- | --- |
| `vertices` | `(N, 3)` | Finite original vertex positions. Geometry uses float64. |
| `faces` | `(F, 3)` | Integer, zero-based triangle indices. |
| `skin` | `(N, J)` | Finite, non-negative predicted weights; required for refinement. |
| `gt_skin` | `(N, J)` | Original ground-truth weights; required for evaluation. |
| `matrix_local` | `(J, 4, 4)` | Optional global rest transforms, despite the legacy field name. |
| `parents` | `(J,)` | Optional integer joint hierarchy. Roots use `-1`; parents precede children. |

Provide both skeleton fields or neither when evaluating. Without a skeleton,
the other metrics are available but DE is `null`. Ground truth is never fused
to create an evaluation reference: the evaluator uses original GT rows at
retained vertex indices for global metrics, including for GT-input analysis.
Local metrics back-project output weights to their original target vertices;
see [metric cohorts and SO comparisons](evaluation.md).

Refinement/evaluation NPZ files are read with `allow_pickle=False`. The supplied
AXL converter requires explicit trust before reading official object records.
Zero-sum skin rows
are handled by validity masks and fallbacks; they are not valid GT annotations.

## Refinement and evaluation

```bash
python -m mesh_refinement refine \
  --input input.npz --output refined.npz \
  --config config/fusion_config.yaml

python -m mesh_refinement evaluate \
  --input input.npz --refined refined.npz --output metrics.json
```

All output files must be new. Existing files and symlinks are not overwritten.
Refinement does not require ground truth; evaluation does.

### Mean comparison

```bash
python -m mesh_refinement refine \
  --input input.npz --output mean.npz \
  --config config/fusion_config.yaml --fusion mean
```

This changes only our skinning-weight fusion to uniform averaging. It does not
run the Ma et al. or Blender baselines.

## Configuration

Edit [config/fusion_config.yaml](../config/fusion_config.yaml). The YAML mapping
accepts the following four fields; omitted fields use `SkinningFusionConfig`
defaults. Explicit `--fusion` and `--bandwidth-scale` override YAML values.

| Field | Default | Meaning |
| --- | --- | --- |
| `policy` | `gaussian_l2` | Self-included Gaussian KDE; `uniform` selects Mean. |
| `adaptive_scale` | `0.5` | Scale `c` in `h = max(c * d_group, floor)`. |
| `adaptive_bandwidth_floor` | `1e-12` | Minimum bandwidth, including zero-dispersion cases. |
| `gaussian_exponent_scale` | `0.5` | Scale `a` in `exp(-a * (distance / h)**2)`. |

`d_group` is the RMS Euclidean distance of valid, normalized weight vectors
from their subgroup mean. With the default exponent scale, the Gaussian
denominator is `2*h**2`. Using `1.0` instead gives denominator `h**2`; it changes
the estimator and is not an equivalent rewriting of the default formula.

The three numeric values must be finite and positive. Unknown fields and Python
object tags are rejected; YAML is read with `yaml.safe_load`. Mean ignores KDE
parameters and records their canonical defaults. Fixed-bandwidth and
leave-one-out variants are not supported.

The full resolved configuration is recorded in the output. Non-default
floor/exponent values also extend the fusion identity tag.

## Python API

```python
from mesh_refinement import SkinningFusionConfig, evaluate_mesh, refine_mesh

result = refine_mesh(vertices, faces, predicted_weights,
                     fusion_config=SkinningFusionConfig())
refined = result["evaluation"]
scores = evaluate_mesh(vertices, faces, gt_weights, refined, skeleton=skeleton)
```

`refine_mesh(vertices, faces, skin, *, fusion_config=None)` uses the same defaults
when the config is omitted. `evaluate_mesh(vertices, faces, gt_skin, refined,
*, skeleton=None, pose_count=64, pose_angle_deg=30.0, seed=321, sample_index=0)`
requires the original geometry, GT and retained-index mapping. The skeleton
dictionary uses `matrix_local` and `parents` as defined above.

## Output structure

The API returns two phases:

- `analysis`: merged vertices with remapped faces before face removal.
- `evaluation`: face cleanup followed by unused-vertex compaction.

Face cleanup removes repeated-index triangles and same-index duplicate or
reversed triangles. Distinct, opposite-facing surfaces with separate vertex
indices can remain. Faces are checked using vertex indices, not only positions.

The selected CLI phase is written as numeric NPZ arrays:

| Field | Meaning |
| --- | --- |
| `vertices`, `faces`, `skin` | Refined geometry and float32 skinning weights. |
| `keep_indices` | Retained original vertex index for each output vertex. |
| `old_to_new` | Mapping for retained original vertices; `-1` otherwise. |
| `merged_old_to_new` | Mapping including merged-away vertices; `-1` if removed by cleanup. |
| `kept_source_face_indices` | Original face index for each retained face. |
| `gt_skin` | If supplied, original GT mapped through `keep_indices`, never fused. |
| `matrix_local`, `parents` | If supplied, unchanged skeleton arrays. |
| `metadata_json` | Version, phase, complete resolved fusion config and runtime. |

Evaluation checks exact retained-coordinate correspondence rather than inferring
nearest reference vertices. Its JSON report includes the producer metadata in
`refinement`, or `null` if the input arrays have no such metadata.

## CLI options

The dataset runner/aggregator and renderer have separate complete option
references in [dataset evaluation](batch_evaluation.md#full-cli-reference) and
[visualization](visualization.md#full-cli-reference). They use the same numeric
NPZs and do not run model inference.

Both commands also accept `-h` / `--help`.

| Command | Option | Default / behavior |
| --- | --- | --- |
| `refine` | `--input PATH` | Required numeric NPZ. |
| `refine` | `--output PATH` | Required new output NPZ. |
| `refine` | `--config PATH` | Optional YAML mapping; omitted fields use defaults. |
| `refine` | `--phase {analysis,evaluation}` | `evaluation`. |
| `refine` | `--fusion {kde,mean}` | Optional policy override; KDE without other settings. |
| `refine` | `--bandwidth-scale FLOAT` | Optional positive scale override; default `0.5`. |
| `evaluate` | `--input PATH` | Required original geometry and GT; optional skeleton. |
| `evaluate` | `--refined PATH` | Required refined arrays and retained indices. |
| `evaluate` | `--output PATH` | Required new JSON file. |
| `evaluate` | `--pose-count INT` | `64`, positive integer. |
| `evaluate` | `--pose-angle-deg FLOAT` | `30`, non-negative rotation-vector component bound. |
| `evaluate` | `--seed INT` | `321`, non-negative base seed. |
| `evaluate` | `--sample-index INT` | `0`, original evaluation ordinal. |

## Scope and limitations

Positions are grouped by exact equality. Partitioning uses deterministic
every-member compatibility, not connected components. Coincident faces are
identified by matching their three positions; a negative normal dot product
prevents merging their opposite sides. There is no extra angle threshold, and
vertices without valid incident faces are protected.

KDE expresses agreement among weight vectors; it does not identify correct
weights or guarantee robustness to outliers. Opposite surface layers are fused
independently and can receive different weights. UV coordinates and supplied
vertex normals are not preserved by this numeric interface. Vertex positions
and the skeleton are not optimized.
