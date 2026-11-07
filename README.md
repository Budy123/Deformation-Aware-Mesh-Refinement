<div align="center">

<h1>Deformation-Aware Mesh Refinement</h1>

<p>Orientation-aware duplicate-vertex refinement for rigged triangle meshes.</p>

<p><a href="#installation">Installation</a> · <a href="#quick-start">Quick start</a> · <a href="#refinement-and-evaluation">Refinement and evaluation</a> · <a href="LICENSE">Apache 2.0</a></p>

</div>

Standalone code for refining duplicate-position vertices in rigged triangle
meshes. It does not import Puppeteer, APF, Blender, PyTorch, or private data
adapters. Licensed under [Apache 2.0](LICENSE); see [NOTICE.md](NOTICE.md) for
attribution information.

## What is included

- Orientation-compatible vertex partitioning, KDE/mean weight fusion, physical
  vertex merging, face cleanup and original-index mappings.
- Paper metrics: Skin L1, DE, FPR, rest-pose vertex CD and NME.
- A numeric NPZ CLI, a synthetic example, regression tests and source provenance.

No datasets, trained weights, cached predictions, experiment logs or third-party
model implementations are included. This package consumes existing skinning
weights; it does **not** predict or repredict them. Dataset-wide adapters and
Blender/[11] baseline implementations are not bundled. The example validates the
pipeline, not the published dataset averages.

## Method

The refinement pipeline has three stages:

1. **Surface orientation partitioning.** Group vertices at exactly the same
   position, then partition each group using incident-face compatibility.
   Opposite-facing duplicate surfaces are kept separate.
2. **Skinning-weight fusion.** Normalize the valid skinning-weight rows in each
   subgroup and fuse them using KDE-based weighting. Uniform averaging is
   available as a baseline.
3. **Vertex merging and face removal.** Merge vertices within each subgroup,
   remap the faces, remove collapsed and duplicate faces based on vertex indices,
   and retain mappings to the original vertices.

Refinement does not move vertex positions or modify the skeleton. See
[Limits](#limits) for the treatment of opposite surface layers and attributes.

## Installation

Verified locally with Python 3.8.10, NumPy 1.24.4 and SciPy 1.10.1. Core refinement
needs only NumPy; SciPy supplies the float64 KD-tree used for vertex CD. Exact
SciPy/NumPy pins are provided in `requirements.txt` as the tested environment,
not a claim that only those versions work. No dependency is downloaded at runtime.

A CPU is sufficient; CUDA, PyTorch and pretrained model weights are not required.
Use Python 3.8 for the pinned environment below.

```bash
git clone https://github.com/Budy123/Deformation-Aware-Mesh-Refinement.git
cd Deformation-Aware-Mesh-Refinement
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install .
```

Your Python installation must provide `venv` and `pip`. If a minimal Python
installation reports that `ensurepip` is unavailable, prepare a Python 3.8
environment with pip before continuing; this is an environment setup issue.
An existing Python 3.8 environment can also be used without creating `.venv`.

The last command installs the package and its `mesh-refinement` console command.
Package metadata declares minimum runtime dependencies; `requirements.txt`
selects the tested versions.

Check the installation from the repository directory:

```bash
python -m unittest discover -s tests -v
python -m mesh_refinement --help
```

The package was also tested after installation from outside the source
directory. Both the source and installed-package suites passed all 22 tests
with the versions listed above. These are functional checks, not a reproduction
of the full-dataset paper results.

## Quick start

Choose a fresh output directory; the commands refuse existing output files.

```bash
python examples/make_example.py --output /tmp/mesh-refinement-example/input.npz
python -m mesh_refinement refine \
  --input /tmp/mesh-refinement-example/input.npz \
  --output /tmp/mesh-refinement-example/refined.npz
python -m mesh_refinement evaluate \
  --input /tmp/mesh-refinement-example/input.npz \
  --refined /tmp/mesh-refinement-example/refined.npz \
  --output /tmp/mesh-refinement-example/metrics.json
```

This synthetic example needs no dataset download. It produces a refined mesh
and a JSON report with Skin L1, DE, FPR, CD and NME. The output files are numeric
NPZ and JSON files, not rendered images or FBX assets.

## Refinement and evaluation

After preparing your own `input.npz` as described below, run:

```bash
# Our refinement with KDE-based fusion.
mesh-refinement refine --input input.npz --output refined.npz

# Evaluate against the original ground-truth weights and skeleton.
mesh-refinement evaluate --input input.npz --refined refined.npz --output metrics.json

# Uniform-averaging baseline; all other refinement settings are unchanged.
mesh-refinement refine --input input.npz --output mean.npz --fusion mean
mesh-refinement evaluate --input input.npz --refined mean.npz --output mean_metrics.json
```

`refine` needs geometry and input skinning weights, but not ground-truth weights.
`evaluate` additionally needs `gt_skin`; DE requires both skeleton fields below.
Without a skeleton, the other metrics are still evaluated and DE is `null`.
All output paths must be new; existing files are never overwritten.

## Data preparation

Prepare a numeric NPZ containing the arrays below. Dataset files and pretrained
skinning predictions are not bundled. The public interface starts from vertex,
face and skinning-weight arrays; it does not load FBX files or run a skinning
prediction model.

NPZ files must contain numeric arrays; they are loaded with `allow_pickle=False`.
Object-array dataset archives must first be converted using trusted dataset
preparation code. Do not enable pickle to consume untrusted archives.

| Field | Shape | Meaning |
| --- | --- | --- |
| `vertices` | `(N, 3)` | Finite original positions; float64 geometry is preserved. |
| `faces` | `(F, 3)` | Integer zero-based vertex indices. |
| `skin` | `(N, J)` | Finite non-negative **input prediction** weights; needed by `refine`. |
| `gt_skin` | `(N, J)` | Original GT weights; needed by `evaluate`, never fused as an evaluation reference. |
| `matrix_local` | `(J, 4, 4)` | Optional invertible **global rest transforms**, despite the legacy field name; needed for DE. |
| `parents` | `(J,)` | Optional integer hierarchy; roots are `-1`, parents precede children; needed for DE. |

Provide both skeleton fields or neither when evaluating. Zero-sum skin rows are
allowed and explicitly filtered/fallback-handled, not treated as valid GT.
The main protocol uses original GT geometry/skeleton and raw predicted `skin`.
GT-input analysis may set `skin=gt_skin` but still compares the output against
the **unmodified original** GT rows at retained indices.

## API and defaults

```python
from mesh_refinement import refine_mesh, evaluate_mesh

result = refine_mesh(vertices, faces, predicted_weights)
final = result["evaluation"]
scores = evaluate_mesh(vertices, faces, gt_weights, final, skeleton=skeleton)
```

`refine_mesh(vertices, faces, skin, *, angle_threshold_deg=180.0,
fusion_config=None)` uses the following paper-final policy when no config is
provided: self-included Gaussian L2, group-RMS bandwidth
`h = max(0.5 * d_group, 1e-12)`. Here `d_group` is the RMS distance of normalized
valid rows from their group mean. The legacy fixed-bandwidth configuration is
available through `SkinningFusionConfig` but is **not** this public API's default.

The production function bodies in `_topology.py` and `fusion.py` are preserved.
The validated wrapper selects the paper configuration and matches production
float32 skin storage; it does not change the estimator or cleanup decisions.

`evaluate_mesh(vertices, faces, gt_skin, refined, *, skeleton=None,
pose_count=64, pose_angle_deg=30.0, seed=321, sample_index=0)` expects the phase
dictionary returned by refinement, including `keep_indices`. It checks exact
coordinate correspondence rather than guessing nearest reference vertices.
`sample_index` is the original main evaluation ordinal, **not** the ordinal of
a newly filtered subset. The original seed rule is
`seed + sample_index*15991 + pose_index*65537`. Preserve that ordinal, prediction
version, input dtype and original joint ordering for dataset reproduction.

### Outputs

The API returns `analysis` and `evaluation` phase dictionaries. Analysis retains
remapped face issues; evaluation removes repeated-index triangles and duplicate
or reversed triangles with the same vertex indices, then compacts unused vertices.
Distinct opposite-facing surface layers can remain separate. The core also
returns merge/protection diagnostics and the fusion identity.

CLI refinement writes the selected phase as numeric NPZ arrays:
`vertices`, `faces`, `skin`, `keep_indices`, `old_to_new`, `merged_old_to_new`,
`kept_source_face_indices`, optional mapped `gt_skin` and unchanged skeleton
arrays. `metadata_json` records the version, phase, fusion configuration and
runtime. `keep_indices` maps output vertices to original retained vertex indices;
`merged_old_to_new` additionally maps merged-away vertices, or `-1` when unused.

## Metric definitions

The public JSON uses descriptive field names rather than substituting one
reduction for another:

| Field | Definition |
| --- | --- |
| `skin_l1` | **Paper metric**: absolute difference of row-normalized output/GT weights, averaged over valid retained vertices **and J joints**. GT validity is `abs(sum(GT_row)-1) < 1e-2`. |
| `skin_l1_joint_sum` | Joint-summed normalized L1 averaged over the same valid vertices; equals `J * skin_l1`, not the paper table column. |
| `skin_l1_joint_sum_raw` | Legacy diagnostic: joint-summed differences before row normalization, averaged over valid vertices. |
| `de` | **Paper DE**: Euclidean distance between paired GT/output LBS positions, averaged over valid retained vertices and sampled poses; exclude zero-sum GT or output rows. |
| `de_mse` | Separate coordinate-MSE diagnostic; not DE. |
| `fpr` | Output face count / original face count. |
| `cd` | Half the sum of bidirectional mean nearest-**vertex** Euclidean distances in the rest pose; not a squared or surface-sampled metric. |
| `cd_squared` | Separate squared-distance diagnostic. |
| `nme` | Number of undirected edges incident to a number of faces other than two; **includes boundary edges**, matching the recorded protocol. |

DE uses 64 poses by default. Each joint's three axis-angle/rotation-vector
components are sampled in `[-30, 30]` degrees; this is **not** an Euler-angle
parameterization or a bound of 30 degrees on the rotation vector's norm.
The original LBS row-sum denominator includes `1e-8`, which is preserved.
This is the local evaluator's UniRig-style pose/LBS approach, not a claim to run
the unchanged UniRig coordinate-MSE training loss as DE.

Unavailable metrics are JSON `null` with validity counts, not silent zeros.
The unchanged low-level legacy kernel's empty-valid-set Skin L1 sentinel of
zero is exposed as `null` by this public wrapper. Without a skeleton only DE
is unavailable. When SciPy is absent, the legacy float32 NN fallback is named
explicitly in `cd_backend`; use the pinned SciPy environment for paper CD.
Published dataset reporting averages per-mesh scores (macro mean), not every
vertex/joint entry across the whole dataset (micro mean). Empty eligible sets
must not contribute an artificial zero.

## Full CLI option reference

All commands also accept `-h` / `--help`.

| Command | Option | Default / behavior |
| --- | --- | --- |
| `refine` | `--input PATH` | Required original numeric NPZ. |
| `refine` | `--output PATH` | Required **new** NPZ file; existing files and symlinks rejected. |
| `refine` | `--phase {analysis,evaluation}` | `evaluation`; which phase to save. |
| `refine` | `--fusion {kde,mean}` | `kde`; `mean` is the uniform averaging baseline. |
| `refine` | `--bandwidth-scale FLOAT` | `0.5`, finite and positive; Gaussian group-RMS scale; unused by uniform fusion. |
| `refine` | `--angle-threshold-deg FLOAT` | `180`, finite within `[0,180]`. |
| `evaluate` | `--input PATH` | Required original geometry/GT NPZ; optional skeleton. |
| `evaluate` | `--refined PATH` | Required refined phase NPZ with retained-index mapping. |
| `evaluate` | `--output PATH` | Required **new** JSON file; no overwrite flag. |
| `evaluate` | `--pose-count INT` | `64`, positive. |
| `evaluate` | `--pose-angle-deg FLOAT` | `30`, finite non-negative rotation-vector component range. |
| `evaluate` | `--seed INT` | `321`, non-negative base seed. |
| `evaluate` | `--sample-index INT` | `0`, non-negative original record ordinal used for the pose seed. |
| Example generator | `--output PATH` | Required new numeric synthetic NPZ. |

## Limits

Coordinates are grouped by exact equality. The partition is deterministic
greedy pairwise-compatible grouping, not a connected component. The estimator
does not detect erroneous weights, guarantee outlier robustness, or guarantee
that opposite surface layers share fused weights. UV coordinates and supplied
vertex normals are not preserved/remapped by this numeric interface. Vertex
coordinates are not optimized; skeleton topology/transforms are not refined.

See [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) for the source versions.

## Code structure

- [mesh_refinement/core.py](mesh_refinement/core.py): validated refinement API
  and paper-default configuration.
- [mesh_refinement/_topology.py](mesh_refinement/_topology.py) and
  [mesh_refinement/fusion.py](mesh_refinement/fusion.py): partitioning, merging,
  face removal and skinning-weight fusion.
- [mesh_refinement/metrics.py](mesh_refinement/metrics.py): metric evaluation.
- [mesh_refinement/cli.py](mesh_refinement/cli.py): NPZ refinement and evaluation
  commands.
- [examples/make_example.py](examples/make_example.py): self-contained synthetic
  input generator.
- [tests/test_release.py](tests/test_release.py): regression and CLI tests.

## Acknowledgements and license

This standalone release was extracted from our research pipeline built around
[Puppeteer](https://github.com/Seed3D/Puppeteer). The evaluation follows the
recorded [UniRig](https://github.com/VAST-AI-Research/UniRig)-style skinning and
pose-sampling conventions, with the metric definitions documented above.
Neither project is a runtime dependency of this package.

The code is released under [Apache 2.0](LICENSE). See [NOTICE.md](NOTICE.md),
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json), and the preserved
[upstream license](licenses/Puppeteer-APACHE-2.0.txt) for attribution and source
details.
