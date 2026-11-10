# Dataset evaluation

[Back to README](../README.md) · [Metric definitions](evaluation.md)

## Prepared dataset

First run the [dataset converter](data_preparation.md). Its manifest lists the
numeric NPZ files, sample IDs and source indices. Then evaluate every listed
mesh, without reloading the raw AXL archive for each mesh:

```bash
OPENBLAS_NUM_THREADS=1 python -m mesh_refinement run-dataset \
  --manifest data/prepared/axl2_test/manifest.json \
  --output-dir outputs/axl2_test --config config/fusion_config.yaml --workers 4
```

The runner refines each input using the supplied weights, evaluates original
and refined meshes against original GT, and uses the same per-mesh poses for
both. It performs no prediction or training. SO is evaluated on common surviving
edge pairs across the two methods. It loads at most one numeric NPZ per worker;
the worker queue and aggregation are bounded/streamed, not a whole-dataset mesh
cache. Limit BLAS threads when increasing worker count.

## Recorded paper populations and pose ordinals

Use the [recorded manifests](../config/evaluation/README.md) rather than deriving
the paper pose index from a reordered converter manifest:

```bash
python -m mesh_refinement run-dataset \
  --manifest config/evaluation/axl2_test.json.gz \
  --input-dir data/prepared/axl2_test --output-dir outputs/paper_axl2_test \
  --config config/fusion_config.yaml

python -m mesh_refinement run-dataset \
  --manifest config/evaluation/axl2_train.json.gz \
  --input-dir data/prepared/axl2_train --output-dir outputs/paper_axl2_train \
  --config config/fusion_config.yaml

python -m mesh_refinement run-dataset \
  --manifest config/evaluation/rignetv1.json.gz \
  --input-dir data/prepared/rignetv1 --output-dir outputs/paper_rignetv1 \
  --config config/fusion_config.yaml

python -m mesh_refinement aggregate \
  --runs outputs/paper_axl2_train outputs/paper_axl2_test \
  --output outputs/paper_axl2_combined.json
```

Recorded manifests require predicted-weight metadata, not GT-input substitutes.
The record's `sample_index` overrides `source_index`; neither is regenerated
from its position in a selected subset. Train shard-local original indices are
preserved. Generic converter manifests use their `source_index`; this is a
valid new evaluation protocol, but not necessarily the paper's original bank.
Reproducing numerical table entries also requires the same predictions and
preprocessing. IDs and pose seeds alone are not a full reproduction guarantee.

For a custom subset, copy selected records into a JSON manifest while keeping
their IDs, file paths and original indices. If present, update `population_size`
to the selected count. Files are relative to `--input-dir` (default: manifest
directory); traversal outside that directory and duplicate IDs/files are rejected.

## Outputs and aggregation

- `refined/<id>.npz`: final mesh, fused weights and original correspondences.
- `records/<id>.json`: original/refined metrics, common SO coverage, input/output
  hashes and the actual pose index.
- `metrics.csv`: one row per mesh and method, in manifest order.
- `summary.json`: global/SO macro means over eligible meshes; OFPR pooled over
  original face pairs; Local scores pooled over valid original target vertices.
  Each metric has its own denominator and unavailable values are not zeros.
- `run.json`: immutable configuration, population/ordinals, manifest hash,
  runtime versions and hashes of the executing package modules.
- `complete.json`: written only on success, with configuration/record hashes.
  A failure writes `failure.json` and stops; completed files are preserved but
  no partial dataset is presented as successful. There is no overwrite/resume flag.

`aggregate` reads original per-mesh records, not averages of split averages.
It requires completed, hash-consistent runs with the same dataset, weights,
configuration, pose protocol and package code, and rejects overlapping sample IDs.
Use a fresh output directory/file for every invocation.

## Full CLI reference

| Command | Option | Default / meaning |
| --- | --- | --- |
| `run-dataset` | `--manifest PATH` | Required converter/custom JSON or recorded JSON.gz. |
| `run-dataset` | `--input-dir PATH` | Prepared NPZ root; defaults to manifest directory. |
| `run-dataset` | `--output-dir PATH` | Required new directory. |
| `run-dataset` | `--config PATH` | Optional fusion YAML; normal defaults otherwise. |
| `run-dataset` | `--fusion {kde,mean}` | Optional YAML policy override. |
| `run-dataset` | `--workers INT` | `1`; positive process count. |
| `run-dataset` | `--seed INT` | `321`; nonnegative base seed. |
| `run-dataset` | `--pose-count INT` | `64`; positive pose count. |
| `run-dataset` | `--pose-angle-deg FLOAT` | `30`; nonnegative rotation-vector component bound. |
| `aggregate` | `--runs DIR [DIR ...]` | Required completed run directories. |
| `aggregate` | `--output PATH` | Required new JSON output. |
| Both | `-h`, `--help` | Print usage. |
