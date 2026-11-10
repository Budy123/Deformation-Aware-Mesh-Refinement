# Data preparation

[Back to README](../README.md)

The converter reads the public formats used by our experiments. You do not
need to construct mesh arrays or manually rename NPZ fields. Download the
data from the original providers; full datasets are not redistributed here.
For a download-free first run, use the bundled [Fish1 example](../examples/axl2_fish/README.md).

## 1. Download

- **RigNetv1:** download the [official preprocessed dataset](https://drive.google.com/file/d/1-B6hJ4423rw1LrTForHp7oaG5qRAbJx3/view),
  linked under [RigNet's Data section](https://github.com/zhan-xu/RigNet#data).
  Use the preprocessed ZIP directly or extract it. The supplied root must
  contain `obj_remesh/<id>.obj` and `rig_info_remesh/<id>.txt`.
  Do not use the FBX download or run RigNet's training-data generation script.
- **AXL 2.0:** download `articulation_xlv2_train.npz` and/or
  `articulation_xlv2_test.npz` from the [official release](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0/tree/main).
  These are the preprocessed main-set records, not the diverse-pose subset.
  Use each provided split separately. The converter also accepts a local
  directory of ordered `shard_*.npz` files with the same `arr_0` record format.

For example, place downloads under `data/` and predictions under `predictions/`.
These local folders are ignored by Git. Respect the original dataset terms.

## 2. Select the input skinning weights

The dataset contains GT weights, not the model predictions used in the paper.
Choose one of the following explicitly; missing predictions never fall back to GT.

**Predicted-input evaluation:** obtain skinning predictions with the original
mesh and GT skeleton fixed. Our experiments use Puppeteer. Its
[skinning instructions](https://github.com/Seed3D/Puppeteer/tree/main/skinning)
describe inference and `--save_skin_npy`; place its saved `<sample_id>_skin.npy`
files in the directory passed to `--predictions-dir`. Use the raw-mesh predictions,
not predictions recomputed after refinement. This package does not run model
inference or install Puppeteer's GPU dependencies.

Each prediction must be an `(N, J)` numeric array in the **same vertex and joint
order as the downloaded record**. The converter checks shape, finite values and
non-negative weights. An NPY array contains no vertex/joint identifiers, so
shape checks alone cannot prove the ordering; do not substitute predictions
from a remeshed object, reordered skeleton or a different sample.

**GT-input use:** pass `--use-gt-weights` instead. This makes the downloaded GT
the refinement input and also retains an independent original GT reference.
It is useful for running the method without a predictor, but does not reproduce
the paper's predicted-input scores.

## 3. Convert

After installing the package as described in README:

```bash
python scripts/prepare_data.py --dataset rignetv1 \
  --input data/ModelResource_RigNetv1_preproccessed.zip \
  --predictions-dir predictions/rignetv1 --output-dir data/prepared/rignetv1

python scripts/prepare_data.py --dataset axl2 \
  --input data/articulation_xlv2_test.npz --trust-axl-pickle \
  --predictions-dir predictions/axl2_test --output-dir data/prepared/axl2_test
```

AXL stores dictionaries in NumPy object arrays. `--trust-axl-pickle` explicitly
allows deserialization **only for trusted files obtained from the official
provider**; pickle can execute code. The normal refinement/evaluation readers
still use `allow_pickle=False`. A compressed object array must be loaded as a
whole, even with `--sample-id`; the large train NPZ therefore needs substantial
memory. If using existing shards, only one shard is resident at a time.

Each run needs a new output directory. An error stops conversion and leaves
any completed files in that directory; choose a new directory after correcting
the input. A successfully completed run writes `manifest.json` listing every
output sample and its original `source_index`. No samples are silently skipped.

The converter preserves positions, duplicate vertices, face connectivity and
joint ordering. OBJ polygons, if present, are fan-triangulated as in our original
loader. GT rows are normalized as in the experiment loader. Rest transforms are
identity rotations with the source joint positions as translations, matching
the evaluator; `parents` comes from the source hierarchy. Non-topological joint
ordering is rejected, not silently rearranged, since reordering would change
the seeded per-joint poses.

## 4. Refine and evaluate one mesh

This example uses an existing RigNetv1 sample and explicit GT input, so no model
prediction is needed. To use predictions, replace `--use-gt-weights` with
`--predictions-dir predictions/rignetv1`.

```bash
python scripts/prepare_data.py --dataset rignetv1 \
  --input data/ModelResource_RigNetv1_preproccessed.zip \
  --sample-id 10034 --use-gt-weights --output-dir data/prepared/rignetv1_one

python -m mesh_refinement refine \
  --input data/prepared/rignetv1_one/10034.npz \
  --output outputs/10034/refined.npz --config config/fusion_config.yaml

python -m mesh_refinement evaluate \
  --input data/prepared/rignetv1_one/10034.npz \
  --refined outputs/10034/refined.npz --output outputs/10034/metrics.json
```

For matched dataset evaluation, use the original evaluation ordinal as
`--sample-index`; do not renumber a filtered subset. The manifest's `source_index`
is relative to the supplied source sequence (sorted OBJ names or AXL records),
not necessarily the ordinal of a separately filtered paper evaluation.
See [evaluation](evaluation.md) for seeds and metric aggregation.

## Full converter options

| Option | Meaning |
| --- | --- |
| `--dataset {rignetv1,axl2}` | Required source format. |
| `--input PATH` | Required RigNet ZIP/extracted root, AXL NPZ or local shard directory. |
| `--output-dir PATH` | Required new directory. Existing directories are refused. |
| `--predictions-dir PATH` | Matching `<sample_id>_skin.npy` files. Mutually exclusive with GT input. |
| `--use-gt-weights` | Explicit GT input instead of predictions. Exactly one weight source is required. |
| `--trust-axl-pickle` | Required for trusted AXL object records; not needed for RigNet. |
| `--sample-id ID` | Optional single sample. Default converts the entire supplied source. |
| `-h`, `--help` | Print usage. |

Each numeric NPZ includes `vertices`, `faces`, `skin`, `gt_skin`, `matrix_local`,
`parents`, `joint_names` and `metadata_json`. Metadata labels the weight source;
GT references are not fused. See [usage](usage.md) for array shapes and outputs.
