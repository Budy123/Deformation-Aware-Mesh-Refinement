<div align="center">

<h1>Deformation-Aware Mesh Refinement<br>for Robust Automatic Rigging</h1>

<p><strong>Hyun Wook Kim · Sanglim Han · Hak Gu Kim</strong><br>Chung-Ang University</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#documentation">Documentation</a>
</p>

</div>

## Overview

Duplicate vertices in rigged meshes can receive different skinning weights and
separate during deformation. Our method combines surface orientation partitioning,
density-weighted skinning fusion and vertex merging to refine these duplicates
without relocating vertices or modifying the skeleton.

```mermaid
flowchart LR
    A["Group by vertex position"] --> B["Partition by face orientation"]
    B --> C["Fuse skinning weights"]
    C --> D["Merge vertices and clean faces"]
```

This CPU-only implementation takes a triangle mesh and existing skinning weights
as input. No training or pretrained checkpoint is required. It provides KDE and
Mean fusion, plus Skin L1, DE, FPR, CD and NME evaluation.

## Installation

Tested with Python 3.8.10, NumPy 1.24.4, SciPy 1.10.1 and PyYAML 6.0.3.

```bash
git clone https://github.com/Budy123/Deformation-Aware-Mesh-Refinement.git
cd Deformation-Aware-Mesh-Refinement

python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install .
```

An existing Python environment can also be used. CUDA and Blender are not required.

## Quick Start

Run the included synthetic demo without downloading a dataset:

```bash
python scripts/make_example.py --output outputs/demo/input.npz

python -m mesh_refinement refine \
  --input outputs/demo/input.npz \
  --output outputs/demo/refined.npz \
  --config config/fusion_config.yaml

python -m mesh_refinement evaluate \
  --input outputs/demo/input.npz \
  --refined outputs/demo/refined.npz \
  --output outputs/demo/metrics.json
```

The commands produce a refined NPZ and a JSON metric report. Choose fresh output
paths when repeating a run; existing files are not overwritten.

For your own mesh, prepare a numeric NPZ with `vertices`, `faces` and predicted
`skin`. Evaluation additionally requires original `gt_skin`; DE also requires
the skeleton. See [data preparation and usage](docs/usage.md) for the full schema.

## Configuration

Edit [config/fusion_config.yaml](config/fusion_config.yaml) to change fusion
settings. The default configuration matches the recorded experiment settings.
Use `--fusion mean` to compare uniform averaging with the same refinement pipeline.

## Repository Structure

```text
config/             YAML settings
docs/               Usage, evaluation and testing reference
mesh_refinement/    Refinement, fusion, metrics and CLI
scripts/            Synthetic demo generation
tests/              Unit and CLI regression tests
licenses/           Retained third-party licenses
```

## Documentation

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Input/output formats, configuration, Python API and CLI options |
| [Evaluation](docs/evaluation.md) | Metric definitions, pose sampling and reproduction scope |
| [Testing](docs/testing.md) | Test commands, numerical checks and source provenance |

Datasets and model predictions are not included. The package processes numeric
NPZ files; it does not run auto-rigging models or import/export FBX.

## Citation

```bibtex
@misc{kim2026deformation,
  title={Deformation-Aware Mesh Refinement for Robust Automatic Rigging},
  author={Kim, Hyun Wook and Han, Sanglim and Kim, Hak Gu},
  year={2026},
  note={Manuscript}
}
```

## Acknowledgements and License

The research pipeline builds on [Puppeteer](https://github.com/Seed3D/Puppeteer),
with evaluation conventions based on [UniRig](https://github.com/VAST-AI-Research/UniRig).
Neither project is a runtime dependency of this package.

Released under [Apache 2.0](LICENSE). See [NOTICE.md](NOTICE.md) for attribution.
