# Attribution and license

Copyright 2026 Budy123 (new code and modifications).

This repository provides mesh-refinement paper code; it is not an official
Puppeteer or UniRig release. The code is provided under the Apache License,
Version 2.0 in `LICENSE`, as selected by the repository owner. Existing
third-party attribution and license notices are retained.

Production source provenance is recorded in `docs/source_provenance.json`. Topology
and fusion were separated from the existing Puppeteer-based research tree;
evaluation kernels were separated from the existing APF evaluator. The standalone
code includes documented topology and interface adaptations; extraction hashes
identify the original sources, not byte identity of every current function.
The dataset converter adapts the same local RigNet OBJ/rig and AXL sparse-weight
loading conventions; it is independent of the original model runtimes.
The source-tree Puppeteer license is
retained verbatim in `licenses/Puppeteer-APACHE-2.0.txt`.

The skin-weight metric uses the UniRig-style vertex/joint entry mean. The pose
sampling and paired LBS follow the existing local evaluator's UniRig-style
approach. DE here is mean Euclidean distance; it is not UniRig's coordinate-MSE
training loss. UniRig model code, weights and datasets are not bundled.

The single example under examples/axl2_fish/ uses Quaternius Fish1 (CC0 1.0)
via Articulation-XL2.0 (CC BY 4.0), with cached Puppeteer predictions. Its numeric
derivative and preview retain CC BY 4.0 attribution, separately from this
repository's Apache-licensed code. See examples/axl2_fish/README.md for source
credits, license links and modifications. Full datasets are not bundled.
