# Testing and source provenance

[Back to README](../README.md)

## Run the tests

From the repository root in the installed environment:

```bash
python -m unittest discover -s tests -v
```

The 57-test suite covers the bundled real-mesh example, dataset conversion,
grouping and surface orientation, KDE/Mean settings,
face cleanup and mappings, original GT references, metric reductions,
deterministic poses, YAML configuration and CLI precedence, input validation,
unsafe YAML-tag rejection, JSON metadata and overwrite refusal. Converter tests
cover RigNet ZIP/extracted roots, AXL sparse/sharded records, explicit pickle
trust, prediction attachment, source ordinals and rejected malformed inputs.
The bundled-example tests verify numeric data, provenance hashes, the expected
1,098-to-274 vertex reduction, face retention and complete 64-pose evaluation.

The tested CPU environment is Python 3.8.10, NumPy 1.24.4, SciPy 1.10.1 and
PyYAML 6.0.3, as pinned in [requirements.txt](../requirements.txt).

## Numerical regression checks

During configuration centralization, 512 synthetic refinement comparisons
matched the pre-change implementation exactly for geometry, skinning weights,
mappings and existing numerical diagnostics. Another 32 paired complete metric
dictionaries matched across both output phases with 64 poses. Added metadata
fields and Mean's explicit resolved scale representation were excluded from
that numerical comparison. Later YAML/path changes did not change the kernels.

These are bounded implementation checks, not a full-dataset rerun or evidence
of improved reconstruction or deformation accuracy.

## Provenance and licensing

[source_provenance.json](source_provenance.json) records original research
source versions, function lists and extraction hashes. The current standalone
modules include documented adaptations; those hashes identify the original
extraction rather than byte identity with every current function.

See [NOTICE.md](../NOTICE.md), [LICENSE](../LICENSE) and the preserved
[Puppeteer license](../licenses/Puppeteer-APACHE-2.0.txt) for attribution.
