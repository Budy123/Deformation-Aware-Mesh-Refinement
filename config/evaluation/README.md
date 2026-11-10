# Recorded evaluation populations

These compressed JSON manifests contain only the sample IDs and recorded
pose-seed ordinals from the main predicted-input experiment logs:

| Manifest | Population |
| --- | ---: |
| `axl2_train.json.gz` | 46,638 |
| `axl2_test.json.gz` | 1,997 |
| `rignetv1.json.gz` | 2,703 |

No meshes, annotations, predictions or checkpoints are stored here. Download
datasets from their original providers and prepare numeric NPZs using the
[dataset converter](../../docs/data_preparation.md). Use each manifest with the
corresponding prepared input directory. These ID lists are evaluation metadata,
not a new train/test split or a training recipe.

`sample_index` is the saved main-run ordinal, not the current row number or
the converter's `source_index`. For sharded training evaluation it can restart
inside each original shard; `partition` preserves that context. Do not replace
it with a globally renumbered sequence. The runner preserves these values.
`provenance.json` records the source-log hashes and per-partition counts.

The manifests fix the evaluated population and pose seeds. They do not provide
the original model predictions or guarantee paper scores for new predictions.
The supplied runner compares original and refined inputs; it does not execute
Ma et al./Blender or the paper's separate ablations.
