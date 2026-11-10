"""Convert downloaded RigNetv1/AXL 2.0 data to the refinement input format."""

import argparse
from pathlib import Path

from mesh_refinement.data import convert_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("rignetv1", "axl2"), required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="RigNetv1 preprocessed ZIP/root, or AXL object NPZ/shard directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--predictions-dir", type=Path, help="Puppeteer <sample_id>_skin.npy files")
    weights.add_argument("--use-gt-weights", action="store_true", help="Explicit GT-input run, not the paper's predicted-input evaluation")
    parser.add_argument("--trust-axl-pickle", action="store_true",
                        help="Allow trusted official AXL object arrays to be deserialized")
    parser.add_argument("--sample-id", help="Convert only this mesh; default converts all records")
    args = parser.parse_args()
    result = convert_dataset(args.dataset, args.input, args.output_dir,
        predictions_dir=args.predictions_dir, use_gt_weights=args.use_gt_weights,
        trust_axl_pickle=args.trust_axl_pickle, sample_id=args.sample_id)
    print("Converted {} {} meshes ({}) to {}".format(
        len(result["records"]), result["dataset"], result["weights"], args.output_dir))


if __name__ == "__main__":
    main()
