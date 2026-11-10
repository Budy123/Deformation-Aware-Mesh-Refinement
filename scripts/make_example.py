"""Create a tiny numeric demo without downloading or redistributing data."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    vertices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 2, 3], [1, 2, 3]], dtype=np.int64)
    skin = np.array([[0.8, 0.2], [0.6, 0.4], [1, 0], [1, 0]], dtype=np.float32)
    gt = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)
    rest = np.repeat(np.eye(4)[None], 2, axis=0)
    rest[1, 0, 3] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, vertices=vertices, faces=faces, skin=skin,
                            gt_skin=gt, matrix_local=rest, parents=np.array([-1, 0]))


if __name__ == "__main__":
    main()
