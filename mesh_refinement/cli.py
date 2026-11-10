"""Portable NPZ CLI; no project paths, pickle loading, inference or overwrites."""

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from . import __version__
from .core import refine_mesh, validate_skin
from .fusion import SkinningFusionConfig
from .metrics import evaluate_mesh


def _load_npz(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(keys) - set(archive.files)
        if missing:
            raise ValueError("Missing NPZ fields: " + ", ".join(sorted(missing)))
        # Ignore unrelated optional attributes rather than loading object arrays.
        arrays = {key: archive[key] for key in keys}
        for key in ("gt_skin", "matrix_local", "parents", "metadata_json"):
            if key in archive.files and key not in arrays:
                arrays[key] = archive[key]
    return arrays


def _fusion_config(args):
    values = {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("fusion config must be a YAML mapping")
    if args.fusion is not None:
        values["policy"] = "uniform" if args.fusion == "mean" else "gaussian_l2"
    if args.bandwidth_scale is not None:
        values["adaptive_scale"] = args.bandwidth_scale
    try:
        return SkinningFusionConfig(**values)
    except TypeError as exc:
        raise ValueError("Unsupported fusion config field: " + str(exc)) from exc


def _fresh_path(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refine = sub.add_parser("refine", help="Refine an NPZ containing vertices, faces and skin")
    refine.add_argument("--input", type=Path, required=True)
    refine.add_argument("--output", type=Path, required=True)
    refine.add_argument("--phase", choices=("analysis", "evaluation"), default="evaluation")
    refine.add_argument("--config", type=Path, help="Optional YAML SkinningFusionConfig; omitted fields use paper defaults")
    refine.add_argument("--fusion", choices=("kde", "mean"), default=None,
                        help="Override the config policy; default without a config is kde")
    refine.add_argument("--bandwidth-scale", type=float, default=None,
                        help="Override the config bandwidth scale; paper default is 0.5")
    evaluate = sub.add_parser("evaluate", help="Evaluate a refined NPZ against original GT")
    evaluate.add_argument("--input", type=Path, required=True, help="Original vertices, faces, gt_skin; optional skeleton")
    evaluate.add_argument("--refined", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True, help="Fresh JSON output path")
    evaluate.add_argument("--pose-count", type=int, default=64)
    evaluate.add_argument("--pose-angle-deg", type=float, default=30.0)
    evaluate.add_argument("--seed", type=int, default=321)
    evaluate.add_argument("--sample-index", type=int, default=0,
                          help="Original main-run ordinal for the per-asset pose seed")
    args = parser.parse_args(argv)
    # Check before loading/computing; exclusive creation below also prevents races.
    output = _fresh_path(args.output)
    runtime = {"release_version": __version__, "python": platform.python_version(),
               "numpy": np.__version__}
    if args.command == "refine":
        data = _load_npz(args.input, ("vertices", "faces", "skin"))
        if "gt_skin" in data:
            validate_skin(data["gt_skin"], len(data["vertices"]), data["skin"].shape[1])
        config = _fusion_config(args)
        result = refine_mesh(data["vertices"], data["faces"], data["skin"], fusion_config=config)
        phase = result[args.phase]
        arrays = {k: v for k, v in phase.items() if isinstance(v, np.ndarray)}
        # Ground truth is never fused: it is mapped via original retained indices.
        if "gt_skin" in data:
            arrays["gt_skin"] = data["gt_skin"][phase["keep_indices"]]
        for key in ("matrix_local", "parents"):
            if key in data:
                arrays[key] = data[key]
        metadata = {**runtime, "phase": args.phase,
                    "fusion": result["skin_fusion_config"],
                    "removed_vertex_count_by_merge": result["removed_vertex_count"],
                    "issue_types": result["issue_types"]}
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
        with output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        print(json.dumps(metadata, allow_nan=False))
    else:
        data = _load_npz(args.input, ("vertices", "faces", "gt_skin"))
        refined = _load_npz(args.refined, ("vertices", "faces", "skin", "keep_indices"))
        has_skeleton = ("matrix_local" in data, "parents" in data)
        if any(has_skeleton) and not all(has_skeleton):
            raise ValueError("Provide both matrix_local and parents, or neither")
        skeleton = {k: data[k] for k in ("matrix_local", "parents")} if all(has_skeleton) else None
        result = evaluate_mesh(data["vertices"], data["faces"], data["gt_skin"], refined,
                    skeleton=skeleton, pose_count=args.pose_count,
                    pose_angle_deg=args.pose_angle_deg, seed=args.seed, sample_index=args.sample_index)
        result["runtime"] = runtime
        result["refinement"] = None
        if "metadata_json" in refined:
            metadata = json.loads(str(refined["metadata_json"]))
            if not isinstance(metadata, dict):
                raise ValueError("refinement metadata must be a JSON object")
            result["refinement"] = metadata
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, allow_nan=False))
