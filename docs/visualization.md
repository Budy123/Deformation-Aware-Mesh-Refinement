# Visualize your output

[Back to README](../README.md)

Install the optional lightweight renderer dependency in your environment:

```bash
python -m pip install ".[visualization]"
```

Only Pillow is added. No CUDA, Blender, display server or prediction model is
needed. Refine the [bundled example](../examples/axl2_fish/README.md), then run:

```bash
python -m mesh_refinement render \
  --input examples/axl2_fish/input.npz --refined outputs/fish/refined.npz \
  --output-dir outputs/fish/visualization --view side
```

The new directory contains `comparison.png`, `comparison.gif`, `motion.npz`
and `render.json`. Each side uses the weights in its supplied NPZ: original
predictions on the left and the provided refined weights on the right. The
renderer **does not rerun inference** and this is not the fresh re-inference
protocol of the separate README video.

Both meshes share the same motion, camera, lighting and framing over the entire
sequence. A thin, light wireframe and backface culling are always applied. Faces
must use consistent outward winding. This is a simple orthographic CPU preview,
not a material/texture renderer; it does not preserve UVs or custom normals.

With a skeleton, the default motion is a deterministic sinusoidal cycle with
sampled per-joint rotation vectors; root joints stay fixed. This is a visual
inspection motion, not an original asset animation or the metric pose bank.
Without a skeleton the result is explicitly a static rest-pose comparison.
Use `--view front` for the X/Z view or `--view side` for a Y-dominant side view.

## Supply an existing motion

`--motion motion.npz` accepts a numeric `rotvec` array shaped `(frames, J, 3)`,
in radians, aligned with the input skeleton's joint order. Both sides receive
the exact same array; the renderer does not select poses by error. The saved
`motion.npz` can be reused in this way, including its procedural motion.
Original/refined coordinates must retain their exact correspondence; any
supplied refined skeleton must match the original. Missing or inconsistent
skeletons/motion cause an error rather than a silently different animation.

## Full CLI reference

| Option | Default / meaning |
| --- | --- |
| `--input PATH` | Required original numeric NPZ with vertices/faces/skin; optional skeleton. |
| `--refined PATH` | Required refined NPZ with vertices/faces/skin/keep_indices. |
| `--output-dir PATH` | Required new directory; never overwritten. |
| `--view {front,side}` | `front`; shared camera for both meshes. |
| `--motion PATH` | Optional numeric NPZ with rotation vectors; requires a skeleton. |
| `--frames INT` | `48`, at least 2; procedural frame count, ignored with supplied motion. |
| `--fps INT` | `12`, integer 1–50; GIF frame delays are quantized by the format. |
| `--seed INT` | `321`; nonnegative procedural-motion seed. |
| `--angle-deg FLOAT` | `30`; nonnegative procedural rotation-vector component bound. |
| `-h`, `--help` | Print usage. |
