"""LIBERO labeling adapter: point at task objects over cached DINO grids -> salient patch indices.

Iterates the per-demo VL/DINO grid caches ({stem}__{demo}.npy, shape (T, N, D)) alongside the HDF5
frames, runs the pointer on the CURRENT frame at a regular stride (Table 4 "Sample Rate"), and
saves per-demo salient indices aligned to the grid cache. Stores INDICES only (grid is the source
of truth); WorkspaceGridDataset gathers grid[idx] at load time.

Offline (VLM at train-time). CLI:
    python -m workspace_models.saliency.libero_adapter \
        --hdf5 /path/*_demo.hdf5 --vl-cache-dir .../vl_cache_goal_dinov2 \
        --out-dir .../workspace_salient_goal --sample-rate 5 --max-patches 8
"""

import glob
import os

import numpy as np

from .patchify import points_to_indices
from .pointing import Pointer, MockPointer
from .label import objects_from_task_language


def _stem(hdf5_path: str) -> str:
    return os.path.basename(hdf5_path).replace("_demo.hdf5", "").replace(".hdf5", "")


def label_libero(hdf5_paths: list[str], vl_cache_dir: str, out_dir: str, pointer: Pointer,
                 img_key: str = "agentview_rgb", max_patches: int = 8, sample_rate: int = 5,
                 objects_per_task: dict[str, list[str]] | None = None, grid_hw=None,
                 max_demos: int | None = None):
    """Point over every demo/frame (strided) and save {out_dir}/{stem}__{demo}.npz with
    frame_idx (L,) + salient_idx (L, m). max_demos caps demos per task. Returns total frames."""
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for path in hdf5_paths:
        stem = _stem(path)
        objects = (objects_per_task or {}).get(stem) or objects_from_task_language(stem.replace("_", " "))
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys())
            if max_demos:
                demos = demos[:max_demos]
            print(f"[{stem}] {len(demos)} demos, objects={objects}", flush=True)
            for di, demo in enumerate(demos):
                out_npz = os.path.join(out_dir, f"{stem}__{demo}.npz")
                if os.path.exists(out_npz):                       # resumable: skip done demos
                    continue
                grid = np.load(os.path.join(vl_cache_dir, f"{stem}__{demo}.npy"), mmap_mode="r")
                imgs = f[f"data/{demo}/obs/{img_key}"]           # (T, H, W, C) uint8
                T, N = grid.shape[0], grid.shape[1]
                frame_idx = np.arange(0, T, sample_rate, dtype=np.int64)
                sal = np.stack([
                    points_to_indices(
                        [p for obj in objects for p in pointer.point(np.asarray(imgs[t]), obj)],
                        N, max_patches, grid_hw)
                    for t in frame_idx])
                np.savez(out_npz, frame_idx=frame_idx, salient_idx=sal)
                total += len(frame_idx)
                if di == 0 or (di + 1) % 5 == 0:
                    print(f"[{stem}] demo {di+1}/{len(demos)}  ({total} frames so far)", flush=True)
    return total


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", nargs="+", required=True, help="LIBERO *_demo.hdf5 files (globs ok)")
    ap.add_argument("--vl-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--img-key", default="agentview_rgb")
    ap.add_argument("--max-patches", type=int, default=8)
    ap.add_argument("--sample-rate", type=int, default=5)
    ap.add_argument("--max-demos-per-task", type=int, default=None, help="cap demos/task (subset)")
    ap.add_argument("--molmo", default=None, help="MolmoPoint model id; omit -> MockPointer (test)")
    ap.add_argument("--objects-json", default=None, help="JSON dict {stem: [objects]} (else heuristic)")
    args = ap.parse_args()

    paths = sorted(p for g in args.hdf5 for p in glob.glob(g))
    objects_per_task = None
    if args.objects_json:
        import json
        objects_per_task = json.load(open(args.objects_json))
    if args.molmo:
        from .pointing import MolmoPointer
        pointer = MolmoPointer(args.molmo)
    else:
        print("[warn] no --molmo: using MockPointer (deterministic, NOT real saliency)")
        pointer = MockPointer()
    n = label_libero(paths, args.vl_cache_dir, args.out_dir, pointer, max_patches=args.max_patches,
                     sample_rate=args.sample_rate, objects_per_task=objects_per_task,
                     max_demos=args.max_demos_per_task)
    print(f"[done] labeled {n} frames across {len(paths)} tasks -> {args.out_dir}")
