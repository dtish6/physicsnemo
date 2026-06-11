#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero-pad variable-sized elasticity samples to a common cube.

The CSV-derived samples produced by ``generate_from_csv.py`` all share the
clamped-base D extent but have heterogeneous (H, W) footprints (one wall per
sample).  A standard DataLoader cannot collate mismatched shapes, and the
depth-``d`` U-Net requires every spatial dim divisible by ``2**d``.

This script zero-pads each field on the HIGH end of every spatial axis up to a
fixed target ``(D, H, W)`` and writes the result to a new output directory
(originals are left untouched).  Padding semantics:

  * ``solid_mask``    -> 0  (padded voxels are void; masked_mse ignores them)
  * ``E_normalized``  -> 0  (no material in padding)
  * ``body_force``    -> 0  (no load in padding)
  * ``wind_pressure`` -> 0
  * ``displacement``  -> 0  (clamped/void -> zero displacement target)

After padding the train set, per-channel z-score statistics are recomputed
(vectorised) over the padded data so the normalizer matches what the model
sees.  Field assembly order mirrors ``ElasticityDataset._load_sample``:
input  = [solid_mask, E, fx, fy, fz, wind]  (6 ch)
output = [ux, uy, uz]                        (3 ch)

Usage
-----
    # --target is the output grid shape in VOXELS (cells), not a physical
    # length: D=16, H=64, W=80.  Physical extent = target * voxel_size (m).
    # Each dim must be >= the largest sample and divisible by 2**model_depth.
    python step1_preprocess/_1_4_pad_to_cube.py --target 16 64 80
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

FIELDS_3D = ("solid_mask", "E_normalized", "wind_pressure")  # (D,H,W)
FIELDS_4D = ("body_force", "displacement")                   # (3,D,H,W)


def pad_field(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """Zero-pad the trailing (D, H, W) axes of ``arr`` up to ``target``."""
    spatial = arr.shape[-3:]
    if any(s > t for s, t in zip(spatial, target)):
        raise ValueError(f"Sample dim {spatial} exceeds target {target}.")
    pad_spatial = [(0, t - s) for s, t in zip(spatial, target)]
    lead = arr.ndim - 3
    pad = [(0, 0)] * lead + pad_spatial
    return np.pad(arr, pad, mode="constant", constant_values=0.0)


def pad_dir(src: Path, dst: Path, target: tuple[int, int, int]) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(src / "*.h5")))
    for fp in files:
        with h5py.File(fp, "r") as f:
            out = {k: pad_field(f[k][:].astype(np.float32), target) for k in (*FIELDS_3D, *FIELDS_4D)}
        with h5py.File(dst / Path(fp).name, "w") as g:
            for k, v in out.items():
                g.create_dataset(k, data=v, compression="gzip")
    return len(files)


def recompute_stats(data_dir: Path, out_path: Path) -> None:
    """Vectorised per-channel mean/std over padded train data (6 in, 3 out)."""
    in_sum = np.zeros(6, np.float64); in_sq = np.zeros(6, np.float64); in_n = 0
    out_sum = np.zeros(3, np.float64); out_sq = np.zeros(3, np.float64); out_n = 0
    for fp in sorted(glob.glob(str(data_dir / "*.h5"))):
        with h5py.File(fp, "r") as f:
            x = np.concatenate([
                f["solid_mask"][:][None],
                f["E_normalized"][:][None],
                f["body_force"][:],
                f["wind_pressure"][:][None],
            ], axis=0).astype(np.float64)          # (6,D,H,W)
            y = f["displacement"][:].astype(np.float64)  # (3,D,H,W)
        in_sum += x.reshape(6, -1).sum(1); in_sq += (x.reshape(6, -1) ** 2).sum(1)
        in_n += x[0].size
        out_sum += y.reshape(3, -1).sum(1); out_sq += (y.reshape(3, -1) ** 2).sum(1)
        out_n += y[0].size
    in_mean = in_sum / in_n
    in_std = np.sqrt(np.maximum(in_sq / in_n - in_mean ** 2, 0.0))
    out_mean = out_sum / out_n
    out_std = np.sqrt(np.maximum(out_sq / out_n - out_mean ** 2, 0.0))
    in_std = np.where(in_std < 1e-8, 1.0, in_std).astype(np.float32)
    out_std = np.where(out_std < 1e-8, 1.0, out_std).astype(np.float32)
    np.savez(
        str(out_path),
        input_mean=in_mean.astype(np.float32),
        input_std=in_std,
        output_mean=out_mean.astype(np.float32),
        output_std=out_std,
    )
    print(f"  input_mean ={np.round(in_mean,4)}")
    print(f"  input_std  ={np.round(in_std,4)}")
    print(f"  output_mean={np.round(out_mean,6)}")
    print(f"  output_std ={np.round(out_std,6)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--target", type=int, nargs=3, default=(16, 64, 80), metavar=("D", "H", "W"))
    args = p.parse_args()

    root = Path(args.data_root)
    target = tuple(args.target)
    print(f"Padding to target (D,H,W)={target}")
    for split in ("train", "val"):
        n = pad_dir(root / split, root / f"{split}_padded", target)
        print(f"  {split}: padded {n} samples -> {root / (split + '_padded')}")

    stats_out = root / "normalization_stats_padded.npz"
    print(f"Recomputing stats over padded train set -> {stats_out}")
    recompute_stats(root / "train_padded", stats_out)
    print("Done.")


if __name__ == "__main__":
    main()
