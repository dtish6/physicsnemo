# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check whether any floorplan CSV exceeds a fixed voxel bounding box.

Target box: 128 voxels per axis at 10 cm/voxel => 1280 cm (12.8 m) per axis.
Reports per-plan extent (full grid and solid-only) in voxels, and flags any
plan whose extent exceeds 128 voxels in x, y, or z.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

CM_PER_VOXEL = 10.0
N_VOX = 128
BOX_CM = CM_PER_VOXEL * N_VOX  # 1280 cm


def _idx(p):
    m = re.search(r"plan_(\d+)\.csv$", os.path.basename(p))
    return int(m.group(1)) if m else -1


def detect_spacing(a):
    u = np.unique(a)
    return float(np.median(np.diff(u))) if u.size > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"D:\MultiOpt\ResPlanData\ResPlan\fields_csv")
    ap.add_argument("--voxel_cm", type=float, default=CM_PER_VOXEL)
    ap.add_argument("--n_vox", type=int, default=N_VOX)
    args = ap.parse_args()

    box_cm = args.voxel_cm * args.n_vox
    files = sorted(
        (f for f in glob.glob(os.path.join(args.csv_dir, "plan_*.csv"))
         if not f.endswith("_geom.csv")),
        key=_idx,
    )
    if not files:
        raise SystemExit(f"No plan_*.csv in {args.csv_dir}")

    print(f"Box: {args.n_vox} vox x {args.voxel_cm} cm = {box_cm:.0f} cm per axis")
    print(f"Checking {len(files)} plans. Coordinate unit auto-detected per axis.\n")

    over = []
    crop_losses = []
    max_span_vox = np.zeros(3)
    # Track global coordinate spacing to infer units (cm vs m).
    spac_samples = []
    for f in files:
        df = pd.read_csv(f, usecols=["x", "y", "z", "value"])
        xyz = df[["x", "y", "z"]].to_numpy(np.float64)
        v = df["value"].to_numpy(np.float64)
        sp = detect_spacing(xyz[:, 0])
        spac_samples.append(sp)
        # Convert coords to cm. If spacing ~0.1 -> metres; if ~10 -> already cm.
        unit_cm = 100.0 if sp < 1.0 else 1.0
        coords_cm = xyz * unit_cm

        full_span = coords_cm.max(0) - coords_cm.min(0)
        full_vox = np.rint(full_span / args.voxel_cm).astype(int) + 1  # inclusive cells
        max_span_vox = np.maximum(max_span_vox, full_vox)

        solid = coords_cm[np.isclose(v, 1.0)]
        if solid.size:
            sspan = solid.max(0) - solid.min(0)
            svox = np.rint(sspan / args.voxel_cm).astype(int) + 1
            # Crop loss: solid voxels whose index (anchored at the plan's min
            # corner, positive direction) falls outside [0, n_vox) in any axis.
            idx = np.rint((solid - solid.min(0)) / args.voxel_cm).astype(int)
            inside = np.all((idx >= 0) & (idx < args.n_vox), axis=1)
            loss_pct = 100.0 * (1.0 - inside.mean())
        else:
            svox = np.zeros(3, int)
            loss_pct = 0.0
        crop_losses.append(loss_pct)

        exceeds = full_vox > args.n_vox
        tag = "OVER " if exceeds.any() else "  ok "
        if exceeds.any():
            over.append((os.path.basename(f), full_vox.tolist(), svox.tolist(), loss_pct))
        print(f"  {tag} {os.path.basename(f):14s} full(vox) x,y,z="
              f"{full_vox.tolist()}  solid(vox)={svox.tolist()}  crop_loss={loss_pct:5.1f}%")

    print("\n--- summary ---")
    print(f"coord spacing seen (axis x): min={np.nanmin(spac_samples):.4g} "
          f"max={np.nanmax(spac_samples):.4g}  "
          f"(=> unit {'metres' if np.nanmin(spac_samples) < 1.0 else 'cm'})")
    print(f"max full-grid extent across all plans (vox D,H,W) = {max_span_vox.astype(int).tolist()}")
    print(f"box limit = {args.n_vox} vox per axis")
    cl = np.array(crop_losses)
    print(f"\ncrop loss (solid voxels cut by origin-anchored {args.n_vox}^3 crop):")
    print(f"  plans with ANY loss : {(cl > 0).sum()} / {len(files)}")
    print(f"  mean loss           : {cl.mean():.1f}%")
    print(f"  median loss         : {np.median(cl):.1f}%")
    print(f"  max loss            : {cl.max():.1f}%  ({os.path.basename(files[int(cl.argmax())])})")
    print(f"  plans losing >25%   : {(cl > 25).sum()}")
    if over:
        print(f"\n{len(over)} / {len(files)} plans EXCEED the {args.n_vox}^3 box.")
    else:
        print(f"\nAll {len(files)} plans FIT within the {args.n_vox}^3 box.")


if __name__ == "__main__":
    main()
