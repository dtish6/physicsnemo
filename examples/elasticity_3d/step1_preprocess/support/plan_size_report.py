# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: floorplan size distribution + box-size vs cropping trade-off.

Helps choose a voxel box that reduces crop loss while keeping real physical
scale. Reports, per candidate horizontal box B: fraction of plans fully
contained, mean fraction of solid wall voxels cropped, and the voxel size B/N
that box implies at a few grid resolutions.
"""

import argparse
import glob
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir",
                    default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1")
    ap.add_argument("--vbox", type=float, default=6.0, help="vertical box [m] for crop calc")
    args = ap.parse_args()

    files = [f for f in glob.glob(args.csv_dir + "/plan_*.csv")
             if not f.endswith("_geom.csv")]
    hx, hz, S = [], [], []
    for fp in files:
        df = pd.read_csv(fp, usecols=["x", "y", "z", "value"])
        s = df[np.isclose(df["value"], 1.0)]
        if len(s) == 0:
            continue
        x = s["x"].to_numpy(); y = s["y"].to_numpy(); z = s["z"].to_numpy()
        hx.append(max(x.max() - x.min(), y.max() - y.min()))
        hz.append(z.max() - z.min())
        S.append((x - x.min(), y - y.min(), z - z.min(), len(s)))
    hx = np.array(hx); hz = np.array(hz)

    print(f"plans={len(S)}")
    print("HORIZONTAL extent max(x,y) [m] percentiles:")
    for p in (50, 75, 90, 95, 100):
        print(f"   p{p:<3} = {np.percentile(hx, p):5.1f}")
    print("VERTICAL extent z [m] percentiles:")
    for p in (50, 90, 95, 100):
        print(f"   p{p:<3} = {np.percentile(hz, p):5.1f}")

    print(f"\nBox sweep (horizontal B, vertical fixed {args.vbox} m):")
    print(f"  {'box[m]':>7} {'%plans full':>11} {'mean crop':>10} {'max crop':>9}"
          f"   vox@32   vox@48   vox@64")
    for B in (12.8, 16.0, 19.2, 22.4, 25.6, float(np.ceil(hx.max()))):
        loss = []; full = 0
        for (xx, yy, zz, n) in S:
            keep = int(((xx < B) & (yy < B) & (zz < args.vbox)).sum())
            loss.append(1 - keep / n); full += int(keep == n)
        loss = np.array(loss)
        print(f"  {B:>7.1f} {100*full/len(S):>10.0f}% {100*loss.mean():>9.1f}%"
              f" {100*loss.max():>8.1f}%   {B/32:5.2f}m   {B/48:5.2f}m   {B/64:5.2f}m")


if __name__ == "__main__":
    main()
