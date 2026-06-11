# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physics-sanity QA over generated FEA samples.

For every sample_*.h5 it checks that the displacement field obeys the physics
we expect from a gravity-loaded, base-clamped structure:

  1. finite        -- no NaN / Inf.
  2. void is zero  -- displacement ~0 outside the solid (direct solver).
  3. clamped base  -- displacement near the z=0 plane << displacement at the top.
  4. sags down     -- mean vertical displacement uz in the solid is negative.
  5. plausible mag -- |u|max within a sane range (flags blow-ups like floating solid).

Prints a per-sample line and an aggregate PASS/FAIL with any flagged files.

Usage:
    python step2_fea/support/qa_dataset.py --data_dir ./data/train
"""

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

UMAX_HI = 1e-2     # m  -- > 1 cm under self-weight is suspicious for stiff material
UMAX_LO = 1e-9     # m  -- essentially no deformation -> something missing
# A 1-element shell of void touching solid is legitimately nonzero (element-
# center averaging shares solid corner nodes). Only DEEP void (no solid
# neighbour) must be ~0, and total void must not BLOW UP past the solid.
DEEP_VOID_TOL = 1e-3   # deep-void |u|max < this * solid |u|max
BLOWUP_TOL = 2.0       # any void |u|max > this * solid |u|max => singular/floating


def qa_file(fp):
    with h5py.File(fp, "r") as f:
        mask = f["solid_mask"][:] > 0.5            # (D,H,W)
        disp = f["displacement"][:].astype(np.float64)  # (3,D,H,W)
    umag = np.sqrt((disp ** 2).sum(axis=0))        # (D,H,W)
    flags = []

    finite = bool(np.isfinite(disp).all())
    if not finite:
        flags.append("NaN/Inf")

    n_solid = int(mask.sum())
    if n_solid == 0:
        flags.append("no-solid")
        return dict(umax=float(umag.max()), flags=flags, n_solid=0,
                    void_ratio=np.nan, uz_mean=np.nan, base_top=np.nan)

    umax_solid = float(umag[mask].max())
    umax_void = float(umag[~mask].max()) if (~mask).any() else 0.0
    void_ratio = umax_void / (umax_solid + 1e-300)
    # Deep void = void voxels sharing NO node with any solid element. A node is
    # shared across a 26-neighbourhood, so erode the void by a full 3x3x3 shell.
    void = ~mask
    deep_void = binary_erosion(void, structure=np.ones((3, 3, 3), bool))
    deep_void_max = float(umag[deep_void].max()) if deep_void.any() else 0.0
    deep_ratio = deep_void_max / (umax_solid + 1e-300)
    if void_ratio > BLOWUP_TOL:
        flags.append(f"void-BLOWUP({void_ratio:.1e})")     # floating/singular
    if deep_ratio > DEEP_VOID_TOL:
        flags.append(f"deep-void-nonzero({deep_ratio:.1e})")

    # gravity: mean vertical displacement in solid should be negative (downward)
    uz_mean = float(disp[2][mask].mean())
    if uz_mean >= 0:
        flags.append(f"not-sagging(uz_mean={uz_mean:.1e})")

    # clamped base: solid displacement in the bottom 25% of occupied height
    # should be much smaller than in the top 25%.
    zs = np.where(mask.any(axis=(1, 2)))[0]
    base_top = np.nan
    if zs.size:
        z0, z1 = zs.min(), zs.max()
        span = max(z1 - z0, 1)
        lo = z0 + int(0.25 * span)
        hi = z1 - int(0.25 * span)
        base_m = mask.copy(); base_m[lo:] = False           # bottom quartile
        top_m = mask.copy();  top_m[:max(hi, lo + 1)] = False  # top quartile
        b = umag[base_m].mean() if base_m.any() else 0.0
        t = umag[top_m].mean() if top_m.any() else 0.0
        base_top = b / (t + 1e-300)
        if base_top > 0.5:
            flags.append(f"base-not-clamped(base/top={base_top:.2f})")

    if umax_solid > UMAX_HI:
        flags.append(f"too-large({umax_solid:.1e})")
    if umax_solid < UMAX_LO:
        flags.append(f"too-small({umax_solid:.1e})")

    return dict(umax=umax_solid, void_ratio=void_ratio, deep_ratio=deep_ratio,
                uz_mean=uz_mean, base_top=base_top, n_solid=n_solid, flags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data/train")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 in {args.data_dir}")

    bad = []
    print(f"QA over {len(files)} samples in {args.data_dir}\n")
    print(f"{'file':22s} {'|u|max':>10s} {'deep_void':>10s} {'uz_mean':>10s} {'base/top':>9s}  flags")
    for fp in files:
        r = qa_file(fp)
        if r["flags"]:
            bad.append((os.path.basename(fp), r["flags"]))
        print(f"{os.path.basename(fp):22s} {r['umax']:10.2e} {r['deep_ratio']:10.1e} "
              f"{r['uz_mean']:10.2e} {r['base_top']:9.2f}  {','.join(r['flags']) or 'ok'}")

    print(f"\n{'='*60}")
    if bad:
        print(f"FAIL: {len(bad)}/{len(files)} samples flagged:")
        for name, fl in bad:
            print(f"  {name}: {', '.join(fl)}")
    else:
        print(f"PASS: all {len(files)} samples physically sane "
              f"(finite, void~0, base-clamped, sagging, plausible magnitude).")


if __name__ == "__main__":
    main()
