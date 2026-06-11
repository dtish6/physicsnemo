# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the direct (active-DOF) solver against the iterative AMG solver.

Runs the SAME rasterised plan through both solvers and compares the
displacement fields -- they should match to round-off (void with Emin
contributes ~nothing), confirming the direct path is correct. Also reports
the wall-clock time of each so we can see the speedup.

Usage:
    python step2_fea/support/validate_direct.py --plan 0 --grid_size 64 --grid_z 24
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1_preprocess._1_1_binarize import rasterize           # noqa: E402
from step1_preprocess.generate_from_csv import build_sample     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1")
    ap.add_argument("--plan", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=64, help="Horizontal H=W.")
    ap.add_argument("--grid_z", type=int, default=24, help="Vertical D.")
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--skip_amg", action="store_true", help="Only run direct (for big grids).")
    args = ap.parse_args()

    fp = os.path.join(args.csv_dir, f"plan_{args.plan}.csv")
    df = pd.read_csv(fp, usecols=["x", "y", "z", "value"])
    grid = (args.grid_z, args.grid_size, args.grid_size)
    rho = rasterize(df, args.voxel_size, grid_size=grid)
    solid = int((rho > 0.5).sum())
    print(f"plan_{args.plan}: grid(D,H,W)={rho.shape} solid={solid} "
          f"({100.0*solid/rho.size:.1f}%)", flush=True)

    # --- Direct ---
    t = time.perf_counter()
    d_dir = build_sample(rho, args.voxel_size, dict(precond="direct", verbose=False))
    t_dir = time.perf_counter() - t
    u_dir = d_dir["displacement"]
    i_dir = d_dir["_info"]
    print(f"\nDIRECT : {t_dir:6.1f}s  backend={i_dir.get('backend')}  "
          f"active_dofs={i_dir.get('n_active_dofs')}/{i_dir.get('n_free_dofs')}  "
          f"rel_resid={i_dir['residual']:.2e}  converged={i_dir['converged']}  "
          f"|u|max={np.abs(u_dir).max():.4e}", flush=True)

    if args.skip_amg:
        return

    # --- AMG (reference) ---
    t = time.perf_counter()
    d_amg = build_sample(rho, args.voxel_size, dict(precond="amg", tol=1e-8, maxiter=5000))
    t_amg = time.perf_counter() - t
    u_amg = d_amg["displacement"]
    i_amg = d_amg["_info"]
    print(f"AMG    : {t_amg:6.1f}s  CG={i_amg['iters']}it  "
          f"rel_resid={i_amg['residual']:.2e}  converged={i_amg['converged']}  "
          f"|u|max={np.abs(u_amg).max():.4e}", flush=True)

    # --- Compare ---
    solid_mask = rho > 0.5                          # (Nz,Ny,Nx) per-element
    sm = np.broadcast_to(solid_mask, u_dir.shape)   # (3,Nz,Ny,Nx)

    def _stats(a, b, mask=None):
        if mask is not None:
            a, b = a[mask], b[mask]
        d = float(np.abs(a - b).max())
        s = float(np.abs(b).max()) + 1e-30
        return d, d / s

    d_all, r_all = _stats(u_dir, u_amg)
    d_sol, r_sol = _stats(u_dir, u_amg, sm)
    print(f"\n--- comparison ---")
    print(f"  WHOLE field : |u|max dir={np.abs(u_dir).max():.3e} amg={np.abs(u_amg).max():.3e}"
          f"  max_abs_diff={d_all:.3e}  rel={r_all:.3e}")
    print(f"  SOLID only  : |u|max dir={np.abs(u_dir[sm]).max():.3e} amg={np.abs(u_amg[sm]).max():.3e}"
          f"  max_abs_diff={d_sol:.3e}  rel={r_sol:.3e}")
    print(f"  speedup (amg/dir) : {t_amg / max(t_dir, 1e-9):.1f}x")
    ok = r_sol < 1e-3
    print(f"\n  {'MATCH in solid (direct is correct)' if ok else 'MISMATCH in solid -- investigate'}")


if __name__ == "__main__":
    main()
