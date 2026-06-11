# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile where time goes in one 128^3 FEA solve.

Rasterises a single plan into the fixed grid, runs the full
assemble -> bcs -> loads -> (precond build + CG) pipeline, and prints the
per-phase wall-clock breakdown so we can see whether the cost is matrix
assembly, AMG setup, or CG iterations -- which decides what's worth reusing.

Usage:
    python step2_fea/support/profile_solve.py --plan 0 --precond amg
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from step1_preprocess._1_1_binarize import rasterize          # noqa: E402
from step1_preprocess.generate_from_csv import build_sample    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1")
    ap.add_argument("--plan", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--precond", default="amg", choices=["jacobi", "amg", "none", "direct"])
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=2000)
    args = ap.parse_args()

    fp = os.path.join(args.csv_dir, f"plan_{args.plan}.csv")
    df = pd.read_csv(fp, usecols=["x", "y", "z", "value"])
    rho = rasterize(df, args.voxel_size, grid_size=args.grid_size)
    solid = int((rho > 0.5).sum())
    print(f"plan_{args.plan}: grid={rho.shape}  solid_voxels={solid} "
          f"({100.0 * solid / rho.size:.1f}% of box)  precond={args.precond}", flush=True)

    data = build_sample(
        rho, args.voxel_size,
        dict(tol=args.tol, maxiter=args.maxiter, precond=args.precond, verbose=False),
    )
    info = data["_info"]

    total = info["t_solve_total_s"] + info["t_assemble_s"] + info["t_bcs_s"] + info["t_loads_s"]
    print("\n--- phase breakdown (seconds) ---", flush=True)
    print(f"  assemble_system (matrix values) : {info['t_assemble_s']:8.1f}")
    print(f"  apply_bcs                        : {info['t_bcs_s']:8.1f}")
    print(f"  apply_loads                      : {info['t_loads_s']:8.1f}")
    print(f"  solve total                      : {info['t_solve_total_s']:8.1f}")
    print(f"     |- preconditioner setup       : {info['precond_setup_s']:8.1f}   <-- AMG 'map'")
    print(f"     |- CG iterations ({info['iters']:4d} its)   : {info['cg_solve_s']:8.1f}")
    print(f"  -------------------------------------------")
    print(f"  TOTAL                            : {total:8.1f}")
    print(f"\n  converged={info['converged']}  residual={info['residual']:.2e}")
    # Fractions
    print("\n--- where the time goes ---", flush=True)
    for name, val in [
        ("assemble", info["t_assemble_s"]),
        ("bcs", info["t_bcs_s"]),
        ("loads", info["t_loads_s"]),
        ("precond_setup", info["precond_setup_s"]),
        ("cg_iters", info["cg_solve_s"]),
    ]:
        print(f"  {name:14s}: {100.0 * val / total:5.1f}%")


if __name__ == "__main__":
    main()
