# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve one floor-plan and export FEA results for Rhino / Grasshopper.

Writes one CSV row per SOLID voxel with everything needed to visualise:
  - voxel centre point     (x, y, z)       original position, metres, Z-up
  - displaced centre point  (x_def, y_def, z_def)  position AFTER deformation
                            (= original + displacement x --scale; see --scale)
  - displacement vector    (ux, uy, uz) in metres + magnitude |u|
  - von Mises stress        vm          in Pa  (scalar -> colour / contour)
  - max principal stress    (px, py, pz) unit direction + signed magnitude ps
                            (force-flow "stress lines": draw a short segment at
                             each point along this direction)

Coordinate convention matches DESIGN.md: x=W, y=H, z=D (vertical), Z-up.

Grasshopper recipe
------------------
  1. Read File  ->  the CSV.  Split each line by ",".  Skip the header row.
  2. Point cloud A = Point3d from (x, y, z)        -- the original geometry.
     Point cloud B = Point3d from (x_def,y_def,z_def) -- the deformed geometry.
  3. Displacement field:  Line / Vector Display between A and B (one vector per
     row -> the deformation field).  (Real deflections are sub-micron, so the
     export exaggerates them via --scale; see below.)
  4. Von Mises:  remap vm to a colour gradient, paint cloud A (or a mesh).
  5. Stress lines:  Vector from (px,py,pz)*ps (or just the unit dir), draw a
     short Line centred on each point -> force-flow field. Optionally feed the
     direction field into a streamline/flow-line component.

Two point clouds (A original, B displaced) are row-aligned in ONE file, so the
i-th row of A pairs with the i-th row of B -- no risk of mismatch from two files.

Usage
-----
    python step2_fea/support/export_rhino.py --plan 0 --material wood \
        --out ./plan0_wood.csv --stride 2
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step1_preprocess._1_1_binarize import rasterize                       # noqa: E402
from step1_preprocess.generate_from_csv import (                           # noqa: E402
    MATERIALS, GRAVITY, DENSITY_THRESHOLD,
)
from step2_fea._2_2_voxel_fem import VoxelFEMSolver                         # noqa: E402


def principal_stress(stress_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max-|magnitude| principal stress per voxel from Voigt stress.

    stress_voigt: (N, 6) = [sxx, syy, szz, sxy, sxz, syz].
    Returns (dirs (N,3) unit vectors, mags (N,) signed principal stress).
    """
    sxx, syy, szz, sxy, sxz, syz = stress_voigt.T
    T = np.empty((stress_voigt.shape[0], 3, 3), dtype=np.float64)
    T[:, 0, 0] = sxx; T[:, 1, 1] = syy; T[:, 2, 2] = szz
    T[:, 0, 1] = T[:, 1, 0] = sxy
    T[:, 0, 2] = T[:, 2, 0] = sxz
    T[:, 1, 2] = T[:, 2, 1] = syz
    w, v = np.linalg.eigh(T)                       # ascending eigenvalues
    idx = np.argmax(np.abs(w), axis=1)             # most extreme principal
    rows = np.arange(T.shape[0])
    mags = w[rows, idx]
    dirs = v[rows, :, idx]                         # eigenvector columns
    return dirs, mags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1")
    ap.add_argument("--plan", type=int, default=0)
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--grid_z", type=int, default=48)
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--material", default="steel", choices=list(MATERIALS))
    ap.add_argument("--E0", type=float, default=None)
    ap.add_argument("--rho_mat", type=float, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="Export every Nth solid voxel (lighten the file for Grasshopper).")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="Displacement exaggeration for the x_def/y_def/z_def columns. "
                         "0 (default) = auto (max deflection ~3 voxels). 1 = true scale "
                         "(sub-micron, points coincide). Does NOT affect ux/uy/uz.")
    ap.add_argument("--out", default=None, help="Output CSV (default plan<n>_<material>.csv).")
    args = ap.parse_args()

    mat = dict(MATERIALS[args.material])
    if args.E0 is not None:      mat["E0"] = args.E0
    if args.rho_mat is not None: mat["rho_mat"] = args.rho_mat
    if args.nu is not None:      mat["nu"] = args.nu

    fp = os.path.join(args.csv_dir, f"plan_{args.plan}.csv")
    df = pd.read_csv(fp, usecols=["x", "y", "z", "value"])
    grid = (args.grid_z, args.grid_size, args.grid_size)
    rho = rasterize(df, args.voxel_size, grid_size=grid)
    Nz, Ny, Nx = rho.shape
    print(f"plan_{args.plan}: grid={rho.shape} material={args.material} "
          f"E0={mat['E0']:.2e} nu={mat['nu']} rho={mat['rho_mat']}", flush=True)

    # Solve (direct) with von Mises.
    body_force = np.zeros((3, Nz, Ny, Nx), dtype=np.float32)
    body_force[2] = (-mat["rho_mat"] * GRAVITY * rho).astype(np.float32)
    solver = VoxelFEMSolver(grid_size=(Nz, Ny, Nx), voxel_size=args.voxel_size, nu=mat["nu"])
    res = solver.run(rho=rho, E0=mat["E0"], body_force=body_force,
                     compute_vm=True, precond="direct", verbose=False)
    info = res["info"]
    print(f"  solved: converged={info['converged']} res={info['residual']:.1e} "
          f"|u|max={np.abs(res['u_elem']).max():.3e} m", flush=True)
    u_elem = res["u_elem"]                       # (3, Nz, Ny, Nx)
    vm = res["sigma_vm"]                          # (Nz, Ny, Nx)
    _, stress = solver.element_strain_stress(res["u_nodal"])  # (6, Nz, Ny, Nx)

    # Solid voxels (optionally strided).
    mask = rho > DENSITY_THRESHOLD
    didx = np.argwhere(mask)                      # (Nsolid, 3) as (d, h, w)
    if args.stride > 1:
        didx = didx[::args.stride]
    d, h, w = didx[:, 0], didx[:, 1], didx[:, 2]

    vs = args.voxel_size
    x = (w + 0.5) * vs; y = (h + 0.5) * vs; z = (d + 0.5) * vs   # metres, Z-up
    ux = u_elem[0][d, h, w]; uy = u_elem[1][d, h, w]; uz = u_elem[2][d, h, w]
    umag = np.sqrt(ux**2 + uy**2 + uz**2)
    vmv = vm[d, h, w]
    stress_voigt = np.stack([stress[c][d, h, w] for c in range(6)], axis=1)  # (N,6)
    pdirs, pmags = principal_stress(stress_voigt)

    # Displaced voxel centres = original + displacement x scale.  Real
    # deformations are sub-micron, so at scale=1 the displaced points sit on top
    # of the originals (zero-length vectors in Grasshopper).  Pick a scale that
    # makes the largest displacement a sensible fraction of a voxel; auto by
    # default ("--scale 0" -> ~3 voxels of max visual deflection).
    umax = float(umag.max()) if umag.size else 0.0
    scale = args.scale
    if scale <= 0.0:
        scale = (3.0 * vs / umax) if umax > 0.0 else 1.0
    xd = x + ux * scale; yd = y + uy * scale; zd = z + uz * scale

    out = args.out or f"plan{args.plan}_{args.material}.csv"
    cols = ["x", "y", "z", "x_def", "y_def", "z_def",
            "ux", "uy", "uz", "umag", "von_mises", "px", "py", "pz", "ps"]
    data = np.column_stack([x, y, z, xd, yd, zd, ux, uy, uz, umag, vmv,
                            pdirs[:, 0], pdirs[:, 1], pdirs[:, 2], pmags])
    pd.DataFrame(data, columns=cols).to_csv(out, index=False, float_format="%.6e")
    print(f"  wrote {len(data)} solid voxels -> {out}", flush=True)
    print(f"  displaced points scaled x{scale:.3g} "
          f"(|u|max {umax:.2e} m -> {umax*scale:.3g} m visual)", flush=True)
    print(f"  |u| range [{umag.min():.2e}, {umag.max():.2e}] m | "
          f"von Mises range [{vmv.min():.2e}, {vmv.max():.2e}] Pa", flush=True)


if __name__ == "__main__":
    main()
