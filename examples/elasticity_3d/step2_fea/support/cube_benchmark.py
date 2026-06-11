# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical verification: a solid cube, fixed base, uniform load on top.

Default case (matches the hand-check):
  geometry : 1.0 x 1.0 x 1.0 m cube, voxel 0.1 m -> 10 x 10 x 10 elements
  support  : bottom face (z=0 plane) fully clamped (u = 0)
  load     : -1000 N total, uniform pressure on the TOP face (z+), pointing -z
  material : isotropic, E = 2700 MPa, nu = 0.3, density 1.25 kg/m^3

The top load is applied as CONSISTENT nodal point loads (interior nodes get the
full tributary force, edge nodes half, corner nodes a quarter) so the resultant
is exactly the requested total and represents a true uniform pressure -- without
the spurious side-face loading that `wind_pressure` would add on the perimeter.

Outputs the FEA setup card + result, compares to the 1D analytical estimate,
and (optionally) writes an .h5 / Grasshopper CSV.

Usage
-----
    python step2_fea/support/cube_benchmark.py
    python step2_fea/support/cube_benchmark.py --n 20 --force -5000 --gravity
    python step2_fea/support/cube_benchmark.py --csv cube.csv      # for Grasshopper
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from step2_fea._2_2_voxel_fem import VoxelFEMSolver   # noqa: E402


def top_face_point_loads(n_el: int, h: float, total_fz: float):
    """Consistent nodal loads for a uniform pressure on the top (z+) face.

    n_el : elements per side (so n_el+1 nodes per side).
    h    : voxel size (m).   total_fz : resultant z-force (N), e.g. -1000.
    Returns a list of (global_node_index, [0, 0, Fz]).
    """
    nn = n_el + 1                       # nodes per axis
    iz_top = n_el                       # top node layer index
    area = (n_el * h) ** 2              # loaded face area (m^2)
    p = total_fz / area                 # equivalent uniform pressure (Pa)

    # Per-axis tributary weight: 1.0 interior, 0.5 on the two boundary nodes.
    w = np.ones(nn); w[0] = w[-1] = 0.5
    loads = []
    for iy in range(nn):
        for ix in range(nn):
            fz = p * (h * h) * w[ix] * w[iy]          # consistent nodal force
            node = iz_top * (nn * nn) + iy * nn + ix  # global node index
            loads.append((node, np.array([0.0, 0.0, fz], dtype=np.float64)))
    return loads, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="Elements per side (default 10 -> 10cm voxels).")
    ap.add_argument("--size", type=float, default=1.0, help="Cube edge length (m, default 1.0).")
    ap.add_argument("--E", type=float, default=2700e6, help="Young's modulus (Pa, default 2700 MPa).")
    ap.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio (default 0.3).")
    ap.add_argument("--rho_mat", type=float, default=1.25, help="Material density (kg/m^3).")
    ap.add_argument("--force", type=float, default=-1000.0, help="Total z-force on top (N, default -1000).")
    ap.add_argument("--gravity", action="store_true", help="Also apply self-weight (-rho*g in z).")
    ap.add_argument("--csv", default=None, help="Also write a Grasshopper CSV here.")
    args = ap.parse_args()

    n = args.n
    h = args.size / n
    grid = (n, n, n)                       # (Nz, Ny, Nx)
    rho = np.ones(grid, dtype=np.float64)  # fully solid cube

    solver = VoxelFEMSolver(grid_size=grid, voxel_size=h, nu=args.nu)
    solver.assemble_system(rho, E0=args.E)
    solver.apply_bcs()                     # clamp z=0 plane (bottom face)

    loads, p = top_face_point_loads(n, h, args.force)
    body_force = None
    if args.gravity:
        body_force = np.zeros((3, *grid))
        body_force[2] = -args.rho_mat * 9.81
    solver.apply_loads(body_force=body_force, point_loads=loads)

    u_nodal, u_elem, info = solver.solve(precond="direct", verbose=False)
    vm = solver.von_mises_stress(u_nodal)
    uz = u_elem[2]                          # element-centre z-displacement
    uz_top = float(u_nodal[2, n, :, :].mean())   # mean uz over the top node face

    area = (n * h) ** 2
    sigma = args.force / area              # nominal axial stress (Pa)
    delta_1d = sigma * args.size / args.E  # free-lateral 1D estimate (m)

    print("=== CUBE BENCHMARK (fixed base, uniform top load) ===")
    print(f"  geometry      : {args.size} m cube, {n}x{n}x{n} elems @ {h:.3f} m")
    print(f"  material      : E={args.E:.3e} Pa  nu={args.nu}  rho={args.rho_mat} kg/m^3")
    print(f"  support       : z=0 plane fully clamped (Fixed)")
    print(f"  load          : Fz_total={args.force:.1f} N over {area:.3f} m^2 "
          f"-> pressure {p:.3f} Pa{'  + self-weight' if args.gravity else ''}")
    print(f"  DOF active/free: {info.get('n_active_dofs')}/{info.get('n_free_dofs')} "
          f"[{info.get('backend')}]  residual={info['residual']:.1e}")
    print("  --- result ---")
    print(f"  |u|max            : {np.abs(u_elem).max():.4e} m")
    print(f"  mean uz @ top face: {uz_top:.4e} m")
    print(f"  von Mises range   : [{vm.min():.3e}, {vm.max():.3e}] Pa")
    print("  --- analytical check (uniaxial, free lateral) ---")
    print(f"  sigma = F/A       : {sigma:.3f} Pa")
    print(f"  delta = sigma*L/E : {delta_1d:.4e} m   (fixed base -> expect slightly less)")
    print(f"  ratio uz_top/delta: {uz_top/delta_1d:.3f}  (~1 means agreement)")

    if args.csv:
        _write_csv(args.csv, rho, u_elem, solver, vm, h)
        print(f"  wrote Grasshopper CSV -> {args.csv}")


def _write_csv(path, rho, u_elem, solver, vm, h):
    import pandas as pd
    didx = np.argwhere(rho > 0.05)
    d, hh, w = didx[:, 0], didx[:, 1], didx[:, 2]
    x = (w + 0.5) * h; y = (hh + 0.5) * h; z = (d + 0.5) * h
    ux = u_elem[0][d, hh, w]; uy = u_elem[1][d, hh, w]; uz = u_elem[2][d, hh, w]
    umag = np.sqrt(ux**2 + uy**2 + uz**2)
    cols = ["x", "y", "z", "ux", "uy", "uz", "umag", "von_mises"]
    data = np.column_stack([x, y, z, ux, uy, uz, umag, vm[d, hh, w]])
    pd.DataFrame(data, columns=cols).to_csv(path, index=False, float_format="%.6e")


if __name__ == "__main__":
    main()
