#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jacobi vs AMG preconditioner comparison on a 64^3 elasticity problem.

Two test cases are run back-to-back:

Case A -- nu = 0 (exact analytical reference)
    The 3D problem decouples along each axis at nu=0, reducing to Nz=64
    independent 1D bars in the z-direction.  Exact solution:

        uz(z) = (rho*g / E) * (z^2/2 - L*z)

    Q1 FEM is superconvergent here: nodal errors are at solver-tolerance level
    regardless of mesh refinement.  Jacobi CG converges in ~Nz iterations (the
    condition number is O(Nz) for the decoupled 1D system), so this case is
    intentionally easy for Jacobi.  Use it to verify correctness of both solvers.

Case B -- nu = 0.3 (realistic SIMP topology optimisation density)
    The full 3-component coupled system with Poisson effects.  Random SIMP
    density field (uniform in [0.3, 1.0]) with p=3 penalisation.  No closed-
    form solution; accuracy measured by solution agreement between solvers.
    This case shows where AMG's iteration-count advantage matters.

Domain   : 64 x 64 x 64 voxel cube, voxel size h = 1/64 m (1 m total side).
Material : Steel E = 210 GPa; body force fz = -rho*g = -76,518 N/m^3.
BC       : z = 0 plane fully clamped.

Usage::

    cd examples/elasticity_3d
    python solvers/compare_preconditioners.py
    python solvers/compare_preconditioners.py --tol 1e-8
    python solvers/compare_preconditioners.py --skip-nu0      # Case B only
    python solvers/compare_preconditioners.py --skip-nu03     # Case A only
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvers.voxel_fem import VoxelFEMSolver

# ---------------------------------------------------------------------------
# Shared problem parameters
# ---------------------------------------------------------------------------

GRID  = (64, 64, 64)
H     = 1.0 / 64          # voxel size [m] -> 1 m total cube
E0    = 2.1e11             # Young's modulus [Pa]
RHO_G = 7_800 * 9.81      # body force density [N/m^3]


# ---------------------------------------------------------------------------
# Exact solution (Case A, nu = 0)
# ---------------------------------------------------------------------------

def uz_exact(z: np.ndarray, L: float) -> np.ndarray:
    """Exact uz for a 1D bar under self-weight (nu=0):
        uz(z) = (rho*g / E) * (z^2/2 - L*z)
    """
    return (RHO_G / E0) * (z**2 / 2.0 - L * z)


# ---------------------------------------------------------------------------
# Single solve
# ---------------------------------------------------------------------------

def run_solve(
    precond: str,
    tol: float,
    maxiter: int,
    nu: float,
    rho: np.ndarray,
    p_simp: float = 1.0,
) -> dict:
    """Assemble and solve; return timing, iteration, and accuracy data."""
    Nz, Ny, Nx = GRID
    L = Nz * H

    t0 = time.perf_counter()
    solver = VoxelFEMSolver(GRID, voxel_size=H, nu=nu)
    solver.assemble_system(rho, E0, Emin=1e-9, p_simp=p_simp)
    solver.apply_bcs()
    bf = np.zeros((3, *GRID))
    bf[2] = -RHO_G
    solver.apply_loads(body_force=bf)
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    u_nodal, _, info = solver.solve(
        tol=tol, maxiter=maxiter, precond=precond, verbose=False
    )
    t_solve = time.perf_counter() - t1

    # Cross-section-averaged uz profile
    nz = Nz + 1
    uz_avg = u_nodal[2].reshape(nz, -1).mean(axis=1)   # (nz,)

    # Accuracy vs exact (only meaningful for nu=0, but computed for both)
    z_nodes = np.arange(nz) * H
    uz_ref  = uz_exact(z_nodes, L)
    abs_err = np.abs(uz_avg - uz_ref)
    l2_ref  = np.linalg.norm(uz_ref)
    rel_l2  = np.linalg.norm(uz_avg - uz_ref) / (l2_ref + 1e-300)

    return {
        "precond":   precond,
        "nu":        nu,
        "t_setup":   t_setup,
        "t_solve":   t_solve,
        "t_total":   t_setup + t_solve,
        "iters":     info["iters"],
        "residual":  info["residual"],
        "converged": info["converged"],
        "rel_l2":    rel_l2,
        "max_err":   abs_err.max(),
        "uz_avg":    uz_avg,
        "uz_ref":    uz_ref,
        "z_nodes":   z_nodes,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _sep(w: int = 70) -> None:
    print("-" * w)


def print_case(title: str, results: list[dict], show_exact: bool) -> None:
    Nz, Ny, Nx = GRID
    n_elem = Nz * Ny * Nx
    n_dof  = 3 * (Nz+1) * (Ny+1) * (Nx+1)
    L      = Nz * H
    nu     = results[0]["nu"]

    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  Grid  : {Nz}x{Ny}x{Nx}  | elements={n_elem:,} | DOFs={n_dof:,}")
    print(f"  E     : {E0:.3e} Pa  |  rho*g: {RHO_G:.0f} N/m^3  |  nu: {nu}")
    if show_exact:
        print(f"  Exact uz_top: {uz_exact(L, L)*1e6:.4f} um  (z = {L:.2f} m)")
    print()

    # Performance
    w = 36
    print(f"  {'Metric':<{w}}  {'Jacobi':>14}  {'AMG':>14}")
    _sep(w + 32)
    for label, key, fmt in [
        ("Total time  [setup + solve]  [s]", "t_total",  "{:.2f}"),
        ("  Setup time              [s]",    "t_setup",  "{:.2f}"),
        ("  Solve time              [s]",    "t_solve",  "{:.2f}"),
        ("CG iterations",                    "iters",    "{:d}"),
        ("Final |r|/|f|",                    "residual", "{:.2e}"),
        ("Converged",                        "converged","{!s}"),
    ]:
        vals = [fmt.format(r[key]) for r in results]
        print(f"  {label:<{w}}  {vals[0]:>14}  {vals[1]:>14}")
    _sep(w + 32)

    # Accuracy
    print()
    if show_exact:
        print(f"  {'Accuracy vs. exact':<{w}}  {'Jacobi':>14}  {'AMG':>14}")
        _sep(w + 32)
        for label, key, fmt in [
            ("Relative L2 error  ||err||/||u||", "rel_l2",  "{:.3e}"),
            ("Max |uz_fem - uz_exact|  [m]",    "max_err", "{:.3e}"),
        ]:
            vals = [fmt.format(r[key]) for r in results]
            print(f"  {label:<{w}}  {vals[0]:>14}  {vals[1]:>14}")
        _sep(w + 32)
        print()
    else:
        j = next(r for r in results if r["precond"] == "jacobi")
        a = next(r for r in results if r["precond"] == "amg")
        agree = np.allclose(j["uz_avg"], a["uz_avg"], rtol=1e-3, atol=1e-20)
        print(f"  Solutions agree (rtol=1e-3): {agree}")
        max_diff = np.max(np.abs(j["uz_avg"] - a["uz_avg"]))
        rel_diff = max_diff / (np.max(np.abs(j["uz_avg"])) + 1e-300)
        print(f"  Max |u_jacobi - u_amg| : {max_diff:.3e} m  (rel: {rel_diff:.3e})")
        print()

    # Nodal displacement profile
    ref = results[0]
    z   = ref["z_nodes"]
    step = max(1, len(z) // 8)
    indices = list(range(0, len(z), step))
    if indices[-1] != len(z) - 1:
        indices.append(len(z) - 1)

    hdr = f"  {'z [m]':>8}  {'uz_exact [um]':>16}" if show_exact else f"  {'z [m]':>8}"
    for r in results:
        hdr += f"  {'uz_'+r['precond']+' [um]':>18}"
        if show_exact:
            hdr += f"  {'err [nm]':>10}"
    print(hdr)
    col_w = 80 + (30 if show_exact else 22) * len(results)
    _sep(col_w)
    for i in indices:
        row = f"  {z[i]:8.4f}"
        if show_exact:
            row += f"  {ref['uz_ref'][i]*1e6:16.6f}"
        for r in results:
            row += f"  {r['uz_avg'][i]*1e6:18.6f}"
            if show_exact:
                err_nm = (r['uz_avg'][i] - r['uz_ref'][i]) * 1e9
                row += f"  {err_nm:10.6f}"
        print(row)
    _sep(col_w)

    # Summary
    j = next(r for r in results if r["precond"] == "jacobi")
    a = next(r for r in results if r["precond"] == "amg")
    iter_ratio = j["iters"] / max(a["iters"], 1)
    time_ratio = j["t_solve"] / max(a["t_solve"], 1e-9)
    print()
    faster_solver = "Jacobi" if time_ratio < 1 else "AMG"
    speedup = 1.0 / time_ratio if time_ratio < 1 else time_ratio
    print(f"  Iteration reduction : {iter_ratio:.0f}x  "
          f"({j['iters']} -> {a['iters']} iters)")
    print(f"  Solve-time winner   : {faster_solver} is {speedup:.1f}x faster  "
          f"(Jacobi {j['t_solve']:.1f}s  vs  AMG {a['t_solve']:.1f}s)")
    print(f"  Note: AMG solve time includes sparse K assembly + hierarchy build")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Jacobi and AMG on a 64^3 elasticity problem"
    )
    parser.add_argument("--tol",       type=float, default=1e-6)
    parser.add_argument("--maxiter",   type=int,   default=5000)
    parser.add_argument("--skip-nu0",  action="store_true",
                        help="Skip Case A (nu=0, exact reference)")
    parser.add_argument("--skip-nu03", action="store_true",
                        help="Skip Case B (nu=0.3, realistic SIMP)")
    args = parser.parse_args()

    print(f"\nVoxelFEMSolver - Preconditioner Comparison  (64^3, tol={args.tol:.0e})")

    rho_solid  = np.ones(GRID, dtype=np.float64)
    rng = np.random.default_rng(42)
    rho_simp   = rng.uniform(0.3, 1.0, size=GRID)

    # ------------------------------------------------------------------ Case A
    if not args.skip_nu0:
        print("\n[Case A] nu=0, full solid, exact solution known ...")
        results_a = []
        for precond in ("jacobi", "amg"):
            print(f"  Running {precond} ...", end=" ", flush=True)
            r = run_solve(precond, args.tol, args.maxiter,
                          nu=0.0, rho=rho_solid, p_simp=1.0)
            conv = "converged" if r["converged"] else "DID NOT CONVERGE"
            print(f"{r['iters']} iters | {r['t_total']:.1f}s total [{conv}]")
            results_a.append(r)
        print_case(
            "Case A: nu=0, full solid cube under self-weight (exact solution)",
            results_a,
            show_exact=True,
        )

    # ------------------------------------------------------------------ Case B
    if not args.skip_nu03:
        print("\n[Case B] nu=0.3, random SIMP density [0.3, 1.0], no exact solution ...")
        results_b = []
        for precond in ("jacobi", "amg"):
            print(f"  Running {precond} ...", end=" ", flush=True)
            r = run_solve(precond, args.tol, args.maxiter,
                          nu=0.3, rho=rho_simp, p_simp=3.0)
            conv = "converged" if r["converged"] else "DID NOT CONVERGE"
            print(f"{r['iters']} iters | {r['t_total']:.1f}s total [{conv}]")
            results_b.append(r)
        print_case(
            "Case B: nu=0.3, random SIMP density (p=3) under self-weight",
            results_b,
            show_exact=False,
        )


if __name__ == "__main__":
    main()
