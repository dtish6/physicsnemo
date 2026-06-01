#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmark for VoxelFEMSolver.

Measures wall-clock time for:
  1. elem_dofs construction (once per grid shape)
  2. assemble_system (SIMP density mapping)
  3. Matrix-free matvec  K·v
  4. Full CG solve (Jacobi preconditioned)
  5. Post-processing (von Mises stress)

Reports throughput (DOFs/s) and estimated time for a 128³ solve.

Usage::

    cd examples/elasticity_3d
    python solvers/benchmark.py          # default 32³ and 64³
    python solvers/benchmark.py --grid 128  # WARNING: ~4+ min per solve
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvers.voxel_fem import VoxelFEMSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _timer(label: str, fn, *args, **kwargs):
    t0  = time.perf_counter()
    out = fn(*args, **kwargs)
    dt  = time.perf_counter() - t0
    print(f"  {label:<40s} {dt:8.3f} s")
    return out, dt


# ---------------------------------------------------------------------------
# Single-grid benchmark
# ---------------------------------------------------------------------------

def benchmark_grid(Nz: int, Ny: int, Nx: int,
                   nu: float = 0.3,
                   E0: float = 2.1e11,
                   tol: float = 1e-6,
                   maxiter: int = 2000,
                   precond: str = "jacobi") -> dict:
    """Run a complete benchmark for one grid size."""
    grid = (Nz, Ny, Nx)
    n_elem  = Nz * Ny * Nx
    n_nodes = (Nz+1)*(Ny+1)*(Nx+1)
    n_dof   = 3 * n_nodes

    _print_section(f"Grid {Nz}x{Ny}x{Nx}  |  elements={n_elem:,}  DOFs={n_dof:,}")

    # 1. Construction
    _, t_init = _timer("Solver construction (K0 + elem_dofs)",
                       VoxelFEMSolver, grid)
    solver = VoxelFEMSolver(grid, voxel_size=1.0, nu=nu)

    # 2. Assemble
    rng = np.random.default_rng(42)
    rho = rng.uniform(0.3, 1.0, size=grid)   # random density field

    _, t_asm = _timer("assemble_system",
                      solver.assemble_system, rho, E0, Emin=1e-9, p_simp=3.0)

    # 3. BCs and loads
    solver.apply_bcs()
    bf = np.zeros((3, *grid))
    bf[2] = -9.81 * 7800.0   # steel self-weight, z-direction
    solver.apply_loads(body_force=bf)

    # 4. Matvec throughput
    v_rand = rng.standard_normal(n_dof)
    def _matvec_bench():
        for _ in range(10):
            solver._matvec(v_rand)

    _, t_mv10 = _timer("10x matrix-free matvec", _matvec_bench)
    t_matvec  = t_mv10 / 10.0
    print(f"    -> 1 matvec ={t_matvec*1e3:.1f} ms  |  "
          f"{n_dof/t_matvec/1e6:.0f} MDOF/s")

    # 5. Full CG solve
    def _solve():
        return solver.solve(tol=tol, maxiter=maxiter, precond=precond, verbose=False)

    (u_nodal, u_elem, info), t_solve = _timer("CG solve (full)", _solve)
    print(f"    -> iters={info['iters']}  |r|/|f|={info['residual']:.2e}  "
          f"converged={info['converged']}")

    # 6. Von Mises
    _, t_vm = _timer("Von Mises stress (post-process)",
                     solver.von_mises_stress, u_nodal)

    # Summary row
    print(f"\n  Summary  {Nz}^3 grid:")
    print(f"    Construction:     {t_init:.3f} s")
    print(f"    Assembly:         {t_asm:.3f} s")
    print(f"    Matvec (1 call):  {t_matvec*1e3:.1f} ms")
    print(f"    Solve:            {t_solve:.1f} s  ({info['iters']} iters)")
    print(f"    Post-process:     {t_vm*1e3:.1f} ms")
    print(f"    Total:            {t_init+t_asm+t_solve+t_vm:.1f} s")

    return {
        "grid":      grid,
        "n_dof":     n_dof,
        "n_elem":    n_elem,
        "t_init":    t_init,
        "t_asm":     t_asm,
        "t_matvec":  t_matvec,
        "t_solve":   t_solve,
        "t_vm":      t_vm,
        "iters":     info["iters"],
        "converged": info["converged"],
    }


# ---------------------------------------------------------------------------
# Scaling extrapolation
# ---------------------------------------------------------------------------

def print_scaling_estimate(results: list[dict]) -> None:
    """Extrapolate timings to larger grids assuming O(n) scaling for matvec.

    Iteration count estimate:
      Jacobi-CG scales as O(h^-1) ~ O(n^1/3), so iters ~ iters0 * (n/n0)^(1/3).
      AMG-CG is grid-independent; we use the measured count directly.
    """
    _print_section("Scaling extrapolation")

    r      = results[-1]
    n0     = r["n_elem"]
    t_mv0  = r["t_matvec"]
    iters0 = r["iters"]

    # Heuristic: if <30 iters assume AMG (grid-independent); else Jacobi (O(n^1/3))
    amg_mode = iters0 < 30
    iter_label = f"~{iters0} iters (AMG)" if amg_mode else "O(n^1/3) iters (Jacobi)"

    targets = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)]
    print(f"  Iteration model: {iter_label}")
    print(f"  {'Grid':<12}  {'Elements':>10}  {'DOFs':>10}  "
          f"{'Matvec (ms)':>12}  {'Est. solve':>22}")
    print(f"  {'-'*72}")
    for (nz, ny, nx) in targets:
        n     = nz * ny * nx
        ratio = n / n0
        t_mv  = t_mv0 * ratio
        if amg_mode:
            est_iters = iters0          # AMG: nearly grid-independent
        else:
            est_iters = max(iters0, int(iters0 * ratio ** (1 / 3)))  # Jacobi: O(n^1/3)
        t_est = t_mv * est_iters
        label = "measured" if (nz, ny, nx) == r["grid"] else "estimated"
        print(f"  {nz}^3{'':<8}  {n:>10,}  {3*(nz+1)**3:>10,}  "
              f"{t_mv*1e3:>10.1f}ms  {t_est:>16.0f} s ({est_iters} iters) [{label}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VoxelFEMSolver performance benchmark")
    parser.add_argument(
        "--grid", type=int, default=None,
        help="Single grid size N for N³ benchmark (default: run 16³ and 32³)"
    )
    parser.add_argument(
        "--precond", type=str, default="jacobi",
        choices=["jacobi", "amg", "none"],
        help="Preconditioner (default: jacobi)"
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=2000)
    args = parser.parse_args()

    print("VoxelFEMSolver - Performance Benchmark")
    print(f"  numpy  {np.__version__}")
    try:
        import scipy; print(f"  scipy  {scipy.__version__}")
    except ImportError:
        pass
    try:
        import pyamg; print(f"  pyamg  {pyamg.__version__}")
    except ImportError:
        print("  pyamg  not installed (Jacobi only)")

    grids = [(args.grid,)*3] if args.grid else [(16, 16, 16), (32, 32, 32)]
    results = []
    for grid in grids:
        r = benchmark_grid(
            *grid, tol=args.tol, maxiter=args.maxiter, precond=args.precond
        )
        results.append(r)

    if results:
        print_scaling_estimate(results)


if __name__ == "__main__":
    main()
