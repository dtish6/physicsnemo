#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert architectural-wall density CSVs into training-ready HDF5 samples.

Input
-----
One CSV per wall (``plan_*.csv``) with columns ``x, y, z, value`` where:
  * ``x, y, z``  are physical coordinates (metres) on a regular grid.
  * ``value``    is a voxel density rho in [0, 1] (1 = fully solid, 0 = void).

The companion ``*_geom.csv`` files (2D wall outlines) are ignored.

What this script does (per wall)
--------------------------------
1.  Rasterise the (x, y, z, value) point cloud onto a dense (Nz, Ny, Nx)
    density grid, snapping physical coords to indices via the voxel size.
    Axis convention (matches DESIGN.md NCDHW): D=z (vertical), H=y, W=x.
2.  Optionally block-average downsample by an integer factor (``--downsample``)
    to keep CPU FEM + training tractable.
3.  Anchor the wall base (lowest z) at the D=0 plane (the clamped plane) and
    zero-pad each spatial dim up to a multiple of ``--pad-multiple`` (= 2**model_depth)
    so the U-Net can pool cleanly.
4.  Build the load case (DEFAULTS): homogeneous material E0, gravity body force
    fz = -rho_mat * g * rho, no wind.
5.  Solve 3D linear elasticity with the built-in VoxelFEMSolver (clamped base)
    to obtain the displacement field -> the supervised TARGET.
6.  Write ``sample_{i:05d}.h5`` with the five fields ElasticityDataset expects.

Usage
-----
    # Quick test: 2 walls, aggressive downsample
    python data/generate_from_csv.py --csv_dir "D:/.../0-1" \
        --out_dir ./data --n_samples 2 --downsample 4

    # Full run (all walls), 80/20 train/val split
    python data/generate_from_csv.py --csv_dir "D:/.../0-1" \
        --out_dir ./data --downsample 2

Outputs ``<out_dir>/train/*.h5`` and ``<out_dir>/val/*.h5``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Make the project root importable so we can use step2_fea.voxel_fem.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step2_fea._2_2_voxel_fem import VoxelFEMSolver  # noqa: E402

# Preprocessing sub-step: binarize CSV -> fixed 128^3 voxel grid (with crop).
# (Fixed-cube 1:1 flow: downsample/pad steps are not used here.)
from step1_preprocess._1_1_binarize import rasterize  # noqa: E402

# ---------------------------------------------------------------------------
# Default material + load case (architectural wall, gravity only)
# ---------------------------------------------------------------------------
E0_PA      = 2.1e11    # Young's modulus, Pa (steel default)
NU         = 0.3       # Poisson's ratio
RHO_MAT    = 7800.0    # material density, kg/m^3 (steel)
GRAVITY    = 9.81      # m/s^2
DENSITY_THRESHOLD = 0.05   # rho above this counts as "solid" for the mask/E field

# Material presets (isotropic). Wood is orthotropic in reality; these are an
# isotropic structural-softwood approximation (along-grain stiffness).
MATERIALS = {
    "steel": dict(E0=2.1e11, rho_mat=7800.0, nu=0.30),
    "wood":  dict(E0=1.1e10, rho_mat=500.0,  nu=0.35),  # structural softwood
}


def _plan_index(path: str) -> int:
    m = re.search(r"plan_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_field_csvs(csv_dir: str) -> list[str]:
    """Return sorted plan_*.csv field files, excluding *_geom.csv."""
    files = [
        f for f in glob.glob(os.path.join(csv_dir, "plan_*.csv"))
        if not f.endswith("_geom.csv")
    ]
    return sorted(files, key=_plan_index)


def build_sample(rho: np.ndarray, voxel_size: float, fem_kwargs: dict,
                 material: dict | None = None, load_factor: float = 1.0) -> dict:
    """Assemble all five HDF5 fields for one plan, solving FEM for displacement.

    ``material`` overrides the default steel properties with keys
    ``E0`` (Pa), ``rho_mat`` (kg/m^3), ``nu`` (Poisson). Defaults to steel.
    ``load_factor`` multiplies the gravity body force (a design safety factor):
    the stored ``body_force`` input channel AND the FEA displacement target are
    both scaled by it, so the sample stays physically self-consistent.
    """
    mat = material or {}
    E0      = float(mat.get("E0", E0_PA))
    rho_mat = float(mat.get("rho_mat", RHO_MAT))
    nu      = float(mat.get("nu", NU))

    Nz, Ny, Nx = rho.shape
    solid = (rho > DENSITY_THRESHOLD).astype(np.float32)

    # Young's modulus field (physical Pa); homogeneous material, zero in void.
    E_field = (E0 * solid).astype(np.float32)

    # Body force: gravity in -z, scaled by local density and the safety factor.
    body_force = np.zeros((3, Nz, Ny, Nx), dtype=np.float32)
    body_force[2] = (-load_factor * rho_mat * GRAVITY * rho).astype(np.float32)

    # Default load case: no wind.
    wind_pressure = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    # --- Solve linear elasticity: clamped base (z=0), gravity load ----------
    solver = VoxelFEMSolver(grid_size=(Nz, Ny, Nx), voxel_size=voxel_size, nu=nu)
    result = solver.run(
        rho=rho,
        E0=E0,
        body_force=body_force,
        wind_pressure=None,          # gravity-only default
        compute_vm=False,
        **fem_kwargs,
    )
    displacement = result["u_elem"].astype(np.float32)  # (3, Nz, Ny, Nx)

    # Full Cauchy stress tensor at element centres, Voigt order
    # [sxx, syy, szz, sxy, sxz, syz], shape (6, Nz, Ny, Nx). Stored so von Mises
    # and principal stress can be derived later WITHOUT re-solving the FEM.
    _, stress = solver.element_strain_stress(result["u_nodal"])
    stress = stress.astype(np.float32)

    return {
        "solid_mask":    rho,            # continuous density (informative input)
        "E_normalized":  E_field,        # raw Pa; ChannelNormalize z-scores it
        "body_force":    body_force,
        "wind_pressure": wind_pressure,
        "displacement":  displacement,
        "stress":        stress,         # (6, Nz, Ny, Nx) Cauchy stress (Pa)
        "_info":         result["info"],
        "_nu":           nu,
    }


def write_h5(path: Path, data: dict, voxel_size: float) -> None:
    with h5py.File(path, "w") as f:
        for key in ("solid_mask", "E_normalized", "body_force",
                    "wind_pressure", "displacement", "stress"):
            f.create_dataset(key, data=data[key], dtype="float32",
                             compression="gzip")
        f.attrs["nu"]         = float(data.get("_nu", NU))
        f.attrs["bc"]         = "z=0 plane fully clamped (Dirichlet u=0)"
        f.attrs["grid_shape"] = np.array(data["solid_mask"].shape, dtype=np.int32)
        f.attrs["voxel_size"] = voxel_size


def _process_plan(task: tuple) -> dict:
    """Worker: rasterize one plan into the fixed grid, solve FEM, write HDF5.

    Module-level + picklable args so it works with multiprocessing on Windows
    (spawn). ``grid_dhw`` is (Nz, Ny, Nx) = (D, H, W).
    """
    plan_path, out_path, voxel_size, grid_dhw, fem_kwargs, material, load_factor = task
    t0 = time.perf_counter()
    df = pd.read_csv(plan_path, usecols=["x", "y", "z", "value"])
    rho = rasterize(df, voxel_size, grid_size=grid_dhw)
    data = build_sample(rho, voxel_size, fem_kwargs, material, load_factor)
    info = data["_info"]
    # Only write CONVERGED samples. A non-converged solve means the cropped
    # geometry has solid disconnected from the clamped base (singular system):
    # the displacement is garbage and must not pollute the dataset.
    written = bool(info["converged"])
    if written:
        write_h5(Path(out_path), data, voxel_size)
    n_solid = int((rho > DENSITY_THRESHOLD).sum())
    return {
        "plan":      os.path.basename(plan_path),
        "out":       os.path.basename(out_path),
        "split":     "val" if f"{os.sep}val{os.sep}" in out_path else "train",
        "shape":     tuple(int(s) for s in rho.shape),
        "iters":     int(info["iters"]),
        "residual":  float(info["residual"]),
        "converged": bool(info["converged"]),
        "written":   written,
        "umax":      float(np.abs(data["displacement"]).max()),
        "n_solid":   n_solid,                                  # solid elements
        "n_active":  int(info.get("n_active_dofs", -1)),       # solved DOFs (direct)
        "n_free":    int(info.get("n_free_dofs", -1)),         # non-clamped DOFs
        "backend":   info.get("backend", "?"),
        "dt":        time.perf_counter() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_dir",
                    default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1",
                    help="Directory with plan_*.csv files.")
    ap.add_argument("--out_dir", default="./data", help="Output root (creates train/ and val/).")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="Limit number of plans processed (default: all).")
    ap.add_argument("--grid_size", type=int, default=128,
                    help="Horizontal voxel grid size (H=W=N). Plans are anchored at the "
                         "origin and CROPPED to fit. 128 @ 0.1 m = 12.8 m in x/y.")
    ap.add_argument("--grid_z", type=int, default=48,
                    help="Vertical voxel grid size (D). Apartments are ~36 tall; 48 @ 0.1 m "
                         "= 4.8 m and is divisible by 2**model_depth (16). Trims dead air.")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of plans held out for validation.")
    ap.add_argument("--voxel_size", type=float, default=0.1,
                    help="Physical voxel edge length in metres (CSV coords are metres; "
                         "spacing is 0.1 m = 10 cm). Fixed, not auto-detected.")
    ap.add_argument("--load_factor", type=float, default=1.0,
                    help="Design safety factor on the gravity body force (fz). 3.0 = solve "
                         "under 3x self-weight; scales both the fz input channel and the "
                         "displacement target (linear, so they stay consistent).")
    ap.add_argument("--fem_tol", type=float, default=1e-6, help="CG tolerance.")
    ap.add_argument("--fem_maxiter", type=int, default=2000, help="CG max iterations.")
    ap.add_argument("--precond", default="amg", choices=["jacobi", "amg", "none", "direct"],
                    help="FEM solve: 'direct' = active-DOF reduction + direct sparse factorize "
                         "(fastest, exact); 'amg'/'jacobi' = iterative CG. amg needs pyamg.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes (AMG solve ~7-15 GB RAM each; 2 fits 32 GB).")
    ap.add_argument("--threads_per_job", type=int, default=5,
                    help="BLAS/OMP threads per worker (jobs*threads <= logical cores).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if the output .h5 exists (default: skip existing, "
                         "so an interrupted run can be resumed).")
    ap.add_argument("--material", default="steel", choices=list(MATERIALS),
                    help="Material preset (steel | wood). Wood = isotropic structural softwood.")
    ap.add_argument("--E0", type=float, default=None, help="Override Young's modulus (Pa).")
    ap.add_argument("--rho_mat", type=float, default=None, help="Override material density (kg/m^3).")
    ap.add_argument("--nu", type=float, default=None, help="Override Poisson's ratio.")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for train/val split.")
    args = ap.parse_args()

    # Resolve material: preset, with optional per-property overrides.
    material = dict(MATERIALS[args.material])
    if args.E0 is not None:      material["E0"] = args.E0
    if args.rho_mat is not None: material["rho_mat"] = args.rho_mat
    if args.nu is not None:      material["nu"] = args.nu

    files = list_field_csvs(args.csv_dir)
    if not files:
        raise SystemExit(f"No plan_*.csv files found in {args.csv_dir}")
    if args.n_samples is not None:
        files = files[: args.n_samples]

    # Deterministic train/val split.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * args.val_frac))) if len(files) > 1 else 0
    val_idx = set(order[:n_val].tolist())

    train_dir = Path(args.out_dir) / "train"
    val_dir   = Path(args.out_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    fem_kwargs = dict(tol=args.fem_tol, maxiter=args.fem_maxiter, precond=args.precond)
    grid_dhw = (args.grid_z, args.grid_size, args.grid_size)  # (D, H, W) = (z, y, x)

    # Pre-assign each plan a split and a global sample index -> output path,
    # so parallel workers never collide on filenames.
    tasks = []
    n_train = n_val_written = n_skip = 0
    for i, fp in enumerate(files):
        if i in val_idx:
            out = val_dir / f"sample_{n_val_written:05d}.h5"; n_val_written += 1
        else:
            out = train_dir / f"sample_{n_train:05d}.h5"; n_train += 1
        if (not args.overwrite) and out.exists():
            n_skip += 1   # resume: skip already-generated samples
            continue
        tasks.append((fp, str(out), args.voxel_size, grid_dhw, fem_kwargs, material,
                      args.load_factor))

    box = tuple(n * args.voxel_size for n in grid_dhw)
    print(f"Found {len(files)} plans. grid(D,H,W)={grid_dhw} @ {args.voxel_size} m "
          f"= {box[0]:.1f}x{box[1]:.1f}x{box[2]:.1f} m (1:1, crop to fit) "
          f"train={n_train} val={n_val_written} | to-do={len(tasks)} skip-existing={n_skip}",
          flush=True)
    print(f"Material: {args.material} E0={material['E0']:.2e} Pa, nu={material['nu']}, "
          f"rho_mat={material['rho_mat']} kg/m^3 | Load: gravity only (-z), clamped base, "
          f"no wind | precond={args.precond} jobs={args.jobs}x{args.threads_per_job}t", flush=True)

    # --- FEA setup card (constant across plans; for nTopology replication) ----
    nz, ny, nx = grid_dhw
    n_nodes = (nz + 1) * (ny + 1) * (nx + 1)
    print(
        "\n=== FEA setup (replicate in nTop) ===\n"
        f"  Element type      : Q1/H8 trilinear hex, 2x2x2 Gauss, 3 DOF/node\n"
        f"  Voxel grid (DxHxW): {nz} x {ny} x {nx}  ({nz*ny*nx} elements max)\n"
        f"  Voxel size        : {args.voxel_size} m  (box {box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} m, Z-up)\n"
        f"  Nodes / total DOF : {n_nodes} / {3*n_nodes}  (solid-only DOF printed per-plan below)\n"
        f"  Young's modulus E : {material['E0']:.3e} Pa   (void uses Emin=1e-9*E, SIMP p=3)\n"
        f"  Poisson ratio nu  : {material['nu']}\n"
        f"  Material density  : {material['rho_mat']} kg/m^3\n"
        f"  Gravity           : {GRAVITY} m/s^2  x load_factor {args.load_factor}  ->  body force fz "
        f"= -{args.load_factor}*rho*g = {-args.load_factor*material['rho_mat']*GRAVITY:.1f} N/m^3\n"
        f"  Boundary condition: z=0 plane fully clamped (Dirichlet u=0); free everywhere else\n"
        f"  Solver            : active-DOF reduction + direct sparse factorize (pardiso/superlu)\n"
        "=====================================\n", flush=True)

    n_failed = [0]   # list so the nested _report can mutate it

    def _report(k: int, r: dict) -> None:
        conv = "OK" if r["converged"] else "NOT-CONV (skipped, not written)"
        if not r["written"]:
            n_failed[0] += 1
        print(f"[{k}/{len(tasks)}] {r['plan']} -> {r['split']}/{r['out']} "
              f"grid={r['shape']} solid_elem={r['n_solid']} "
              f"DOF active/free={r['n_active']}/{r['n_free']} [{r['backend']}] "
              f"res={r['residual']:.1e} {conv} "
              f"|u|max={r['umax']:.2e} m ({r['dt']:.1f}s)", flush=True)

    t_start = time.perf_counter()
    if args.jobs <= 1:
        for k, task in enumerate(tasks, 1):
            _report(k, _process_plan(task))
    else:
        # Limit BLAS/OMP threads per worker BEFORE spawning so jobs*threads
        # don't oversubscribe the CPU (children inherit these env vars).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(args.threads_per_job)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            for k, r in enumerate(pool.imap_unordered(_process_plan, tasks), 1):
                _report(k, r)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed/max(len(tasks),1):.1f}s/sample avg). "
          f"Wrote {len(tasks) - n_failed[0]}/{len(tasks)} samples to {args.out_dir} "
          f"(skip-existing={n_skip}, not-converged/dropped={n_failed[0]}).", flush=True)


if __name__ == "__main__":
    main()
