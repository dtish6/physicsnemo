#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert architectural-wall CSVs into training-ready HDF5 samples.

Input
-----
One CSV per wall (``plan_*.csv``) with columns ``x, y, z, value`` where:
  * ``x, y, z``  are physical coordinates (metres) on a regular grid.
  * ``value``    is a category label; only ``value == 1.0`` is solid wall,
                 every other value is void (see step1_preprocess._1_1_binarize).

The companion ``*_geom.csv`` files (2D wall outlines) are ignored.

What this script does (per wall)
--------------------------------
1.  Rasterise the (x, y, z, value) point cloud onto a FIXED (Nz, Ny, Nx) binary
    grid, anchored at the plan's min corner and CROPPED to fit (no scaling).
    Axis convention (matches DESIGN.md NCDHW): D=z (vertical), H=y, W=x.
2.  Build the load case (DEFAULTS): homogeneous material E0, gravity body force
    fz = -load_factor * rho_mat * g * rho, no wind.
3.  Solve 3D linear elasticity with the built-in VoxelFEMSolver (clamped base)
    to obtain the displacement + stress fields -> the supervised TARGET.
4.  Write ``sample_{i:05d}.h5`` with the six fields ElasticityDataset expects.

Configuration
-------------
Edit the USER CONFIG block below (input/output folders, material, grid size),
then run with no arguments. Any command-line flag overrides the config value.

Usage
-----
    # Use the USER CONFIG defaults
    python step1_preprocess/generate_from_csv.py

    # Override on the command line (e.g. 2 walls, wood, custom folders)
    python step1_preprocess/generate_from_csv.py --csv_dir "D:/.../0-1" \
        --out_dir ./data_wood --material wood --n_samples 2

Outputs ``<out_dir>/train/*.h5`` and ``<out_dir>/val/*.h5``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Make the project root importable so we can use step2_fea.voxel_fem.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from step2_fea._2_2_voxel_fem import VoxelFEMSolver  # noqa: E402

# Preprocessing sub-step: binarize CSV -> fixed (Nz, Ny, Nx) voxel grid (1:1,
# crop to fit; no downsample/pad). Grid size comes from USER CONFIG below.
from step1_preprocess._1_1_binarize import rasterize  # noqa: E402

# ---------------------------------------------------------------------------
# Physics constants + material presets (architectural wall, gravity only)
# ---------------------------------------------------------------------------
GRAVITY    = 9.81          # m/s^2
DENSITY_THRESHOLD = 0.05   # rho above this counts as "solid" for the mask/E field

# Material presets (isotropic), the single source of truth for E0/rho_mat/nu.
# Wood is orthotropic in reality; this is an isotropic structural-softwood
# approximation (along-grain stiffness). "steel" is also the fallback default.
MATERIALS = {
    "steel": dict(E0=2.1e11, rho_mat=7800.0, nu=0.30),
    "wood":  dict(E0=1.1e10, rho_mat=500.0,  nu=0.35),  # structural softwood
}


# ===========================================================================
# USER CONFIG  --  edit these 5 lines, then just run:
#     python step1_preprocess/generate_from_csv.py
# Any command-line flag still OVERRIDES the matching value here.
# ===========================================================================
INPUT_FOLDER  = r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\within_12p8m_1000"  # plan_*.csv live here
OUTPUT_FOLDER = r"D:\Nemo\physicsnemo\examples\elasticity_3d\step1_preprocess\hdf5_data\061611"        # writes OUTPUT_FOLDER/train/*.h5 and /val/*.h5
MATERIAL      = "wood"          # which preset above: "steel" or "wood"
LOAD_FACTOR   = 3.0              # safety factor on gravity fz (e.g. 3.0 = solve under 3x self-weight)
GRID_XY       = 128              # horizontal grid H=W (e.g. 128 or 32). Plans cropped to fit.
GRID_Z        = 48               # vertical grid D (height)
PRECOND       = "direct"         # FEA solver: "direct" = exact + fastest (PARDISO); "amg"/"jacobi" = iterative
JOBS          = 2                # plans solved in parallel (separate processes)
THREADS_PER_JOB = 4              # CPU threads each solve may use internally
# SAFE grid numbers (divisible by 16, so the training step is happy): 16, 32, 48, 64, 96, 128
# JOBS x THREADS_PER_JOB ~= physical cores. Raise JOBS first until RAM is full
# (each 128^3 direct solve ~13 GB), then give the rest to THREADS_PER_JOB.
# On a 32 GB machine, keep JOBS <= 2 for PRECOND="direct" at 128^3 or PARDISO OOMs.
# ===========================================================================


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
    """Assemble all six HDF5 fields for one plan, solving FEM for displacement.

    ``material`` overrides the default steel properties with keys
    ``E0`` (Pa), ``rho_mat`` (kg/m^3), ``nu`` (Poisson). Defaults to steel.
    ``load_factor`` multiplies the gravity body force (a design safety factor):
    the stored ``body_force`` input channel AND the FEA displacement target are
    both scaled by it, so the sample stays physically self-consistent.
    """
    # Fill any missing key from the steel preset (single source of truth).
    mat = {**MATERIALS["steel"], **(material or {})}
    E0      = float(mat["E0"])
    rho_mat = float(mat["rho_mat"])
    nu      = float(mat["nu"])

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
        "solid_mask":    rho,            # binary mask 0/1 (informative input)
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
        f.attrs["nu"]         = float(data.get("_nu", MATERIALS["steel"]["nu"]))
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


def format_fea_setup(args, material: dict, grid_dhw: tuple, box: tuple,
                     backends: set | None = None) -> str:
    """Render the FEA setup card from the values ACTUALLY used this run.

    Shared by the console print and the run_description.txt file so the two can
    never drift. ``backends`` (e.g. {"pardiso"}) is only known after the run, so
    it is omitted (None) for the pre-run console print and filled in for the file.
    """
    nz, ny, nx = grid_dhw
    n_nodes = (nz + 1) * (ny + 1) * (nx + 1)
    fz_coef = -args.load_factor * material["rho_mat"] * GRAVITY
    if args.precond == "direct":
        solver = "active-DOF reduction + direct sparse factorize (pardiso/superlu)"
    else:
        solver = (f"active-DOF reduction + CG (precond={args.precond}, "
                  f"tol={args.fem_tol}, maxiter={args.fem_maxiter})")
    backend_line = (f"  Backend (actual)  : {', '.join(sorted(backends))}\n"
                    if backends else "")
    return (
        f"  Element type      : Q1/H8 trilinear hex, 2x2x2 Gauss, 3 DOF/node\n"
        f"  Voxel grid (DxHxW): {nz} x {ny} x {nx}  ({nz*ny*nx} elements max)\n"
        f"  Voxel size        : {args.voxel_size} m  (box {box[0]:.1f} x {box[1]:.1f} x {box[2]:.1f} m, Z-up)\n"
        f"  Nodes / total DOF : {n_nodes} / {3*n_nodes}\n"
        f"  Material          : {args.material}\n"
        f"  Young's modulus E : {material['E0']:.3e} Pa   (void uses Emin=1e-9*E, SIMP p=3)\n"
        f"  Poisson ratio nu  : {material['nu']}\n"
        f"  Material density  : {material['rho_mat']} kg/m^3\n"
        f"  Gravity           : {GRAVITY} m/s^2  x load_factor {args.load_factor}\n"
        f"  Body force fz     : {fz_coef:.1f} * rho  N/m^3  (gravity in -z, no wind)\n"
        f"  Density threshold : {DENSITY_THRESHOLD}  (rho above this counts as solid)\n"
        f"  Boundary condition: z=0 plane fully clamped (Dirichlet u=0); free elsewhere\n"
        f"  Solver            : {solver}\n"
        f"{backend_line}"
    )


def write_run_description(path: Path, args, material: dict, grid_dhw: tuple,
                          box: tuple, *, n_found: int, n_train: int, n_val: int,
                          n_tasks: int, n_skip: int, n_written: int,
                          n_dropped: int, elapsed: float, results: list) -> None:
    """Write a human-readable summary of this generation run to ``path``.

    Lists the FEA setup actually used (resolved material incl. overrides, real
    solver backend) plus run accounting (plans found/run/written, timing).
    """
    backends = {r["backend"] for r in results} if results else set()
    umaxes = [r["umax"] for r in results if r["written"]]
    iters = [r["iters"] for r in results if r["written"]]

    lines = [
        "FEA dataset generation -- run description",
        "=" * 55,
        f"Generated         : {datetime.now().isoformat(timespec='seconds')}",
        f"Script            : {Path(__file__).name}",
        "",
        "--- Paths ---",
        f"Input  (csv_dir)  : {args.csv_dir}",
        f"Output (out_dir)  : {args.out_dir}",
        "",
        "--- FEA setup (values actually used) ---",
        format_fea_setup(args, material, grid_dhw, box, backends=backends).rstrip("\n"),
        "",
        "--- Run results ---",
        f"Plans found       : {n_found}",
        f"Train / Val split : {n_train} / {n_val}  (val_frac={args.val_frac}, seed={args.seed})",
        f"Skipped (existing): {n_skip}",
        f"Plans attempted   : {n_tasks}",
        f"Samples written   : {n_written}",
        f"Dropped (non-conv): {n_dropped}",
    ]
    if umaxes:
        lines.append(f"|u|max (written)  : min={min(umaxes):.2e}  "
                     f"max={max(umaxes):.2e}  mean={sum(umaxes)/len(umaxes):.2e} m")
    if iters and max(iters) > 0:
        lines.append(f"CG iters (written): min={min(iters)}  max={max(iters)}")
    lines += [
        f"Total time        : {elapsed/60:.2f} min ({elapsed:.1f} s)",
        f"Avg per attempted : {elapsed/max(n_tasks,1):.2f} s",
        "",
        "--- Parallelism (this run) ---",
        f"Workers (jobs)    : {args.jobs}",
        f"Threads per job   : {args.threads_per_job}",
        f"Total CPU threads : {args.jobs * args.threads_per_job}  (jobs x threads_per_job)",
        "",
        "Rule of thumb for picking jobs x threads_per_job:",
        "  1. Target jobs x threads_per_job = physical cores; never exceed logical cores.",
        "  2. Raise JOBS first until RAM is the limit (each 128^3 direct solve ~13 GB),",
        "     then spend any leftover cores on THREADS_PER_JOB.",
        "  3. JOBS scales throughput near-linearly; extra THREADS_PER_JOB has",
        "     diminishing returns -- so prefer more jobs over more threads.",
        "  Cap: direct @ 128^3 on 32 GB -> JOBS <= 2 (more OOMs PARDISO).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_dir",
                    default=INPUT_FOLDER,
                    help="Directory with plan_*.csv files. [default: USER CONFIG INPUT_FOLDER]")
    ap.add_argument("--out_dir", default=OUTPUT_FOLDER,
                    help="Output root (creates train/ and val/). [default: USER CONFIG OUTPUT_FOLDER]")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="Limit number of plans processed (default: all).")
    ap.add_argument("--grid_size", type=int, default=GRID_XY,
                    help="Horizontal voxel grid size (H=W=N). Plans are anchored at the "
                         "origin and CROPPED to fit. 128 @ 0.1 m = 12.8 m in x/y. "
                         "[default: USER CONFIG GRID_XY]")
    ap.add_argument("--grid_z", type=int, default=GRID_Z,
                    help="Vertical voxel grid size (D). Apartments are ~36 tall; 48 @ 0.1 m "
                         "= 4.8 m and is divisible by 2**model_depth (16). Trims dead air. "
                         "[default: USER CONFIG GRID_Z]")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of plans held out for validation.")
    ap.add_argument("--voxel_size", type=float, default=0.1,
                    help="Physical voxel edge length in metres (CSV coords are metres; "
                         "spacing is 0.1 m = 10 cm). Fixed, not auto-detected.")
    ap.add_argument("--load_factor", type=float, default=LOAD_FACTOR,
                    help="Design safety factor on the gravity body force (fz). 3.0 = solve "
                         "under 3x self-weight; scales both the fz input channel and the "
                         "displacement target (linear, so they stay consistent). "
                         "[default: USER CONFIG LOAD_FACTOR]")
    ap.add_argument("--fem_tol", type=float, default=1e-6, help="CG tolerance.")
    ap.add_argument("--fem_maxiter", type=int, default=2000, help="CG max iterations.")
    ap.add_argument("--precond", default=PRECOND, choices=["jacobi", "amg", "none", "direct"],
                    help="FEM solve: 'direct' = active-DOF reduction + direct sparse factorize "
                         "(fastest, exact); 'amg'/'jacobi' = iterative CG. amg needs pyamg. "
                         "[default: USER CONFIG PRECOND]")
    ap.add_argument("--jobs", type=int, default=JOBS,
                    help="Parallel worker processes (each direct 128^3 solve ~13 GB; 2 fits "
                         "32 GB). [default: USER CONFIG JOBS]")
    ap.add_argument("--threads_per_job", type=int, default=THREADS_PER_JOB,
                    help="BLAS/OMP threads per worker (jobs*threads <= physical cores). "
                         "[default: USER CONFIG THREADS_PER_JOB]")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if the output .h5 exists (default: skip existing, "
                         "so an interrupted run can be resumed).")
    ap.add_argument("--material", default=MATERIAL, choices=list(MATERIALS),
                    help="Material preset (steel | wood). Wood = isotropic structural softwood. "
                         "[default: USER CONFIG MATERIAL]")
    ap.add_argument("--E0", type=float, default=None, help="Override Young's modulus (Pa).")
    ap.add_argument("--rho_mat", type=float, default=None, help="Override material density (kg/m^3).")
    ap.add_argument("--nu", type=float, default=None, help="Override Poisson's ratio.")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed for train/val split.")
    ap.add_argument("--no_stats", action="store_true",
                    help="Skip computing normalization_stats.npz at the end "
                         "(otherwise it is written into out_dir, ready for train.py).")
    args = ap.parse_args()

    # --- Check the USER CONFIG values before doing any slow work ------------
    # Guard 1: material name must be a known preset. (argparse does not check
    # the default against the choices, so a typo here would otherwise crash
    # later with a confusing error.) Stop now with a clear message.
    if args.material not in MATERIALS:
        raise SystemExit(
            f"MATERIAL = {args.material!r} is not a known material. "
            f"Use one of: {', '.join(MATERIALS)}.")
    # Guard 2: grid numbers must be positive (zero/negative is always wrong).
    if args.grid_size <= 0 or args.grid_z <= 0:
        raise SystemExit(
            f"GRID_XY and GRID_Z must be positive numbers "
            f"(got GRID_XY={args.grid_size}, GRID_Z={args.grid_z}).")
    # Heads-up (not an error): the later training step prefers grid numbers
    # that divide evenly by 16 (e.g. 16, 32, 48, 64, 128). Warn but continue.
    for cfg_name, val in (("GRID_XY", args.grid_size), ("GRID_Z", args.grid_z)):
        if val % 16 != 0:
            print(f"NOTE: {cfg_name}={val} is not divisible by 16. This script "
                  f"still runs fine, but the training step may need a multiple "
                  f"of 16 (e.g. 16, 32, 48, 64, 128).", flush=True)

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
    print("\n=== FEA setup (replicate in nTop) ===\n"
          + format_fea_setup(args, material, grid_dhw, box)
          + "=====================================\n", flush=True)

    n_failed = [0]    # list so the nested _report can mutate it
    results = []      # per-plan result dicts, for the run-description summary

    def _report(k: int, r: dict) -> None:
        conv = "OK" if r["converged"] else "NOT-CONV (skipped, not written)"
        if not r["written"]:
            n_failed[0] += 1
        results.append(r)
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
    n_written = len(tasks) - n_failed[0]
    print(f"\nDone in {elapsed/60:.1f} min ({elapsed/max(len(tasks),1):.1f}s/sample avg). "
          f"Wrote {n_written}/{len(tasks)} samples to {args.out_dir} "
          f"(skip-existing={n_skip}, not-converged/dropped={n_failed[0]}).", flush=True)

    # Write a human-readable description of this run (FEA setup actually used +
    # run accounting) alongside the samples.
    desc_path = Path(args.out_dir) / "run_description.txt"
    write_run_description(
        desc_path, args, material, grid_dhw, box,
        n_found=len(files), n_train=n_train, n_val=n_val_written,
        n_tasks=len(tasks), n_skip=n_skip, n_written=n_written,
        n_dropped=n_failed[0], elapsed=elapsed, results=results)
    print(f"Wrote run description -> {desc_path}", flush=True)

    # Compute normalization stats over the train split we just wrote, so the
    # dataset is immediately ready for train.py (skip with --no_stats). We hand
    # compute_stats() THIS run's out_dir, so it always reads the data just
    # generated -- never the standalone DATA_DIR default in compute_stats.py.
    # Wrapped so a stats failure can't undo a completed (multi-hour) run.
    if not args.no_stats:
        try:
            from step1_preprocess.compute_stats import compute_stats  # noqa: E402
            stats_path, _, n_stat = compute_stats(args.out_dir)
            print(f"Wrote normalization stats -> {stats_path} "
                  f"(over {n_stat} train samples)", flush=True)
        except Exception as exc:
            print(f"Stats: SKIPPED/FAILED ({exc}). Generation is fine; run "
                  f"compute_stats.py to make normalization_stats.npz.", flush=True)


if __name__ == "__main__":
    main()
