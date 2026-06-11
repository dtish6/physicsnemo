# DESIGN_PIPELINE.md — 3D Elasticity Surrogate, organized by pipeline stage

> Companion to [DESIGN.md](DESIGN.md). DESIGN.md describes the system **by
> component** (channels, model, loss, …). This document describes the **same
> system as a chronological pipeline** and maps every stage to the `stepN_*`
> folders and their numbered `_N_M_` sub-step modules. If the two disagree,
> this file reflects the *current code*.

## Folder convention

- `stepN_*/` — one folder per pipeline stage (1 preprocess → 4 predict).
- `_N_M_name.py` — a numbered sub-step *module* within a stage, in execution
  order. (Underscores, not dots: `_2.1_` is an illegal Python module name.)
- `support/` — files that are NOT on the main path (tests, benchmarks,
  alternate backends, mock/synthetic helpers).

## Overview

```
raw apartment floor-plan CSVs
   │
   ▼  step1_preprocess/ generate_from_csv.py     (1:1 — no downsample, no padding)
①  _1_1_binarize: rasterize -> fixed 48x128x128 grid (10 cm voxels), crop to fit
   │
   ▼  step2_fea/ _2_2_voxel_fem.run(precond="direct")
②  active-DOF reduction (solid only) + DIRECT sparse solve ──▶ displacement = TARGET
   │                                                            (~16 s/plan, exact)
   ▼  step3_training/
③  pair (x, y) ─▶ normalize ─▶ U-Net ─▶ combined loss ─▶ backprop
   │
   ▼  step4_predict/
④  trained surrogate ─▶ displacement in one forward pass ─▶ metrics / topo-opt
```

Each `sample_NNNNN.h5` holds 5 fields (see [DESIGN.md §5](DESIGN.md)):
`solid_mask`, `E_normalized`, `body_force`, `wind_pressure`, `displacement`.
Tensor layout is **NCDHW**; `D = z` (vertical), base plane `D=0` is clamped.
Grid is a fixed **48 × 128 × 128** (D×H×W) at 10 cm/voxel = 4.8 × 12.8 × 12.8 m.

---

## Step 1 — Preprocess  ·  `step1_preprocess/`

Turns raw architectural-wall point clouds into the model's 6-channel input
**x**, and prepares the geometry that step 2 will solve. The three per-wall
sub-steps are separate modules, chained inside `generate_from_csv.py`.

| File | Role |
|------|------|
| `_1_1_binarize.py` | `rasterize()` — snap CSV points 1:1 into the fixed grid; solid iff `value==1.0`; crop to box. |
| `generate_from_csv.py` | Driver: per plan, binarize → step 2 (FEA) → write `.h5`. Parallel (`--jobs`), resumable (skips existing), drops non-converged samples. |
| `support/check_bbox.py` | Diagnostic: which plans exceed the box / how much crop loss. |
| `support/generate_mock_hdf5.py` | Synthetic samples for smoke-testing (no CSV/FEM needed). |
| `support/_1_2_downsample.py`, `_1_3_pad.py`, `_1_4_pad_to_cube.py` | **Retired.** From the older downsample + pad-to-cube flow; kept for reference, not used by the 1:1 pipeline. |

### ① Binarize — `_1_1_binarize.rasterize()`
Each `plan_*.csv` is `(x, y, z, value)` in metres (10 cm grid spacing).
**`value` is a label, not a density:** only `value == 1.0` is solid wall; all
other values (0, 0.1, 0.5, …) mean void. Points are snapped (1:1, no
downsampling) into a **fixed `(48, 128, 128)` grid**, anchored at the origin
(base on the clamped `z=0` plane). Anything outside the box is **cropped**.
The result is strictly binary `{0, 1}` (no fractional voxels).

**Box sizing:** 128 voxels × 10 cm = 12.8 m horizontally; 48 × 10 cm = 4.8 m
vertically (apartments are ~36 voxels / 3.6 m tall, plus margin; 48 is ÷16 for
the depth-4 U-Net). ~80% of plans exceed 12.8 m in x/y and are cropped (mean
~19% solid loss); `check_bbox.py` quantifies this per plan.

**Output:** `data/train/*.h5`, `data/val/*.h5` (deterministic 80/20 split).

---

## Step 2 — FEA simulation  ·  `step2_fea/`

The ground-truth solver: a **matrix-free Q1 hexahedral FEM** for 3D linear
elasticity. It produces the supervised **target** displacement field `y` for
each preprocessed geometry. Invoked from `generate_from_csv.py` via
`build_sample()`.

| File | Role |
|------|------|
| `_2_1_hex_stiffness.py` | The 24×24 unit element stiffness `K0` and strain operator `B`. |
| `_2_2_voxel_fem.py` | `VoxelFEMSolver` — the solver (the heart of step 2). |
| `support/fem_interface.py` | Abstract `FEMSolverInterface` + `MockFEMSolver` (swap-in backend point). |
| `support/benchmark.py` | Wall-clock performance benchmark of the solver. |
| `support/compare_preconditioners.py` | Jacobi vs AMG comparison at 64³. |
| `support/tests/test_voxel_fem.py` | Correctness tests (patch tests, cantilever, etc.). |

### How `VoxelFEMSolver` works
**Discretization:** `Nz×Ny×Nx` voxel *elements*; nodes at the 8 corners,
3 DOFs each (ux,uy,uz); trilinear hex (Q1/H8) with 2×2×2 Gauss quadrature.

**Material (SIMP):** per-element modulus `E_e = Emin + ρ_e^p · (E0 − Emin)`
(`p≈3`, `Emin` avoids a singular matrix), Poisson `ν = 0.3` fixed.

**Matrix-free solve:** no global stiffness matrix is assembled. Instead the
matrix–vector product `K·v` is computed element-wise,
`Kv += Σ_e E_e · (K0 · v_local_e)`, via gather/`bincount` scatter — O(n_elem·24)
memory instead of O(n_dof·bandwidth). Solved with **Conjugate Gradient** on the
free (non-clamped) DOFs. `K0` comes from `_2_1_hex_stiffness`.

### Two solve methods (`solve()` picks via `precond`)

**A. Direct — `precond="direct"` (default for generation, ~10× faster).**
The box is ~93% void, so solving over every DOF wastes effort. The direct path
(`_solve_direct`) does **active-DOF reduction**: it assembles `K` over only the
**solid** elements (+ clamped base), shrinking ~2.4 M DOFs → ~0.5 M, then
**factorizes and solves exactly** with a direct sparse solver — Intel MKL
PARDISO via `pypardiso` if installed, else SciPy SuperLU. No preconditioner, no
iterations. Result matches the iterative solve to round-off in the solid and is
cleanly **zero in the void**. ~16 s/plan at 48×128×128. *Caveat:* solid not
connected to the clamped base → singular → flagged `converged=False` (the
generator drops such samples rather than writing garbage).

**B. Iterative — `precond="amg"` / `"jacobi"`.** Matrix-free Conjugate Gradient,
no assembled `K`: `Kv += Σ_e E_e · (K0 · v_local_e)` via gather/`bincount`.
AMG (smoothed aggregation, needs `pyamg`) preconditions it to ~48 iters but its
setup is costly (~150 s/plan); Jacobi is cheap but needs far more iterations.
Kept as a fallback / reference.

### The methods, in call order (`run()` orchestrates them)
1. `assemble_system(rho, E0)` — per-element SIMP moduli + `elem_dofs` connectivity.
2. `apply_bcs()` — clamp the `z=0` plane (Dirichlet `u=0`); partitions free/fixed DOFs.
3. `apply_loads()` — gravity body force, optional point loads / wind pressure traction.
4. `solve()` — direct active-DOF factorization (default) **or** matrix-free CG.
5. post-process — `_element_center_displacement` → `u_elem`, optional `von_mises_stress`.

`run()` returns `{u_nodal, u_elem, sigma_vm?, info}`; the generator stores
`u_elem` (shape `(3,Nz,Ny,Nx)`) as the `displacement` target.

### Extending step 2
`support/fem_interface.py` is the seam for a real backend (FEniCSx, FEBio):
subclass `FEMSolverInterface`, implement `solve()`/`is_available()`.
`MockFEMSolver` returns cheap noise for fast tests.

---

## Step 3 — Training (pairing)  ·  `step3_training/`

Pairs each input `x` with its FEA target `y`, normalizes, and trains the U-Net.

| File | Role |
|------|------|
| `_3_1_dataset.py` | `ElasticityDataset` — reads `.h5`, assembles 6-ch `x` + 3-ch `y`. |
| `_3_2_transforms.py` | `ChannelNormalize` — per-channel z-score (+ denormalize for inference). |
| `_3_3_elasticity_unet.py` | `ElasticityUNet` — wraps `physicsnemo.models.unet.UNet`. |
| `_3_4_losses.py` | `displacement_mse`, `masked_mse`, `equilibrium_residual`, `combined_loss`. |
| `_3_5_trainer.py` | `Trainer` — epoch loop, AMP, checkpointing, validation. |

### The pairing
- **x** (6 ch): `[solid_mask, E, fx, fy, fz, wind]` — `_load_sample()`.
- **y** (3 ch): `displacement` — the FEA answer from step 2.
- `ChannelNormalize` z-scores both with the stats `.npz`.

### Loss (`combined_loss`)
`L = w_mse·mse + w_masked·masked_mse + w_eq·eq_residual` (defaults 1,1,0).
`masked_mse` weights only **solid** voxels. Channel 0 is z-score **normalized**
(so void < 0, solid > 0 — not a {0,1} mask), so the trainer **denormalizes
channel 0 and thresholds `>0.5`** to recover a true {0,1} mask (otherwise
negative weights drive the loss negative and it diverges).

### Run it
`conf/config.yaml` defaults to the generated dataset, so:
```
python train.py                 # uses data/train, grid_size [48,128,128]
```
See the **train.py I/O** section below for the full input/output contract.

---

## Step 4 — Predicting  ·  `step4_predict/`

Use the trained surrogate as a fast forward model.

| File | Role |
|------|------|
| `_4_1_metrics.py` | `relative_l2_error`, `von_mises_stress_error`, `max_displacement_error`, `compute_all_metrics`. |
| `support/topo_opt_stub.py` | `TopologyOptimizer` — planned downstream use (surrogate inside a topology optimizer). |

Inference = one forward pass `ElasticityUNet(x) → pred`, then
`ChannelNormalize.denormalize_output(pred)` to recover physical metres
(denorm stats are moved to the prediction's device). Quality is judged by the
metrics above (`relative_l2` is the headline number).

---

## train.py — input/output contract

`train.py` is the **Hydra entry point** for step 3. It is a *training* driver:
its job is to turn a config + a dataset into a trained model.

### Inputs
| Input | Source | Notes |
|-------|--------|-------|
| Config | `conf/config.yaml` (+ CLI overrides) | All hyperparameters: dataset paths, model depth/width, epochs, lr, batch size, AMP, loss weights, checkpoint/log dirs. Override inline, e.g. `python train.py training.epochs=50 training.lr=5e-5`. |
| Training data | `dataset.train_path` (`./data/train`) | `sample_*.h5` files → each yields `(x: 6×48×128×128, y: 3×48×128×128)`. |
| Validation data | `dataset.val_path` (`./data/val`) | Same format; used for metrics each epoch. |
| Normalization stats | `dataset.stats_path` (`./data/normalization_stats.npz`) | Per-channel mean/std for z-scoring. |
| Resume checkpoint | `checkpoint.ckpt_path` (`./checkpoints`) | If a checkpoint exists there, training **resumes** from it; else starts from scratch. |

### Outputs
| Output | Location | Notes |
|--------|----------|-------|
| Model weights | `checkpoints/ElasticityUNet.0.<epoch>.mdlus` | PhysicsNeMo `.mdlus` archive (weights + args + metadata); load with `Module.from_checkpoint()`. Saved every `save_every_n_epochs` and at the end. |
| Training state | `checkpoints/checkpoint.0.<epoch>.pt` | Optimizer + scheduler + AMP scaler — for exact resume. |
| TensorBoard logs | `logging.log_dir` (`./tensorboard_logs`) | Scalar curves (losses, val metrics). |
| Console / `train.log` | stdout + Hydra run dir | Per-epoch `mse / masked_mse / eq_residual / total` and val `rel_l2 / von_mises_err / max_disp_err`. |
| Hydra run dir | `outputs/<timestamp>/` | Resolved config snapshot + `train.log` for the run. |

### What it does (the loop)
1. Build `Trainer` (`_3_5_trainer`) → datasets, dataloaders, model, optimizer, scheduler.
2. `load_checkpoint()` → resume epoch (0 if fresh).
3. For each epoch: `train_epoch()` (forward → `combined_loss` → backward → step),
   then `validate()` (metrics), log both, checkpoint on schedule.
4. Save a final checkpoint; log `Training complete.`

**In one line:** *in* = config + paired `(x, y)` HDF5 data; *out* = a trained
`ElasticityUNet` checkpoint (+ logs/metrics) that predicts displacement from the
6-channel input.

---

## Appendix — folder ↔ DESIGN.md topic map

| Stage folder | DESIGN.md section |
|--------------|-------------------|
| `step1_preprocess/` | §3 input channels, §5 HDF5 layout |
| `step2_fea/` | §1 governing equations, §7.3 equilibrium |
| `step3_training/` | §6 model, §7 loss, §8 pipeline, §9 checkpoint |
| `step4_predict/` | §10 metrics, §12 future (topo-opt) |

Data artifacts live in `data/` (not a code package): `train/ val/` +
`normalization_stats.npz`. (`data/_obsolete_16x64x80/` holds the earlier
wall-based dataset, kept as a backup.)
