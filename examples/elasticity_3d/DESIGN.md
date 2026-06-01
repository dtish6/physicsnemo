# DESIGN.md — 3D Linear Elasticity Surrogate (PhysicsNeMo)

## 1. Problem Statement

A data-driven surrogate solver for 3D linear elasticity.  Given a voxelized
description of geometry, material properties, loads, and boundary conditions on
a regular Cartesian grid, the model predicts the full displacement field u(x).

**Target application:** structural analysis of architectural/civil geometries
subject to gravity body forces and lateral wind loads, with a clamped base.
The primary use case is as a fast forward model inside a topology optimizer.

**Governing equations (linear elastic, static equilibrium):**

```
div(σ) + f = 0         (momentum balance)
σ = λ tr(ε) I + 2μ ε  (isotropic Hooke's law)
ε = ½(∇u + ∇uᵀ)       (small-strain kinematics)

Lamé parameters:
    λ = E·ν / ((1+ν)(1−2ν))
    μ = E / (2(1+ν))
    ν = 0.30 (constant)

Boundary conditions:
    u = 0  on z = 0 plane  (fully clamped base, Dirichlet)
    σ·n = p·n  on wind-exposed exterior surface  (Neumann)
```

---

## 2. Tensor Conventions

All tensors follow PyTorch **NCDHW** layout: `(N, C, D, H, W)`.

| Symbol | Meaning |
|--------|---------|
| N | batch size |
| C | channel (feature) dimension |
| D | depth (z-direction, vertical in physical space) |
| H | height (y-direction) |
| W | width  (x-direction) |

**Coordinate system:** right-handed Cartesian.  Tensor index `[…, d, h, w]`
corresponds to physical coordinate `(x = w·Δx, y = h·Δy, z = d·Δz)` where
`Δx = Δy = Δz = voxel_size` (isotropic).

**Base plane:** `D = 0` slice → `z = 0` (clamped, `u = 0`).

---

## 3. Input Channel Layout (`C_in = 6`)

| Ch | Field | Physical meaning | Units | Notes |
|----|-------|-----------------|-------|-------|
| 0 | `solid_mask` | Voxel occupancy | — | Binary float: 1 = solid, 0 = void |
| 1 | `E_normalized` | Young's modulus | Pa | z-score normalized using dataset stats |
| 2 | `fx` | Body force, x | N/m³ | z-score normalized |
| 3 | `fy` | Body force, y | N/m³ | z-score normalized |
| 4 | `fz` | Body force, z | N/m³ | Gravity-dominated: `−ρg·solid_mask` |
| 5 | `wind_pressure` | Surface pressure magnitude | Pa | Non-zero at exterior voxels only; direction inferred from geometry |

**Fixed physics (not model inputs):**
- Poisson's ratio: **ν = 0.30** (constant across all samples)
- Boundary condition: **z = 0 plane fully clamped** (`u = 0` Dirichlet) — the
  model learns this constraint implicitly from training data.

---

## 4. Output Channel Layout (`C_out = 3`)

| Ch | Field | Physical meaning | Units |
|----|-------|-----------------|-------|
| 0 | `ux` | x-displacement | m (normalized during training) |
| 1 | `uy` | y-displacement | m |
| 2 | `uz` | z-displacement | m |

---

## 5. HDF5 File Layout

Each training sample is stored as one HDF5 file `sample_{i:05d}.h5`:

```
sample_00042.h5
├── solid_mask        (D, H, W)      float32
├── E_normalized      (D, H, W)      float32
├── body_force        (3, D, H, W)   float32
├── wind_pressure     (D, H, W)      float32
└── displacement      (3, D, H, W)   float32

attrs:
    nu        = 0.3   (Poisson's ratio, reference only)
    bc        = "z=0 plane fully clamped (Dirichlet u=0)"
    grid_size = 128
```

**Normalization statistics** (`normalization_stats.npz`):

```
input_mean   shape (6,)   float32
input_std    shape (6,)   float32
output_mean  shape (3,)   float32
output_std   shape (3,)   float32
```

Generated once by `ElasticityDataset.compute_statistics()`.  Applied at load
time by `ChannelNormalize`.

---

## 6. Model Architecture

```
Input (N, 6, D, H, W)
        │
        ▼
ElasticityUNet  (physicsnemo.core.module.Module)
    └── UNet backbone  (physicsnemo.models.unet.UNet)
        ├── Encoder
        │   ├── Level 0: Conv3DBlock × 2  (ch: 32)   → MaxPool3d(2)
        │   ├── Level 1: Conv3DBlock × 2  (ch: 64)   → MaxPool3d(2)
        │   ├── Level 2: Conv3DBlock × 2  (ch: 128)  → MaxPool3d(2)
        │   └── Level 3: Conv3DBlock × 2  (ch: 256)  ← bottleneck
        └── Decoder (skip connections from encoder levels)
            ├── Level 2: ConvTranspose3d(2) + Conv3DBlock × 2
            ├── Level 1: ConvTranspose3d(2) + Conv3DBlock × 2
            └── Level 0: ConvTranspose3d(2) + Conv3DBlock × 2
                                             + 1×1×1 conv → out_channels=3
        │
        ▼
Output (N, 3, D, H, W)
```

**Spatial sizes** (128³ input, depth=4):

| Level | Spatial (DHW) | Channels |
|-------|--------------|---------|
| Input | 128³ | 6 |
| After L0 pool | 64³ | 32 |
| After L1 pool | 32³ | 64 |
| After L2 pool | 16³ | 128 |
| Bottleneck | 16³ | 256 |
| After L2 up | 32³ | 64 |
| After L1 up | 64³ | 32 |
| After L0 up | 128³ | 32 |
| Output | 128³ | 3 |

**Normalization:** GroupNorm (default) with `num_groups=1` (equivalent to
InstanceNorm).  Stable with variable batch sizes; no batch-size dependency.

**Gradient checkpointing:** enabled by default — halves peak activation memory
at the cost of ~30% extra compute per backward pass.

---

## 7. Loss Functions

### 7.1  `displacement_mse`

```
L_mse = ||û − u||²_F / (N·3·D·H·W)
```

Dense signal over all voxels.

### 7.2  `masked_mse`

```
L_masked = Σ_{i∈Solid} ||û_i − u_i||² / |Solid|
```

MSE restricted to solid voxels (`solid_mask == 1`).  Prevents the model from
wasting capacity fitting void-region values that are trivially near zero.

### 7.3  `equilibrium_residual`  (optional physics-informed term)

```
r_i = Σ_j ∂σ_ij/∂x_j      (stress divergence)
L_eq = ||r||²

σ derived from û via central finite differences:
    ε_ij = ½(∂û_i/∂x_j + ∂û_j/∂x_i)
    σ_ij = λ·tr(ε)·δ_ij + 2μ·ε_ij
```

**Warm-up strategy:** Set `loss_weights.eq_residual: 0.0` for the first
≈20 epochs.  Raw network output early in training produces large, noisy stress
divergence values that destabilize learning.  Enable gradually (e.g. 0.01 →
0.1 → 1.0) once `L_mse` has converged below ~0.1.

### 7.4  Combined loss

```
L = w_mse · L_mse  +  w_masked · L_masked  +  w_eq · L_eq
```

Default weights: `w_mse = 1.0`, `w_masked = 1.0`, `w_eq = 0.0`.

---

## 8. Data Pipeline

```
HDF5 files on disk (*.h5, one file = one sample)
        │
        ▼  (num_workers DataLoader worker processes)
ElasticityDataset.__getitem__()
   ├── Opens .h5 file fresh each call  (h5py fork-safety requirement)
   ├── Reads 5 datasets → assembles (6, D, H, W) input tensor
   └── Returns (x: float32, y: float32)
        │
        ▼
ChannelNormalize.__call__()
   ├── Loads mean/std from normalization_stats.npz
   └── Applies z-score per channel: x_norm = (x − μ) / σ
        │
        ▼
DataLoader (batch_size=N, pin_memory=True)
        │
        ▼
Trainer.train_epoch() / validate()
   ├── autocast("cuda")  +  GradScaler  (AMP)
   ├── model(x)  →  pred
   ├── combined_loss(pred, y, ...)  →  L
   └── L.backward()  →  optimizer.step()
```

---

## 9. Checkpoint Strategy

Uses `physicsnemo.utils.checkpoint.save_checkpoint` / `load_checkpoint`:

```
checkpoints/
├── ElasticityUNet.0.<epoch>.mdlus   ← model weights (zip archive)
└── checkpoint.0.<epoch>.pt         ← optimizer + scheduler + scaler state
```

- Saved every `cfg.checkpoint.save_every_n_epochs` epochs (default 5).
- Loaded automatically on resume; `load_checkpoint()` returns starting epoch.
- `.mdlus` format stores `model.pt` + `args.json` + `metadata.json` — supports
  `Module.from_checkpoint()` for inference-only loading without re-instantiation.

---

## 10. Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Relative L2 | `∥û−u∥₂ / ∥u∥₂` | Per-sample; averaged over batch |
| Von Mises error | `∥σ_vm(û) − σ_vm(u)∥₂ / ∥σ_vm(u)∥₂` | FD-derived stress field |
| Max displacement error | `max ∥û_mag − u_mag∥` | Magnitude per voxel |

---

## 11. Scaling Path

| Grid | `model_depth` | `feature_map_channels` | Approx. params | Peak GPU RAM (fp16, batch=2) |
|------|-------------|----------------------|----------------|------------------------------|
| 32³ (smoke test) | 3 | [16,16,32,32,64,64] | ~0.5 M | < 1 GB |
| 128³ (default) | 4 | [32,32,64,64,128,128,256,256] | ~8 M | ~6–8 GB |
| 256³ | 5 | [32,32,64,64,128,128,256,256,512,512] | ~30 M | ~24–32 GB |

For 256³: set `model_depth: 5` and extend `feature_map_channels` to 10 entries
in `conf/config.yaml`.  No code changes required.

**Memory reduction levers:**
- `gradient_checkpointing: true` (default) — halves activation memory
- `amp: true` — halves weight+activation memory (fp16)
- Reduce `batch_size` to 1 if OOM

---

## 12. Future Extensions

### Topology Optimization
`topo_opt/topo_opt_stub.py` exposes the full planned API.  Key steps:
1. Parameterize solid density `ρ ∈ (0,1)` as a learnable sigmoid-gated field.
2. Scale Young's modulus via SIMP: `E_eff = E₀ · ρᵖ` (p ≈ 3).
3. Assemble model input from `ρ`, `E_eff`, and fixed loads.
4. Forward pass through frozen surrogate → displacement → compliance objective.
5. Backpropagate to update `ρ`; project to volume-fraction constraint.

### FEM Data Generation
Subclass `FEMSolverInterface` (`solvers/fem_interface.py`) with a concrete
backend and use it inside `data/generate_mock_hdf5.py`:
- **FEniCSx:** `from dolfinx.fem import ...` — full PDE solver in Python
- **FEBio:** REST API or subprocess with `.feb` XML input files

### Multi-material
Add a one-hot material ID channel group (expanding `C_in` beyond 6).
No architectural changes needed — increase `in_channels` in the config.

### Anisotropic Materials
Extend `equilibrium_residual` to use a full 6-component or 21-component
constitutive tensor stored as additional input channels.
