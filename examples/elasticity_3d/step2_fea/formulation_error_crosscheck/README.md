# Formulation-error cross-check — cube benchmark

A clean, fully-controlled case for validating our `VoxelFEMSolver` against an
**independent** method (nTopology, Abaqus, or closed-form theory). A floor plan
is a poor cross-check (cropped, irregular); a solid cube with a known load is
ideal — it isolates *formulation* correctness from geometry/meshing noise.

**Units: mm – N – MPa** (consistent CAD system, matches Rhino/nTop). Lengths in
mm, forces in N, modulus/stress in MPa (= N/mm²) → displacement in **mm**,
stress in **MPa**.

## The case
| | |
|---|---|
| Geometry | 1000 × 1000 × 1000 mm cube, voxel 100 mm → 10×10×10 elements |
| Material | isotropic, **E = 2700 MPa**, **ν = 0.3**, density 1.25 kg/m³ |
| Support | bottom face (z=0) **fully clamped (Fixed)** |
| Load | **−1000 N** total, uniform pressure on the **top** face (−1×10⁻³ MPa) |

## Files
| File | Role |
|---|---|
| `make_cube_case.py` | Generator: builds the voxel field, solves, writes the two artifacts below. Edit the constants at the top to change the case. |
| `cube_voxel_field.csv` | **Input** voxel field — one row per voxel: `x, y, z` centre (mm) + `density`. |
| `cube_result.h5` | **Result** — `displacement` (mm), `strain`, `stress` (MPa), `von_mises` (MPa) + material/loading/max values as attributes. |
| `gh_read_h5.py` | Grasshopper (Rhino 8 Python 3) component: reads `cube_result.h5` → original points `A`, displaced points `B`, and a `log` of material + loading + max disp/strain/stress. |

## Run
```bash
python step2_fea/formulation_error_crosscheck/make_cube_case.py
```

## Reference result (our solver)
```
max |u|     : 3.42e-4 mm  (0.342 µm)
max strain  : 4.08e-7
max vM      : 1.094e-3 MPa  (1.094 kPa)
```

## Cross-check targets
- **Analytical (1D, free lateral):** δ = σL/E = (1e-3 MPa · 1000 mm) / 2700 MPa
  = **3.70e-4 mm (0.370 µm)**. Our 3.42e-4 mm is ~3% stiffer — expected, because
  the *fixed* base suppresses Poisson expansion near the bottom (a frictionless
  base would match 0.370 µm).
- **nTop / external FEM:** same cube, E=2700 MPa, ν=0.3, **Fixed** bottom face,
  1000 N uniform on top. Expect max displacement ≈ 3.6e-4 mm, stress ≈ 1e-3 MPa.
  Keep the bottom BC = *Fixed* (not roller) or you'll see the ~3% offset above.

## Grasshopper note
`gh_read_h5.py` needs **Rhino 8's Python 3** with `h5py` (auto-installed via the
`# r: h5py` directive). If h5py won't load in your Rhino, convert first:
`python step2_fea/support/h5_to_csv.py --h5 .../cube_result.h5` and use the
CSV reader component instead (no extra packages).
