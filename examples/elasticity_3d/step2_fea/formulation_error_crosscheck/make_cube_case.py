# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the cube cross-check case: voxel field CSV + solved-result HDF5.

Canonical formulation cross-check (compare our solver against nTop / theory):
  geometry : 1000 x 1000 x 1000 mm cube, voxel 100 mm -> 10 x 10 x 10 elements
  support  : bottom face (z=0 plane) fully clamped (Fixed)
  load     : -1000 N total, uniform on the TOP face (z+), pointing -z
  material : isotropic, E = 2700 MPa, nu = 0.3, density = 1.25 kg/m^3

UNITS: a consistent **mm - N - MPa** system (matches Rhino / nTop). Lengths in
mm, forces in N, modulus/stress in MPa (= N/mm^2). Therefore displacement comes
out in **mm** and stress in **MPa** directly. (Density is informational only --
it is not used unless self-weight is enabled, which needs mass units too.)

Writes, into this folder:
  1. cube_voxel_field.csv  -- the INPUT voxel field (one row per voxel:
                              x, y, z centre in mm + density).
  2. cube_result.h5        -- the FEA RESULT: displacement (mm), strain,
                              stress (MPa), von Mises (MPa), + info as attrs.

Run:
    python step2_fea/formulation_error_crosscheck/make_cube_case.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))            # examples/elasticity_3d

from step2_fea._2_2_voxel_fem import VoxelFEMSolver   # noqa: E402

# --- Case definition (edit here to change the cross-check) ------------------
# Consistent mm - N - MPa units (see module docstring).
N        = 10           # elements per side
SIZE     = 1000.0       # cube edge length (mm)
E0       = 2700.0       # Young's modulus (MPa = N/mm^2)
NU       = 0.3          # Poisson's ratio
RHO_MAT  = 1.25         # density (kg/m^3, informational; unused without gravity)
FORCE_Z  = -1000.0      # total vertical force on the top face (N)
GRAVITY  = False        # add self-weight?  (needs mass units; off for this check)

VOX = SIZE / N          # voxel size (mm)


def principal_strains(strain_voigt):
    """Principal strains per element from a Voigt strain field (6, Nz, Ny, Nx).

    Voigt order [exx, eyy, ezz, gxy, gxz, gyz] with ENGINEERING shears
    (g = 2 x tensor shear), so the tensor off-diagonals are g/2.
    Returns (3, Nz, Ny, Nx) ordered [e1 >= e2 >= e3] (descending).
    """
    sh = strain_voigt.shape[1:]
    n = int(np.prod(sh))
    e = strain_voigt.reshape(6, n)
    T = np.zeros((n, 3, 3))
    T[:, 0, 0] = e[0]; T[:, 1, 1] = e[1]; T[:, 2, 2] = e[2]
    T[:, 0, 1] = T[:, 1, 0] = e[3] / 2.0
    T[:, 0, 2] = T[:, 2, 0] = e[4] / 2.0
    T[:, 1, 2] = T[:, 2, 1] = e[5] / 2.0
    w = np.linalg.eigvalsh(T)[:, ::-1]          # descending [e1, e2, e3]
    return w.T.reshape(3, *sh)


def vm_equivalent_strain(strain_voigt):
    """von Mises equivalent strain field from Voigt strain (6, Nz, Ny, Nx)."""
    exx, eyy, ezz, gxy, gxz, gyz = [strain_voigt[i] for i in range(6)]
    return np.sqrt(2.0 / 9.0 * ((exx - eyy) ** 2 + (eyy - ezz) ** 2 + (ezz - exx) ** 2)
                   + 1.0 / 3.0 * (gxy ** 2 + gxz ** 2 + gyz ** 2))


def top_face_point_loads(n_el, h, total_fz):
    """Consistent nodal loads for a uniform pressure on the top (z+) face."""
    nn = n_el + 1
    iz_top = n_el
    area = (n_el * h) ** 2
    p = total_fz / area
    w = np.ones(nn); w[0] = w[-1] = 0.5            # tributary weights
    loads = []
    for iy in range(nn):
        for ix in range(nn):
            fz = p * (h * h) * w[ix] * w[iy]
            node = iz_top * (nn * nn) + iy * nn + ix
            loads.append((node, np.array([0.0, 0.0, fz])))
    return loads, p


def main():
    grid = (N, N, N)                               # (Nz, Ny, Nx)
    rho = np.ones(grid, dtype=np.float64)          # fully solid cube

    # --- 1. write the input voxel field as CSV -----------------------------
    didx = np.argwhere(rho > 0.0)
    d, hh, w = didx[:, 0], didx[:, 1], didx[:, 2]
    xf = (w + 0.5) * VOX; yf = (hh + 0.5) * VOX; zf = (d + 0.5) * VOX
    pd.DataFrame(
        np.column_stack([xf, yf, zf, rho[d, hh, w]]),
        columns=["x", "y", "z", "density"],
    ).to_csv(_HERE / "cube_voxel_field.csv", index=False, float_format="%.6e")

    # --- 2. solve ----------------------------------------------------------
    solver = VoxelFEMSolver(grid_size=grid, voxel_size=VOX, nu=NU)
    solver.assemble_system(rho, E0=E0)
    solver.apply_bcs()                             # clamp z=0 (bottom face)

    loads, pressure = top_face_point_loads(N, VOX, FORCE_Z)
    body_force = None
    if GRAVITY:
        body_force = np.zeros((3, *grid)); body_force[2] = -RHO_MAT * 9.81
    solver.apply_loads(body_force=body_force, point_loads=loads)

    u_nodal, u_elem, info = solver.solve(precond="direct", verbose=False)
    strain, stress = solver.element_strain_stress(u_nodal)   # (6,...) each
    vm = solver.von_mises_stress(u_nodal)                    # (Nz,Ny,Nx)

    pstrain   = principal_strains(strain)        # (3,Nz,Ny,Nx) = [e1, e2, e3]
    vm_strain = vm_equivalent_strain(strain)     # (Nz,Ny,Nx)

    max_disp   = float(np.abs(u_elem).max())
    max_strain = float(np.abs(strain).max())                 # max|Voigt comp| (legacy)
    max_principal_strain = float(pstrain[0].max())           # e1, most tensile (~nTop)
    min_principal_strain = float(pstrain[2].min())           # e3, most compressive
    max_abs_principal    = float(np.abs(pstrain).max())
    max_vm_strain        = float(vm_strain.max())
    max_stress = float(vm.max())

    material_str = ("isotropic linear elastic | E = %.0f MPa | nu = %.2f | "
                    "density = %.3g kg/m^3" % (E0, NU, RHO_MAT))
    loading_str  = ("Fz = %.0f N uniform on top face (%.4g MPa pressure)%s | "
                    "bottom face fully clamped (Fixed)"
                    % (FORCE_Z, pressure, "  + self-weight" if GRAVITY else ""))

    # --- 3. write the result HDF5 ------------------------------------------
    with h5py.File(_HERE / "cube_result.h5", "w") as f:
        f.create_dataset("solid_mask",   data=rho.astype(np.float32),    compression="gzip")
        f.create_dataset("displacement", data=u_elem.astype(np.float32), compression="gzip")
        f.create_dataset("strain",       data=strain.astype(np.float32), compression="gzip")
        f.create_dataset("principal_strain", data=pstrain.astype(np.float32),   compression="gzip")  # (3,..)=[e1,e2,e3]
        f.create_dataset("vm_strain",    data=vm_strain.astype(np.float32), compression="gzip")
        f.create_dataset("stress",       data=stress.astype(np.float32), compression="gzip")
        f.create_dataset("von_mises",    data=vm.astype(np.float32),     compression="gzip")
        f.attrs["units"]        = "mm-N-MPa"
        f.attrs["voxel_size"]   = VOX                  # mm
        f.attrs["grid_shape"]   = np.array(grid, dtype=np.int32)
        f.attrs["E_mpa"]        = E0
        f.attrs["nu"]           = NU
        f.attrs["density"]      = RHO_MAT
        f.attrs["force_total_n"] = FORCE_Z
        f.attrs["pressure_mpa"] = pressure
        f.attrs["material_str"] = material_str
        f.attrs["loading_str"]  = loading_str
        f.attrs["max_disp_mm"]   = max_disp            # mm
        f.attrs["max_strain"]    = max_strain          # max|Voigt component| (legacy)
        f.attrs["max_principal_strain"] = max_principal_strain   # e1 (compare to nTop)
        f.attrs["min_principal_strain"] = min_principal_strain   # e3
        f.attrs["max_abs_principal_strain"] = max_abs_principal
        f.attrs["max_vm_strain"] = max_vm_strain       # von Mises equivalent strain
        f.attrs["max_stress_mpa"] = max_stress         # MPa

    print("Wrote:")
    print("  ", _HERE / "cube_voxel_field.csv", "(%d voxels)" % len(didx))
    print("  ", _HERE / "cube_result.h5")
    print("Material:", material_str)
    print("Loading :", loading_str)
    print("Result  : max|u| = %.4e mm | max vM stress = %.4e MPa" % (max_disp, max_stress))
    print("Strain  : max principal e1 = %.4e | min principal e3 = %.4e | "
          "max|principal| = %.4e | vM-equiv = %.4e | max|Voigt| = %.4e"
          % (max_principal_strain, min_principal_strain, max_abs_principal,
             max_vm_strain, max_strain))


if __name__ == "__main__":
    main()
