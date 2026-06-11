# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an already-solved sample_*.h5 into a Rhino/Grasshopper CSV.

No re-solve.  Reads the FEA displacement (always) and, if the file stores the
Cauchy ``stress`` tensor, also exports von Mises + principal-stress lines --
the same columns as ``export_rhino.py`` but WITHOUT redoing the ~30s solve.

Output (one row per SOLID voxel):
  x, y, z              original voxel-centre position (metres, Z-up)
  x_def, y_def, z_def  displaced position = original + u * --scale
  ux, uy, uz, umag     displacement vector (metres) + magnitude
  von_mises            equivalent stress (Pa)            [if stress in file]
  p1x,p1y,p1z,p1       1st principal: dir + signed stress (Pa), most TENSILE
  p2x,p2y,p2z,p2       2nd principal: dir + signed stress (Pa), intermediate
  p3x,p3y,p3z,p3       3rd principal: dir + signed stress (Pa), most COMPRESSIVE
                       (the three dirs are orthogonal -> trajectories cross 90 deg)
                                                          [if stress in file]

Grasshopper: Point3d from (x,y,z) = cloud A, from (x_def,y_def,z_def) = cloud B;
wire A and B into a Line / Vector-Display component for the deformation field.
Colour cloud A by von_mises; draw short lines along (px,py,pz) for force-flow.

Usage
-----
    python step2_fea/support/h5_to_csv.py --h5 data_wood/train/sample_00000.h5
    python step2_fea/support/h5_to_csv.py --h5 data_wood/val/sample_00003.h5 \
        --out sample3.csv --scale 1
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DENSITY_THRESHOLD = 0.05   # matches generate_from_csv: rho above this = solid


def principal_stresses(stress_voigt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All three principal stresses per voxel, ordered s1 >= s2 >= s3.

    stress_voigt: (N, 6) = [sxx, syy, szz, sxy, sxz, syz].
    Returns
      dirs : (N, 3, 3) unit eigenvectors; dirs[:, k, :] is the direction of the
             (k+1)-th principal stress (k=0 -> s1 most tensile, k=2 -> s3 most
             compressive). The three are mutually orthogonal -> trajectories
             cross at 90 deg.
      vals : (N, 3) signed principal stresses [s1, s2, s3] (descending).
    """
    sxx, syy, szz, sxy, sxz, syz = stress_voigt.T
    T = np.empty((stress_voigt.shape[0], 3, 3), dtype=np.float64)
    T[:, 0, 0] = sxx; T[:, 1, 1] = syy; T[:, 2, 2] = szz
    T[:, 0, 1] = T[:, 1, 0] = sxy
    T[:, 0, 2] = T[:, 2, 0] = sxz
    T[:, 1, 2] = T[:, 2, 1] = syz
    w, v = np.linalg.eigh(T)                       # ascending: w[:,0]<=..<=w[:,2]
    vals = w[:, ::-1]                              # descending [s1, s2, s3]
    dirs = v[:, :, ::-1].transpose(0, 2, 1)        # (N,3,3): dirs[:,k,:] = k-th evec
    return dirs, vals


def von_mises_from_voigt(s: np.ndarray) -> np.ndarray:
    """von Mises scalar from Voigt stress (N,6)=[sxx,syy,szz,sxy,sxz,syz]."""
    sxx, syy, szz, sxy, sxz, syz = s.T
    return np.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
               + 6.0 * (sxy**2 + sxz**2 + syz**2)) + 1e-300
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to a sample_*.h5 file.")
    ap.add_argument("--out", default=None, help="Output CSV (default: <h5 stem>.csv).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Export every Nth solid voxel (lighten the file).")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="Displacement exaggeration for x_def/y_def/z_def. "
                         "0 (default) = auto (max deflection ~3 voxels); 1 = true scale.")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        rho = f["solid_mask"][...]                      # (Nz, Ny, Nx)
        u   = f["displacement"][...]                    # (3, Nz, Ny, Nx)
        vs  = float(f.attrs.get("voxel_size", 0.1))
        has_stress = "stress" in f
        stress = f["stress"][...] if has_stress else None   # (6, Nz, Ny, Nx)
    Nz, Ny, Nx = rho.shape
    print(f"{Path(args.h5).name}: grid={rho.shape} voxel={vs} m "
          f"stress={'yes' if has_stress else 'NO (displacement only)'}", flush=True)

    # Solid voxels (optionally strided).
    mask = rho > DENSITY_THRESHOLD
    didx = np.argwhere(mask)                            # (Nsolid, 3) as (d,h,w)
    if args.stride > 1:
        didx = didx[::args.stride]
    d, h, w = didx[:, 0], didx[:, 1], didx[:, 2]

    x = (w + 0.5) * vs; y = (h + 0.5) * vs; z = (d + 0.5) * vs   # metres, Z-up
    ux = u[0][d, h, w]; uy = u[1][d, h, w]; uz = u[2][d, h, w]
    umag = np.sqrt(ux**2 + uy**2 + uz**2)

    umax = float(umag.max()) if umag.size else 0.0
    scale = args.scale
    if scale <= 0.0:
        scale = (3.0 * vs / umax) if umax > 0.0 else 1.0
    xd = x + ux * scale; yd = y + uy * scale; zd = z + uz * scale

    cols = ["x", "y", "z", "x_def", "y_def", "z_def", "ux", "uy", "uz", "umag"]
    columns = [x, y, z, xd, yd, zd, ux, uy, uz, umag]

    if has_stress:
        sv = np.stack([stress[c][d, h, w] for c in range(6)], axis=1)  # (N,6)
        vm = von_mises_from_voigt(sv)
        pdirs, pvals = principal_stresses(sv)        # (N,3,3), (N,3) -> [s1,s2,s3]
        cols += ["von_mises"]
        columns += [vm]
        # p1 = most tensile (s1), p2 = intermediate, p3 = most compressive (s3).
        for k in range(3):
            cols += [f"p{k+1}x", f"p{k+1}y", f"p{k+1}z", f"p{k+1}"]
            columns += [pdirs[:, k, 0], pdirs[:, k, 1], pdirs[:, k, 2], pvals[:, k]]

    out = args.out or str(Path(args.h5).with_suffix(".csv"))
    pd.DataFrame(np.column_stack(columns), columns=cols).to_csv(
        out, index=False, float_format="%.6e")
    print(f"  wrote {len(x)} solid voxels -> {out}", flush=True)
    print(f"  displaced points scaled x{scale:.3g} "
          f"(|u|max {umax:.2e} m -> {umax*scale:.3g} m visual)", flush=True)
    if has_stress:
        print(f"  von Mises range [{vm.min():.2e}, {vm.max():.2e}] Pa", flush=True)
    else:
        print("  (no stress in file -> use export_rhino.py for von Mises/stress lines)",
              flush=True)


if __name__ == "__main__":
    main()
