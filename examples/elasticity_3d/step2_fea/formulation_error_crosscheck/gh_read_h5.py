# r: h5py
# r: numpy
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# GhPython component script  --  NOT run from the command line.
# Reads the cube_result.h5 written by make_cube_case.py and outputs the original
# and displaced voxel points plus a summary log (material, loading, and the max
# displacement / strain / stress).
#
# *** REQUIRES Rhino 8's Python 3 component (ScriptEditor) -- NOT IronPython. ***
# HDF5 needs the `h5py` package; the `# r: h5py` / `# r: numpy` directives at the
# top tell Rhino 8 to pip-install them on first run. (If your Rhino can't load
# h5py, instead run  h5_to_csv.py  on the .h5 to get a CSV, and use the CSV
# reader component -- that path needs no extra packages.)
#
# COMPONENT INPUTS (right-click each -> set type):
#   path  : str    file path to cube_result.h5
#   scale : float  displacement exaggeration for the displaced points
#                  (0 = auto: max deflection ~3 voxels ; 1 = true scale)
#
# COMPONENT OUTPUTS:
#   A    : original voxel points   (Point3d list)
#   B    : displaced voxel points  (Point3d list ; A + u * scale)
#   log  : summary string (material + loading + max disp/strain/stress)
# ============================================================================

import h5py
import numpy as np
import Rhino.Geometry as rg


def _txt(v):
    """h5py attr -> str (handles bytes from some writers)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


A = []      # original points
B = []      # displaced points
log = ""

if path:
    with h5py.File(path, "r") as f:
        rho = f["solid_mask"][:]                       # (Nz, Ny, Nx)
        u   = f["displacement"][:]                     # (3, Nz, Ny, Nx)
        vs  = float(f.attrs.get("voxel_size", 0.1))
        material = _txt(f.attrs.get("material_str", ""))
        loading  = _txt(f.attrs.get("loading_str", ""))
        max_disp   = float(f.attrs.get("max_disp_mm", 0.0))     # mm
        max_e1     = float(f.attrs.get("max_principal_strain", 0.0))  # most tensile
        max_e3     = float(f.attrs.get("min_principal_strain", 0.0))  # most compressive
        max_vm_eps = float(f.attrs.get("max_vm_strain", 0.0))   # von Mises equiv strain
        max_stress = float(f.attrs.get("max_stress_mpa", 0.0))  # MPa

    idx = np.argwhere(rho > 0.05)                      # solid voxels (d, h, w)

    # Auto-scale so the largest displacement is visible (~3 voxels), unless given.
    if scale is None or scale <= 0.0:
        umag = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        umax = float(umag.max()) if umag.size else 0.0
        scale = (3.0 * vs / umax) if umax > 0.0 else 1.0

    for (d, h, w) in idx:
        x = (w + 0.5) * vs; y = (h + 0.5) * vs; z = (d + 0.5) * vs
        A.append(rg.Point3d(x, y, z))
        B.append(rg.Point3d(x + float(u[0, d, h, w]) * scale,
                            y + float(u[1, d, h, w]) * scale,
                            z + float(u[2, d, h, w]) * scale))

    log = "\n".join([
        "MATERIAL : " + material,
        "LOADING  : " + loading,
        "-" * 40,
        "max displacement      : %.4e mm  (%.4f um)" % (max_disp, max_disp * 1e3),
        "max principal strain  : %.4e  (e1, most tensile)" % max_e1,
        "min principal strain  : %.4e  (e3, most compressive)" % max_e3,
        "von Mises equiv strain: %.4e" % max_vm_eps,
        "max stress (vM)       : %.4e MPa  (%.4f kPa)" % (max_stress, max_stress * 1e3),
        "display scale         : x%.4g" % scale,
        "points                : %d" % len(A),
    ])
    print(log)
