# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# GhPython component script  --  NOT run from the command line.
# Paste this into a GhPython (or Python 3) component inside Grasshopper to read
# the CSV written by h5_to_csv.py / export_rhino.py.
#
# Works in Rhino 7 (IronPython 2.7) and Rhino 8 (CPython 3).
#
# COMPONENT INPUTS  (right-click each input -> set type):
#   path  : str    (file path to the .csv ; "Item Access")
#   scale : float  (extra multiplier on the displacement vectors ; default 1.0)
#
# COMPONENT OUTPUTS:
#   A     : original voxel points        (Point3d list)
#   B     : displaced voxel points       (Point3d list ; from x_def,y_def,z_def)
#   disp  : displacement vectors         (Vector3d list ; A -> B, * scale)
#   vm    : von Mises stress per point   (float list, Pa ; empty if not in CSV)
#   ps1   : 1st principal vectors        (Vector3d ; most TENSILE family)
#   ps2   : 2nd principal vectors        (Vector3d ; intermediate family)
#   ps3   : 3rd principal vectors        (Vector3d ; most COMPRESSIVE family)
#          -> ps1, ps2, ps3 are mutually orthogonal (trajectories cross 90 deg)
#   log   : summary string  (max displacement + max stress, with units)
#
# Then, downstream in Grasshopper:
#   * Line(A, B) or a Vector Display on (A, disp) -> deformation field
#   * Gradient remap of vm -> colour the A points / a mesh
#   * Feed (A, ps1) and (A, ps3) into a streamline / flow-line component
#     -> the interwoven principal-stress trajectories (force flow)
#   * Panel on log -> read the max displacement / max stress
# ============================================================================

import csv
import Rhino.Geometry as rg

if scale is None:
    scale = 1.0

A = []      # original points
B = []      # displaced points
disp = []   # displacement vectors (A -> B)
vm = []     # von Mises scalars (Pa)
ps1 = []    # 1st principal vectors (most tensile)
ps2 = []    # 2nd principal vectors (intermediate)
ps3 = []    # 3rd principal vectors (most compressive)

_umag = []  # raw displacement magnitudes (m), for the max in the log


def _pvec(row, k):
    """Vector for the k-th principal (1..3): unit dir * signed magnitude."""
    s = float(row.get("p%d" % k, 1.0))
    return rg.Vector3d(float(row["p%dx" % k]) * s,
                       float(row["p%dy" % k]) * s,
                       float(row["p%dz" % k]) * s)

if path:
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)          # uses the header row for column names
        for row in reader:
            x = float(row["x"]);  y = float(row["y"]);  z = float(row["z"])
            pa = rg.Point3d(x, y, z)
            A.append(pa)

            # Displaced point: prefer the precomputed x_def/y_def/z_def columns;
            # fall back to original + (ux,uy,uz) if they are absent.
            if "x_def" in row:
                pb = rg.Point3d(float(row["x_def"]),
                                float(row["y_def"]),
                                float(row["z_def"]))
            else:
                pb = rg.Point3d(x + float(row["ux"]),
                                y + float(row["uy"]),
                                z + float(row["uz"]))
            B.append(pb)

            v = rg.Vector3d(pb - pa)
            v *= scale
            disp.append(v)

            if "umag" in row:
                _umag.append(float(row["umag"]))

            if "von_mises" in row:
                vm.append(float(row["von_mises"]))

            # Three principal-stress families (new CSVs). Fall back to the single
            # px/py/pz/ps column set written by older exports.
            if "p1x" in row:
                ps1.append(_pvec(row, 1))
                ps2.append(_pvec(row, 2))
                ps3.append(_pvec(row, 3))
            elif "px" in row:
                s = float(row.get("ps", 1.0))
                ps1.append(rg.Vector3d(float(row["px"]) * s,
                                       float(row["py"]) * s,
                                       float(row["pz"]) * s))

# --- Summary log: max displacement + max stress, with units -----------------
max_u  = max(_umag) if _umag else 0.0
max_vm = max(vm) if vm else 0.0
log = (
    "points: %d\n"
    "max displacement |u| : %.4e m  (%.3f mm)\n"
    "max von Mises stress : %.4e Pa  (%.3f MPa)"
    % (len(A), max_u, max_u * 1e3, max_vm, max_vm / 1e6)
)
print(log)
