# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# GhPython component script  --  NOT run from the command line.
# Principal-stress TRAJECTORY integrator: turns the discrete per-voxel principal
# directions in the CSV (from h5_to_csv.py) into continuous "stress line" curves
# by marching through the field.  Native Grasshopper can't streamline a sampled
# vector list, so this does it.
#
# Works in Rhino 7 (IronPython 2.7) and Rhino 8 (CPython 3).
#
# COMPONENT INPUTS  (right-click each input -> set type):
#   path     : str    file path to the .csv (must have p1x..p3 columns)
#   family   : int    which principal family to trace: 1 = tensile (default),
#                      3 = compressive, 2 = intermediate
#   vox      : float  voxel size in metres (default 0.1 ; must match the data)
#   step     : float  march step length (default = vox ; smaller = smoother)
#   max_steps: int    max steps per direction from a seed (default 300)
#   seed_skip: int    seed every Nth solid voxel (default 200 ; lower = denser)
#   min_stress: float only seed where |principal stress| >= this (Pa ; default 0)
#
# COMPONENT OUTPUTS:
#   lines : trajectory curves   (PolylineCurve list)
#   seeds : the seed points used (Point3d list)
#   log   : summary string (#voxels, #seeds, #curves)
#
# Tip: trace family 1 AND family 3 (two components, or run twice) and draw both
# -> the interwoven, 90-deg-crossing isostatic picture.  Colour family 1 red
# (tension) and family 3 blue (compression).
# ============================================================================

import csv
import math
import Rhino.Geometry as rg

# ---- inputs / defaults -----------------------------------------------------
if family is None:    family = 1
if vox is None:       vox = 0.1
if step is None or step <= 0.0:  step = vox
if max_steps is None: max_steps = 300
if seed_skip is None or seed_skip < 1:  seed_skip = 200
if min_stress is None: min_stress = 0.0
family = int(family)

dx, dy, dz = "p%dx" % family, "p%dy" % family, "p%dz" % family
dmag = "p%d" % family

# ---- build voxel lookup: (iz,iy,ix) -> (unit dir tuple, signed magnitude) ---
grid = {}
solid_pts = []   # (Point3d, key, |mag|) for seeding


def _key(X, Y, Z):
    ix = int(round(X / vox - 0.5))
    iy = int(round(Y / vox - 0.5))
    iz = int(round(Z / vox - 0.5))
    return (iz, iy, ix)


lines = []
seeds = []
n_vox = 0

if path:
    with open(path, "r") as fh:
        for row in csv.DictReader(fh):
            if dx not in row:
                continue   # CSV has no principal columns -> nothing to trace
            X = float(row["x"]); Y = float(row["y"]); Z = float(row["z"])
            ux = float(row[dx]); uy = float(row[dy]); uz = float(row[dz])
            mag = float(row.get(dmag, 0.0))
            k = _key(X, Y, Z)
            grid[k] = (ux, uy, uz)
            solid_pts.append((rg.Point3d(X, Y, Z), k, abs(mag)))
            n_vox += 1


def _dir_at(p):
    """Unit principal direction at point p, or None if outside the solid."""
    d = grid.get(_key(p.X, p.Y, p.Z))
    if d is None:
        return None
    v = rg.Vector3d(d[0], d[1], d[2])
    if not v.Unitize():
        return None
    return v


def _dot(a, b):
    return a.X * b.X + a.Y * b.Y + a.Z * b.Z


def _march(seed, sense):
    """March from seed; sense=+1 forward, -1 backward. Returns list of points."""
    pts = [seed]
    cur = seed
    prev = None
    for _ in range(max_steps):
        d = _dir_at(cur)
        if d is None:
            break
        if prev is None:
            if sense < 0:                       # start the backward branch
                d = rg.Vector3d(-d.X, -d.Y, -d.Z)
        else:                                   # keep the curve coherent across
            if _dot(d, prev) < 0.0:             # arbitrary eigenvector sign flips
                d = rg.Vector3d(-d.X, -d.Y, -d.Z)
        nxt = rg.Point3d(cur.X + d.X * step, cur.Y + d.Y * step, cur.Z + d.Z * step)
        pts.append(nxt)
        prev = d
        cur = nxt
    return pts


# ---- seed + integrate ------------------------------------------------------
n_seed = 0
for i in range(0, len(solid_pts), seed_skip):
    p, k, m = solid_pts[i]
    if m < min_stress:
        continue
    seeds.append(p)
    n_seed += 1

    fwd = _march(p, +1)
    bwd = _march(p, -1)
    # full curve = reversed backward branch + forward branch (drop shared seed)
    full = list(reversed(bwd))[:-1] + fwd
    if len(full) < 2:
        continue
    pl = rg.Polyline()
    for q in full:
        pl.Add(q)
    crv = pl.ToPolylineCurve()
    if crv:
        lines.append(crv)

log = ("family: p%d  (%s)\n"
       "solid voxels: %d\n"
       "seeds: %d\n"
       "trajectory curves: %d\n"
       "step=%.3g m  max_steps=%d  seed_skip=%d"
       % (family,
          {1: "most tensile", 2: "intermediate", 3: "most compressive"}.get(family, "?"),
          n_vox, n_seed, len(lines), step, max_steps, seed_skip))
print(log)
