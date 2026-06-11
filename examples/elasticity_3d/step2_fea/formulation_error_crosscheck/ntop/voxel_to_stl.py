# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a plan's voxel field to a watertight STL mesh for nTop (or any FEM).

Builds the BLOCKY surface mesh of the solid voxels -- every exposed voxel face
becomes two triangles -- so the geometry is exactly what our VoxelFEMSolver
used. nTop can import this STL, generate a volume (tet) mesh, and run FEA, giving
an independent cross-check of our solver on the SAME geometry.

Source of the voxel field:
  * --h5 path        : read the `solid_mask` from a generated sample .h5, OR
  * --plan N         : rasterize plan_N.csv from --csv_dir (same as generation).

Units: STL is unitless; we write in MILLIMETRES by default (voxel 0.1 m -> 100
mm) so it lands at the right scale in a mm nTop/Rhino document. Use --units m for
metres.

Usage (run from the examples/elasticity_3d root)
-----
    python step2_fea/formulation_error_crosscheck/ntop/voxel_to_stl.py \
        --h5 data_wood/val/sample_00000.h5
    python step2_fea/formulation_error_crosscheck/ntop/voxel_to_stl.py \
        --plan 0 --out plan0.stl --units m
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# face dir -> (outward normal, 4 corner offsets (dx,dy,dz) added to (ix,iy,iz),
# CCW seen from outside so the triangle normal points outward).
FACES = {
    "x+": ((1, 0, 0),  [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    "x-": ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    "y+": ((0, 1, 0),  [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    "y-": ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    "z+": ((0, 0, 1),  [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    "z-": ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
}


def _exposed(is_solid, face):
    """Boolean grid: voxels whose `face` is exposed (neighbour void/outside)."""
    nb = np.zeros_like(is_solid)
    if face == "x+":   nb[:, :, :-1] = is_solid[:, :, 1:]
    elif face == "x-": nb[:, :, 1:]  = is_solid[:, :, :-1]
    elif face == "y+": nb[:, :-1, :] = is_solid[:, 1:, :]
    elif face == "y-": nb[:, 1:, :]  = is_solid[:, :-1, :]
    elif face == "z+": nb[:-1, :, :] = is_solid[1:, :, :]
    elif face == "z-": nb[1:, :, :]  = is_solid[:-1, :, :]
    return is_solid & ~nb


def voxel_surface_triangles(is_solid, s):
    """Return (T,3,3) vertices and (T,3) normals for the blocky surface, scale s."""
    verts, norms = [], []
    for face, (normal, offs) in FACES.items():
        idx = np.argwhere(_exposed(is_solid, face))     # (E,3) = (iz,iy,ix)
        if idx.size == 0:
            continue
        iz, iy, ix = idx[:, 0], idx[:, 1], idx[:, 2]
        # 4 corners in world coords (X=ix, Y=iy, Z=iz) * s
        c = []
        for (dx, dy, dz) in offs:
            c.append(np.stack([(ix + dx) * s, (iy + dy) * s, (iz + dz) * s], axis=1))
        # two triangles per face: (c0,c1,c2) and (c0,c2,c3)
        for (a, b, cc) in [(0, 1, 2), (0, 2, 3)]:
            tri = np.stack([c[a], c[b], c[cc]], axis=1)  # (E,3,3)
            verts.append(tri)
            norms.append(np.tile(np.array(normal, dtype=np.float32), (len(idx), 1)))
    if not verts:
        return np.zeros((0, 3, 3), np.float32), np.zeros((0, 3), np.float32)
    return (np.concatenate(verts, axis=0).astype(np.float32),
            np.concatenate(norms, axis=0).astype(np.float32))


def write_binary_stl(path, verts, norms):
    dt = np.dtype([("n", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3),
                   ("v3", "<f4", 3), ("attr", "<u2")])
    rec = np.zeros(len(verts), dtype=dt)
    rec["n"]  = norms
    rec["v1"] = verts[:, 0]; rec["v2"] = verts[:, 1]; rec["v3"] = verts[:, 2]
    with open(path, "wb") as f:
        f.write(b"voxel_to_stl blocky mesh".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(verts)))
        f.write(rec.tobytes())


def load_solid_mask(args):
    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as f:
            mask = f["solid_mask"][:]
            vs = float(f.attrs.get("voxel_size", args.voxel_size))
        return mask, vs
    # else rasterize a plan CSV
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # examples/elasticity_3d
    from step1_preprocess._1_1_binarize import rasterize
    fp = Path(args.csv_dir) / f"plan_{args.plan}.csv"
    df = pd.read_csv(fp, usecols=["x", "y", "z", "value"])
    grid = (args.grid_z, args.grid_size, args.grid_size)
    return rasterize(df, args.voxel_size, grid_size=grid), args.voxel_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=None, help="Sample .h5 (uses solid_mask).")
    ap.add_argument("--plan", type=int, default=None, help="Rasterize plan_N.csv instead.")
    ap.add_argument("--csv_dir", default=r"D:\Summer2026_ResPlan\ResPlan\fields_csv\3D_field\0-1")
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--grid_z", type=int, default=48)
    ap.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size in metres.")
    ap.add_argument("--units", choices=["mm", "m"], default="mm", help="STL output units.")
    ap.add_argument("--out", default=None, help="Output STL path.")
    args = ap.parse_args()

    if not args.h5 and args.plan is None:
        raise SystemExit("Give --h5 <file> or --plan <N>.")

    mask, vs = load_solid_mask(args)
    is_solid = mask > 0.05
    n_solid = int(is_solid.sum())
    scale = vs * (1000.0 if args.units == "mm" else 1.0)   # world units per voxel

    verts, norms = voxel_surface_triangles(is_solid, scale)
    out = args.out or (f"plan{args.plan}.stl" if args.plan is not None
                       else str(Path(args.h5).with_suffix(".stl")))
    write_binary_stl(out, verts, norms)

    box = tuple(d * scale for d in is_solid.shape)         # (Z,Y,X) world extent
    print(f"solid voxels: {n_solid}  grid(D,H,W)={is_solid.shape}  voxel={vs} m")
    print(f"wrote {len(verts)} triangles -> {out}  [{args.units}]")
    print(f"bounding box ({args.units}): X<= {box[2]:.0f}  Y<= {box[1]:.0f}  Z<= {box[0]:.0f}")


if __name__ == "__main__":
    main()
