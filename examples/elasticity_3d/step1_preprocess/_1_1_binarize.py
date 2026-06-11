# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1.1 — Binarize a wall CSV point cloud onto a dense voxel grid.

``value`` in the CSV is NOT a continuous density.  It is a binary mask where
only ``value == 1.0`` is solid wall; every other value (0, 0.01, 0.1, 0.2,
0.5, ...) is a human-readable sub-category label that means VOID.  We therefore
binarise: solid = (value == 1.0), else empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_voxel_size(df: pd.DataFrame) -> float:
    """Median nearest-neighbour spacing along x (grid is isotropic)."""
    ux = np.unique(df["x"].to_numpy(np.float64))
    return float(np.median(np.diff(ux)))


def rasterize(
    df: pd.DataFrame,
    voxel_size: float,
    grid_size: int | tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Snap (x, y, z, value) points onto a dense (Nz, Ny, Nx) binary mask.

    Axis convention (matches DESIGN.md NCDHW): D=z (vertical), H=y, W=x.
    Solid iff ``value == 1.0``.

    Parameters
    ----------
    voxel_size : float
        Physical edge length of one voxel (same units as the CSV coords).
    grid_size : int | (Nz, Ny, Nx) | None
        If given, rasterise into a FIXED grid of this size, anchored at the
        plan's min corner (positive coordinate direction).  Any voxel whose
        index falls outside ``[0, N)`` on any axis is CROPPED (dropped).
        If ``None``, the grid is sized to the data extent (legacy behaviour).
    """
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    z = df["z"].to_numpy(np.float64)
    v = df["value"].to_numpy(np.float64)

    # Indices anchored at the plan's min corner -> origin (positive direction).
    ix = np.rint((x - x.min()) / voxel_size).astype(np.int64)
    iy = np.rint((y - y.min()) / voxel_size).astype(np.int64)
    iz = np.rint((z - z.min()) / voxel_size).astype(np.int64)

    if grid_size is None:
        Nz, Ny, Nx = iz.max() + 1, iy.max() + 1, ix.max() + 1
    elif isinstance(grid_size, int):
        Nz = Ny = Nx = grid_size
    else:
        Nz, Ny, Nx = grid_size

    rho = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    solid = np.isclose(v, 1.0)
    # Crop: keep only solid voxels that land inside the fixed box.
    inb = (
        (iz >= 0) & (iz < Nz)
        & (iy >= 0) & (iy < Ny)
        & (ix >= 0) & (ix < Nx)
    )
    keep = solid & inb
    rho[iz[keep], iy[keep], ix[keep]] = 1.0   # binary: solid iff value==1
    return rho
