# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1.2 — Block-average downsample the density grid.

Coarsens the rasterised grid by an integer factor to keep the CPU FEM solve
and training tractable.  NOTE: averaging produces fractional edge voxels, so
the result is no longer strictly {0, 1} — downstream code that needs a true
binary mask must threshold (see step3 ``masked_mse``).
"""

from __future__ import annotations

import numpy as np


def block_downsample(rho: np.ndarray, f: int) -> np.ndarray:
    """Average-pool the density field by an integer factor (crops remainder)."""
    if f <= 1:
        return rho
    Nz, Ny, Nx = rho.shape
    Nz, Ny, Nx = (Nz // f) * f, (Ny // f) * f, (Nx // f) * f
    cropped = rho[:Nz, :Ny, :Nx]
    return cropped.reshape(Nz // f, f, Ny // f, f, Nx // f, f).mean(axis=(1, 3, 5))
