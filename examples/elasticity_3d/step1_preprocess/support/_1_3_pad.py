# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1.3 — Pad each dim up to a multiple of ``2**model_depth``.

Per-wall padding so the U-Net can pool cleanly.  The base (D=0) plane is
preserved: padding is added at the high-index end of every axis only, so the
clamped base stays at slice 0.  (Uniform-cube padding across all samples is a
separate later step; see ``_1_4_pad_to_cube``.)
"""

from __future__ import annotations

import numpy as np


def pad_to_multiple(rho: np.ndarray, mult: int) -> np.ndarray:
    """Zero-pad each dim up to a multiple of ``mult`` (high-index end only)."""
    Nz, Ny, Nx = rho.shape
    pz = (-Nz) % mult
    py = (-Ny) % mult
    px = (-Nx) % mult
    return np.pad(rho, ((0, pz), (0, py), (0, px)), mode="constant")
