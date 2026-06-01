#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate synthetic HDF5 samples for pipeline testing.

Produces geometrically plausible (but physically approximate) samples that
exercise the full data → model → loss pipeline without requiring a real FEM
solver.

Usage::

    python generate_mock_hdf5.py --out_dir ./data/train --n_samples 20 --grid 32
    python generate_mock_hdf5.py --out_dir ./data/val   --n_samples 5  --grid 32

The generated displacement field is not a real FEM solution — it is a smooth
random field constructed to respect the clamped-base BC (u=0 at D=0).

HDF5 file layout
----------------
Each file ``sample_{i:05d}.h5`` contains:

    solid_mask    (D, H, W)    float32  — binary geometry (box with random cutouts)
    E_normalized  (D, H, W)    float32  — Young's modulus, z-score normalized
    body_force    (3, D, H, W) float32  — (fx, fy, fz) body force density
    wind_pressure (D, H, W)    float32  — scalar pressure at surface voxels
    displacement  (3, D, H, W) float32  — displacement (ux, uy, uz)

File-level attributes::

    nu        float  0.3  (Poisson's ratio, constant)
    bc        str    "z=0 plane fully clamped (Dirichlet u=0)"
    grid_size int    D (=H=W)
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def _make_solid_mask(
    rng: np.random.Generator,
    grid: int,
    void_fraction: float = 0.15,
) -> np.ndarray:
    """Generate a binary solid mask with a few spherical void cutouts."""
    mask = np.ones((grid, grid, grid), dtype=np.float32)
    n_voids = rng.integers(1, 5)
    for _ in range(n_voids):
        cx, cy, cz = rng.uniform(0.1 * grid, 0.9 * grid, size=3)
        r = rng.uniform(0.05 * grid, 0.15 * grid)
        dz, dy, dx = np.mgrid[0:grid, 0:grid, 0:grid]
        dist = np.sqrt((dz - cz) ** 2 + (dy - cy) ** 2 + (dx - cx) ** 2)
        mask[dist < r] = 0.0
    return mask


def _smooth_random_field(
    rng: np.random.Generator,
    grid: int,
    std: float = 1.0,
    smooth_passes: int = 3,
) -> np.ndarray:
    """Uniform random field blurred with a box filter."""
    field = rng.standard_normal((grid, grid, grid)).astype(np.float32) * std
    for _ in range(smooth_passes):
        from scipy.ndimage import uniform_filter  # optional dep at generation time
        field = uniform_filter(field, size=5, mode="nearest").astype(np.float32)
    return field


def _make_displacement(
    rng: np.random.Generator,
    grid: int,
    solid_mask: np.ndarray,
) -> np.ndarray:
    """Smooth displacement field satisfying the clamped-base BC (u=0 at D=0).

    Constructed as a smooth random field multiplied by a z-ramp that zeros
    the displacement at the base (D=0 slice).
    """
    try:
        from scipy.ndimage import gaussian_filter
        smooth = lambda f: gaussian_filter(f, sigma=3.0)
    except ImportError:
        smooth = lambda f: f

    # z-ramp: linearly increases from 0 at base (D=0) to 1 at top (D=grid-1)
    z_ramp = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    z_ramp = z_ramp[:, None, None]   # broadcast to (D, H, W)

    disp = np.zeros((3, grid, grid, grid), dtype=np.float32)
    for c in range(3):
        field = rng.standard_normal((grid, grid, grid)).astype(np.float32)
        field = smooth(field) * 1e-3
        field *= z_ramp * solid_mask
        disp[c] = field

    return disp


def generate_sample(
    rng: np.random.Generator,
    grid: int,
    E_mean: float = 2.1e11,
    E_std: float = 2.0e10,
) -> dict[str, np.ndarray]:
    """Generate all fields for one sample."""
    solid_mask = _make_solid_mask(rng, grid)

    # Young's modulus: spatially varying around steel (~210 GPa)
    E_raw = E_mean + E_std * rng.standard_normal((grid, grid, grid)).astype(np.float32)
    E_raw = np.clip(E_raw, E_mean * 0.5, E_mean * 1.5)
    E_normalized = ((E_raw - E_mean) / E_std).astype(np.float32)

    # Body force: gravity-dominated (-z), small random lateral component
    body_force = np.zeros((3, grid, grid, grid), dtype=np.float32)
    body_force[2] = -9.81 * 7800.0 * solid_mask  # ρg in z-direction (steel)
    for c in range(2):
        body_force[c] = rng.standard_normal((grid, grid, grid)).astype(np.float32) \
                        * 500.0 * solid_mask

    # Wind pressure: non-zero only at exterior (surface) voxels of the solid.
    # Simple approximation: exterior = solid voxel adjacent to at least one void.
    from scipy.ndimage import binary_dilation  # type: ignore
    is_solid  = solid_mask > 0.5
    dilated   = binary_dilation(is_solid, iterations=1)
    surface   = dilated & ~is_solid  # void voxels adjacent to solid
    # Assign a random positive pressure on the surface voxels.
    pressure_magnitude = abs(rng.standard_normal()) * 1000.0 + 500.0  # Pa
    wind_pressure = (surface.astype(np.float32) * pressure_magnitude)

    displacement = _make_displacement(rng, grid, solid_mask)

    return {
        "solid_mask":    solid_mask,
        "E_normalized":  E_normalized,
        "body_force":    body_force,
        "wind_pressure": wind_pressure,
        "displacement":  displacement,
    }


def write_sample(path: Path, data: dict[str, np.ndarray], grid: int) -> None:
    """Write one sample to an HDF5 file."""
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, dtype="float32", compression="gzip")
        f.attrs["nu"]        = 0.3
        f.attrs["bc"]        = "z=0 plane fully clamped (Dirichlet u=0)"
        f.attrs["grid_size"] = grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HDF5 samples for elasticity surrogate testing."
    )
    parser.add_argument("--out_dir",   type=str, default="./data/train",
                        help="Output directory for .h5 files.")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--grid",      type=int, default=32,
                        help="Voxel grid size per side (isotropic).")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Base random seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.n_samples} samples at {args.grid}^3 → {out_dir}")

    for i in range(args.n_samples):
        data = generate_sample(rng, args.grid)
        path = out_dir / f"sample_{i:05d}.h5"
        write_sample(path, data, args.grid)
        if (i + 1) % max(1, args.n_samples // 10) == 0:
            print(f"  Written {i + 1}/{args.n_samples}")

    print("Done.")


if __name__ == "__main__":
    main()
