# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract FEM solver interface and a mock implementation for testing.

To integrate a real FEM backend, subclass FEMSolverInterface and implement
``solve()`` and ``is_available()``.  See the TODO comments below for FEniCSx
and FEBio integration hooks.
"""

import abc
from typing import Any

import numpy as np


class FEMSolverInterface(abc.ABC):
    """Abstract interface for a voxel-based Finite Element solver.

    Implementations wrap an external FEM backend (FEniCSx, FEBio, OpenSees,
    etc.) that accepts a voxelized geometry description and returns the
    full displacement field u(x).

    All arrays use CDHW layout (channels first) matching the model tensor
    convention, except for scalar fields which are plain (D, H, W).

    Expected ``input_dict`` keys
    ----------------------------
    solid_mask : ndarray, shape (D, H, W), uint8 or float32
        Binary voxel occupancy: 1 = solid, 0 = void.
    E : ndarray, shape (D, H, W), float32
        Young's modulus in Pa (physical, not normalized).
    nu : float
        Poisson's ratio (constant 0.3 in the current problem setup).
    body_force : ndarray, shape (3, D, H, W), float32
        Body force density vector (fx, fy, fz) in N/m³.
    wind_pressure : ndarray, shape (D, H, W), float32
        Scalar wind pressure magnitude in Pa at exterior surface voxels.
        The traction direction is outward-normal (inferred from geometry).
    voxel_size : float
        Physical edge length of one voxel in metres.

    Returned ``output_dict`` keys
    -----------------------------
    displacement : ndarray, shape (3, D, H, W), float32
        Displacement field (ux, uy, uz) in metres.
    """

    @abc.abstractmethod
    def solve(self, input_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run the FEM solver and return the displacement field."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the solver backend is importable and reachable."""
        ...


class MockFEMSolver(FEMSolverInterface):
    """Mock FEM solver for CI / unit-testing.

    Returns a displacement field of zeros plus optional Gaussian noise.
    No FEM installation required.

    Parameters
    ----------
    noise_std : float
        Standard deviation of additive Gaussian noise (default 1e-4 m).
    seed : int
        Random seed for reproducibility (default 42).

    Future integration hooks
    ------------------------
    FEniCSx::

        # TODO: Replace MockFEMSolver with FEniCSxSolver
        # from fenics import *
        # from dolfinx.mesh import create_box
        # mesh = create_box(MPI.COMM_WORLD, ...)
        # V = VectorFunctionSpace(mesh, ("Lagrange", 1))
        # u = Function(V)
        # ...

    FEBio (REST API or subprocess)::

        # TODO: FEBioSolver subclass
        # import subprocess, json, tempfile
        # subprocess.run(["febio4", "-i", feb_file, "-o", log_file], ...)
        # parse xplt output -> displacement field
    """

    def __init__(self, noise_std: float = 1e-4, seed: int = 42) -> None:
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def solve(self, input_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        solid_mask: np.ndarray = np.asarray(input_dict["solid_mask"])
        D, H, W = solid_mask.shape
        displacement = self._rng.normal(
            loc=0.0, scale=self.noise_std, size=(3, D, H, W)
        ).astype(np.float32)
        # Enforce clamped BC: zero displacement at z=0 plane (D=0 slice).
        displacement[:, 0, :, :] = 0.0
        return {"displacement": displacement}

    def is_available(self) -> bool:
        return True
