# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoxelFEMSolver: matrix-free Q1 hex FEM solver for 3D linear elasticity.

Discretisation
--------------
- Grid of (Nz × Ny × Nx) voxel *elements* on a regular Cartesian mesh.
- Nodes at the 8 corners of each voxel: (Nz+1) × (Ny+1) × (Nx+1) nodes.
- Each node has 3 DOFs: (ux, uy, uz).
- Element type: trilinear hexahedron (Q1/H8), 2×2×2 Gauss quadrature.

Material
--------
SIMP-penalised Young's modulus per element:

    E_e = Emin + rho_e^p × (E0 - Emin)

where rho_e ∈ [0, 1] is the element density (one value per voxel),
E0 may vary per element, Emin avoids a singular stiffness matrix, and
p is the penalisation exponent (typically 3).  Poisson ratio ν is fixed.

Solver
------
Matrix-free Conjugate Gradient on the reduced (free DOF) system.
No global stiffness matrix is assembled; instead K·v is evaluated as:

    Kv[global] += Σ_e  E_e × (K0 × v_local_e)

using numpy gather / np.bincount scatter.  This uses O(n_elem × 24)
memory rather than O(n_dof × bandwidth) for the matrix.

Preconditioning
---------------
- Jacobi (diagonal scaling): always available.
- AMG (smoothed aggregation): used if ``pyamg`` is installed AND the
  ``precond='amg'`` option is selected.  AMG requires assembling K for the
  free DOFs, which uses additional memory (~1 GB at 64³).

Coordinate convention
---------------------
  Tensor index   Physical coordinate   Spatial meaning
  z (dim 0)  →  iz × h               vertical (up direction)
  y (dim 1)  →  iy × h               depth
  x (dim 2)  →  ix × h               width

The clamped base corresponds to the iz = 0 (D = 0) plane.

API
---
>>> solver = VoxelFEMSolver(grid_size=(Nz, Ny, Nx))
>>> solver.assemble_system(rho, E0)
>>> solver.apply_bcs()                   # clamp z=0 plane (default)
>>> solver.apply_loads(body_force=f)
>>> u_node, u_elem, info = solver.solve()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._2_1_hex_stiffness import (
    FACE_LOCAL_NODES,
    FACE_NORMALS,
    compute_B_at_center,
    compute_unit_stiffness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoxelFEMSolver
# ---------------------------------------------------------------------------

class VoxelFEMSolver:
    """Matrix-free Q1 hexahedral FEM solver for 3D linear elasticity.

    Parameters
    ----------
    grid_size : tuple of int
        (Nz, Ny, Nx) number of *elements* (voxels) per spatial dimension.
    voxel_size : float
        Physical edge length of one voxel in metres (default 1.0).
    nu : float
        Poisson's ratio, fixed across all samples (default 0.3).
    dtype : numpy dtype
        Floating-point precision for the solve (default float64).
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        voxel_size: float = 1.0,
        nu: float = 0.3,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.Nz, self.Ny, self.Nx = grid_size
        self.h    = float(voxel_size)
        self.nu   = float(nu)
        self.dtype = np.dtype(dtype)

        # Node counts per dimension
        self.nz = self.Nz + 1
        self.ny = self.Ny + 1
        self.nx = self.Nx + 1

        self.n_nodes = self.nz * self.ny * self.nx
        self.n_dof   = 3 * self.n_nodes
        self.n_elem  = self.Nz * self.Ny * self.Nx

        # Precompute element stiffness for E = 1
        self.K0: np.ndarray = compute_unit_stiffness(nu=nu, h=voxel_size)  # (24, 24)
        self.K0_diag: np.ndarray = np.diag(self.K0)                         # (24,)

        # B matrix at element centre for post-processing
        self._B_centre: np.ndarray = compute_B_at_center(h=voxel_size)     # (6, 24)

        # Element-to-global-DOF connectivity (built once, reused for all solves)
        self.elem_dofs: np.ndarray = self._build_elem_dofs()   # (n_elem, 24) int32

        # State populated by assemble_system / apply_bcs / apply_loads
        self.E_elem: np.ndarray | None = None        # (n_elem,)
        self.free_dofs: np.ndarray | None = None     # (n_free,) int
        self.constrained_dofs: np.ndarray | None = None
        self.f: np.ndarray | None = None             # (n_dof,)

        logger.info(
            "VoxelFEMSolver: grid %s, nodes=%d, DOFs=%d, elements=%d",
            grid_size, self.n_nodes, self.n_dof, self.n_elem,
        )

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def _build_elem_dofs(self) -> np.ndarray:
        """Build the (n_elem, 24) global-DOF index table.

        For element (ez, ey, ex), the 8 corner nodes follow the ordering in
        ``hex_stiffness.py`` and the DOFs are interleaved: [u_Ix, u_Iy, u_Iz].
        """
        # Element multi-indices
        ez, ey, ex = np.meshgrid(
            np.arange(self.Nz),
            np.arange(self.Ny),
            np.arange(self.Nx),
            indexing="ij",
        )
        ez = ez.ravel().astype(np.int32)
        ey = ey.ravel().astype(np.int32)
        ex = ex.ravel().astype(np.int32)

        # Global node index: (iz, iy, ix) → iz*ny*nx + iy*nx + ix
        ny, nx = self.ny, self.nx

        def _node(diz: int, diy: int, dix: int) -> np.ndarray:
            return (ez + diz) * ny * nx + (ey + diy) * nx + (ex + dix)

        # 8 corner nodes per element (local nodes 0-7, see hex_stiffness.py)
        offsets = [(0,0,0),(0,0,1),(0,1,1),(0,1,0),
                   (1,0,0),(1,0,1),(1,1,1),(1,1,0)]
        elem_nodes = np.stack([_node(*d) for d in offsets], axis=1)  # (n_elem, 8)

        # Expand to DOFs: node_idx * 3 + {0, 1, 2}
        elem_dofs = (
            elem_nodes[:, :, np.newaxis] * 3
            + np.array([0, 1, 2], dtype=np.int32)
        ).reshape(self.n_elem, 24)                                     # (n_elem, 24)

        return elem_dofs.astype(np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble_system(
        self,
        rho: np.ndarray,
        E0: float | np.ndarray,
        Emin: float = 1e-9,
        p_simp: float = 3.0,
    ) -> "VoxelFEMSolver":
        """Compute element Young's moduli via SIMP penalisation.

        E_e = Emin + rho_e^p × (E0 - Emin)

        Parameters
        ----------
        rho : ndarray, shape (Nz, Ny, Nx)
            Element density field, values in [0, 1].
        E0 : float or ndarray, shape (Nz, Ny, Nx)
            Young's modulus of the fully solid material (Pa).
            May vary per element for heterogeneous material.
        Emin : float
            Residual stiffness for void elements (avoids singular K).
            Defaults to 1e-9 × E0 effectively.
        p_simp : float
            SIMP penalisation exponent (default 3).

        Returns
        -------
        self (for method chaining)
        """
        rho_flat = np.asarray(rho, dtype=np.float64).ravel()
        if rho_flat.size != self.n_elem:
            raise ValueError(
                f"rho has {rho_flat.size} elements; expected {self.n_elem}."
            )

        E0_flat = (
            np.full(self.n_elem, float(E0), dtype=np.float64)
            if np.isscalar(E0)
            else np.asarray(E0, dtype=np.float64).ravel()
        )
        if E0_flat.size != self.n_elem:
            raise ValueError(
                f"E0 has {E0_flat.size} elements; expected {self.n_elem}."
            )

        self.E_elem = (Emin + np.clip(rho_flat, 0.0, 1.0) ** p_simp
                       * (E0_flat - Emin)).astype(self.dtype)
        logger.debug(
            "assemble_system: E range [%.3e, %.3e]",
            self.E_elem.min(), self.E_elem.max(),
        )
        return self

    def apply_bcs(
        self,
        fixed_nodes: np.ndarray | None = None,
    ) -> "VoxelFEMSolver":
        """Identify constrained (zero-displacement) degrees of freedom.

        Parameters
        ----------
        fixed_nodes : ndarray of int, optional
            Global node indices to clamp.  If None (default), all nodes on
            the iz = 0 plane are clamped (fully clamped base: ux = uy = uz = 0).

        Returns
        -------
        self
        """
        if fixed_nodes is None:
            # Base plane: iz = 0  →  node(0, iy, ix) = iy*nx + ix
            iy_grid, ix_grid = np.meshgrid(
                np.arange(self.ny), np.arange(self.nx), indexing="ij"
            )
            fixed_nodes = (iy_grid * self.nx + ix_grid).ravel()

        # Each fixed node contributes 3 constrained DOFs (ux, uy, uz)
        fixed_nodes = np.asarray(fixed_nodes, dtype=np.int64)
        constrained = (fixed_nodes[:, np.newaxis] * 3
                       + np.array([0, 1, 2])).ravel()

        all_dofs = np.arange(self.n_dof, dtype=np.int64)
        self.constrained_dofs = constrained
        self.free_dofs = np.setdiff1d(all_dofs, constrained)
        logger.debug(
            "apply_bcs: %d constrained DOFs, %d free DOFs",
            len(self.constrained_dofs), len(self.free_dofs),
        )
        return self

    def apply_loads(
        self,
        body_force: np.ndarray | None = None,
        point_loads: list[tuple[int | np.ndarray, np.ndarray]] | None = None,
        wind_pressure: np.ndarray | None = None,
        rho: np.ndarray | None = None,
        rho_threshold: float = 0.5,
    ) -> "VoxelFEMSolver":
        """Assemble the global external force vector.

        Three load types are supported and may be combined:

        1. **Body force**: volumetric body force density f(x) in N/m³.
           Applied with lumped integration: each of the 8 element nodes
           receives f_e × h³ / 8 in each direction.

        2. **Point loads**: discrete forces applied directly to nodes.

        3. **Wind pressure**: scalar surface pressure magnitude (Pa) per
           element.  The traction direction is the outward face normal,
           detected from the solid/void interface (element density vs
           ``rho_threshold``).  Lumped: each of the 4 face nodes receives
           p_e × h² / 4 in the normal direction.  Requires ``rho`` to be
           provided for surface detection.

        Parameters
        ----------
        body_force : ndarray, shape (3, Nz, Ny, Nx) or None
            Body force density (fx, fy, fz) in physical units.
        point_loads : list of (node_idx, force_vector) or None
            ``node_idx`` is the scalar global node index; ``force_vector``
            is a length-3 array [Fx, Fy, Fz].
        wind_pressure : ndarray, shape (Nz, Ny, Nx) or None
            Scalar pressure field (Pa).  Non-zero only at surface voxels.
            Requires ``rho`` for surface detection.
        rho : ndarray, shape (Nz, Ny, Nx) or None
            Density field used for surface detection (required if
            ``wind_pressure`` is not None).
        rho_threshold : float
            Density threshold separating solid (> threshold) from void.

        Returns
        -------
        self
        """
        self.f = np.zeros(self.n_dof, dtype=self.dtype)
        h3_over8 = self.h ** 3 / 8.0
        h2_over4 = self.h ** 2 / 4.0

        # -- 1. Body force ---------------------------------------------------
        if body_force is not None:
            bf = np.asarray(body_force, dtype=np.float64)  # (3, Nz, Ny, Nx)
            if bf.shape != (3, self.Nz, self.Ny, self.Nx):
                raise ValueError(
                    f"body_force shape {bf.shape} != (3, {self.Nz}, {self.Ny}, {self.Nx})"
                )
            for dir_idx in range(3):  # x=0, y=1, z=2
                # Nodal force from each element: h³/8 × f_e (lumped)
                f_elem = (bf[dir_idx].ravel() * h3_over8).astype(self.dtype)
                # Scatter to all 8 element nodes (DOF = node*3 + dir_idx)
                dofs = self.elem_dofs[:, dir_idx::3].ravel()   # (n_elem*8,) DOFs for dir
                vals = np.repeat(f_elem, 8)                     # replicate per node
                self.f += np.bincount(dofs, weights=vals, minlength=self.n_dof)

        # -- 2. Point loads --------------------------------------------------
        if point_loads is not None:
            for node_idx, force in point_loads:
                node_idx = int(node_idx)
                force = np.asarray(force, dtype=self.dtype)
                for d in range(3):
                    self.f[3 * node_idx + d] += force[d]

        # -- 3. Wind pressure (surface-normal traction) ----------------------
        if wind_pressure is not None:
            if rho is None:
                raise ValueError(
                    "``rho`` must be provided for wind_pressure surface detection."
                )
            p_field  = np.asarray(wind_pressure, dtype=np.float64).reshape(self.Nz, self.Ny, self.Nx)
            rho_grid = np.asarray(rho, dtype=np.float64).reshape(self.Nz, self.Ny, self.Nx)
            is_solid = rho_grid > rho_threshold

            # Neighbour masks along each axis (False = void / outside domain)
            padded_zm = np.pad(is_solid, ((1,0),(0,0),(0,0)), constant_values=False)
            padded_zp = np.pad(is_solid, ((0,1),(0,0),(0,0)), constant_values=False)
            padded_ym = np.pad(is_solid, ((0,0),(1,0),(0,0)), constant_values=False)
            padded_yp = np.pad(is_solid, ((0,0),(0,1),(0,0)), constant_values=False)
            padded_xm = np.pad(is_solid, ((0,0),(0,0),(1,0)), constant_values=False)
            padded_xp = np.pad(is_solid, ((0,0),(0,0),(0,1)), constant_values=False)

            exposed = [
                is_solid & ~padded_xm[:, :, :-1],   # x- face
                is_solid & ~padded_xp[:, :,  1:],   # x+ face
                is_solid & ~padded_ym[:, :-1, :],   # y- face
                is_solid & ~padded_yp[:, 1:,  :],   # y+ face
                is_solid & ~padded_zm[:-1, :, :],   # z- face (base)
                is_solid & ~padded_zp[ 1:, :, :],   # z+ face
            ]

            elem_idx_grid = np.arange(self.n_elem, dtype=np.int32).reshape(
                self.Nz, self.Ny, self.Nx
            )

            for face_i, (exp_mask, face_nodes, normal) in enumerate(
                zip(exposed, FACE_LOCAL_NODES, FACE_NORMALS)
            ):
                exp_elems = elem_idx_grid[exp_mask]   # indices of exposed elements
                if exp_elems.size == 0:
                    continue
                p_face = p_field.ravel()[exp_elems] * h2_over4  # per-node force magnitude

                for local_node in face_nodes:
                    for dir_idx in range(3):
                        n_comp = normal[dir_idx]
                        if n_comp == 0.0:
                            continue
                        dofs = self.elem_dofs[exp_elems, 3 * local_node + dir_idx]
                        vals = n_comp * p_face
                        self.f += np.bincount(
                            dofs, weights=vals.astype(self.dtype),
                            minlength=self.n_dof,
                        )

        return self

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        tol: float = 1e-8,
        maxiter: int = 3000,
        precond: str = "jacobi",
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Solve the static equilibrium system K u = f using CG.

        Requires prior calls to :meth:`assemble_system`, :meth:`apply_bcs`,
        and :meth:`apply_loads`.

        Parameters
        ----------
        tol : float
            Relative residual tolerance for CG (default 1e-8).
        maxiter : int
            Maximum CG iterations (default 3000).
        precond : str
            Preconditioner: ``"jacobi"`` (default) or ``"amg"``
            (requires ``pyamg``; triggers sparse assembly).
        verbose : bool
            Log progress.

        Returns
        -------
        u_nodal : ndarray, shape (3, nz, ny, nx)
            Displacement at mesh nodes.
        u_elem : ndarray, shape (3, Nz, Ny, Nx)
            Displacement averaged to element centres (mean of 8 corner nodes).
        info : dict
            ``"iters"``: CG iterations,
            ``"residual"``: final relative residual,
            ``"converged"``: bool,
            ``"solve_time_s"``: wall-clock time.
        """
        if self.E_elem is None:
            raise RuntimeError("Call assemble_system() before solve().")
        if self.free_dofs is None:
            raise RuntimeError("Call apply_bcs() before solve().")
        if self.f is None:
            raise RuntimeError("Call apply_loads() before solve().")

        # Direct solver on the ACTIVE (solid-only) DOFs -- bypasses CG entirely.
        if precond == "direct":
            return self._solve_direct(verbose=verbose)

        n_free = len(self.free_dofs)
        f_free = self.f[self.free_dofs]

        # Linear operator wrapping the matrix-free matvec
        K_op = spla.LinearOperator(
            shape=(n_free, n_free),
            matvec=self._matvec_free,
            dtype=self.dtype,
        )

        # Preconditioner (timed separately: AMG setup can dominate at large
        # grids and is NOT part of the CG-iteration time below).
        _tp = time.perf_counter()
        M = self._build_preconditioner(precond, n_free, f_free)
        precond_setup_s = time.perf_counter() - _tp

        if verbose:
            logger.info(
                "CG: n_free=%d, tol=%.1e, maxiter=%d, precond=%s",
                n_free, tol, maxiter, precond,
            )

        # Iteration counter callback
        iteration_count = [0]

        def _callback(xk):
            iteration_count[0] += 1
            if verbose and iteration_count[0] % 100 == 0:
                res = np.linalg.norm(self._matvec_free(xk) - f_free)
                logger.info("  CG iter %d  |r| = %.3e", iteration_count[0], res)

        t0 = time.perf_counter()
        u_free, info_code = spla.cg(
            K_op, f_free, M=M, rtol=tol, maxiter=maxiter, callback=_callback
        )
        dt = time.perf_counter() - t0

        # Full displacement vector (free DOFs filled; constrained = 0)
        u_full = np.zeros(self.n_dof, dtype=self.dtype)
        u_full[self.free_dofs] = u_free

        # Relative residual
        res_full = self._matvec(u_full) - self.f
        rel_res = (np.linalg.norm(res_full[self.free_dofs])
                   / (np.linalg.norm(f_free) + 1e-300))

        converged = info_code == 0 and rel_res < tol * 10  # allow small overshoot
        info = {
            "iters":          iteration_count[0],
            "residual":       float(rel_res),
            "converged":      converged,
            "solve_time_s":   dt,
            "precond_setup_s": precond_setup_s,
            "cg_solve_s":     dt,
        }
        if verbose or not converged:
            logger.info(
                "CG %s: iters=%d  |r|/|f|=%.2e  t=%.1fs",
                "converged" if converged else "NOT CONVERGED",
                iteration_count[0], rel_res, dt,
            )

        # Reshape to nodal field: (3, nz, ny, nx)
        u_nodal = u_full.reshape(self.n_nodes, 3).T.reshape(
            3, self.nz, self.ny, self.nx
        )

        # Element-centre displacement: average of 8 corner nodes
        u_elem = self._element_center_displacement(u_nodal)

        return u_nodal, u_elem, info

    # ------------------------------------------------------------------
    # Matrix-free matvec
    # ------------------------------------------------------------------

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute K · v for the full DOF vector v (including constrained)."""
        # Gather: extract local displacement for all elements simultaneously
        v_local = v[self.elem_dofs]                               # (n_elem, 24)
        # Apply element stiffness: f_local_e = E_e × K0 × v_local_e
        f_local = self.E_elem[:, np.newaxis] * (v_local @ self.K0.T)  # (n_elem, 24)
        # Scatter: accumulate to global force vector
        result = np.bincount(
            self.elem_dofs.ravel(),
            weights=f_local.ravel().astype(np.float64),
            minlength=self.n_dof,
        ).astype(self.dtype)
        return result

    def _matvec_free(self, v_free: np.ndarray) -> np.ndarray:
        """Compute K_free · v_free (reduced system, constrained DOFs = 0)."""
        v_full = np.zeros(self.n_dof, dtype=self.dtype)
        v_full[self.free_dofs] = v_free
        return self._matvec(v_full)[self.free_dofs]

    # ------------------------------------------------------------------
    # Preconditioner
    # ------------------------------------------------------------------

    def _build_preconditioner(
        self, precond: str, n_free: int, f_free: np.ndarray
    ) -> spla.LinearOperator | None:
        """Build the requested preconditioner for the free-DOF system."""
        if precond == "jacobi":
            return self._jacobi_preconditioner(n_free)
        if precond == "amg":
            return self._amg_preconditioner(n_free, f_free)
        if precond == "none":
            return None
        raise ValueError(f"Unknown preconditioner: {precond!r}. Choose 'jacobi', 'amg', or 'none'.")

    def _jacobi_preconditioner(self, n_free: int) -> spla.LinearOperator:
        """Diagonal (Jacobi) preconditioner: M⁻¹ = diag(K)⁻¹."""
        # Diagonal of K assembled via bincount (same pattern as matvec)
        diag_vals = (self.E_elem[:, np.newaxis] * self.K0_diag[np.newaxis, :]).ravel()
        diag_K = np.bincount(
            self.elem_dofs.ravel(),
            weights=diag_vals.astype(np.float64),
            minlength=self.n_dof,
        ).astype(self.dtype)
        # Restrict to free DOFs and invert
        diag_free = diag_K[self.free_dofs]
        inv_diag  = 1.0 / np.maximum(np.abs(diag_free), 1e-300)
        return spla.LinearOperator(
            shape=(n_free, n_free),
            matvec=lambda v: inv_diag * v,
            dtype=self.dtype,
        )

    def _amg_preconditioner(
        self, n_free: int, f_free: np.ndarray
    ) -> spla.LinearOperator | None:
        """AMG preconditioner using pyamg (falls back to Jacobi if unavailable).

        Uses smoothed-aggregation AMG with the 6 rigid-body modes (3 translations
        + 3 rotations) as the near-null space.  Supplying these modes is critical
        for elasticity: without them SA-AMG treats the problem as scalar and the
        convergence rate degrades by 3-5x on large grids.
        """
        try:
            import pyamg  # type: ignore[import]
        except ImportError:
            logger.warning(
                "pyamg not installed - falling back to Jacobi preconditioner."
            )
            return self._jacobi_preconditioner(n_free)

        logger.info("Building AMG preconditioner (sparse assembly + near-null space)")
        K_sparse = self._assemble_sparse_free()
        B = self._build_near_nullspace()
        # Jacobi prolongation smoother: O(n) setup cost, same convergence quality
        # as energy-minimising smoother when the near-null space B is supplied.
        # 'energy' smoother is 10-30x more expensive to build with no benefit here.
        ml = pyamg.smoothed_aggregation_solver(
            K_sparse,
            B=B,                               # (n_free, 6) rigid body modes
            smooth=("jacobi", {"omega": 4.0 / 3.0}),
            strength="symmetric",
            max_coarse=500,
        )
        logger.info(
            "AMG hierarchy: %d levels, coarsest=%d DOFs",
            len(ml.levels), ml.levels[-1].A.shape[0],
        )
        return ml.aspreconditioner()

    def _build_near_nullspace(self) -> np.ndarray:
        """Return the 6 rigid-body modes restricted to free DOFs.

        Rigid body modes for 3D linear elasticity (DOF layout ux,uy,uz per node):
          mode 0  — translation x  : ux=1
          mode 1  — translation y  : uy=1
          mode 2  — translation z  : uz=1
          mode 3  — rotation about z: ux=-y, uy=x
          mode 4  — rotation about x: uy=-z, uz=y
          mode 5  — rotation about y: ux=z,  uz=-x

        Coordinates are centred at the domain mid-point for better conditioning.

        Returns
        -------
        B : ndarray, shape (n_free, 6), float64
        """
        h = self.h
        # Physical node coordinates, centred
        iz, iy, ix = np.mgrid[0:self.nz, 0:self.ny, 0:self.nx]
        x = (ix.ravel() - (self.nx - 1) / 2.0) * h   # (n_nodes,)
        y = (iy.ravel() - (self.ny - 1) / 2.0) * h
        z = (iz.ravel() - (self.nz - 1) / 2.0) * h

        B = np.zeros((self.n_dof, 6), dtype=np.float64)
        # Translations
        B[0::3, 0] = 1.0   # tx
        B[1::3, 1] = 1.0   # ty
        B[2::3, 2] = 1.0   # tz
        # Rotations: u = omega x r
        B[0::3, 3] = -y;  B[1::3, 3] = x    # Rz: omega=(0,0,1), u=(-y, x, 0)
        B[1::3, 4] = -z;  B[2::3, 4] = y    # Rx: omega=(1,0,0), u=(0, -z, y)
        B[0::3, 5] =  z;  B[2::3, 5] = -x   # Ry: omega=(0,1,0), u=(z, 0, -x)

        return B[self.free_dofs, :].astype(self.dtype)

    def _assemble_sparse_free(self) -> sp.csr_matrix:
        """Assemble the sparse stiffness matrix restricted to free DOFs.

        Used only for AMG preconditioning.  Memory cost: O(n_elem × 576).

        Uses a pre-allocated output buffer filled in chunks to avoid the
        growing-list + concatenate pattern (which copies all data twice and
        is ~3x slower than writing directly into a fixed buffer).
        """
        n_free = len(self.free_dofs)
        # Map global DOF → free DOF index (−1 for constrained)
        dof_map = np.full(self.n_dof, -1, dtype=np.int32)
        dof_map[self.free_dofs] = np.arange(n_free, dtype=np.int32)

        # Pre-allocate full COO buffer: n_elem × 24 × 24 entries (upper bound).
        # Writing into a fixed buffer instead of growing lists avoids the
        # list.append overhead and the final np.concatenate copy.
        n_nnz_max = self.n_elem * 576
        rows_buf = np.empty(n_nnz_max, dtype=np.int32)
        cols_buf = np.empty(n_nnz_max, dtype=np.int32)
        vals_buf = np.empty(n_nnz_max, dtype=np.float64)
        offset = 0

        # Process in chunks to cap per-iteration peak memory (~300 MB per chunk).
        # Each chunk's temporary (c, 24, 24) arrays are freed after the write.
        chunk = min(self.n_elem, 32_000)
        for start in range(0, self.n_elem, chunk):
            end = min(start + chunk, self.n_elem)
            ed  = self.elem_dofs[start:end]                    # (c, 24)
            Ee  = self.E_elem[start:end]                       # (c,)

            # Build COO triplets: shape (c, 24, 24) → (c*576,)
            # row[e,I,J] = ed[e,I],  col[e,I,J] = ed[e,J]
            row_local = ed[:, :, np.newaxis].repeat(24, axis=2).ravel()
            col_local = ed[:, np.newaxis, :].repeat(24, axis=1).ravel()
            val_local = (Ee[:, np.newaxis, np.newaxis]
                         * self.K0[np.newaxis]).ravel()

            # Restrict to free-DOF pairs
            row_free = dof_map[row_local]
            col_free = dof_map[col_local]
            mask = (row_free >= 0) & (col_free >= 0)
            n_valid = int(mask.sum())

            # Write directly into pre-allocated buffer (no copy, no list append)
            rows_buf[offset:offset + n_valid] = row_free[mask]
            cols_buf[offset:offset + n_valid] = col_free[mask]
            vals_buf[offset:offset + n_valid] = val_local[mask]
            offset += n_valid

        return sp.coo_matrix(
            (vals_buf[:offset], (rows_buf[:offset], cols_buf[:offset])),
            shape=(n_free, n_free),
        ).tocsr()

    # ------------------------------------------------------------------
    # Direct solve over active (solid-only) DOFs
    # ------------------------------------------------------------------

    def _solve_direct(
        self, verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Solve K u = f directly, over ACTIVE (solid-only) free DOFs.

        Skips the ~93% void: assembles the stiffness over only the elements
        with real material, then factorizes and solves exactly with a direct
        sparse solver (Intel MKL PARDISO via ``pypardiso`` if installed, else
        SciPy SuperLU).  No preconditioner, no CG iterations.

        Void with ``Emin`` contributes nothing physically, so the result
        matches the iterative (Emin-smeared) solve to round-off.  Void and
        clamped DOFs are filled with zero displacement on scatter-back.

        Caveat: solid not connected to the clamped base makes the reduced
        system singular; the returned ``residual`` will be large and
        ``converged`` False in that case.
        """
        t0 = time.perf_counter()
        # Active elements: real material (void sits at ~Emin << E0).
        e_thresh = 1e-6 * float(self.E_elem.max())
        active_elem = np.nonzero(self.E_elem > e_thresh)[0]

        free_mask = np.zeros(self.n_dof, dtype=bool)
        free_mask[self.free_dofs] = True
        if active_elem.size:
            active_dofs = np.unique(self.elem_dofs[active_elem].ravel())
            active_free = active_dofs[free_mask[active_dofs]]
        else:
            active_free = np.empty(0, dtype=np.int64)

        n_red = int(active_free.size)
        u_full = np.zeros(self.n_dof, dtype=self.dtype)
        info: dict[str, Any] = {
            "iters": 0, "residual": 0.0, "converged": True,
            "solve_time_s": 0.0, "precond_setup_s": 0.0, "cg_solve_s": 0.0,
            "n_active_dofs": n_red, "n_free_dofs": int(len(self.free_dofs)),
            "backend": "none",
        }

        if n_red > 0:
            # Reduced index map: global DOF -> [0, n_red) ; -1 if not active-free
            red = np.full(self.n_dof, -1, dtype=np.int64)
            red[active_free] = np.arange(n_red, dtype=np.int64)

            # Assemble K over ACTIVE elements only, restricted to active-free.
            t_asm = time.perf_counter()
            ed = self.elem_dofs[active_elem]                       # (a, 24)
            Ee = self.E_elem[active_elem].astype(np.float64)       # (a,)
            row_g = ed[:, :, np.newaxis].repeat(24, axis=2).ravel()
            col_g = ed[:, np.newaxis, :].repeat(24, axis=1).ravel()
            val   = (Ee[:, np.newaxis, np.newaxis]
                     * self.K0[np.newaxis].astype(np.float64)).ravel()
            r = red[row_g]
            c = red[col_g]
            m = (r >= 0) & (c >= 0)
            K = sp.coo_matrix((val[m], (r[m], c[m])), shape=(n_red, n_red)).tocsc()
            f_red = self.f[active_free].astype(np.float64)
            asm_s = time.perf_counter() - t_asm

            # Direct factorize + solve.
            t_solve = time.perf_counter()
            try:
                from pypardiso import spsolve as _direct_spsolve
                backend = "pardiso"
            except Exception:
                from scipy.sparse.linalg import spsolve as _direct_spsolve
                backend = "superlu"
            u_red = np.asarray(_direct_spsolve(K, f_red)).ravel()
            solve_s = time.perf_counter() - t_solve

            u_full[active_free] = u_red.astype(self.dtype)
            resid = K @ u_red - f_red
            rel = float(np.linalg.norm(resid) / (np.linalg.norm(f_red) + 1e-300))
            info.update(
                residual=rel, converged=bool(rel < 1e-6),
                precond_setup_s=asm_s, cg_solve_s=solve_s,
                solve_time_s=time.perf_counter() - t0, backend=backend,
            )
            if verbose:
                logger.info(
                    "direct[%s]: n_active=%d/%d  asm=%.1fs solve=%.1fs  rel=%.2e",
                    backend, n_red, len(self.free_dofs), asm_s, solve_s, rel,
                )

        u_nodal = u_full.reshape(self.n_nodes, 3).T.reshape(
            3, self.nz, self.ny, self.nx
        )
        u_elem = self._element_center_displacement(u_nodal)
        return u_nodal, u_elem, info

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _element_center_displacement(
        self, u_nodal: np.ndarray
    ) -> np.ndarray:
        """Average nodal displacements to element centres.

        Parameters
        ----------
        u_nodal : ndarray, shape (3, nz, ny, nx)

        Returns
        -------
        u_elem : ndarray, shape (3, Nz, Ny, Nx)
        """
        u_flat = u_nodal.reshape(3, self.n_nodes)   # (3, n_nodes)
        # For each element, average 8 corner node values
        node_ids = self.elem_dofs[:, ::3] // 3     # (n_elem, 8)  node indices
        u_corners = u_flat[:, node_ids]             # (3, n_elem, 8)
        u_center  = u_corners.mean(axis=2)          # (3, n_elem)
        return u_center.reshape(3, self.Nz, self.Ny, self.Nx)

    def element_strain_stress(
        self, u_nodal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Voigt strains and Cauchy stresses at element centres.

        Parameters
        ----------
        u_nodal : ndarray, shape (3, nz, ny, nx)

        Returns
        -------
        strain : ndarray, shape (6, Nz, Ny, Nx)
            [εxx, εyy, εzz, γxy, γxz, γyz] at each element centre.
        stress : ndarray, shape (6, Nz, Ny, Nx)
            [σxx, σyy, σzz, σxy, σxz, σyz] in physical units (Pa if E in Pa).
        """
        if self.E_elem is None:
            raise RuntimeError("Call assemble_system() first.")

        # Global DOF vector
        u_flat = u_nodal.reshape(3, self.n_nodes)

        # Gather the 24 DOF values at the 8 corners of each element
        node_ids = self.elem_dofs[:, ::3] // 3    # (n_elem, 8) node indices
        u_local  = np.empty((self.n_elem, 24), dtype=np.float64)
        for dir_idx in range(3):
            u_local[:, dir_idx::3] = u_flat[dir_idx][node_ids]   # (n_elem, 8)

        # Strain at element centre: ε = B_c × u_local  →  (6, n_elem)
        strain_flat = (self._B_centre @ u_local.T)   # (6, n_elem)

        # Lamé parameters per element (in physical units)
        nu  = self.nu
        lam = self.E_elem * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # (n_elem,)
        mu  = self.E_elem / (2.0 * (1.0 + nu))

        tr_eps = strain_flat[:3].sum(axis=0)          # volumetric strain (n_elem,)
        stress_flat = np.empty_like(strain_flat)
        # Normal stresses: σ_ii = λ tr(ε) + 2μ ε_ii
        for i in range(3):
            stress_flat[i] = lam * tr_eps + 2.0 * mu * strain_flat[i]
        # Shear stresses: σ_ij = μ γ_ij  (Voigt γ = 2ε so σ = μ γ)
        for i in range(3, 6):
            stress_flat[i] = mu * strain_flat[i]

        shape = (self.Nz, self.Ny, self.Nx)
        return strain_flat.reshape(6, *shape), stress_flat.reshape(6, *shape)

    def von_mises_stress(self, u_nodal: np.ndarray) -> np.ndarray:
        """Compute the von Mises equivalent stress at element centres.

        σ_vm = √(½[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)²
                    + 6(σxy² + σxz² + σyz²)])

        Parameters
        ----------
        u_nodal : ndarray, shape (3, nz, ny, nx)

        Returns
        -------
        sigma_vm : ndarray, shape (Nz, Ny, Nx)
        """
        _, stress = self.element_strain_stress(u_nodal)
        sxx, syy, szz = stress[0], stress[1], stress[2]
        sxy, sxz, syz = stress[3], stress[4], stress[5]
        vm = np.sqrt(
            0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
                   + 6.0 * (sxy**2 + sxz**2 + syz**2))
            + 1e-300
        )
        return vm

    # ------------------------------------------------------------------
    # Convenience: full pipeline in one call
    # ------------------------------------------------------------------

    def run(
        self,
        rho: np.ndarray,
        E0: float | np.ndarray,
        *,
        body_force: np.ndarray | None = None,
        point_loads: list | None = None,
        wind_pressure: np.ndarray | None = None,
        fixed_nodes: np.ndarray | None = None,
        Emin: float = 1e-9,
        p_simp: float = 3.0,
        rho_threshold: float = 0.5,
        tol: float = 1e-8,
        maxiter: int = 3000,
        precond: str = "jacobi",
        verbose: bool = False,
        compute_vm: bool = True,
    ) -> dict[str, np.ndarray | dict]:
        """Execute the full assemble → BC → load → solve → post-process pipeline.

        Parameters
        ----------
        (See individual method docstrings for full parameter descriptions.)

        Returns
        -------
        dict with keys:
            ``u_nodal``  (3, nz, ny, nx),
            ``u_elem``   (3, Nz, Ny, Nx),
            ``sigma_vm`` (Nz, Ny, Nx)  if ``compute_vm=True``,
            ``info``     dict from :meth:`solve`.
        """
        _t = time.perf_counter
        _t0 = _t(); self.assemble_system(rho, E0, Emin=Emin, p_simp=p_simp)
        t_assemble = _t() - _t0
        _t0 = _t(); self.apply_bcs(fixed_nodes=fixed_nodes)
        t_bcs = _t() - _t0
        _t0 = _t(); self.apply_loads(
            body_force=body_force,
            point_loads=point_loads,
            wind_pressure=wind_pressure,
            rho=rho,
            rho_threshold=rho_threshold,
        )
        t_loads = _t() - _t0
        _t0 = _t(); u_nodal, u_elem, info = self.solve(
            tol=tol, maxiter=maxiter, precond=precond, verbose=verbose
        )
        t_solve_total = _t() - _t0
        # Phase breakdown (assemble = matrix values, solve_total = precond build
        # + CG iterations; see info['precond_setup_s'] / info['cg_solve_s']).
        info.update(
            t_assemble_s=t_assemble,
            t_bcs_s=t_bcs,
            t_loads_s=t_loads,
            t_solve_total_s=t_solve_total,
        )
        result: dict = {"u_nodal": u_nodal, "u_elem": u_elem, "info": info}
        if compute_vm:
            result["sigma_vm"] = self.von_mises_stress(u_nodal)
        return result
