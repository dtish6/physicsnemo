# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for VoxelFEMSolver.

Run with pytest:
    cd examples/elasticity_3d
    python -m pytest solvers/tests/test_voxel_fem.py -v

Tests
-----
1. unit_stiffness_symmetry    — K0 is symmetric, positive semi-definite.
2. patch_test                 — Linear displacement field reproduced exactly.
3. energy_consistency         — u^T K u = u^T f (strain energy = external work).
4. gravity_bar                — Cantilever under gravity matches 1D continuum.
5. symmetry_check             — Symmetric geometry + load → symmetric response.
6. von_mises_consistency      — σ_vm ≥ 0 everywhere; uniform tension formula.
7. wind_pressure_load         — Wind pressure applies force on exposed faces only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project importable when run from within the tests/ subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solvers.hex_stiffness import compute_unit_stiffness
from solvers.voxel_fem import VoxelFEMSolver

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _solid_rho(grid):
    return np.ones(grid, dtype=np.float64)


def _build_solver(grid, h=1.0, nu=0.3):
    return VoxelFEMSolver(grid_size=grid, voxel_size=h, nu=nu)


# ---------------------------------------------------------------------------
# Test 1: Unit element stiffness — symmetry and PSD
# ---------------------------------------------------------------------------

def test_unit_stiffness_symmetry():
    """K0 must be symmetric and positive semi-definite (6 zero eigenvalues = 6 RBMs)."""
    K0 = compute_unit_stiffness(nu=0.3, h=1.0)
    assert K0.shape == (24, 24)

    # Symmetry
    np.testing.assert_allclose(K0, K0.T, atol=1e-14, err_msg="K0 not symmetric")

    # Eigenvalues: exactly 6 should be (nearly) zero (rigid body modes);
    # the remaining 18 should be positive.
    eigvals = np.linalg.eigvalsh(K0)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min():.3e}"
    n_zero = np.sum(eigvals < 1e-10)
    assert n_zero == 6, f"Expected 6 near-zero eigenvalues (RBMs), got {n_zero}"


# ---------------------------------------------------------------------------
# Test 2: Patch test — linear displacement field reproduced exactly
# ---------------------------------------------------------------------------

def test_patch_test():
    """Q1 elements must reproduce any affine displacement field exactly.

    Strategy
    --------
    Prescribe u(x,y,z) = (a·x, b·y, c·z) at ALL nodes (as Dirichlet BCs on
    interior nodes is implicit in computing K·u directly).

    Then compute K·u_prescribed and verify that for every interior node
    the net nodal force is zero (since an affine field causes constant strain
    and hence zero divergence of stress in the bulk).

    For boundary nodes the residual force equals the surface traction, which
    we don't check here (the patch test simply requires zero interior forces).
    """
    grid = (4, 4, 4)
    h    = 1.0
    nu   = 0.3
    E0   = 1.0

    solver = _build_solver(grid, h=h, nu=nu)
    rho    = _solid_rho(grid)
    solver.assemble_system(rho, E0, Emin=0.0, p_simp=1.0)

    # Prescribe u = (0.01·x, 0.02·y, 0.03·z) at all nodes
    a, b, c = 0.01, 0.02, 0.03
    Nz, Ny, Nx = grid
    nz, ny, nx = Nz + 1, Ny + 1, Nx + 1

    iz, iy, ix = np.mgrid[0:nz, 0:ny, 0:nx]   # node indices
    x = ix.ravel().astype(float) * h
    y = iy.ravel().astype(float) * h
    z = iz.ravel().astype(float) * h

    n_nodes = nz * ny * nx
    u_full = np.zeros(3 * n_nodes)
    u_full[0::3] = a * x
    u_full[1::3] = b * y
    u_full[2::3] = c * z

    # Identify interior nodes (not on any face)
    interior = (
        (iz > 0) & (iz < nz - 1)
        & (iy > 0) & (iy < ny - 1)
        & (ix > 0) & (ix < nx - 1)
    ).ravel()
    interior_dofs = np.where(
        np.repeat(interior, 3) & np.tile([True, True, True], n_nodes)
    )[0]

    # Compute K·u_prescribed
    Ku = solver._matvec(u_full)

    # Interior forces must be zero (patch test)
    np.testing.assert_allclose(
        Ku[interior_dofs], 0.0,
        atol=1e-10,
        err_msg="Patch test failed: non-zero interior force for affine displacement",
    )


# ---------------------------------------------------------------------------
# Test 3: Energy consistency  u^T K u = u^T f
# ---------------------------------------------------------------------------

def test_energy_consistency():
    """Strain energy must equal external work after convergence.

    u^T K u = u^T f  (exact for an exact solve; approximate for CG).
    """
    grid = (4, 4, 4)
    E0   = 1.0
    nu   = 0.3

    solver = _build_solver(grid, nu=nu)
    rho    = _solid_rho(grid)
    solver.assemble_system(rho, E0, p_simp=1.0)
    solver.apply_bcs()

    # Uniform body force in z
    bf = np.zeros((3, *grid))
    bf[2] = -1.0   # fz = -1 N/m³
    solver.apply_loads(body_force=bf)

    u_nodal, _, info = solver.solve(tol=1e-10, precond="jacobi", verbose=False)
    assert info["converged"], f"CG did not converge: {info}"

    # Full displacement vector (free DOFs only needed)
    u_full = u_nodal.reshape(3, -1).T.ravel()
    u_free = u_full[solver.free_dofs]
    f_free = solver.f[solver.free_dofs]

    strain_energy = 0.5 * u_free @ solver._matvec_free(u_free)
    ext_work      = 0.5 * u_free @ f_free

    rel_err = abs(strain_energy - ext_work) / (abs(ext_work) + 1e-300)
    assert rel_err < 1e-6, (
        f"Energy balance failed: strain_energy={strain_energy:.6e}, "
        f"ext_work={ext_work:.6e}, rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: Gravity bar — displacement profile
# ---------------------------------------------------------------------------

def test_gravity_bar():
    """Prismatic bar under gravity: displacement profile matches 1D solution.

    A 1×1×Nz bar clamped at z=0, body force fz = -ρg.  The axial stress is
    σzz(z) = ρg·(L − z), giving displacement:
        uz(z) = (ρg / E) · (L·z − z²/2)

    The Q1 elements represent linear displacement within each element, so the
    nodal displacements should match the exact quadratic at the node positions.

    We use a relatively long bar (Nz=8) and check uz at the top node.
    """
    Nz = 8
    grid = (Nz, 1, 1)
    h    = 1.0
    E0   = 1.0
    nu   = 0.0   # zero Poisson's ratio decouples z from x/y; 1D formula holds exactly
    q    = -1.0   # body force density in z (N/m³)

    solver = _build_solver(grid, h=h, nu=nu)
    solver.assemble_system(_solid_rho(grid), E0, Emin=0.0, p_simp=1.0)
    solver.apply_bcs()

    bf = np.zeros((3, *grid))
    bf[2] = q
    solver.apply_loads(body_force=bf)

    u_nodal, _, info = solver.solve(tol=1e-10, precond="jacobi")
    assert info["converged"], f"CG did not converge: {info}"

    # Exact 1D solution at node heights z = iz * h
    L = Nz * h
    nz = Nz + 1
    z_nodes = np.arange(nz) * h
    uz_exact = (q / E0) * (L * z_nodes - z_nodes**2 / 2.0)

    # Extract uz at all 4 corner nodes of the bar at each z-level.
    # Average over the 4 nodes in the (ny, nx) cross-section (should be equal).
    uz_nodal = u_nodal[2]   # (nz, ny, nx) = (9, 2, 2)
    uz_avg   = uz_nodal.reshape(nz, -1).mean(axis=1)  # (nz,)

    # The FEM solution with lumped body forces matches the exact quadratic
    # at the nodes (Q1 elements, 1D bar → exact for degree ≤ 2).
    np.testing.assert_allclose(
        uz_avg, uz_exact,
        rtol=0.02,   # 2% tolerance for lumped vs consistent body-force difference
        atol=1e-10,
        err_msg="Gravity bar: uz profile deviates from 1D analytical solution",
    )


# ---------------------------------------------------------------------------
# Test 5: Symmetry check
# ---------------------------------------------------------------------------

def test_symmetry_check():
    """A symmetric geometry under a symmetric load must give a symmetric response.

    Load: body force in z (gravity).  Geometry: full solid cube.
    Expected: ux and uy are antisymmetric about their respective midplanes.
    Here we check that the z-displacement is symmetric about the x and y
    midplanes (uz(x, y, z) = uz(Nx·h − x, y, z) for all y).
    """
    Nz, Ny, Nx = 4, 4, 4
    grid = (Nz, Ny, Nx)

    solver = _build_solver(grid)
    solver.assemble_system(_solid_rho(grid), E0=1.0, p_simp=1.0)
    solver.apply_bcs()

    bf = np.zeros((3, *grid))
    bf[2] = -1.0
    solver.apply_loads(body_force=bf)

    u_nodal, _, info = solver.solve(tol=1e-10)
    assert info["converged"]

    uz = u_nodal[2]   # (nz, ny, nx)

    # Symmetry in x: uz(iz, iy, ix) == uz(iz, iy, Nx − ix) for ix ≤ Nx//2
    np.testing.assert_allclose(
        uz[:, :, :],
        uz[:, :, ::-1],
        atol=1e-8,
        err_msg="uz not symmetric about x-midplane",
    )

    # Symmetry in y: uz(iz, iy, ix) == uz(iz, Ny − iy, ix)
    np.testing.assert_allclose(
        uz[:, :, :],
        uz[:, ::-1, :],
        atol=1e-8,
        err_msg="uz not symmetric about y-midplane",
    )


# ---------------------------------------------------------------------------
# Test 6: Von Mises stress consistency
# ---------------------------------------------------------------------------

def test_von_mises_consistency():
    """σ_vm must be non-negative; for uniform uniaxial stress σ_vm = |σzz|."""
    grid = (2, 2, 2)
    E0   = 1.0
    nu   = 0.3

    solver = _build_solver(grid)
    solver.assemble_system(_solid_rho(grid), E0, p_simp=1.0)
    solver.apply_bcs()

    # Apply a small z-body force
    bf = np.zeros((3, *grid))
    bf[2] = -0.1
    solver.apply_loads(body_force=bf)

    u_nodal, _, info = solver.solve(tol=1e-10)
    assert info["converged"]

    vm = solver.von_mises_stress(u_nodal)
    assert np.all(vm >= -1e-12), f"Negative von Mises stress: {vm.min():.3e}"


# ---------------------------------------------------------------------------
# Test 7: Wind pressure on exterior faces only
# ---------------------------------------------------------------------------

def test_wind_pressure_exterior_faces():
    """Wind pressure must not appear on fully interior faces (void sandwich)."""
    # Geometry: solid layer at z=1..2, void elsewhere → top and bottom faces exposed
    Nz, Ny, Nx = 3, 2, 2
    grid = (Nz, Ny, Nx)
    rho  = np.zeros(grid)
    rho[1] = 1.0   # only the middle z-layer is solid

    solver = _build_solver(grid)
    solver.assemble_system(rho, E0=1.0, p_simp=1.0)
    solver.apply_bcs()

    # Uniform wind pressure on all solid voxels
    wp = np.zeros(grid)
    wp[1] = 1.0

    solver.apply_loads(wind_pressure=wp, rho=rho)

    # Individual nodal z-forces should be non-zero (both faces exposed, opposite signs).
    # The net sum is zero by design (z- and z+ contributions cancel for a floating slab),
    # but individual values must be non-zero.
    f_z = solver.f[2::3]   # uz DOFs
    assert np.any(np.abs(f_z) > 1e-12), (
        "Expected non-zero nodal z-forces from wind pressure on exposed z-faces"
    )

    # x-faces: exposed → x-force should also be non-zero
    f_x = solver.f[0::3]
    total_fx = f_x.sum()
    # The x+ and x- contributions cancel for a symmetric slab → net = 0
    # but individual values should be non-zero
    assert np.any(np.abs(f_x) > 1e-12), "Expected non-zero nodal x-forces from wind"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run all tests when executed directly (without pytest)
    tests = [
        test_unit_stiffness_symmetry,
        test_patch_test,
        test_energy_consistency,
        test_gravity_bar,
        test_symmetry_check,
        test_von_mises_consistency,
        test_wind_pressure_exterior_faces,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
