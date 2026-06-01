# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Q1 trilinear hexahedral element stiffness matrix.

Computes the 24×24 unit element stiffness K0 (E=1) for a cube of side h.
Scaled by element Young's modulus during assembly: K_e = E_e * K0.

Node ordering (consistent with VoxelFEMSolver connectivity):

    Local   diz  diy  dix   ξ    η    ζ
    ─────────────────────────────────────
       0      0    0    0   -1   -1   -1
       1      0    0    1   +1   -1   -1
       2      0    1    1   +1   +1   -1
       3      0    1    0   -1   +1   -1
       4      1    0    0   -1   -1   +1
       5      1    0    1   +1   -1   +1
       6      1    1    1   +1   +1   +1
       7      1    1    0   -1   +1   +1

DOF layout: [u0x, u0y, u0z, u1x, u1y, u1z, ..., u7x, u7y, u7z]  (24 DOFs)

ξ ↔ x-direction (ix/W),  η ↔ y-direction (iy/H),  ζ ↔ z-direction (iz/D, vertical)

Voigt strain convention: [εxx, εyy, εzz, γxy, γxz, γyz]
where γ = 2ε (engineering shear strains).
"""

import numpy as np

# Natural coordinates for each of the 8 nodes.
_NODE_XI  = np.array([-1, +1, +1, -1, -1, +1, +1, -1], dtype=np.float64)
_NODE_ETA = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=np.float64)
_NODE_ZET = np.array([-1, -1, -1, -1, +1, +1, +1, +1], dtype=np.float64)

# Face local node indices (4 nodes per face) and outward unit normals.
# Face order: x-, x+, y-, y+, z-, z+
FACE_LOCAL_NODES: list[list[int]] = [
    [0, 3, 7, 4],   # x-  (ξ = -1)
    [1, 2, 6, 5],   # x+  (ξ = +1)
    [0, 1, 5, 4],   # y-  (η = -1)
    [3, 2, 6, 7],   # y+  (η = +1)
    [0, 1, 2, 3],   # z-  (ζ = -1)  — base / clamped face
    [4, 5, 6, 7],   # z+  (ζ = +1)
]

FACE_NORMALS: np.ndarray = np.array([
    [-1,  0,  0],   # x-
    [+1,  0,  0],   # x+
    [ 0, -1,  0],   # y-
    [ 0, +1,  0],   # y+
    [ 0,  0, -1],   # z-
    [ 0,  0, +1],   # z+
], dtype=np.float64)

# Indices [0..5] for each face label
FACE_IDX = {"x-": 0, "x+": 1, "y-": 2, "y+": 3, "z-": 4, "z+": 5}


def compute_unit_stiffness(nu: float = 0.3, h: float = 1.0) -> np.ndarray:
    """Compute the 24×24 unit element stiffness matrix K0 (E = 1).

    Uses 2×2×2 Gauss quadrature (exact for the trilinear interpolation with a
    uniform Jacobian).

    Parameters
    ----------
    nu : float
        Poisson's ratio (default 0.3).
    h : float
        Voxel side length in physical units (default 1.0).

    Returns
    -------
    K0 : ndarray, shape (24, 24), float64
        Symmetric, positive semi-definite element stiffness for E = 1.
        For a physical element: K_e = E_e * K0.
    """
    # Lamé parameters (E = 1)
    mu  = 1.0 / (2.0 * (1.0 + nu))
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Constitutive matrix D (6×6, Voigt notation)
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    # 2-point Gauss quadrature on [-1, 1]: points ±1/√3, weights 1.
    gp = 1.0 / np.sqrt(3.0)
    gauss = [(-gp, 1.0), (gp, 1.0)]

    # For the uniform Cartesian element mapped from [-1,1]³ to [0,h]³:
    #   Jacobian J = (h/2) * I₃  →  det(J) = (h/2)³
    #   Physical gradient:  ∂N/∂x = (2/h) * ∂N/∂ξ
    det_J  = (h / 2.0) ** 3
    scale  = 2.0 / h          # dN/dξ → dN/dx conversion factor

    K0 = np.zeros((24, 24), dtype=np.float64)
    B  = np.zeros((6, 24),  dtype=np.float64)

    for xi, wi in gauss:
        for eta, wj in gauss:
            for zeta, wk in gauss:
                weight = wi * wj * wk * det_J

                # Shape function natural-coordinate derivatives at this Gauss point.
                dN_dxi  = _NODE_XI  * (1.0 + _NODE_ETA * eta ) * (1.0 + _NODE_ZET * zeta) / 8.0
                dN_deta = _NODE_ETA * (1.0 + _NODE_XI  * xi  ) * (1.0 + _NODE_ZET * zeta) / 8.0
                dN_dzet = _NODE_ZET * (1.0 + _NODE_XI  * xi  ) * (1.0 + _NODE_ETA * eta ) / 8.0

                # Physical derivatives
                dNdx = scale * dN_dxi    # ∂N/∂x (= ∂N/∂W direction)
                dNdy = scale * dN_deta   # ∂N/∂y (= ∂N/∂H direction)
                dNdz = scale * dN_dzet   # ∂N/∂z (= ∂N/∂D direction, vertical)

                # Build strain-displacement matrix B (6×24)
                B[:] = 0.0
                for I in range(8):
                    c = 3 * I
                    # εxx = ∂ux/∂x
                    B[0, c]     = dNdx[I]
                    # εyy = ∂uy/∂y
                    B[1, c + 1] = dNdy[I]
                    # εzz = ∂uz/∂z
                    B[2, c + 2] = dNdz[I]
                    # γxy = ∂ux/∂y + ∂uy/∂x
                    B[3, c]     = dNdy[I]
                    B[3, c + 1] = dNdx[I]
                    # γxz = ∂ux/∂z + ∂uz/∂x
                    B[4, c]     = dNdz[I]
                    B[4, c + 2] = dNdx[I]
                    # γyz = ∂uy/∂z + ∂uz/∂y
                    B[5, c + 1] = dNdz[I]
                    B[5, c + 2] = dNdy[I]

                K0 += weight * (B.T @ D @ B)

    return K0


def compute_B_at_center(h: float = 1.0) -> np.ndarray:
    """Return the 6×24 B matrix evaluated at the element centre (ξ=η=ζ=0).

    Used for post-processing strains and stresses at element centres.

    Parameters
    ----------
    h : float
        Voxel side length.

    Returns
    -------
    B : ndarray, shape (6, 24), float64
    """
    xi, eta, zeta = 0.0, 0.0, 0.0
    scale = 2.0 / h

    dN_dxi  = _NODE_XI  * (1.0 + _NODE_ETA * eta ) * (1.0 + _NODE_ZET * zeta) / 8.0
    dN_deta = _NODE_ETA * (1.0 + _NODE_XI  * xi  ) * (1.0 + _NODE_ZET * zeta) / 8.0
    dN_dzet = _NODE_ZET * (1.0 + _NODE_XI  * xi  ) * (1.0 + _NODE_ETA * eta ) / 8.0

    dNdx = scale * dN_dxi
    dNdy = scale * dN_deta
    dNdz = scale * dN_dzet

    B = np.zeros((6, 24), dtype=np.float64)
    for I in range(8):
        c = 3 * I
        B[0, c]     = dNdx[I]
        B[1, c + 1] = dNdy[I]
        B[2, c + 2] = dNdz[I]
        B[3, c]     = dNdy[I];  B[3, c + 1] = dNdx[I]
        B[4, c]     = dNdz[I];  B[4, c + 2] = dNdx[I]
        B[5, c + 1] = dNdz[I];  B[5, c + 2] = dNdy[I]
    return B
