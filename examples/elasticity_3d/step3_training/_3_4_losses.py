# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for the 3D linear elasticity surrogate.

All tensors use NCDHW layout: (N, C, D, H, W).
Poisson's ratio is treated as a global constant (ν = 0.3).
"""

from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------

def _fd_gradient(u: torch.Tensor, dim: int, voxel_size: float = 1.0) -> torch.Tensor:
    """Central finite differences along a spatial dimension.

    Uses central differences in the interior and one-sided (forward/backward)
    differences at the two boundary planes.

    Parameters
    ----------
    u : torch.Tensor
        Shape ``(B, C, D, H, W)``.
    dim : int
        Spatial dimension index: 2=D, 3=H, 4=W.
    voxel_size : float
        Physical voxel edge length (denominator of the difference quotient).

    Returns
    -------
    torch.Tensor
        Same shape as ``u``.
    """
    n = u.shape[dim]
    # Shift +1 and -1 along dim with zero-padding at boundaries.
    u_fwd = torch.roll(u, -1, dims=dim)
    u_bwd = torch.roll(u,  1, dims=dim)

    # Interior: central difference  (u[i+1] - u[i-1]) / (2*h)
    grad = (u_fwd - u_bwd) / (2.0 * voxel_size)

    # Boundary at index 0: forward difference  (u[1] - u[0]) / h
    idx0 = [slice(None)] * u.ndim
    idx0[dim] = slice(0, 1)
    idx1 = [slice(None)] * u.ndim
    idx1[dim] = slice(1, 2)
    grad[tuple(idx0)] = (u[tuple(idx1)] - u[tuple(idx0)]) / voxel_size

    # Boundary at index n-1: backward difference  (u[n-1] - u[n-2]) / h
    idxN  = [slice(None)] * u.ndim
    idxN[dim]  = slice(n - 1, n)
    idxN1 = [slice(None)] * u.ndim
    idxN1[dim] = slice(n - 2, n - 1)
    grad[tuple(idxN)] = (u[tuple(idxN)] - u[tuple(idxN1)]) / voxel_size

    return grad


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def displacement_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Weighted mean squared error over all voxels and displacement channels.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted displacement, shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Ground-truth displacement, shape ``(B, 3, D, H, W)``.
    weight : float
        Scalar multiplier applied to the loss.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    return weight * F.mse_loss(pred, target)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    solid_mask: torch.Tensor,
    weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE restricted to solid voxels (solid_mask == 1).

    Prevents the model from wasting gradient signal on void regions that
    always have near-zero displacement.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    solid_mask : torch.Tensor
        Binary float mask, shape ``(B, 1, D, H, W)``.  Broadcast over C.
    weight : float
        Scalar multiplier.
    eps : float
        Added to denominator to avoid NaN when the mask is empty.

    Returns
    -------
    torch.Tensor
        Scalar loss value (0.0 if mask is entirely empty).
    """
    mask = solid_mask.expand_as(pred)          # (B, 3, D, H, W)
    diff_sq = (pred - target) ** 2 * mask
    denom = mask.sum() + eps
    return weight * diff_sq.sum() / denom


def equilibrium_residual(
    pred: torch.Tensor,
    E: torch.Tensor,
    voxel_size: float = 1.0,
    weight: float = 1.0,
    nu: float = 0.3,
) -> torch.Tensor:
    """Optional physics-informed loss: divergence of Cauchy stress.

    Computes the linear momentum residual:
        r_i = Σ_j  ∂σ_ij/∂x_j

    where the Cauchy stress is derived from the predicted displacement via
    the linear elastic constitutive law with Lamé parameters:
        λ = E·ν / ((1+ν)·(1−2ν))
        μ = E / (2·(1+ν))

    .. note::
        Set ``weight=0.0`` to skip this loss entirely.  When ``weight==0``
        the function returns ``tensor(0.)`` without performing any
        finite-difference computation, avoiding memory and compute overhead
        during early training epochs.

    .. warning::
        Activate this loss only after the data-driven losses have converged
        (~20 epochs).  Raw network output early in training produces large,
        noisy stress divergence values that destabilize training.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted displacement, shape ``(B, 3, D, H, W)``.
    E : torch.Tensor
        Young's modulus field, shape ``(B, 1, D, H, W)`` (normalized or
        physical; only the spatial pattern matters for weighting the residual).
    voxel_size : float
        Physical voxel edge length in metres.
    weight : float
        Scalar multiplier.  Set to 0.0 to disable.
    nu : float
        Poisson's ratio (constant, default 0.3).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if weight == 0.0:
        return pred.new_zeros(1).squeeze()

    # Lamé parameters derived from E and constant ν.
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))   # (B,1,D,H,W)
    mu  = E / (2.0 * (1.0 + nu))                       # (B,1,D,H,W)

    ux, uy, uz = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]  # each (B,1,D,H,W)

    # Displacement gradients: du_i/dx_j  (i=row, j=col)
    # Spatial dims: 2=D(z), 3=H(y), 4=W(x)
    dux_dz = _fd_gradient(ux, 2, voxel_size)
    dux_dy = _fd_gradient(ux, 3, voxel_size)
    dux_dx = _fd_gradient(ux, 4, voxel_size)

    duy_dz = _fd_gradient(uy, 2, voxel_size)
    duy_dy = _fd_gradient(uy, 3, voxel_size)
    duy_dx = _fd_gradient(uy, 4, voxel_size)

    duz_dz = _fd_gradient(uz, 2, voxel_size)
    duz_dy = _fd_gradient(uz, 3, voxel_size)
    duz_dx = _fd_gradient(uz, 4, voxel_size)

    # Symmetric strain tensor (Voigt: exx, eyy, ezz, exy, exz, eyz)
    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyz = 0.5 * (duy_dz + duz_dy)

    # Volumetric strain: tr(ε)
    tr_eps = exx + eyy + ezz

    # Cauchy stress (linear elastic isotropic)
    sxx = lam * tr_eps + 2.0 * mu * exx
    syy = lam * tr_eps + 2.0 * mu * eyy
    szz = lam * tr_eps + 2.0 * mu * ezz
    sxy = 2.0 * mu * exy
    sxz = 2.0 * mu * exz
    syz = 2.0 * mu * eyz

    # Divergence of stress: r_i = Σ_j ∂σ_ij/∂x_j
    # r_x = ∂σ_xx/∂x + ∂σ_xy/∂y + ∂σ_xz/∂z
    rx = (_fd_gradient(sxx, 4, voxel_size)
          + _fd_gradient(sxy, 3, voxel_size)
          + _fd_gradient(sxz, 2, voxel_size))
    # r_y = ∂σ_xy/∂x + ∂σ_yy/∂y + ∂σ_yz/∂z
    ry = (_fd_gradient(sxy, 4, voxel_size)
          + _fd_gradient(syy, 3, voxel_size)
          + _fd_gradient(syz, 2, voxel_size))
    # r_z = ∂σ_xz/∂x + ∂σ_yz/∂y + ∂σ_zz/∂z
    rz = (_fd_gradient(sxz, 4, voxel_size)
          + _fd_gradient(syz, 3, voxel_size)
          + _fd_gradient(szz, 2, voxel_size))

    residual = torch.cat([rx, ry, rz], dim=1)   # (B, 3, D, H, W)
    return weight * residual.pow(2).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    solid_mask: torch.Tensor,
    E: torch.Tensor,
    loss_weights: dict[str, Any],
    nu: float = 0.3,
    voxel_size: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute all active loss terms and return their weighted sum.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted displacement, shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Ground-truth displacement, shape ``(B, 3, D, H, W)``.
    solid_mask : torch.Tensor
        Binary float mask extracted from input channel 0,
        shape ``(B, 1, D, H, W)``.
    E : torch.Tensor
        Young's modulus field from input channel 1,
        shape ``(B, 1, D, H, W)``.
    loss_weights : dict
        Keys: ``"mse"``, ``"masked_mse"``, ``"eq_residual"`` — float values.
    nu : float
        Poisson's ratio (constant, default 0.3).
    voxel_size : float
        Physical voxel edge length in metres.

    Returns
    -------
    total : torch.Tensor
        Scalar, sum of weighted loss terms.
    components : dict[str, float]
        Per-term scalar values (detached), for logging.
    """
    w_mse   = float(loss_weights.get("mse", 1.0))
    w_mmse  = float(loss_weights.get("masked_mse", 1.0))
    w_eq    = float(loss_weights.get("eq_residual", 0.0))

    l_mse  = displacement_mse(pred, target, weight=w_mse)
    l_mmse = masked_mse(pred, target, solid_mask, weight=w_mmse)
    l_eq   = equilibrium_residual(pred, E, voxel_size=voxel_size,
                                   weight=w_eq, nu=nu)

    total = l_mse + l_mmse + l_eq
    components = {
        "mse":          l_mse.item(),
        "masked_mse":   l_mmse.item(),
        "eq_residual":  l_eq.item(),
        "total":        total.item(),
    }
    return total, components
