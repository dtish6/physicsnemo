# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation metrics for the 3D linear elasticity surrogate.

All tensors use NCDHW layout: (N, C, D, H, W).
Metrics are computed per-sample (returning shape (B,)) so callers can
accumulate and average over a validation DataLoader.
Poisson's ratio is treated as a global constant (ν = 0.3).
"""

import torch

from step3_training._3_4_losses import _fd_gradient


# ---------------------------------------------------------------------------
# Displacement metrics
# ---------------------------------------------------------------------------

def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample relative L2 error on the displacement field.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    eps : float
        Stability term added to the denominator.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.  Caller averages over batch.
    """
    B = pred.shape[0]
    diff_norm = (pred - target).reshape(B, -1).norm(dim=1)
    tgt_norm  = target.reshape(B, -1).norm(dim=1)
    return diff_norm / (tgt_norm + eps)


def max_displacement_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample maximum absolute error of the displacement magnitude.

    Displacement magnitude = ||u||_2 = sqrt(ux² + uy² + uz²).

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Shape ``(B, 3, D, H, W)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.
    """
    pred_mag   = pred.pow(2).sum(dim=1).sqrt()    # (B, D, H, W)
    target_mag = target.pow(2).sum(dim=1).sqrt()  # (B, D, H, W)
    B = pred.shape[0]
    return (pred_mag - target_mag).abs().reshape(B, -1).max(dim=1).values


# ---------------------------------------------------------------------------
# Stress-derived metrics
# ---------------------------------------------------------------------------

def von_mises_stress(
    displacement: torch.Tensor,
    E: torch.Tensor,
    nu: float = 0.3,
    voxel_size: float = 1.0,
) -> torch.Tensor:
    """Von Mises equivalent stress derived from a displacement field.

    Uses central finite differences (see ``training.losses._fd_gradient``)
    to compute the full strain tensor, then converts to Cauchy stress via
    isotropic linear elasticity, and finally computes the von Mises scalar.

    Parameters
    ----------
    displacement : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    E : torch.Tensor
        Young's modulus field, shape ``(B, 1, D, H, W)``.
    nu : float
        Poisson's ratio (constant, default 0.3).
    voxel_size : float
        Physical voxel edge length in metres.

    Returns
    -------
    torch.Tensor
        Von Mises stress field, shape ``(B, D, H, W)``.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))

    ux, uy, uz = displacement[:, 0:1], displacement[:, 1:2], displacement[:, 2:3]

    exx = _fd_gradient(ux, 4, voxel_size)
    eyy = _fd_gradient(uy, 3, voxel_size)
    ezz = _fd_gradient(uz, 2, voxel_size)
    exy = 0.5 * (_fd_gradient(ux, 3, voxel_size) + _fd_gradient(uy, 4, voxel_size))
    exz = 0.5 * (_fd_gradient(ux, 2, voxel_size) + _fd_gradient(uz, 4, voxel_size))
    eyz = 0.5 * (_fd_gradient(uy, 2, voxel_size) + _fd_gradient(uz, 3, voxel_size))

    tr_eps = exx + eyy + ezz
    sxx = lam * tr_eps + 2.0 * mu * exx
    syy = lam * tr_eps + 2.0 * mu * eyy
    szz = lam * tr_eps + 2.0 * mu * ezz
    sxy = 2.0 * mu * exy
    sxz = 2.0 * mu * exz
    syz = 2.0 * mu * eyz

    # von Mises: σ_vm = sqrt(0.5 * [(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)²
    #                               + 6(σxy² + σxz² + σyz²)])
    vm = torch.sqrt(
        0.5 * (
            (sxx - syy).pow(2)
            + (syy - szz).pow(2)
            + (szz - sxx).pow(2)
            + 6.0 * (sxy.pow(2) + sxz.pow(2) + syz.pow(2))
        )
        + 1e-12   # numerical safety under sqrt
    )
    return vm.squeeze(1)   # (B, D, H, W)


def von_mises_stress_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    E: torch.Tensor,
    nu: float = 0.3,
    voxel_size: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample relative L2 error on the von Mises stress field.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    E : torch.Tensor
        Young's modulus field, shape ``(B, 1, D, H, W)``.
    nu : float
        Poisson's ratio (constant, default 0.3).
    voxel_size : float
        Physical voxel edge length in metres.
    eps : float
        Denominator stability term.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``.
    """
    vm_pred   = von_mises_stress(pred,   E, nu, voxel_size)   # (B,D,H,W)
    vm_target = von_mises_stress(target, E, nu, voxel_size)

    B = pred.shape[0]
    diff_norm = (vm_pred - vm_target).reshape(B, -1).norm(dim=1)
    tgt_norm  = vm_target.reshape(B, -1).norm(dim=1)
    return diff_norm / (tgt_norm + eps)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    E: torch.Tensor,
    nu: float = 0.3,
    voxel_size: float = 1.0,
) -> dict[str, float]:
    """Compute all evaluation metrics and return mean values over the batch.

    Parameters
    ----------
    pred : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    target : torch.Tensor
        Shape ``(B, 3, D, H, W)``.
    E : torch.Tensor
        Young's modulus field, shape ``(B, 1, D, H, W)``.
    nu : float
        Poisson's ratio (constant, default 0.3).
    voxel_size : float
        Physical voxel edge length in metres.

    Returns
    -------
    dict[str, float]
        Keys: ``"rel_l2"``, ``"von_mises_err"``, ``"max_disp_err"``.
    """
    with torch.no_grad():
        rel_l2    = relative_l2_error(pred, target).mean().item()
        vm_err    = von_mises_stress_error(pred, target, E, nu, voxel_size).mean().item()
        max_derr  = max_displacement_error(pred, target).mean().item()

    return {
        "rel_l2":        rel_l2,
        "von_mises_err": vm_err,
        "max_disp_err":  max_derr,
    }
