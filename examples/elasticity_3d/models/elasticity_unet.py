# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ElasticityUNet: 3D U-Net surrogate for linear elasticity on voxel grids."""

from dataclasses import dataclass
from typing import Sequence

import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.unet import UNet


@dataclass
class ElasticityMetaData(ModelMetaData):
    """PhysicsNeMo metadata for the elasticity surrogate model."""

    amp: bool = True
    cuda_graphs: bool = True
    jit: bool = False


class ElasticityUNet(Module):
    """3D U-Net surrogate for linear elasticity on voxel grids.

    Maps a 6-channel voxelized input field to a 3-channel displacement field
    on a regular (D, H, W) grid.

    Input channel layout (C_in = 6):
        [0]   solid_mask         — binary float: 1=solid, 0=void
        [1]   E_normalized       — Young's modulus (z-score normalized)
        [2:5] body_force (fx,fy,fz) — N/m³ (z-score normalized)
        [5]   wind_pressure      — scalar pressure magnitude at exterior surface
                                   voxels (Pa, z-score); traction direction is
                                   inferred from geometry (surface normal
                                   implicit in solid_mask)

    Fixed physics constants (not channels):
        ν = 0.3  (Poisson's ratio, constant across all samples)
        BC: z=0 plane fully clamped (u=0 Dirichlet), learned implicitly

    Output channel layout (C_out = 3):
        [0] ux, [1] uy, [2] uz  — displacement components (m, normalized)

    Tensor layout: (N, C, D, H, W)  — NCDHW, right-handed Cartesian.
    Base plane = D=0 slice (z = 0 in physical coordinates).

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 6.
    out_channels : int
        Number of output channels (displacement components). Default 3.
    model_depth : int
        U-Net depth (number of encoder levels). Default 4.
        At 128^3 input, bottleneck spatial size = 128 / 2^(depth-1) = 16^3.
        Increase to 5 for 256^3 grids (bottleneck = 16^3 still).
    feature_map_channels : sequence of int
        Per-block channel counts; length must equal model_depth * 2
        (two Conv3DBlocks per encoder level, mirrored in decoder).
        Default (32, 32, 64, 64, 128, 128, 256, 256) for depth=4.
    normalization : str or None
        Normalization applied inside Conv3DBlocks.
        One of "groupnorm" (default), "batchnorm", "layernorm", or None.
    gradient_checkpointing : bool
        Enable activation recomputation to reduce peak GPU memory at the
        cost of one extra forward pass per layer. Default True.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        model_depth: int = 4,
        feature_map_channels: Sequence[int] = (32, 32, 64, 64, 128, 128, 256, 256),
        normalization: str | None = "groupnorm",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__(meta=ElasticityMetaData())

        # Store all args as plain Python types for Module JSON serialization.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_depth = model_depth
        self.feature_map_channels = list(feature_map_channels)
        self.normalization = normalization
        self.gradient_checkpointing = gradient_checkpointing

        self.backbone = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            model_depth=model_depth,
            feature_map_channels=self.feature_map_channels,
            normalization=normalization,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 6, D, H, W)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 3, D, H, W)``, predicted displacement (ux, uy, uz).
        """
        return self.backbone(x)
