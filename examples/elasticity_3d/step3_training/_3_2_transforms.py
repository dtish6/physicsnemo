# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-channel z-score normalization transform for elasticity samples."""

from pathlib import Path

import numpy as np
import torch


class ChannelNormalize:
    """Apply z-score normalization to (x, y) samples from ElasticityDataset.

    Loads per-channel mean and standard deviation from a ``normalization_stats.npz``
    file produced by :meth:`~data.dataset.ElasticityDataset.compute_statistics`.

    Normalization:
        x_norm[c] = (x[c] - μ_in[c]) / σ_in[c]
        y_norm[c] = (y[c] - μ_out[c]) / σ_out[c]

    Parameters
    ----------
    stats_path : str or Path
        Path to ``normalization_stats.npz``.  Must contain keys
        ``input_mean``, ``input_std``, ``output_mean``, ``output_std``.
    device : str or torch.device
        Device on which the normalization buffers are placed.
        Use ``"cpu"`` for DataLoader workers (default).
    """

    def __init__(
        self,
        stats_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        stats = np.load(str(stats_path))
        # Shapes: (C_in,) and (C_out,) — will be broadcast over spatial dims.
        self.in_mean  = torch.tensor(stats["input_mean"],  dtype=torch.float32, device=device)
        self.in_std   = torch.tensor(stats["input_std"],   dtype=torch.float32, device=device)
        self.out_mean = torch.tensor(stats["output_mean"], dtype=torch.float32, device=device)
        self.out_std  = torch.tensor(stats["output_std"],  dtype=torch.float32, device=device)

    def __call__(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize a single (x, y) sample.

        Parameters
        ----------
        sample : tuple[Tensor, Tensor]
            ``x`` shape ``(6, D, H, W)``, ``y`` shape ``(3, D, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Normalized ``(x_norm, y_norm)``.
        """
        x, y = sample
        # Reshape means/stds to (C, 1, 1, 1) for broadcasting.
        in_mean = self.in_mean.view(-1, 1, 1, 1)
        in_std  = self.in_std.view(-1, 1, 1, 1)
        out_mean = self.out_mean.view(-1, 1, 1, 1)
        out_std  = self.out_std.view(-1, 1, 1, 1)

        x_norm = (x - in_mean)  / in_std
        y_norm = (y - out_mean) / out_std
        return x_norm, y_norm

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Invert output normalization to recover physical displacement values.

        Parameters
        ----------
        y_norm : torch.Tensor
            Normalized prediction, shape ``(B, 3, D, H, W)`` or
            ``(3, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Displacement in physical units, same shape as input.
        """
        # Handle both batched and unbatched tensors.  Move stats to the
        # prediction's device (the normalizer buffers live on CPU for the
        # DataLoader workers, but inference runs on GPU).
        out_mean = self.out_mean.to(y_norm.device)
        out_std  = self.out_std.to(y_norm.device)
        if y_norm.ndim == 5:
            out_mean = out_mean.view(1, -1, 1, 1, 1)
            out_std  = out_std.view(1,  -1, 1, 1, 1)
        elif y_norm.ndim == 4:
            out_mean = out_mean.view(-1, 1, 1, 1)
            out_std  = out_std.view(-1,  1, 1, 1)

        return y_norm * out_std + out_mean

    def denormalize_input_channel(
        self, x_norm: torch.Tensor, ch: int
    ) -> torch.Tensor:
        """Invert normalization for a single input channel.

        Useful for recovering physical Young's modulus values during eval.

        Parameters
        ----------
        x_norm : torch.Tensor
            Normalized input, arbitrary shape where the channel ``ch``
            is the first dimension.
        ch : int
            Channel index (0–5).

        Returns
        -------
        torch.Tensor
            De-normalized values, same shape as ``x_norm``.
        """
        return x_norm * self.in_std[ch].to(x_norm.device) + self.in_mean[ch].to(x_norm.device)
