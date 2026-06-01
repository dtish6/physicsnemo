# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HDF5-backed dataset for 3D linear elasticity samples.

Each sample is one HDF5 file.  The file layout is documented in DESIGN.md.
Channel conventions are exposed as class-level slice constants so downstream
code can reference them symbolically rather than hardcoding indices.
"""

import glob
import os
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Poisson's ratio is constant across all samples (ν = 0.3).
# Stored as a file-level HDF5 attribute for reference; not a model input.
POISSON_RATIO: float = 0.3


class ElasticityDataset(Dataset):
    """PyTorch Dataset streaming 3D elasticity samples from HDF5 files.

    Each ``.h5`` file must contain the following top-level datasets:

    ============================================================
    Key                   Shape                  dtype
    ============================================================
    solid_mask            (D, H, W)              float32
    E_normalized          (D, H, W)              float32
    body_force            (3, D, H, W)           float32
    wind_pressure         (D, H, W)              float32
    displacement          (3, D, H, W)           float32
    ============================================================

    Optional HDF5 file-level attributes (metadata only, not model inputs):
        ``nu``    — Poisson's ratio (constant 0.3)
        ``bc``    — boundary condition description string

    The dataset stacks the five input fields into a 6-channel tensor:
        x[:, 0]   = solid_mask
        x[:, 1]   = E_normalized
        x[:, 2:5] = body_force (fx, fy, fz)
        x[:, 5]   = wind_pressure

    Parameters
    ----------
    data_path : str or Path
        Directory containing ``*.h5`` sample files.
    file_pattern : str
        Glob pattern used to find files (default ``"*.h5"``).
    grid_size : int or tuple of int
        Expected voxel grid size per side for validation.  If a single int,
        assumed isotropic.  Set to ``None`` to skip shape checking.
    stats_path : str or Path or None
        Path to ``normalization_stats.npz`` produced by
        :meth:`compute_statistics`.  When provided the dataset stores the
        paths for use by :class:`~data.transforms.ChannelNormalize`.
    transform : callable or None
        Optional callable applied to ``(x, y)`` after assembly.
        Typically a :class:`~data.transforms.ChannelNormalize` instance.
    """

    # Channel layout constants — use these instead of magic integers.
    CHANNEL_SOLID_MASK:    slice = slice(0, 1)
    CHANNEL_E:             slice = slice(1, 2)
    CHANNEL_BODY_FORCE:    slice = slice(2, 5)
    CHANNEL_WIND_PRESSURE: slice = slice(5, 6)

    IN_CHANNELS:  int = 6
    OUT_CHANNELS: int = 3

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_pattern: str = "*.h5",
        grid_size: int | tuple[int, int, int] | None = 128,
        stats_path: str | Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.data_path   = Path(data_path)
        self.stats_path  = Path(stats_path) if stats_path is not None else None
        self.transform   = transform

        if isinstance(grid_size, int):
            self.grid_size: tuple[int, int, int] | None = (grid_size,) * 3
        else:
            self.grid_size = grid_size  # type: ignore[assignment]

        self._files: list[Path] = sorted(
            Path(p) for p in glob.glob(str(self.data_path / file_pattern))
        )
        if len(self._files) == 0:
            raise FileNotFoundError(
                f"No files matching '{file_pattern}' found in '{self.data_path}'."
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample and return ``(x, y)`` float32 tensors.

        Parameters
        ----------
        idx : int

        Returns
        -------
        x : torch.Tensor
            Shape ``(6, D, H, W)``, float32.
        y : torch.Tensor
            Shape ``(3, D, H, W)``, float32 — displacement (ux, uy, uz).
        """
        filepath = self._files[idx]
        x, y = self._load_sample(filepath)

        if self.grid_size is not None:
            expected = tuple(self.grid_size)
            actual = tuple(y.shape[1:])
            if actual != expected:
                raise ValueError(
                    f"Sample {filepath.name}: expected grid {expected}, got {actual}."
                )

        if self.transform is not None:
            x, y = self.transform((x, y))

        return x, y

    @staticmethod
    def _load_sample(filepath: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one HDF5 file and assemble input/output tensors.

        .. important::
            Opens the file handle inside this method (not cached) so the
            dataset is safe for ``num_workers > 0`` DataLoader workers.
            h5py is not fork-safe with persistent handles.
        """
        with h5py.File(filepath, "r") as f:
            solid_mask    = torch.from_numpy(f["solid_mask"][:].astype(np.float32))      # (D,H,W)
            E_normalized  = torch.from_numpy(f["E_normalized"][:].astype(np.float32))    # (D,H,W)
            body_force    = torch.from_numpy(f["body_force"][:].astype(np.float32))      # (3,D,H,W)
            wind_pressure = torch.from_numpy(f["wind_pressure"][:].astype(np.float32))   # (D,H,W)
            displacement  = torch.from_numpy(f["displacement"][:].astype(np.float32))    # (3,D,H,W)

        # Assemble 6-channel input: [solid_mask, E, fx, fy, fz, wind_pressure]
        x = torch.cat([
            solid_mask.unsqueeze(0),     # (1,D,H,W)
            E_normalized.unsqueeze(0),   # (1,D,H,W)
            body_force,                  # (3,D,H,W)
            wind_pressure.unsqueeze(0),  # (1,D,H,W)
        ], dim=0)                        # → (6,D,H,W)

        return x, displacement

    @classmethod
    def compute_statistics(
        cls,
        data_path: str | Path,
        output_path: str | Path,
        file_pattern: str = "*.h5",
        num_samples: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute per-channel mean and standard deviation over the dataset.

        Streams through all (or ``num_samples``) files, accumulating online
        statistics.  Writes a ``normalization_stats.npz`` file that can be
        loaded by :class:`~data.transforms.ChannelNormalize`.

        Parameters
        ----------
        data_path : str or Path
        output_path : str or Path
            Destination ``.npz`` file path.
        file_pattern : str
        num_samples : int or None
            If set, only the first ``num_samples`` files are used.

        Returns
        -------
        dict with keys ``input_mean``, ``input_std``, ``output_mean``,
        ``output_std`` — all shape ``(C,)`` float32 arrays.
        """
        files = sorted(
            Path(p) for p in glob.glob(str(Path(data_path) / file_pattern))
        )
        if num_samples is not None:
            files = files[:num_samples]
        if len(files) == 0:
            raise FileNotFoundError(f"No files found in '{data_path}'.")

        # Welford online mean/variance (numerically stable, single pass).
        in_ch, out_ch = cls.IN_CHANNELS, cls.OUT_CHANNELS
        in_count  = np.zeros(in_ch,  dtype=np.float64)
        in_mean   = np.zeros(in_ch,  dtype=np.float64)
        in_M2     = np.zeros(in_ch,  dtype=np.float64)
        out_count = np.zeros(out_ch, dtype=np.float64)
        out_mean  = np.zeros(out_ch, dtype=np.float64)
        out_M2    = np.zeros(out_ch, dtype=np.float64)

        for fp in files:
            x, y = cls._load_sample(fp)
            # x: (6,D,H,W), y: (3,D,H,W)
            for c in range(in_ch):
                vals = x[c].numpy().ravel()
                for v in vals:
                    in_count[c] += 1
                    delta = v - in_mean[c]
                    in_mean[c] += delta / in_count[c]
                    in_M2[c]   += delta * (v - in_mean[c])

            for c in range(out_ch):
                vals = y[c].numpy().ravel()
                for v in vals:
                    out_count[c] += 1
                    delta = v - out_mean[c]
                    out_mean[c] += delta / out_count[c]
                    out_M2[c]   += delta * (v - out_mean[c])

        in_std  = np.sqrt(in_M2  / np.maximum(in_count  - 1, 1)).astype(np.float32)
        out_std = np.sqrt(out_M2 / np.maximum(out_count - 1, 1)).astype(np.float32)
        in_mean  = in_mean.astype(np.float32)
        out_mean = out_mean.astype(np.float32)

        # Avoid division by zero for constant channels (e.g. solid_mask).
        in_std  = np.where(in_std  < 1e-8, 1.0, in_std)
        out_std = np.where(out_std < 1e-8, 1.0, out_std)

        stats = {
            "input_mean":  in_mean,
            "input_std":   in_std,
            "output_mean": out_mean,
            "output_std":  out_std,
        }
        np.savez(str(output_path), **stats)
        return stats
