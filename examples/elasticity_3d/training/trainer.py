# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer class for the 3D linear elasticity surrogate.

Follows the training pattern established in
``examples/structural_mechanics/deforming_plate/train.py``:
  - torch.amp.autocast + GradScaler for mixed precision
  - physicsnemo.utils.checkpoint for save/load
  - TensorBoard SummaryWriter for logging
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
    _TENSORBOARD_AVAILABLE = False

from data.dataset import ElasticityDataset
from data.transforms import ChannelNormalize
from eval.metrics import compute_all_metrics
from models.elasticity_unet import ElasticityUNet
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint
from training.losses import combined_loss

logger = logging.getLogger(__name__)


class _NullWriter:
    """No-op TensorBoard writer used when tensorboard is not installed."""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class Trainer:
    """Self-contained training loop for the elasticity surrogate.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra configuration object.
    device : torch.device
        Target device (``cuda:0`` for single-GPU workstation).
    """

    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device
        self.amp    = bool(cfg.training.amp) and device.type == "cuda"

        # --- Model -----------------------------------------------------------
        self.model = ElasticityUNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            model_depth=cfg.model.model_depth,
            feature_map_channels=list(cfg.model.feature_map_channels),
            normalization=cfg.model.normalization,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        ).to(device)

        # --- Optimizer & scheduler -------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.lr_min,
        )
        self.scaler = GradScaler(enabled=self.amp)

        # --- Data ------------------------------------------------------------
        self.normalizer = (
            ChannelNormalize(cfg.dataset.stats_path, device="cpu")
            if Path(cfg.dataset.stats_path).exists()
            else None
        )
        if self.normalizer is None:
            logger.warning(
                "stats_path '%s' not found — running without normalization. "
                "Run data/generate_mock_hdf5.py and compute_statistics() first.",
                cfg.dataset.stats_path,
            )

        self.train_loader = self._build_dataloader(
            cfg.dataset.train_path, shuffle=True
        )
        self.val_loader = self._build_dataloader(
            cfg.dataset.val_path, shuffle=False
        )

        # --- Logging ---------------------------------------------------------
        Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
        if _TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=cfg.logging.log_dir)
        else:
            logger.warning(
                "tensorboard not installed — scalar logging disabled.  "
                "Install with: pip install tensorboard"
            )
            self.writer = _NullWriter()
        self.log_every  = int(cfg.logging.log_every_n_steps)
        self._global_step = 0

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one full pass over the training DataLoader.

        Returns
        -------
        dict[str, float]
            Averaged loss component values for the epoch.
        """
        self.model.train()
        accum: dict[str, float] = {}
        n_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.amp):
                pred = self.model(x)
                # Channel 0 is z-score normalized, so it is NOT binary here
                # (void < 0, solid > 0).  masked_mse requires a true {0,1}
                # mask — negative weights would let the loss go negative and
                # diverge.  Recover physical density and threshold.
                if self.normalizer is not None:
                    rho = self.normalizer.denormalize_input_channel(x[:, 0:1], 0)
                    solid_mask = (rho > 0.5).float()
                else:
                    solid_mask = (x[:, 0:1] > 0.5).float()
                E          = x[:, 1:2]
                loss, comps = combined_loss(
                    pred, y, solid_mask, E,
                    loss_weights=self.cfg.training.loss_weights,
                )

            self.scaler.scale(loss).backward()

            # Gradient clipping (applied after unscaling).
            if self.cfg.training.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.training.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate for epoch-level reporting.
            for k, v in comps.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

            # Step-level TensorBoard logging.
            if self._global_step % self.log_every == 0:
                for k, v in comps.items():
                    self.writer.add_scalar(f"train_step/{k}", v, self._global_step)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], self._global_step
                )
            self._global_step += 1

        self.scheduler.step()

        means = {k: v / max(n_batches, 1) for k, v in accum.items()}
        for k, v in means.items():
            self.writer.add_scalar(f"train_epoch/{k}", v, epoch)
        return means

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self, epoch: int) -> dict[str, float]:
        """Run inference on the validation DataLoader.

        Predictions are de-normalized before computing physical metrics.

        Returns
        -------
        dict[str, float]
            Keys: ``"rel_l2"``, ``"von_mises_err"``, ``"max_disp_err"``.
        """
        self.model.eval()
        accum: dict[str, float] = {}
        n_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.amp):
                    pred = self.model(x)

                # De-normalize predictions and targets for physical metrics.
                if self.normalizer is not None:
                    pred_phys = self.normalizer.denormalize_output(pred)
                    y_phys    = self.normalizer.denormalize_output(y)
                else:
                    pred_phys, y_phys = pred, y

                E = x[:, 1:2]  # still in normalized space; pattern is sufficient
                metrics = compute_all_metrics(pred_phys, y_phys, E)
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                n_batches += 1

        means = {k: v / max(n_batches, 1) for k, v in accum.items()}
        for k, v in means.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        return means

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_checkpoint(self, epoch: int) -> None:
        """Save model, optimizer, scheduler, and scaler state.

        Delegates to :func:`physicsnemo.utils.checkpoint.save_checkpoint`.
        """
        ckpt_path = self.cfg.checkpoint.ckpt_path
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            path=ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
        )
        logger.info("Checkpoint saved at epoch %d → %s", epoch, ckpt_path)

    def load_checkpoint(self) -> int:
        """Load the most recent checkpoint if one exists.

        Returns
        -------
        int
            Starting epoch (0 if no checkpoint found).
        """
        ckpt_path = self.cfg.checkpoint.ckpt_path
        if not Path(ckpt_path).exists():
            logger.info("No checkpoint found at '%s' — starting from scratch.", ckpt_path)
            return 0
        try:
            epoch = load_checkpoint(
                path=ckpt_path,
                models=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=str(self.device),
            )
            logger.info("Loaded checkpoint: resuming from epoch %d.", epoch)
            return epoch
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s — starting from scratch.", exc)
            return 0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_dataloader(self, data_path: str, shuffle: bool) -> DataLoader:
        """Construct DataLoader for a given data directory."""
        cfg = self.cfg
        if not Path(data_path).exists():
            logger.warning("Data path '%s' does not exist — returning empty loader.", data_path)
            return DataLoader([])

        dataset = ElasticityDataset(
            data_path=data_path,
            grid_size=cfg.dataset.grid_size,
            stats_path=cfg.dataset.stats_path if self.normalizer is None else None,
            transform=self.normalizer,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.dataset.pin_memory and self.device.type == "cuda",
            drop_last=shuffle,
        )
