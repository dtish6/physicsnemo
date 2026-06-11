#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra training entry point for the 3D linear elasticity surrogate.

Usage::

    # Standard run (128^3 grid, production config)
    python train.py

    # Quick sanity run on 32^3 mock data
    python train.py dataset.grid_size=32 training.epochs=2 training.batch_size=1

    # Dry-run: print resolved config without training
    python train.py --cfg job

    # Override checkpoint directory and learning rate
    python train.py checkpoint.ckpt_path=/tmp/ckpt training.lr=5e-5
"""

import logging
import sys

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Force UTF-8 console output. On Windows the default cp1252 codec cannot encode
# the non-ASCII characters used in log messages (e.g. "→", "—"), which makes the
# logging StreamHandler raise UnicodeEncodeError and spam tracebacks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Make project subpackages importable when running from the project root.
# (No install required — just `cd examples/elasticity_3d && python train.py`.)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Elasticity surrogate — starting training.")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Resolve relative paths relative to the original working directory.
    cfg.dataset.train_path = to_absolute_path(cfg.dataset.train_path)
    cfg.dataset.val_path   = to_absolute_path(cfg.dataset.val_path)
    cfg.dataset.stats_path = to_absolute_path(cfg.dataset.stats_path)
    cfg.checkpoint.ckpt_path = to_absolute_path(cfg.checkpoint.ckpt_path)
    cfg.logging.log_dir      = to_absolute_path(cfg.logging.log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(device))

    # Import here so sys.path manipulations (if any) take effect first.
    from step3_training._3_5_trainer import Trainer

    trainer = Trainer(cfg, device)
    start_epoch = trainer.load_checkpoint()

    for epoch in range(start_epoch, cfg.training.epochs):
        train_metrics = trainer.train_epoch(epoch)
        log.info(
            "Epoch %d/%d — train: %s",
            epoch + 1,
            cfg.training.epochs,
            {k: f"{v:.4f}" for k, v in train_metrics.items()},
        )

        val_metrics = trainer.validate(epoch)
        log.info(
            "Epoch %d/%d — val:   %s",
            epoch + 1,
            cfg.training.epochs,
            {k: f"{v:.4f}" for k, v in val_metrics.items()},
        )

        if (epoch + 1) % cfg.checkpoint.save_every_n_epochs == 0:
            trainer.save_checkpoint(epoch + 1)

    # Save final checkpoint.
    trainer.save_checkpoint(cfg.training.epochs)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
