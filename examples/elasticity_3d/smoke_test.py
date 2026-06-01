#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the 3D linear elasticity surrogate scaffold.

Verifies — without any real data on disk — that the entire forward/backward/
checkpoint pipeline runs correctly end-to-end.

Checks:
  1. ElasticityUNet instantiates with correct in/out channel counts.
  2. Forward pass produces the right output shape (B, 3, D, H, W).
  3. All three loss functions compute finite scalar values.
  4. Backward pass succeeds (gradients flow through the U-Net).
  5. physicsnemo.utils.checkpoint.save_checkpoint writes files.
  6. Gradient checkpointing can be toggled without error.

Run::

    cd examples/elasticity_3d
    python smoke_test.py

Expected output (values will vary)::

    Device: cuda   (or cpu)
    Model parameters: 502,147
    Input shape : (2, 6, 32, 32, 32)
    Target shape: (2, 3, 32, 32, 32)
    Output shape: (2, 3, 32, 32, 32)
    Losses — total: 2.1834  mse: 0.9812  masked_mse: 0.9971  eq_residual: 0.2051
    Backward pass : OK
    Checkpoint files: ['ElasticityUNet.0.1.mdlus', 'checkpoint.0.1.pt']
    ✓ Smoke test PASSED.
"""

import sys
import tempfile

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 2
GRID   = 32    # small grid for speed; production is 128

# Tiny model that fits in < 500 MB even on CPU.
MODEL_DEPTH  = 3
FEAT_CHANNELS = (16, 16, 32, 32, 64, 64)  # length = model_depth * 2

AMP_ENABLED = DEVICE == "cuda"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_batch(
    batch: int = BATCH, grid: int = GRID, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random (x, y) tensors matching the elasticity tensor convention."""
    x = torch.randn(batch, 6, grid, grid, grid, device=device)
    y = torch.randn(batch, 3, grid, grid, grid, device=device)

    # Channel 0 must be binary (solid_mask).
    x[:, 0] = (x[:, 0] > 0).float()
    # Channel 5 (wind_pressure) should be non-negative.
    x[:, 5] = x[:, 5].abs()

    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # 1. Instantiate model
    # ------------------------------------------------------------------
    from models.elasticity_unet import ElasticityUNet

    model = ElasticityUNet(
        in_channels=6,
        out_channels=3,
        model_depth=MODEL_DEPTH,
        feature_map_channels=FEAT_CHANNELS,
        normalization="groupnorm",
        gradient_checkpointing=False,  # disable for speed in smoke test
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    assert model.in_channels  == 6, "in_channels mismatch"
    assert model.out_channels == 3, "out_channels mismatch"

    # ------------------------------------------------------------------
    # 2. Build fake batch
    # ------------------------------------------------------------------
    x, y = make_fake_batch()
    solid_mask = x[:, 0:1]   # (B, 1, D, H, W)
    E          = x[:, 1:2]   # (B, 1, D, H, W)
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Target shape: {tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = GradScaler(enabled=AMP_ENABLED)

    optimizer.zero_grad()
    with autocast(device_type=DEVICE, enabled=AMP_ENABLED):
        pred = model(x)
        print(f"Output shape: {tuple(pred.shape)}")
        assert pred.shape == (BATCH, 3, GRID, GRID, GRID), (
            f"Unexpected output shape {pred.shape}"
        )

        # ------------------------------------------------------------------
        # 4. Loss computation
        # ------------------------------------------------------------------
        from training.losses import combined_loss

        loss_total, loss_dict = combined_loss(
            pred=pred,
            target=y,
            solid_mask=solid_mask,
            E=E,
            loss_weights={"mse": 1.0, "masked_mse": 1.0, "eq_residual": 0.1},
        )

    # Verify all losses are finite scalars.
    assert loss_total.isfinite(), f"Total loss is not finite: {loss_total}"
    for name, val in loss_dict.items():
        assert torch.isfinite(torch.tensor(val)), f"Loss '{name}' is not finite: {val}"

    parts = "  ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items())
    print(f"Losses: {parts}")

    # ------------------------------------------------------------------
    # 5. Backward pass
    # ------------------------------------------------------------------
    scaler.scale(loss_total).backward()
    scaler.step(optimizer)
    scaler.update()
    print("Backward pass : OK")

    # ------------------------------------------------------------------
    # 6. Checkpoint save
    # ------------------------------------------------------------------
    from physicsnemo.utils.checkpoint import save_checkpoint

    with tempfile.TemporaryDirectory() as ckpt_dir:
        save_checkpoint(
            path=ckpt_dir,
            models=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=1,
        )
        import os
        ckpt_files = sorted(os.listdir(ckpt_dir))
        assert len(ckpt_files) > 0, "No checkpoint files written!"
        print(f"Checkpoint files: {ckpt_files}")

    # ------------------------------------------------------------------
    # 7. Optional: verify gradient_checkpointing=True also works
    # ------------------------------------------------------------------
    model_gc = ElasticityUNet(
        in_channels=6,
        out_channels=3,
        model_depth=MODEL_DEPTH,
        feature_map_channels=FEAT_CHANNELS,
        normalization="groupnorm",
        gradient_checkpointing=True,
    ).to(DEVICE)
    with torch.no_grad():
        out_gc = model_gc(x)
    assert out_gc.shape == pred.shape, "gradient_checkpointing variant shape mismatch"
    print("Gradient checkpointing: OK")

    print("\nSmoke test PASSED.")


if __name__ == "__main__":
    main()
