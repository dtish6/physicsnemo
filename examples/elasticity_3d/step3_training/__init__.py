# Step 3 — Training (pairing).
#
# Pairs each preprocessed input x with its FEA target y, normalizes, and
# trains the U-Net surrogate.
from ._3_1_dataset import ElasticityDataset
from ._3_2_transforms import ChannelNormalize
from ._3_3_elasticity_unet import ElasticityUNet
from ._3_4_losses import (
    displacement_mse,
    masked_mse,
    equilibrium_residual,
    combined_loss,
)

# Trainer is imported lazily to avoid pulling in optional deps (tensorboard)
# at package import time.  Use: from step3_training._3_5_trainer import Trainer
__all__ = [
    "ElasticityDataset",
    "ChannelNormalize",
    "ElasticityUNet",
    "displacement_mse",
    "masked_mse",
    "equilibrium_residual",
    "combined_loss",
]
