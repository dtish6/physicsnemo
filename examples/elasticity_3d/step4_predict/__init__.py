# Step 4 — Prediction (inference + downstream use).
#
# Evaluation metrics for surrogate predictions, plus the topology-optimization
# driver that uses the trained surrogate as a fast forward model.
from ._4_1_metrics import (
    relative_l2_error,
    von_mises_stress,
    von_mises_stress_error,
    max_displacement_error,
    compute_all_metrics,
)
from .support.topo_opt_stub import TopologyOptimizer

__all__ = [
    "relative_l2_error",
    "von_mises_stress",
    "von_mises_stress_error",
    "max_displacement_error",
    "compute_all_metrics",
    "TopologyOptimizer",
]
