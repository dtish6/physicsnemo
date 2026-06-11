# Step 2 — FEA simulation (the ground-truth solver).
#
# 3D linear-elasticity voxel FEM used by step1_preprocess to generate the
# supervised displacement targets.
from .support.fem_interface import FEMSolverInterface, MockFEMSolver
from ._2_2_voxel_fem import VoxelFEMSolver

__all__ = ["FEMSolverInterface", "MockFEMSolver", "VoxelFEMSolver"]
