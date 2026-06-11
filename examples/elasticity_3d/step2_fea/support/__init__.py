# Supporting files for step 2 (not part of the data-generation path):
#   fem_interface.py          — abstract backend interface + MockFEMSolver
#   benchmark.py              — solver performance benchmark
#   compare_preconditioners.py — Jacobi vs AMG comparison
#   tests/                    — solver correctness tests
from .fem_interface import FEMSolverInterface, MockFEMSolver

__all__ = ["FEMSolverInterface", "MockFEMSolver"]
