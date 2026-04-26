"""
TechPulse Spindle RDO Framework
================================

A modular framework for Robust Design Optimization of CNC lathe spindles.

Modules:
    - design_variables: Design space definition
    - lhs_sampler: Latin Hypercube Sampling
    - fea_pool_runner: Batch FEA execution
    - ml_surrogate: ML surrogate models
    - robust_optimizer: GA with Taguchi S/N ratio
    - inverse_design: DNN inverse mapping
"""

__version__ = "1.0.0"
__author__ = "Manus AI"

from .design_variables import DesignSpace, VariableBounds
from .lhs_sampler import LHSSampler
from .ml_surrogate import SurrogateModel
from .robust_optimizer import RobustOptimizer
from .inverse_design import InverseDesignEngine

__all__ = [
    "DesignSpace",
    "VariableBounds",
    "LHSSampler",
    "SurrogateModel",
    "RobustOptimizer",
    "InverseDesignEngine",
]
