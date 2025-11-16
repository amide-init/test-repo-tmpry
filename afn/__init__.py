"""
AFN (Adaptive Fidelity Nexus) Framework - Ensemble Implementation
"""

from .afn_core import AFNCore, EnsembleRegressor
from .comparison_algorithms import GA, PSO, ACO

__version__ = "1.0.0"
__author__ = "AFN Research Team"

__all__ = [
    "AFNCore",
    "EnsembleRegressor", 
    "GA",
    "PSO",
    "ACO"
]