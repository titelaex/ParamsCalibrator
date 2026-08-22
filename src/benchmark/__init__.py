"""
Benchmark harness: runs all six calibration methods (GA, PSO, LM, EKF,
Bayesian Optimization, MLP) through the identical interface every method
already exposes (src/baselines/, src/ml/model.py) and scores them on the
five axes from the project proposal (Section 7): accuracy, speed,
convergence robustness, scalability (1-node vs 2-node), and streaming
suitability.
"""

from src.benchmark import harness, metrics, registry
from src.benchmark.registry import METHOD_METADATA, build_methods

__all__ = ["harness", "metrics", "registry", "build_methods", "METHOD_METADATA"]
