"""
Classical calibration baselines, one module per method, each exposing the
same two-function interface: `calibrate_one_node(...)` and
`calibrate_two_node(...)`, both returning a `CalibrationResult`
(src/baselines/base.py). This uniform shape is what lets the benchmark
harness (src/benchmark/, WIP) loop over every method identically.
"""

from src.baselines import bayesopt, ekf, ga, lm, pso
from src.baselines.base import CalibrationResult

METHODS = {
    "ga": ga,
    "pso": pso,
    "lm": lm,
    "ekf": ekf,
    "bayesopt": bayesopt,
}

__all__ = ["CalibrationResult", "METHODS", "ga", "pso", "lm", "ekf", "bayesopt"]
