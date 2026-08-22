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
