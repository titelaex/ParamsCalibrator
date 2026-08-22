"""
The ML calibration model: physics-motivated feature extraction
(src/ml/features.py) feeding a small MLP (src/ml/model.py).

`MLPCalibrator` exposes the same `calibrate_one_node` / `calibrate_two_node`
interface as every classical baseline in src/baselines/, so the benchmark
harness can run all six methods through identical code.
"""

from src.ml.features import (
    ONE_NODE_FEATURE_NAMES,
    ONE_NODE_TARGET_NAMES,
    TWO_NODE_FEATURE_NAMES,
    TWO_NODE_TARGET_NAMES,
    build_one_node_xy,
    build_two_node_xy,
    extract_one_node_features,
    extract_two_node_features,
)
from src.ml.model import MLPCalibrator, regression_metrics

__all__ = [
    "MLPCalibrator",
    "regression_metrics",
    "build_one_node_xy",
    "build_two_node_xy",
    "extract_one_node_features",
    "extract_two_node_features",
    "ONE_NODE_FEATURE_NAMES",
    "TWO_NODE_FEATURE_NAMES",
    "ONE_NODE_TARGET_NAMES",
    "TWO_NODE_TARGET_NAMES",
]
