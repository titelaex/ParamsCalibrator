"""
Registry of all six calibration methods (5 classical baselines + the MLP),
exposed uniformly as {"one_node": calibrate_fn, "two_node": calibrate_fn}
dicts so the benchmark harness (src/benchmark/harness.py) can loop over
every method identically, regardless of whether it is a stateless module
function (GA/PSO/LM/EKF/BayesOpt) or a bound method on a trained model
(MLP).

Method metadata records what the "convergence robustness" and "streaming
suitability" axes (proposal, Section 7, metrics 3 and 5) need without
re-deriving it from behavior:

- `stochastic`: whether the method's outcome depends on a random draw (a
  randomized initial guess/population/swarm), so repeated calls under the
  *same* run and noise level can legitimately land on different estimates.
  The MLP is the one method for which this is False -- a trained network is
  a fixed function of its input, so it has zero restart-to-restart variance
  by construction. That is itself a headline robustness result, not an
  artifact of not testing it properly.
- `streaming_capable`: whether the method updates its estimate on one new
  sample without re-solving from scratch. Only EKF is True here: it
  maintains a running state and processes each sample in O(1) work. The MLP
  is fast to re-run, but every re-run still recomputes features over the
  whole current window rather than folding in just the new sample -- a
  real but different kind of cheapness, not the same thing.
"""

from src.baselines import bayesopt, ekf, ga, lm, pso

METHOD_METADATA = {
    "ga":       {"family": "population heuristic",       "stochastic": True,  "streaming_capable": False},
    "pso":      {"family": "population heuristic",       "stochastic": True,  "streaming_capable": False},
    "lm":       {"family": "gradient least-squares",      "stochastic": True,  "streaming_capable": False},
    "bayesopt": {"family": "sample-efficient global",     "stochastic": True,  "streaming_capable": False},
    "ekf":      {"family": "sequential/online",           "stochastic": True,  "streaming_capable": True},
    "mlp":      {"family": "learned (offline-trained)",   "stochastic": False, "streaming_capable": False},
}

_BASELINE_MODULES = {"ga": ga, "pso": pso, "lm": lm, "ekf": ekf, "bayesopt": bayesopt}


def build_methods(mlp_one_node=None, mlp_two_node=None) -> dict:
    """Build the {method_name: {"one_node": fn, "two_node": fn}} registry.

    `mlp_one_node` / `mlp_two_node` are trained `MLPCalibrator` instances
    (src/ml/model.py); pass None to omit the ML method entirely (e.g. when
    no trained model is available yet).
    """
    methods = {
        name: {"one_node": mod.calibrate_one_node, "two_node": mod.calibrate_two_node}
        for name, mod in _BASELINE_MODULES.items()
    }
    if mlp_one_node is not None or mlp_two_node is not None:
        methods["mlp"] = {
            "one_node": mlp_one_node.calibrate_one_node if mlp_one_node is not None else None,
            "two_node": mlp_two_node.calibrate_two_node if mlp_two_node is not None else None,
        }
    return methods
