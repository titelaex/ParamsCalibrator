"""
FastAPI microservice wrapping the trained ML calibrator -- and, for
comparison/demo purposes, every classical baseline -- behind one HTTP API.

This is the "single forward pass" half of the project's central claim made
concrete: POST a window of sensor data, get a calibrated constant back in
low-single-digit milliseconds via the MLP, versus the hundreds of
milliseconds to several seconds every classical baseline needs for the same
window (see reports/benchmark_results/ for the measured numbers). Selecting
`method` to a classical baseline is left available specifically so a live
demo can show that latency gap request-for-request, rather than only citing
the offline benchmark -- GA/PSO/BayesOpt genuinely take that long per call,
which is itself part of the demo.
"""

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import (
    CalibrationResponse,
    HealthResponse,
    MethodInfo,
    MethodsResponse,
    OneNodeCalibrationRequest,
    TwoNodeCalibrationRequest,
)
from src.benchmark.registry import METHOD_METADATA, build_methods
from src.ml.model import MLPCalibrator

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Mutable at runtime (populated at startup / by _load_models), not a
# constant -- deliberately module-level rather than FastAPI app.state so
# tests can call _load_models() directly without spinning up the app.
_state = {"methods": {}, "models_loaded": {"one_node": False, "two_node": False}}


def _load_models():
    mlp_one_node = mlp_two_node = None
    path_1 = os.path.join(MODEL_DIR, "mlp_one_node.joblib")
    path_2 = os.path.join(MODEL_DIR, "mlp_two_node.joblib")
    if os.path.exists(path_1):
        mlp_one_node = MLPCalibrator.load(path_1)
        _state["models_loaded"]["one_node"] = True
    if os.path.exists(path_2):
        mlp_two_node = MLPCalibrator.load(path_2)
        _state["models_loaded"]["two_node"] = True
    _state["methods"] = build_methods(mlp_one_node=mlp_one_node, mlp_two_node=mlp_two_node)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="ParamsCalibrator",
    description="Calibrates motor-winding thermal constants (hA, k_wh) from a window of sensor data.",
    version="0.1.0",
    lifespan=lifespan,
)


def _dispatch(testbed, method, run):
    """Look up `method` for `testbed`, run it, and translate failures into HTTP errors."""
    fns = _state["methods"].get(method)
    if fns is None:
        raise HTTPException(400, f"unknown method {method!r}")
    fn = fns.get(testbed)
    if fn is None:
        raise HTTPException(
            503,
            f"method {method!r} is not available for {testbed} "
            f"(its trained model was not found under models/ -- run scripts/train_ml.py)",
        )
    try:
        return run(fn)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 -- surfaced to the client as a 422, not a stack trace
        raise HTTPException(422, f"calibration failed: {e}") from e


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models_loaded=dict(_state["models_loaded"]))


@app.get("/methods", response_model=MethodsResponse)
def methods():
    out = {}
    for name, meta in METHOD_METADATA.items():
        fns = _state["methods"].get(name, {})
        available = [tb for tb in ("one_node", "two_node") if fns.get(tb) is not None]
        out[name] = MethodInfo(available_for=available, **meta)
    return MethodsResponse(methods=out)


def _check_lengths(req):
    try:
        req.check_lengths()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/calibrate/one_node", response_model=CalibrationResponse)
def calibrate_one_node(req: OneNodeCalibrationRequest):
    _check_lengths(req)
    t = np.asarray(req.t, dtype=float)
    I = np.asarray(req.I, dtype=float)
    T = np.asarray(req.T_measured, dtype=float)

    def run(fn):
        result = fn(t, I, T, req.T_ambient, rng=np.random.default_rng(0))
        return CalibrationResponse(
            testbed="one_node", method=req.method, params=result.params,
            runtime_ms=result.runtime_s * 1000.0, n_evals=result.n_evals, converged=result.converged,
        )

    return _dispatch("one_node", req.method, run)


@app.post("/calibrate/two_node", response_model=CalibrationResponse)
def calibrate_two_node(req: TwoNodeCalibrationRequest):
    _check_lengths(req)
    t = np.asarray(req.t, dtype=float)
    I = np.asarray(req.I, dtype=float)
    T_w = np.asarray(req.T_w_measured, dtype=float)
    T_h = np.asarray(req.T_h_measured, dtype=float)

    def run(fn):
        result = fn(t, I, T_w, T_h, req.T_ambient, rng=np.random.default_rng(0))
        return CalibrationResponse(
            testbed="two_node", method=req.method, params=result.params,
            runtime_ms=result.runtime_s * 1000.0, n_evals=result.n_evals, converged=result.converged,
        )

    return _dispatch("two_node", req.method, run)
