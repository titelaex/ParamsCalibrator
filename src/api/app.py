

import os

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from src.api import demo_data

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
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


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


# --------------------------------------------------------------------------
# Demo UI: a small static page (static/index.html) that replays runs from the
# held-out test set through the same /calibrate endpoints an external client
# would call, and shows the estimate next to the known ground truth.
# --------------------------------------------------------------------------


@app.get("/demo/runs/{testbed}")
def demo_runs(testbed: str):
    """How many held-out test runs are available for `testbed`."""
    if testbed not in ("one_node", "two_node"):
        raise HTTPException(404, f"unknown testbed {testbed!r}")
    if not demo_data.available(testbed):
        raise HTTPException(
            503,
            f"no test dataset for {testbed} under data/ -- run scripts/generate_datasets.py",
        )
    return {"testbed": testbed, "n_runs": demo_data.n_runs(testbed)}


@app.get("/demo/sample/{testbed}")
def demo_sample(
    testbed: str,
    index: int = Query(0, ge=0, description="which held-out test run to replay"),
    window: float = Query(1.0, gt=0.0, le=1.0, description="fraction of the run to expose"),
):
    """One held-out run, truncated to the first `window` fraction of its samples.

    Includes the ground-truth constants so the UI can score the estimate; a
    real client would only have the sensor channels.
    """
    if testbed not in ("one_node", "two_node"):
        raise HTTPException(404, f"unknown testbed {testbed!r}")
    if not demo_data.available(testbed):
        raise HTTPException(
            503,
            f"no test dataset for {testbed} under data/ -- run scripts/generate_datasets.py",
        )
    try:
        return demo_data.sample_run(testbed, index, window)
    except IndexError as e:
        raise HTTPException(404, str(e)) from e


class ReconstructRequest(BaseModel):
    """Demo-only: replay the physics with an estimated parameter set."""

    t: list[float] = Field(..., min_length=2)
    I: list[float] = Field(..., min_length=2)
    T_ambient: float
    T0: list[float] = Field(..., min_length=1, description="initial temperature(s), degC")
    hA: float
    k_wh: float | None = None


@app.post("/demo/reconstruct/{testbed}")
def demo_reconstruct(testbed: str, req: ReconstructRequest):
    """The temperature curve the estimated constants predict, for overlay on the plot."""
    if testbed not in ("one_node", "two_node"):
        raise HTTPException(404, f"unknown testbed {testbed!r}")
    if len(req.I) != len(req.t):
        raise HTTPException(422, "t and I must have equal length")
    if testbed == "two_node" and req.k_wh is None:
        raise HTTPException(422, "k_wh is required for the two_node testbed")
    try:
        return demo_data.reconstruct(
            testbed, req.t, req.I, req.T_ambient, req.T0, {"hA": req.hA, "k_wh": req.k_wh}
        )
    except Exception as e:  # noqa: BLE001 -- bad parameter values, not a server fault
        raise HTTPException(422, f"reconstruction failed: {e}") from e


@app.get("/demo/synth/{testbed}")
def demo_synth(
    testbed: str,
    hA: float = Query(..., gt=0, description="ground-truth hA to simulate, W/K"),
    k_wh: float = Query(30.0, gt=0, description="ground-truth k_wh, W/K (2-node only)"),
    T_ambient: float = Query(25.0, description="ambient temperature, degC"),
    noise_std: float = Query(0.5, ge=0, le=10, description="sensor noise std, degC"),
    profile: str = Query("step", description=f"load profile, one of {list(demo_data.PROFILES)}"),
    level: float = Query(8.0, ge=0, le=40, description="load current level, A"),
    duration: float = Query(3000.0, gt=0, le=20000, description="run length, seconds"),
    seed: int = Query(0, ge=0, description="sensor-noise seed"),
):
    """Simulate a motor with caller-chosen constants, so the demo can check
    whether the calibrators recover the numbers the caller just dialed in."""
    if testbed not in ("one_node", "two_node"):
        raise HTTPException(404, f"unknown testbed {testbed!r}")
    if profile not in demo_data.PROFILES:
        raise HTTPException(422, f"unknown profile {profile!r}; expected one of {list(demo_data.PROFILES)}")
    return demo_data.synth_run(testbed, hA, k_wh, T_ambient, noise_std, profile, level, duration, seed)


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/demo")
