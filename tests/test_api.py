"""
Checks for the FastAPI microservice (src/api/app.py).

Uses FastAPI's TestClient (starlette + httpx under the hood) so the whole
app -- routing, pydantic validation, model loading via the lifespan hook,
dispatch into src/benchmark/registry.py -- is exercised exactly as it would
run under uvicorn, without starting a real server or network socket.

Same standalone-runnable convention as the rest of the suite: `pytest
tests/ -v`, or `python3 tests/test_api.py`. Skips (rather than fails) if the
trained model artifacts under models/ are not present, since those are
gitignored build products (`scripts/train_ml.py` creates them) and a clean
checkout legitimately might not have them yet.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from src.api.app import app
from src.simulator.data_generator import time_grid
from src.simulator.load_profiles import generate_load_profile
from src.simulator.motor_thermal import simulate_one_node, simulate_two_node
from src.simulator.params import OneNodeParams, TwoNodeParams
from src.simulator.sensors import add_noise

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_HAS_ONE_NODE_MODEL = os.path.exists(os.path.join(MODEL_DIR, "mlp_one_node.joblib"))
_HAS_TWO_NODE_MODEL = os.path.exists(os.path.join(MODEL_DIR, "mlp_two_node.joblib"))


def _one_node_payload(hA=15.0, T_ambient=25.0, method="mlp", n=None):
    t = time_grid()
    rng = np.random.default_rng(7)
    I_t, _ = generate_load_profile("step", t, rng)
    T = simulate_one_node(OneNodeParams(hA=hA, T_ambient=T_ambient), t, I_t, T0=T_ambient)
    T_meas = add_noise(T, 0.5, rng)
    sl = slice(0, n) if n else slice(None)
    return {
        "t": t[sl].tolist(), "I": I_t[sl].tolist(), "T_measured": T_meas[sl].tolist(),
        "T_ambient": T_ambient, "method": method,
    }


def _two_node_payload(hA=15.0, k_wh=35.0, T_ambient=25.0, method="mlp"):
    t = time_grid()
    rng = np.random.default_rng(8)
    I_t, _ = generate_load_profile("duty_cycle", t, rng)
    T = simulate_two_node(TwoNodeParams(hA=hA, k_wh=k_wh, T_ambient=T_ambient), t, I_t,
                          T0=np.array([T_ambient, T_ambient]))
    T_w = add_noise(T[:, 0], 0.5, rng)
    T_h = add_noise(T[:, 1], 0.5, rng)
    return {
        "t": t.tolist(), "I": I_t.tolist(), "T_w_measured": T_w.tolist(), "T_h_measured": T_h.tolist(),
        "T_ambient": T_ambient, "method": method,
    }


# ---------------------------------------------------------------------------

def test_health_reports_model_availability():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"]["one_node"] == _HAS_ONE_NODE_MODEL
        assert body["models_loaded"]["two_node"] == _HAS_TWO_NODE_MODEL


def test_methods_lists_all_six_with_metadata():
    with TestClient(app) as client:
        r = client.get("/methods")
        assert r.status_code == 200
        methods = r.json()["methods"]
        assert set(methods) == {"ga", "pso", "lm", "ekf", "bayesopt", "mlp"}
        assert methods["ekf"]["streaming_capable"] is True
        assert methods["mlp"]["stochastic"] is False
        # every classical baseline is always available; mlp only if trained
        assert set(methods["lm"]["available_for"]) == {"one_node", "two_node"}
        expected_mlp = ({"one_node"} if _HAS_ONE_NODE_MODEL else set()) | ({"two_node"} if _HAS_TWO_NODE_MODEL else set())
        assert set(methods["mlp"]["available_for"]) == expected_mlp


def test_calibrate_one_node_with_lm_returns_hA():
    """LM needs no trained model, so this exercises the full request path
    (validation, dispatch, calibration, response schema) regardless of
    whether models/ has been populated yet."""
    with TestClient(app) as client:
        r = client.post("/calibrate/one_node", json=_one_node_payload(hA=15.0, method="lm"))
        assert r.status_code == 200
        body = r.json()
        assert body["testbed"] == "one_node"
        assert body["method"] == "lm"
        assert "hA" in body["params"]
        assert abs(body["params"]["hA"] - 15.0) / 15.0 < 0.2
        assert body["runtime_ms"] >= 0
        assert body["n_evals"] > 0


def test_calibrate_two_node_with_lm_returns_both_constants():
    with TestClient(app) as client:
        r = client.post("/calibrate/two_node", json=_two_node_payload(hA=15.0, k_wh=35.0, method="lm"))
        assert r.status_code == 200
        params = r.json()["params"]
        assert set(params) == {"hA", "k_wh"}
        assert abs(params["hA"] - 15.0) / 15.0 < 0.3
        assert abs(params["k_wh"] - 35.0) / 35.0 < 0.3


def test_calibrate_one_node_with_mlp_when_available():
    if not _HAS_ONE_NODE_MODEL:
        print("SKIP  test_calibrate_one_node_with_mlp_when_available (no trained model)")
        return
    with TestClient(app) as client:
        r = client.post("/calibrate/one_node", json=_one_node_payload(hA=15.0, method="mlp"))
        assert r.status_code == 200
        body = r.json()
        assert body["n_evals"] == 0  # headline: zero simulator rollouts
        assert abs(body["params"]["hA"] - 15.0) / 15.0 < 0.2


def test_calibrate_rejects_unknown_method():
    with TestClient(app) as client:
        r = client.post("/calibrate/one_node", json=_one_node_payload(method="not_a_real_method"))
        assert r.status_code == 422  # pydantic field_validator rejection


def test_calibrate_rejects_mismatched_lengths():
    with TestClient(app) as client:
        payload = _one_node_payload(method="lm")
        payload["I"] = payload["I"][:-5]  # now shorter than t / T_measured
        r = client.post("/calibrate/one_node", json=payload)
        assert r.status_code == 422


def test_calibrate_one_node_mlp_unavailable_returns_503_if_untrained():
    """If a model genuinely is not trained, requesting it must fail loudly
    (503) rather than silently falling back to a different method -- a
    caller asking for 'mlp' needs to know it did not get 'mlp'."""
    with TestClient(app) as client:
        from src.api import app as app_module
        original = app_module._state["methods"].get("mlp")
        app_module._state["methods"]["mlp"] = {"one_node": None, "two_node": None}
        try:
            r = client.post("/calibrate/one_node", json=_one_node_payload(method="mlp"))
            assert r.status_code == 503
        finally:
            if original is not None:
                app_module._state["methods"]["mlp"] = original
            else:
                del app_module._state["methods"]["mlp"]


def test_calibrate_one_node_short_window_still_works():
    """A short/streaming-style window (well under the full 601-sample run)
    must still produce a response -- this is the partial-convergence regime
    the whole project is framed around."""
    with TestClient(app) as client:
        r = client.post("/calibrate/one_node", json=_one_node_payload(hA=15.0, method="lm", n=60))
        assert r.status_code == 200
        assert "hA" in r.json()["params"]


def _run_all():
    tests = [
        test_health_reports_model_availability,
        test_methods_lists_all_six_with_metadata,
        test_calibrate_one_node_with_lm_returns_hA,
        test_calibrate_two_node_with_lm_returns_both_constants,
        test_calibrate_one_node_with_mlp_when_available,
        test_calibrate_rejects_unknown_method,
        test_calibrate_rejects_mismatched_lengths,
        test_calibrate_one_node_mlp_unavailable_returns_503_if_untrained,
        test_calibrate_one_node_short_window_still_works,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} tests FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
