# ParamsCalibrator

Siemens Curious Minds Software Summer School 2026 — track: **Procesare Semnal & Algoritmi**.

ML-based calibration of physical constants (motor winding heat-transfer coefficient) from
sensor data, benchmarked against classical calibration methods (GA, PSO, Levenberg-Marquardt,
EKF, Bayesian Optimization) across accuracy, speed, convergence robustness, scalability, and
real-time/streaming suitability. See `docs/proposal_en.md` / `docs/proposal_ro.md` for the full
technical proposal.

## Status

- [x] Physics simulator (1-node & 2-node motor thermal model) + synthetic data generator
- [x] Baseline calibration methods (GA, PSO, LM, EKF, Bayesian Optimization)
- [x] MLP calibration model
- [x] Benchmark harness
- [x] FastAPI microservice + Docker

## Project layout

```
src/
  simulator/    physics model, load profiles, synthetic dataset generation
  baselines/    GA / PSO / LM / EKF / Bayesian Optimization calibrators
  ml/           physics-motivated feature extraction + MLP calibration model
  benchmark/    benchmark harness across all methods x all 5 evaluation axes
  api/          FastAPI microservice wrapping every calibrator
tests/          pytest sanity checks
scripts/        CLI entry points (generate data, train model, run benchmark)
data/           generated synthetic datasets (gitignored)
models/         trained model artifacts (gitignored)
reports/
  figures/            validation + benchmark comparison plots
  benchmark_results/  raw per-run CSVs + summary.json (gitignored)
```

Every calibrator — the five classical baselines and the MLP alike — exposes the same
`calibrate_one_node(...)` / `calibrate_two_node(...)` interface and returns the same
`CalibrationResult`, so the benchmark harness can run all six methods through identical code.

## Setup

Managed with [uv](https://docs.astral.sh/uv/). This creates an isolated `.venv/` and installs
the exact versions pinned in `uv.lock`:

```
uv sync
```

(A plain `pip install -r requirements.txt` into your own virtualenv also works, if you'd rather
not use uv.)

## Generate the synthetic datasets

```
uv run python scripts/generate_datasets.py
```

## Train the ML calibration model

```
uv run python scripts/train_ml.py
```

## Run the benchmark

```
uv run python scripts/run_benchmark.py --quick   # ~1 min, pipeline smoke-test only
uv run python scripts/run_benchmark.py           # ~20-30 min, presentation-quality numbers
```

Runs all six methods (GA, PSO, LM, EKF, Bayesian Optimization, MLP) through the identical
`calibrate_one_node` / `calibrate_two_node` interface on the same held-out test runs, and scores
all five axes from the proposal (accuracy, speed, convergence robustness, scalability,
streaming/real-time suitability). Writes raw per-calibration CSVs and a `summary.json` to
`reports/benchmark_results/`, and a 4-panel comparison figure per testbed to `reports/figures/`.
Most of the runtime is Bayesian Optimization (each calibration fits a fresh Gaussian Process) and
the population heuristics (GA/PSO run hundreds of full simulator rollouts per calibration).

## Run the API

```
uv run uvicorn src.api.app:app --reload
```

```
curl -X POST http://localhost:8000/calibrate/one_node \
  -H "Content-Type: application/json" \
  -d '{"t": [0, 5, 10, ...], "I": [8, 8, 8, ...], "T_measured": [25.1, 25.4, ...], "T_ambient": 25.0}'
```

With the server running, `http://localhost:8000/demo` serves a browser UI built for live demos.
Press **Start the motor** and the run replays as an animation -- the plot reveals itself sample by
sample while a little motor mascot heats up, glows and starts sweating in step with its own winding
temperature -- then every ticked method is calibrated on that same window through the ordinary
`/calibrate/...` endpoints.

Two data sources:

- **Held-out test set** -- runs from `data/motor_*_test.npz`: virtual motors with constants the
  models never saw during training. This is the honest accuracy story. There is deliberately no run
  picker -- each press draws fresh runs at random, so nothing on stage looks hand-picked; the pills
  above the plot name which runs came up.
- **Design a motor** -- you dial `hA` (and `k_wh`), the load profile and current, ambient, sensor
  noise and run length; `/demo/synth/...` simulates that motor through the same physics used to
  build the training set, and the calibrators have to recover the numbers you just chose. Dial
  `hA = 18.5` and watching the MLP answer `18.56` in a few milliseconds is the most convincing
  thirty seconds of the whole demo.

**Motors in this batch** runs the same comparison over N motors instead of one (N different
held-out runs, or N noise realisations of your own motor). The table then reports median and worst
error per method, and an extra **Error spread** panel plots one dot per motor so the distribution
is visible rather than a single lucky or unlucky draw.

The other panels: **Thermogram** shows the measured channels, the noise-free truth dashed
underneath, and the temperature curve each method's *estimated* constants predict, re-integrated
through the simulator via `/demo/reconstruct/...` -- a bad calibration is visible as a curve peeling
away from the sensor data. **Load profile** shows the driving current `I(t)`. **Latency race** is
log-scale bars, where the MLP at single-digit milliseconds sits next to PSO/GA at tens of seconds.

Note that GA, PSO and BayesOpt genuinely take tens of seconds per request; leave them unticked
(and keep the batch size small) unless the latency gap is the point you want to make -- the UI warns
you with a rough time estimate when a slow method is ticked.

The held-out mode needs the generated datasets under `data/` (`scripts/generate_datasets.py`),
which it reads through `/demo/sample/...` purely so it can score the answer -- the calibration
endpoints themselves stay data-free.

`method` defaults to `"mlp"` (the fast, offline-trained default) but can be set to any of `ga`,
`pso`, `lm`, `ekf`, `bayesopt` -- useful for a live demo showing the same latency gap the
benchmark measures offline, request for request. `GET /methods` lists every method's family,
whether it is stochastic, and whether it is streaming-capable; `GET /health` reports which
trained MLP models were found under `models/`.

```
docker build -t paramscalibrator .
docker run -p 8000:8000 paramscalibrator
```

The image generates its own training data and trains its own models during the build (no
pre-built artifacts or external data are copied in), so it is fully reproducible from source.

## Run tests

```
uv run pytest tests/ -v
```

## ML model: where the learning actually helps

The MLP does not predict the physical constants directly. It predicts a *correction* to a
closed-form estimator derived from the energy balance of the same ODE the simulator
integrates — recovered as the slope of a straight-line fit over the whole window, which makes
it exact for any load profile and lets the unknown initial temperature drop out as the fitted
intercept. Anchoring the network to that estimator instead of the raw constant is what makes
it beat the physics rather than re-derive it.

That framing also locates the ML model's value honestly. On a full, settled 3000s window the
closed form is already near-optimal and the MLP only matches it; the gap opens up as the
window shortens and the run no longer approaches steady state — which is exactly the regime a
streaming calibration deployment operates in:

| Window | `hA` closed-form MAPE | `hA` MLP MAPE |
|---|---|---|
| 10% (300s) | 58.3% | **23.5%** |
| 20% (600s) | 21.1% | **13.5%** |
| 50% (1500s) | 5.0% | **4.7%** |
| 100% (3005s) | **1.8%** | 1.8% |

The advantage is also data-hungry, with a sharp knee: the correction is worth nothing at 800
training runs and ~35% at 1500+.

## Modeling assumptions

No real Siemens hardware or proprietary data is used. The testbed is a synthetic motor winding
thermal model with constants drawn from illustrative, representative ranges for a small/medium
industrial motor (order-of-magnitude consistent with published motor thermal design literature,
e.g. IEC 60034 thermal classes) — not exact manufacturer or standard figures. This is stated
explicitly as a modeling assumption in the project proposal.
