# ParamsCalibrator — Technical Proposal

**Siemens Curious Minds Software Summer School 2026 — Track: Procesare Semnal & Algoritmi**
**Author:** Alexandra | **Deadline:** 28 August 2026, 13:00

---

## 1. Track Justification

The project was initially scoped under Digital Twins & Platforme, but on review the fit is weak: that track's description centers on *creating the virtual replica itself* — a real-time-running simulation of a physical system, with GPU acceleration and large data flows. This project doesn't build a running replica; it calibrates the physical constants that feed into one. The literal match is Procesare Semnal & Algoritmi: "achiziția și procesarea datelor, optimizare matematică și dezvoltare de modele simplificate pentru fenomene complexe" — which is a near-exact description of what this project does (raw sensor data in, mathematical optimization/calibration, an ML surrogate as the simplified model). Repositioning under this track removes the need to stretch the framing and lets the project be evaluated on what it actually is.

## 2. Problem Statement

Calibrating physical constants (heat transfer coefficients, thermal resistances, damping ratios, friction coefficients, etc.) from sensor data is a recurring engineering problem: a model is only as good as the parameters fed into it, and those parameters drift with wear, manufacturing variance, and changing operating conditions. Today this calibration typically relies on iterative optimization — genetic algorithms, gradient-based least squares, sequential filters, or other heuristic/numerical solvers — each re-run from scratch, at real computational cost, every time new data needs to be processed.

**Opportunity:** replace the repeated iterative search with a trained ML model that maps a window of raw sensor data directly to calibrated constants in a single forward pass, and rigorously benchmark it against the standard families of classical calibration methods — not just genetic algorithms, but the full landscape actually used in engineering practice.

## 3. Proposed Solution

A machine learning model trained to perform parameter calibration, benchmarked against five classical calibration methods across multiple evaluation axes, with a fast callable API on top:

- **Input:** a window of raw/noisy sensor readings (e.g. temperature over time under a known load profile) plus known excitation data (applied load, ambient conditions).
- **Output:** the calibrated physical constant(s) (e.g. heat transfer coefficient, thermal resistance).
- **Claim to validate:** the ML approach matches or exceeds classical calibration methods on accuracy, while being substantially faster — because it does its expensive work once, offline, during training, and every subsequent calibration is a cheap forward pass. This is the core comparison the project must demonstrate, not just assert.

## 4. Testbed: Motor Winding Thermal Model

No proprietary Siemens data or hardware is available for this project, so the testbed uses **synthetic data generated from a physically realistic model**, with parameters drawn from ranges representative of published motor design literature (e.g. thermal classes and time constants referenced in IEC 60034) rather than confidential figures — stated explicitly as a modeling assumption.

**System:** an electric motor winding heating under load, cooled convectively — a concrete, well-documented instance of the general calibration problem, relevant to Siemens' motors & drives business.

**1-parameter model (lumped, single node):**

```
C · dT/dt = I(t)² · R_winding − h·A · (T − T_ambient)
```

Calibration target: `h·A` (lumped heat transfer coefficient).

**Multi-parameter model (2-node: winding + housing):** adds a second thermal mass and a winding→housing conduction path, giving 2–3 simultaneous constants to calibrate — this variant drives the *scalability* comparison (Section 6).

**Data generation:** simulate multiple load profiles (step loads, duty cycles, ramps), sample ground-truth constants from realistic ranges, add Gaussian sensor noise at multiple noise levels, and hold out a test set the ML model never trains on.

## 5. Baseline Calibration Methods

Five methods, chosen to span the distinct families actually used for calibration in engineering practice, so the comparison covers the field rather than one convenient baseline:

| Method | Family | Why included |
|---|---|---|
| Genetic Algorithm (GA) | Population-based heuristic | The original baseline; standard evolutionary approach |
| Particle Swarm Optimization (PSO) | Population-based heuristic | A second, distinct metaheuristic — shows the ML model beats the heuristic family, not one member of it |
| Levenberg-Marquardt (LM) | Classical nonlinear regression (gradient-based least squares) | The default choice for most engineers doing curve-fit calibration; fits the known physics equation's exact form to each new curve, one calibration at a time — the most direct conceptual sibling to the ML model (Section 6) |
| Extended Kalman Filter (EKF) | Sequential/online estimation | The classical method actually designed for real-time, streaming parameter updates — the most direct competitor to the ML model's speed and streaming claims |
| Bayesian Optimization | Modern sample-efficient, gradient-free | Represents modern practice for expensive-to-evaluate calibration problems |

## 6. ML Model: Design

**Architecture: a small MLP over physically-motivated engineered features**, not a raw-signal deep network. The underlying physics here is a simple, well-behaved thermal response (a rise toward a plateau), so there's little to gain from a heavier architecture (e.g. CNN) built for finding patterns in complex or high-dimensional raw waveforms — a compact feature set is both sufficient and far easier to train well with limited time.

**Input features**, extracted per sensor window:

- initial slope of the temperature rise
- estimated steady-state temperature (when the load is held long enough)
- estimated thermal time constant (time to reach ~63% of the total temperature change — the standard first-order-system characterization)
- applied current and ambient temperature for that run
- noise-level indicator (local variance), so the model can gauge data reliability

These aren't arbitrary: physics already gives a sanity-checkable relationship at steady state, `T_ss − T_ambient = I²R / (h·A)`, so `h·A` can in principle be solved for directly with algebra when the load is held steady long enough. The MLP's job is to learn the more general, robust version of that relationship — one that still holds under noisy data, short/incomplete windows, and non-steady-state conditions where the closed form doesn't directly apply. That framing also makes the model easy to explain in the presentation: it's not a black box, it's a learned generalization of a known physical formula.

**Network:** features (≈8–12 inputs) → Dense(32, ReLU) → Dense(16, ReLU) → output layer with one unit per calibrated constant (1 for the single-node testbed, 2–3 for the multi-node one). Trained with MSE loss, normalized per constant (since different constants live on very different numeric scales), using Adam. Small enough to train in seconds to a couple of minutes on CPU.

**Conceptual note for the presentation:** this model, and LM (Section 5), are both technically "nonlinear regression" — LM fits the known physics equation's exact form to each new curve from scratch every time; the MLP learns a flexible general-purpose approximation once, from thousands of simulated examples, and then only ever evaluates. Same mathematical category, different strategy — a clean way to frame the comparison for a non-technical audience.

**Fallback if needed:** an MLP over the flattened raw time-series window (no feature engineering) is a simpler but heavier-data, harder-to-explain alternative — kept as Plan B only.

**Stretch goal (not core):** Gaussian Process Regression as an alternative nonlinear regression model, which gives a confidence interval on each calibrated constant for free — useful if there's time to explore uncertainty-aware calibration, but not required for the core pitch.

## 7. Evaluation Metrics

All six methods (5 baselines + ML model) are benchmarked on the same synthetic test cases across:

1. **Accuracy** — calibration error vs. ground-truth constant(s).
2. **Speed / latency** — wall-clock time to produce a calibrated value.
3. **Convergence robustness** — success rate and variance across different random initial guesses and noise levels (heuristic and gradient methods can fail to converge or get stuck; this exposes that honestly).
4. **Scalability** — how each metric changes moving from the 1-parameter to the 2–3 parameter testbed.
5. **Real-time/streaming suitability** — whether the method can update incrementally as new sensor data arrives vs. requiring a full batch re-run. EKF is expected to be genuinely competitive here, which strengthens the story's credibility rather than weakening it — a "the ML model wins everything" result would actually be less convincing.

## 8. System Architecture

```
┌─────────────────────┐
│  Physics Simulator   │  motor thermal model (1-node & 2-node)
│  + Synthetic Data Gen │  → load profiles, noisy sensor data, ground truth
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│              Benchmark Harness                │
│  runs GA / PSO / LM / EKF / BayesOpt / MLP    │
│  on identical test cases, logs all 5 metrics  │
└──────────┬────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐      ┌───────────────────────────┐
│  Results & Charts     │      │  Feature Extraction         │
│  (for presentation)   │      │  → Trained MLP               │
└──────────────────────┘      │  → FastAPI wrapper           │
                               │  → Dockerized microservice   │
                               │  (live demo: POST sensor     │
                               │   data → calibrated value)   │
                               └───────────────────────────┘
```

## 9. Tech Stack

- **Simulation & baselines:** Python, NumPy/SciPy (`scipy.optimize.least_squares` for LM), a lightweight custom or library-based GA/PSO, `filterpy` for EKF, `scikit-optimize` or `bayes_opt` for Bayesian Optimization.
- **ML model:** small MLP over engineered features (scikit-learn `MLPRegressor` or a minimal PyTorch model — either is light enough here; scikit-learn is the simpler choice given the small network size).
- **Microservice:** FastAPI + Docker.
- **Visualization:** Python (matplotlib/plotly) for benchmark charts; final presentation slides.

## 10. Milestone Plan (12 days to 28 Aug, 13:00)

| Days | Milestone |
|---|---|
| 1–2 | Physics simulator + synthetic data generator working end-to-end; validated against sanity checks |
| 3–4 | Baseline methods implemented (GA, PSO, LM) and producing calibrations on synthetic data |
| 5 | EKF + Bayesian Optimization baselines added |
| 6–7 | Feature extraction pipeline + MLP trained and validated on held-out data |
| 8 | Full benchmark harness run across all 6 methods × 5 metrics × both testbed variants |
| 9 | Dockerized FastAPI microservice wrapping the MLP; live demo working |
| 10 | Result charts finalized; sanity-check numbers against expectations |
| 11 | Presentation built (10 minutes, emphasis on problem + solution per assignment requirements) |
| 12 | Buffer / rehearsal |

## 11. Deliverables

1. Working code repository (`ParamsCalibrator`): simulator, baselines, ML model, benchmark harness.
2. Dockerized microservice with a working API demo.
3. Benchmark results and comparison charts.
4. 10-minute presentation.

## 12. Risks & Mitigations

- **Scope too large for 12 days** → build in the milestone order above; a working 1-node testbed with all 6 methods compared on accuracy + speed is the minimum viable result if time runs short — robustness/scalability/streaming axes can be trimmed first if needed.
- **"ML wins everything" reads as unconvincing** → deliberately keep and present the EKF streaming result even if it's competitive with the MLP; a nuanced result is more credible than a sweep.
- **No real Siemens data** → stated explicitly as a modeling assumption; testbed grounded in realistic, publicly-referenced parameter ranges rather than invented ones.
- **Track change from what was validated with Siemens** → worth a short heads-up to whoever validated the Digital Twins direction, since the underlying technical work is unchanged but the framing has shifted to a track that fits it more literally.
