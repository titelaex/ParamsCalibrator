"""
Physical constants and sampling ranges for the motor winding thermal testbed.

IMPORTANT — modeling assumption: no proprietary Siemens data or hardware was
available for this project. The values below are illustrative, representative
orders of magnitude for a small/medium industrial motor, chosen to be
consistent with published motor thermal design literature (e.g. the general
scale of thermal classes and time constants referenced in IEC 60034). They are
NOT exact manufacturer or standard figures and should not be cited as such.
This is stated explicitly in the project proposal (docs/proposal_en.md).

Two testbed variants share the same physical scale:

1-node ("lumped"): the whole motor (winding + housing) is treated as a single
thermal mass. The single unknown to calibrate is the lumped heat-transfer
coefficient `hA` (housing/ambient interface, W/K).

2-node: winding and housing are separate thermal masses connected by a
conduction path. Two unknowns to calibrate: `k_wh` (winding->housing
conduction, W/K) and `hA` (housing->ambient convection, W/K).

Known / fixed quantities (assumed measurable independently, e.g. via a
standard DC winding-resistance test or from material properties) are kept
separate from the *unknown* quantities that the whole project is about
recovering from sensor data.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Known / fixed constants (assumed measurable independently of calibration)
# ---------------------------------------------------------------------------

R_WINDING_OHM = 1.8          # winding electrical resistance (DC resistance test)
C_WINDING_J_PER_K = 2500.0   # winding thermal capacitance (mass x specific heat of copper)
C_HOUSING_J_PER_K = 12000.0  # housing thermal capacitance (mass x specific heat of aluminium)

# 1-node model lumps both masses together.
C_LUMPED_J_PER_K = C_WINDING_J_PER_K + C_HOUSING_J_PER_K


# ---------------------------------------------------------------------------
# Unknown constants — this is what the whole project calibrates.
# Ranges represent realistic drift due to dust/fouling (lowers hA), degraded
# ventilation, insulation aging (affects k_wh), etc.
# ---------------------------------------------------------------------------

HA_RANGE_W_PER_K = (8.0, 25.0)      # housing -> ambient heat transfer coefficient
KWH_RANGE_W_PER_K = (15.0, 60.0)    # winding -> housing conduction coefficient


# ---------------------------------------------------------------------------
# Operating-condition ranges used when generating synthetic runs
# ---------------------------------------------------------------------------

CURRENT_RANGE_A = (0.0, 12.0)       # applied current range (A)
AMBIENT_TEMP_RANGE_C = (15.0, 35.0)  # ambient temperature range (deg C)
INITIAL_TEMP_OFFSET_RANGE_C = (0.0, 5.0)  # motor may start slightly above ambient

# Sensor noise (Gaussian, deg C). Per-run noise_std is sampled from this range
# so the benchmark can be stratified by noise level (see proposal, metric 3:
# convergence robustness).
NOISE_STD_RANGE_C = (0.1, 2.0)

# Fixed time grid shared by every synthetic run (uniform sampling simplifies
# storage as dense arrays). duration/dt chosen so that fast-hA runs
# (tau ~ C/hA as low as ~580s for the 1-node model) get close to steady
# state while slow ones only get partway there — that partial-convergence
# regime is deliberately included since it's what a real streaming
# calibration use case looks like.
SIM_DURATION_S = 3000.0
SIM_DT_S = 5.0


@dataclass(frozen=True)
class OneNodeParams:
    """Ground-truth parameters for a single synthetic 1-node run."""
    hA: float           # UNKNOWN — calibration target
    T_ambient: float
    C: float = C_LUMPED_J_PER_K
    R_winding: float = R_WINDING_OHM


@dataclass(frozen=True)
class TwoNodeParams:
    """Ground-truth parameters for a single synthetic 2-node run."""
    hA: float            # UNKNOWN — calibration target (housing -> ambient)
    k_wh: float          # UNKNOWN — calibration target (winding -> housing)
    T_ambient: float
    C_w: float = C_WINDING_J_PER_K
    C_h: float = C_HOUSING_J_PER_K
    R_winding: float = R_WINDING_OHM
