"""
1-node ("lumped"): the whole motor (winding + housing) is treated as a single
thermal mass. The single unknown to calibrate is the lumped heat-transfer
coefficient `hA` (housing/ambient interface, W/K).

2-node: winding and housing are separate thermal masses connected by a
conduction path. Two unknowns to calibrate: `k_wh` (winding->housing
conduction, W/K) and `hA` (housing->ambient convection, W/K).

"""

from dataclasses import dataclass


# Constante cunoscute masurate de senzori, nu se calibreaza

R_WINDING_OHM = 1.8        
C_WINDING_J_PER_K = 2500.0   # (mass x specific heat of copper)
C_HOUSING_J_PER_K = 12000.0  # (mass x specific heat of aluminium)

# pentru 1-node model
C_LUMPED_J_PER_K = C_WINDING_J_PER_K + C_HOUSING_J_PER_K


# interval constante de calibrare (W/K) pentru a genera date

HA_RANGE_W_PER_K = (8.0, 25.0)      
KWH_RANGE_W_PER_K = (15.0, 60.0)   

# Conditii de mediu pentru generare de date primite

CURRENT_RANGE_A = (0.0, 12.0)       # in Amperi
AMBIENT_TEMP_RANGE_C = (15.0, 35.0)  # in grade Celsius
INITIAL_TEMP_OFFSET_RANGE_C = (0.0, 5.0)  # temperatura initiala motor

# eroare de masurare a temperaturii (simulare senzori)
NOISE_STD_RANGE_C = (0.1, 2.0)


SIM_DURATION_S = 3000.0 #50 min 
SIM_DT_S = 5.0 #pasul intre esantioane


@dataclass(frozen=True)
class OneNodeParams:
    """Ground-truth parameters for a single 1-node run."""
    hA: float           # UNKNOWN
    T_ambient: float
    C: float = C_LUMPED_J_PER_K
    R_winding: float = R_WINDING_OHM


@dataclass(frozen=True)
class TwoNodeParams:
    """Ground-truth parameters for a single 2-node run."""
    hA: float            # UNKNOWN
    k_wh: float          # UNKNOWN
    T_ambient: float
    C_w: float = C_WINDING_J_PER_K
    C_h: float = C_HOUSING_J_PER_K
    R_winding: float = R_WINDING_OHM
