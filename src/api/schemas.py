"""Pydantic request/response models for the calibration microservice."""

from pydantic import BaseModel, Field, field_validator

from src.benchmark.registry import METHOD_METADATA

VALID_METHODS = sorted(METHOD_METADATA)


class OneNodeCalibrationRequest(BaseModel):
    """A window of sensor data from the 1-node (lumped) testbed.

    `t` must be a uniform time grid in seconds (only `t[1]-t[0]` is used as
    the step size); `I` is the applied current (A) at each grid point;
    `T_measured` is the (noisy) winding temperature (degC) at each grid
    point; `T_ambient` is the ambient temperature (degC) for this run.
    """

    t: list[float] = Field(..., min_length=2, description="uniform time grid, seconds")
    I: list[float] = Field(..., min_length=2, description="applied current, amps")
    T_measured: list[float] = Field(..., min_length=2, description="measured winding temperature, degC")
    T_ambient: float = Field(..., description="ambient temperature, degC")
    method: str = Field(
        "mlp", description=f"one of {VALID_METHODS}; 'mlp' is the fast, offline-trained default"
    )

    @field_validator("method")
    @classmethod
    def _method_is_known(cls, v):
        if v not in VALID_METHODS:
            raise ValueError(f"unknown method {v!r}; expected one of {VALID_METHODS}")
        return v

    def check_lengths(self):
        n = len(self.t)
        if not (len(self.I) == len(self.T_measured) == n):
            raise ValueError(f"t, I, T_measured must have equal length (got {n}, {len(self.I)}, {len(self.T_measured)})")


class TwoNodeCalibrationRequest(BaseModel):
    """A window of sensor data from the 2-node (winding + housing) testbed."""

    t: list[float] = Field(..., min_length=2, description="uniform time grid, seconds")
    I: list[float] = Field(..., min_length=2, description="applied current, amps")
    T_w_measured: list[float] = Field(..., min_length=2, description="measured winding temperature, degC")
    T_h_measured: list[float] = Field(..., min_length=2, description="measured housing temperature, degC")
    T_ambient: float = Field(..., description="ambient temperature, degC")
    method: str = Field(
        "mlp", description=f"one of {VALID_METHODS}; 'mlp' is the fast, offline-trained default"
    )

    @field_validator("method")
    @classmethod
    def _method_is_known(cls, v):
        if v not in VALID_METHODS:
            raise ValueError(f"unknown method {v!r}; expected one of {VALID_METHODS}")
        return v

    def check_lengths(self):
        n = len(self.t)
        if not (len(self.I) == len(self.T_w_measured) == len(self.T_h_measured) == n):
            raise ValueError(
                f"t, I, T_w_measured, T_h_measured must have equal length "
                f"(got {n}, {len(self.I)}, {len(self.T_w_measured)}, {len(self.T_h_measured)})"
            )


class CalibrationResponse(BaseModel):
    testbed: str
    method: str
    params: dict[str, float]
    runtime_ms: float
    n_evals: int
    converged: bool


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


class MethodInfo(BaseModel):
    family: str
    stochastic: bool
    streaming_capable: bool
    available_for: list[str]


class MethodsResponse(BaseModel):
    methods: dict[str, MethodInfo]
