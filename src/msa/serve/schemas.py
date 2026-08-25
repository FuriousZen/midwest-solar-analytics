"""Request/response models for the prediction service.

Validation policy
-----------------
Everything a caller can send is bounded, because every unbounded field on a
public endpoint is either a crash or a memory amplifier waiting to happen:

* ``extra="forbid"``: a misspelled field name is a 422, not a silently
  ignored value that makes the caller think they supplied it.
* ``allow_inf_nan=False`` on every float: JSON's ``Infinity``/``NaN`` tokens
  are accepted by Python's parser, and ``ge=0`` happily passes ``+inf``. An
  infinite ``reported_cost_usd`` propagates through ``log1p`` and
  ``cost_per_kw`` into a non-finite prediction. Reject at the edge instead.
* every numeric has an upper bound as well as a lower one, chosen to be far
  outside anything a real distributed-solar permit carries.
* every string is length-capped: they are categorical lookup keys, never
  payloads.
* ``start_date`` is bounded to a plausible window; a permit dated year 0001 or
  9999 produces nonsense calendar features and an unrepresentable response date.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: No municipal solar register in this corpus starts before this.
MIN_START_DATE = date(1990, 1, 1)
#: How far into the future a caller may date a project. Beyond this the
#: seasonality features are extrapolation and the completion date arithmetic
#: risks overflowing ``datetime.date``.
MAX_START_DATE_YEARS_AHEAD = 10

#: Largest batch accepted in one call.
MAX_BATCH_ITEMS = 1000
#: ``?explain=true`` is only honoured up to this batch size; see
#: ``msa.serve.app.MAX_EXPLAIN_BATCH``.
MAX_EXPLAIN_BATCH_ITEMS = 50


class PermitRequest(BaseModel):
    """One solar project whose remaining timeline should be predicted.

    Only ``jurisdiction`` and ``start_date`` are required; everything else is
    optional because open-permit records are routinely incomplete, and XGBoost
    consumes missing numerics natively rather than needing imputation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    jurisdiction: str = Field(
        ...,
        min_length=1,
        max_length=120,
        examples=["Chicago"],
        description="City whose register the permit sits in, e.g. 'Minneapolis'.",
    )
    start_date: date = Field(
        ...,
        description=(
            "The milestone the clock starts from: permit issue date for the "
            "permit_to_completion model, application date for application_to_issue."
        ),
    )
    state: str | None = Field(
        default=None,
        examples=["IL"],
        min_length=2,
        max_length=2,
        pattern=r"^[A-Za-z]{2}$",
    )
    system_size_kw: float | None = Field(
        default=None, ge=0, le=1_000, allow_inf_nan=False
    )
    reported_cost_usd: float | None = Field(
        default=None, ge=0, le=1e9, allow_inf_nan=False
    )
    permit_fee_usd: float | None = Field(
        default=None, ge=0, le=1e7, allow_inf_nan=False
    )
    building_fee_usd: float | None = Field(
        default=None, ge=0, le=1e7, allow_inf_nan=False
    )
    zoning_fee_usd: float | None = Field(
        default=None,
        ge=0,
        le=1e7,
        allow_inf_nan=False,
        description=(
            "A non-zero zoning fee is the signal that zoning review was "
            "triggered, which is one of the stronger predictors of a slow permit."
        ),
    )
    other_fee_usd: float | None = Field(
        default=None, ge=0, le=1e7, allow_inf_nan=False
    )
    sector: str | None = Field(default=None, max_length=60, examples=["residential"])
    review_track: str | None = Field(
        default=None, max_length=120, examples=["Express Permit Program"]
    )
    permit_class: str | None = Field(default=None, max_length=120)
    district: str | None = Field(default=None, max_length=60)
    contractor: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Installer / general contractor. Matched against the contractors seen "
            "in training to look up their historical turnaround."
        ),
    )
    zip_code: str | None = Field(default=None, max_length=10)
    latitude: float | None = Field(
        default=None, ge=-90, le=90, allow_inf_nan=False
    )
    longitude: float | None = Field(
        default=None, ge=-180, le=180, allow_inf_nan=False
    )
    mount_type: str | None = Field(default=None, max_length=40, examples=["roof"])
    roof_type: str | None = Field(default=None, max_length=40, examples=["pitched"])
    battery: int | None = Field(default=None, ge=0, le=1)
    n_parties: float | None = Field(
        default=None, ge=0, le=1_000, allow_inf_nan=False
    )
    has_expeditor: int | None = Field(default=None, ge=0, le=1)
    has_engineer: int | None = Field(default=None, ge=0, le=1)
    has_design_professional: int | None = Field(default=None, ge=0, le=1)
    description_length: float | None = Field(
        default=None, ge=0, le=1e6, allow_inf_nan=False
    )

    @field_validator("state")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v

    @field_validator("start_date")
    @classmethod
    def _plausible_start_date(cls, v: date) -> date:
        latest = date.today() + timedelta(days=365 * MAX_START_DATE_YEARS_AHEAD)
        if v < MIN_START_DATE or v > latest:
            raise ValueError(
                f"start_date must be between {MIN_START_DATE} and {latest}"
            )
        return v


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PermitRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description=(
            f"Up to {MAX_BATCH_ITEMS} projects. `?explain=true` is only honoured "
            f"for batches of {MAX_EXPLAIN_BATCH_ITEMS} or fewer; TreeSHAP over "
            "a large batch costs far more than the prediction itself."
        ),
    )


class Attribution(BaseModel):
    feature: str
    value: float | str | None
    shap: float


class PredictionResponse(BaseModel):
    predicted_days: float = Field(..., description="Point prediction, in days.")
    p10_days: float | None = Field(
        default=None, description="Lower end of the 80% prediction interval."
    )
    p90_days: float | None = Field(
        default=None, description="Upper end of the 80% prediction interval."
    )
    predicted_completion_date: date
    horizon_days: int = Field(
        ..., description="Upper bound of the range this model was fitted on."
    )
    segment: str
    model_version: str
    attributions: list[Attribution] | None = Field(
        default=None,
        description="Top SHAP attributions, present when ?explain=true.",
    )


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Liveness. Answered even when no model could be loaded."""

    status: str = Field(
        ..., description="'ok' when a model is serving, 'degraded' when none is."
    )
    segment: str
    model_loaded: bool
    model_version: str | None = None
    artifact_schema_version: int | None = None
    trained_at: str | None = None
    n_features: int | None = None
    trained_on: int | None = None
    train_window: list[str] | None = None
    jurisdictions: list[str] | None = None
    detail: str | None = Field(
        default=None, description="Why the service is degraded, when it is."
    )


class ReadinessResponse(BaseModel):
    """Readiness: the model is loaded *and* it just answered a canary request."""

    status: str
    segment: str
    model_loaded: bool
    model_version: str | None = None
    trained_at: str | None = None
    checks: dict[str, bool]
    detail: str | None = None


class ErrorResponse(BaseModel):
    """The only error body shape this service emits."""

    error: str = Field(..., description="Stable machine-readable error code.")
    detail: str = Field(..., description="Short human-readable summary. Never a traceback.")
    request_id: str | None = None


__all__ = [
    "MAX_BATCH_ITEMS",
    "MAX_EXPLAIN_BATCH_ITEMS",
    "MIN_START_DATE",
    "Attribution",
    "BatchRequest",
    "BatchResponse",
    "ErrorResponse",
    "HealthResponse",
    "PermitRequest",
    "PredictionResponse",
    "ReadinessResponse",
]
