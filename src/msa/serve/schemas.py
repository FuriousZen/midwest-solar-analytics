"""Request/response models for the prediction service."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class PermitRequest(BaseModel):
    """One solar project whose remaining timeline should be predicted.

    Only ``jurisdiction`` and ``start_date`` are required; everything else is
    optional because open-permit records are routinely incomplete, and XGBoost
    consumes missing numerics natively rather than needing imputation.
    """

    # Free-text fields are length-capped. They are only ever used as categorical
    # lookup keys, so nothing legitimate needs more than a short string, and an
    # uncapped string field on a public endpoint is a cheap memory-amplification
    # vector when combined with the 1000-item batch endpoint.
    jurisdiction: str = Field(
        ...,
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
    state: str | None = Field(default=None, examples=["IL"], max_length=2)
    system_size_kw: float | None = Field(default=None, ge=0, le=1000)
    reported_cost_usd: float | None = Field(default=None, ge=0)
    permit_fee_usd: float | None = Field(default=None, ge=0)
    building_fee_usd: float | None = Field(default=None, ge=0)
    zoning_fee_usd: float | None = Field(
        default=None,
        ge=0,
        description=(
            "A non-zero zoning fee is the signal that zoning review was "
            "triggered, which is one of the stronger predictors of a slow permit."
        ),
    )
    other_fee_usd: float | None = Field(default=None, ge=0)
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
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    mount_type: str | None = Field(default=None, max_length=40, examples=["roof"])
    roof_type: str | None = Field(default=None, max_length=40, examples=["pitched"])
    battery: int | None = Field(default=None, ge=0, le=1)
    n_parties: float | None = Field(default=None, ge=0)
    has_expeditor: int | None = Field(default=None, ge=0, le=1)
    has_engineer: int | None = Field(default=None, ge=0, le=1)
    has_design_professional: int | None = Field(default=None, ge=0, le=1)
    description_length: float | None = Field(default=None, ge=0)

    @field_validator("state")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class BatchRequest(BaseModel):
    items: list[PermitRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Up to 1000 projects. `?explain=true` is only honoured for batches "
            "of 50 or fewer -- TreeSHAP over a large batch costs far more than "
            "the prediction itself."
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
    status: str
    segment: str
    model_version: str
    n_features: int
    trained_on: int
    train_window: list[str]
    jurisdictions: list[str]
