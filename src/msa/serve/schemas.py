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

    jurisdiction: str = Field(
        ...,
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
    sector: str | None = Field(default=None, examples=["residential"])
    review_track: str | None = Field(default=None, examples=["Express Permit Program"])
    permit_class: str | None = None
    district: str | None = None
    contractor: str | None = Field(
        default=None,
        description=(
            "Installer / general contractor. Matched against the contractors seen "
            "in training to look up their historical turnaround."
        ),
    )
    zip_code: str | None = Field(default=None, max_length=10)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    mount_type: str | None = Field(default=None, examples=["roof"])
    roof_type: str | None = Field(default=None, examples=["pitched"])
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
    items: list[PermitRequest] = Field(..., min_length=1, max_length=1000)


class Attribution(BaseModel):
    feature: str
    value: float | str | None
    shap: float


class PredictionResponse(BaseModel):
    predicted_days: float = Field(..., description="Point prediction, in days.")
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
