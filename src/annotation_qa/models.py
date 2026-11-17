"""The model response contract. Semantic correctness is checked separately."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DraftDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Keep category as an unrestricted string so unsupported labels are visible
    # as semantic failures instead of being coerced by the output schema.
    category: str
    bbox_xyxy_norm: list[float] = Field(min_length=4, max_length=4)
    rank_score: float

    @field_validator("bbox_xyxy_norm", mode="before")
    @classmethod
    def reject_boolean_coordinates(cls, value):
        if not isinstance(value, list):
            raise ValueError("box coordinates must be a JSON array")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("box coordinates must be JSON numbers")
        return value

    @field_validator("rank_score", mode="before")
    @classmethod
    def reject_boolean_score(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("ranking score must be a JSON number")
        return value


class DraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detections: list[DraftDetection] = Field(max_length=100)


def output_json_schema() -> dict:
    return DraftResponse.model_json_schema()

