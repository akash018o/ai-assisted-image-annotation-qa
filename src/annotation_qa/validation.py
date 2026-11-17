"""Strict parsing and semantic validation; no automatic repairs are performed."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import ValidationError

from .models import DraftResponse


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number is not supported: {value}")
    return parsed


def validate_raw_output(raw_output: str, allowed_categories: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "parse_failure",
        "parse_errors": [],
        "schema_errors": [],
        "semantic_errors": [],
        "parsed_output": None,
        "detections": [],
    }
    try:
        parsed = json.loads(
            raw_output,
            parse_constant=_reject_nonstandard_constant,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        result["parse_errors"] = [str(exc)]
        return result

    try:
        response = DraftResponse.model_validate(parsed)
    except ValidationError as exc:
        result["status"] = "schema_failure"
        result["schema_errors"] = [
            {"type": error["type"], "loc": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors(include_url=False)
        ]
        result["parsed_output"] = parsed
        return result

    payload = response.model_dump(mode="json")
    result["parsed_output"] = payload
    errors: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(response.detections):
        if detection.category not in allowed_categories:
            errors.append({
                "type": "invalid_category",
                "detection_index": detection_index,
                "value": detection.category,
            })
        coordinates = detection.bbox_xyxy_norm
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            errors.append({
                "type": "non_finite_coordinate",
                "detection_index": detection_index,
                "value": repr(coordinates),
            })
        elif any(coordinate < 0 or coordinate > 1 for coordinate in coordinates):
            errors.append({
                "type": "coordinate_out_of_range",
                "detection_index": detection_index,
                "value": coordinates,
            })
        elif coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            errors.append({
                "type": "invalid_box_geometry",
                "detection_index": detection_index,
                "value": coordinates,
            })
        if not math.isfinite(detection.rank_score):
            errors.append({
                "type": "non_finite_rank_score",
                "detection_index": detection_index,
                "value": repr(detection.rank_score),
            })
        elif not 0 <= detection.rank_score <= 1:
            errors.append({
                "type": "rank_score_out_of_range",
                "detection_index": detection_index,
                "value": detection.rank_score,
            })

    if errors:
        result["status"] = "semantic_failure"
        result["semantic_errors"] = errors
        return result

    result["status"] = "valid"
    result["detections"] = payload["detections"]
    return result

