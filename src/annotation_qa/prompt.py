"""Versioned prompt construction."""

from __future__ import annotations

from .io_utils import sha256_text

PROMPT_VERSION = "draft-v1"
DEFAULT_PROMPT_TEMPLATE = """You are producing draft object-detection annotations for one image.

Image dimensions: width=<<width>>, height=<<height>> pixels.
Allowed category names (use exact spelling and capitalization): <<categories>>

Return only a JSON object matching the supplied JSON Schema. Include every clearly
visible instance belonging to an allowed category. Do not infer objects that are
not visible. Use an empty detections array if none are visible. Return at most 100
detections. For each box, use bbox_xyxy_norm=[x_min,y_min,x_max,y_max], normalized
to the full original image width and height in the inclusive range [0,1]. Do not
clip or repair a box. rank_score is only a relative ordering score from 0 to 1;
it is not a calibrated probability. Give higher scores to detections you rank
more confidently. Do not return explanations or markdown."""


def build_prompt(
    category_names: list[str],
    width: int,
    height: int,
    template_text: str | None = None,
) -> str:
    template = template_text if template_text is not None else DEFAULT_PROMPT_TEMPLATE
    required_tokens = {"<<categories>>", "<<width>>", "<<height>>"}
    missing = required_tokens - {token for token in required_tokens if token in template}
    if missing:
        raise ValueError(f"Prompt template is missing placeholders: {', '.join(sorted(missing))}")
    return (
        template.replace("<<categories>>", ", ".join(category_names))
        .replace("<<width>>", str(width))
        .replace("<<height>>", str(height))
    )


def prompt_sha(prompt: str) -> str:
    return sha256_text(prompt)

