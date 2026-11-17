"""Run one locked manifest split and append every request/response to the audit."""

from __future__ import annotations

import hashlib
import platform
import time
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .io_utils import append_jsonl, ensure_new_directory, load_json, sha256_file, utc_now, write_json
from .manifest import split_images
from .models import output_json_schema
from .ollama_local import LocalOllama, OllamaError
from .prompt import DEFAULT_PROMPT_TEMPLATE, PROMPT_VERSION, build_prompt, prompt_sha
from .validation import validate_raw_output


def run_inference(
    *,
    manifest_path: Path,
    images_dir: Path,
    output_dir: Path,
    model_name: str,
    split: str = "evaluation",
    ollama_host: str = "http://127.0.0.1:11434",
    timeout_seconds: int = 600,
    transport_retries: int = 1,
    prompt_file: Path | None = None,
    prompt_version: str = PROMPT_VERSION,
    generation_seed: int = 20261117,
) -> dict[str, Any]:
    if transport_retries < 0:
        raise ValueError("transport_retries cannot be negative")
    manifest = load_json(manifest_path)
    selected = split_images(manifest, split)
    if not selected:
        raise ValueError(f"Manifest has no images in split '{split}'.")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    client = LocalOllama(ollama_host, timeout_seconds=timeout_seconds)
    model_details = client.model_details(model_name)
    ollama_version = client.version()
    category_names = [str(row["name"]) for row in manifest["categories"]]
    allowed_categories = set(category_names)
    schema = output_json_schema()
    template_text = prompt_file.read_text(encoding="utf-8") if prompt_file else DEFAULT_PROMPT_TEMPLATE
    generation_options = {"temperature": 0, "seed": generation_seed, "num_predict": 4096}
    run_id = str(uuid.uuid4())
    ensure_new_directory(output_dir)
    metadata = {
        "run_id": run_id,
        "created_at": utc_now(),
        "project_version": __version__,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "annotations_sha256": manifest["annotations_sha256"],
        "split": split,
        "sample_image_ids": [int(row["image_id"]) for row in selected],
        "model": model_details,
        "ollama_version": ollama_version,
        "prompt_version": prompt_version,
        "prompt_template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        "prompt_file": str(prompt_file.resolve()) if prompt_file else None,
        "schema_version": "draft-response-v1",
        "json_schema": schema,
        "options": generation_options,
        "transport_retries": transport_retries,
        "retry_policy": "retry transport failures only; never retry invalid model content",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "offline_host": ollama_host,
    }
    write_json(output_dir / "run_metadata.json", metadata, exclusive=True)
    attempts_path = output_dir / "attempts.jsonl"
    predictions_path = output_dir / "predictions.jsonl"

    for image_row in selected:
        image_id = int(image_row["image_id"])
        filename = Path(str(image_row["file_name"]))
        image_path = (images_dir / filename).resolve()
        if images_dir.resolve() not in image_path.parents:
            raise ValueError(f"Manifest image path escapes image directory: {filename}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image {image_id} is missing: {image_path}")
        actual_sha = sha256_file(image_path)
        if actual_sha != image_row["sha256"]:
            raise ValueError(f"Image checksum changed since sampling for image {image_id}.")

        prompt = build_prompt(category_names, int(image_row["width"]), int(image_row["height"]), template_text)
        started_at = utc_now()
        response: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        elapsed_seconds = 0.0
        raw_output = ""
        for attempt_number in range(1, transport_retries + 2):
            attempt_started = time.monotonic()
            attempt_at = utc_now()
            try:
                response = client.chat(model_name, image_path, prompt, schema, generation_options)
                raw_output = str(response.get("message", {}).get("content", ""))
                attempt_elapsed_seconds = time.monotonic() - attempt_started
                elapsed_seconds += attempt_elapsed_seconds
                error = None
                append_jsonl(attempts_path, {
                    "run_id": run_id,
                    "image_id": image_id,
                    "image_sha256": actual_sha,
                    "split": split,
                    "attempt": attempt_number,
                    "started_at": attempt_at,
                    "elapsed_seconds": attempt_elapsed_seconds,
                    "request": {
                        "model": model_name,
                        "prompt_version": prompt_version,
                        "prompt_sha256": prompt_sha(prompt),
                        "prompt": prompt,
                        "options": generation_options,
                    },
                    "raw_output": raw_output,
                    "response": response,
                    "error": None,
                })
                break
            except OllamaError as exc:
                attempt_elapsed_seconds = time.monotonic() - attempt_started
                elapsed_seconds += attempt_elapsed_seconds
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "details": exc.details,
                    "retryable": exc.retryable,
                }
                append_jsonl(attempts_path, {
                    "run_id": run_id,
                    "image_id": image_id,
                    "image_sha256": actual_sha,
                    "split": split,
                    "attempt": attempt_number,
                    "started_at": attempt_at,
                    "elapsed_seconds": attempt_elapsed_seconds,
                    "request": {
                        "model": model_name,
                        "prompt_version": prompt_version,
                        "prompt_sha256": prompt_sha(prompt),
                        "prompt": prompt,
                        "options": generation_options,
                    },
                    "raw_output": None,
                    "response": None,
                    "error": error,
                })
                if not exc.retryable or attempt_number > transport_retries:
                    break

        if error is not None and response is None:
            validation = {
                "status": "inference_failure",
                "parse_errors": [],
                "schema_errors": [],
                "semantic_errors": [],
                "parsed_output": None,
                "detections": [],
            }
        else:
            validation = validate_raw_output(raw_output, allowed_categories)
        append_jsonl(predictions_path, {
            "run_id": run_id,
            "image_id": image_id,
            "split": split,
            "file_name": image_row["file_name"],
            "image_sha256": actual_sha,
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "ollama_load_duration_ns": response.get("load_duration") if response else None,
            "status": validation["status"],
            "raw_output": raw_output if response is not None else None,
            "response": response,
            "error": error,
            "validation": validation,
        })

    metadata["completed_at"] = utc_now()
    write_json(output_dir / "run_metadata.json", metadata)
    return metadata

