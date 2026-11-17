"""A local benchmark review screen; reference labels are intentionally visible."""

from __future__ import annotations

import json
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw

from annotation_qa.coco_data import coco_api_subset
from annotation_qa.io_utils import append_jsonl, load_json, read_jsonl, sha256_file, utc_now


@lru_cache(maxsize=8)
def _cached_sha256(path_text: str, modified_ns: int, size_bytes: int) -> str:
    del modified_ns, size_bytes
    return sha256_file(Path(path_text))


@lru_cache(maxsize=4)
def _cached_coco_dataset(path_text: str, modified_ns: int, size_bytes: int, image_ids: tuple[int, ...]):
    del modified_ns, size_bytes
    return coco_api_subset(Path(path_text), image_ids).dataset


def _manifest_image_map(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in manifest["images"]}


def _draw_boxes(
    image_path: Path,
    model_detections: list[dict[str, Any]],
    references: list[dict[str, Any]],
    category_names: dict[int, str],
    *,
    show_model: bool,
    show_reference: bool,
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if show_reference:
        for annotation in references:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            x2, y2 = x + box_width, y + box_height
            name = category_names.get(int(annotation["category_id"]), "unknown")
            color = "deepskyblue" if not annotation.get("iscrowd", 0) else "orange"
            draw.rectangle((x, y, x2, y2), outline=color, width=3)
            draw.text((x + 2, y + 2), f"REF {name}", fill=color, stroke_width=2, stroke_fill="black")
    if show_model:
        for detection in model_detections:
            x1, y1, x2, y2 = (float(value) for value in detection["bbox_xyxy_norm"])
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            label = str(detection["category"])
            draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
            draw.text((x1 + 2, y1 + 2), f"DRAFT {label}", fill="lime", stroke_width=2, stroke_fill="black")
    return image


def _human_corrections(raw_json: str, categories: set[str]) -> tuple[list[dict[str, Any]] | None, list[str]]:
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON: {exc}"]
    if not isinstance(value, list):
        return None, ["Correction must be a JSON array of {category, bbox_xyxy_norm} objects."]
    if len(value) > 100:
        return None, ["At most 100 annotations are allowed per image."]
    errors: list[str] = []
    for index, annotation in enumerate(value):
        if not isinstance(annotation, dict):
            errors.append(f"Item {index}: expected an object.")
            continue
        if set(annotation) != {"category", "bbox_xyxy_norm"}:
            errors.append(f"Item {index}: expected exactly category and bbox_xyxy_norm.")
            continue
        if not isinstance(annotation["category"], str) or annotation["category"] not in categories:
            errors.append(f"Item {index}: category must exactly match one of the COCO names.")
        box = annotation["bbox_xyxy_norm"]
        if not isinstance(box, list) or len(box) != 4:
            errors.append(f"Item {index}: bbox_xyxy_norm must contain four numbers.")
            continue
        if any(isinstance(item, bool) for item in box):
            errors.append(f"Item {index}: boolean values are not valid box coordinates.")
            continue
        if any(not isinstance(item, (int, float)) for item in box):
            errors.append(f"Item {index}: all box coordinates must be JSON numbers.")
            continue
        try:
            coordinates = [float(item) for item in box]
        except (TypeError, ValueError):
            errors.append(f"Item {index}: all box coordinates must be numeric.")
            continue
        if any(not 0 <= item <= 1 for item in coordinates):
            errors.append(f"Item {index}: coordinates must be within [0,1].")
        elif coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            errors.append(f"Item {index}: x_min < x_max and y_min < y_max are required.")
    return (value, []) if not errors else (None, errors)


def _render_study(
    *,
    plan: dict[str, Any],
    events_path: Path,
    reviewer: str,
    images_dir: Path,
    image_map: dict[int, dict[str, Any]],
    category_names: dict[int, str],
    allowed_categories: set[str],
    prediction_by_id: dict[int, dict[str, Any]],
) -> None:
    tasks = [row for row in plan["tasks"] if row["reviewer_id"] == reviewer]
    if not tasks:
        st.error(f"Reviewer '{reviewer}' has no tasks in this plan.")
        return
    submitted = [
        row for row in read_jsonl(events_path)
        if row.get("event_type") == plan["mode"] and row.get("plan_id") == plan["plan_id"]
    ]
    completed = {str(row["task_id"]) for row in submitted}
    pending = [row for row in tasks if row["task_id"] not in completed]
    if not pending:
        st.success(f"All {len(tasks)} tasks assigned to {reviewer} are complete.")
        return

    task_id = st.selectbox("Study task", [row["task_id"] for row in pending])
    task = next(row for row in pending if row["task_id"] == task_id)
    image_id = int(task["image_id"])
    image = image_map[image_id]
    image_path = (images_dir / str(image["file_name"])).resolve()
    if images_dir.resolve() not in image_path.parents or not image_path.is_file():
        st.error(f"Image file missing or outside the selected directory: {image['file_name']}")
        return
    if sha256_file(image_path) != image["sha256"]:
        st.error("Image checksum differs from the locked sample manifest.")
        return

    timer_key = f"timer:{plan['plan_id']}:{task_id}"
    if timer_key not in st.session_state:
        st.session_state[timer_key] = {
            "active": True,
            "accumulated": 0.0,
            "started_at": time.monotonic(),
        }
    timer = st.session_state[timer_key]
    if timer["active"]:
        if st.button("Pause active-time timer", key=f"pause:{task_id}"):
            timer["accumulated"] += time.monotonic() - timer["started_at"]
            timer["active"] = False
            st.rerun()
    elif st.button("Resume active-time timer", key=f"resume:{task_id}"):
        timer["active"] = True
        timer["started_at"] = time.monotonic()
        st.rerun()
    active_seconds = timer["accumulated"]
    if timer["active"]:
        active_seconds += time.monotonic() - timer["started_at"]
    st.caption(f"Recorded active time: {active_seconds:.1f} seconds. Pause the timer during interruptions.")

    condition = str(task["condition"])
    prediction = prediction_by_id.get(image_id, {})
    validation = prediction.get("validation", {})
    draft = validation.get("detections", []) if validation.get("status") == "valid" else []
    st.subheader("Independent annotation task")
    if plan["mode"] == "workflow_pilot":
        st.write("Condition: **Draft correction**" if condition == "assisted" else "Condition: **Annotate from scratch**")
    if plan["mode"] == "workflow_pilot" and condition == "assisted":
        st.image(
            _draw_boxes(image_path, draft, [], category_names, show_model=True, show_reference=False),
            use_container_width=True,
        )
        initial = [
            {"category": row["category"], "bbox_xyxy_norm": row["bbox_xyxy_norm"]}
            for row in draft
        ]
    else:
        st.image(
            _draw_boxes(image_path, [], [], category_names, show_model=False, show_reference=False),
            use_container_width=True,
        )
        initial = []
    st.info("Reference annotations are hidden until this task is submitted.")
    input_key = f"study_annotations:{plan['plan_id']}:{task_id}"
    if input_key not in st.session_state:
        st.session_state[input_key] = json.dumps(initial, indent=2)
    with st.form(f"study_form:{plan['plan_id']}:{task_id}"):
        annotation_text = st.text_area(
            "Final annotations JSON",
            key=input_key,
            height=280,
            help='Use [] for no objects; each item: {"category":"person","bbox_xyxy_norm":[0.1,0.2,0.4,0.8]}',
        )
        note = st.text_input("Optional ambiguity or process note")
        submit = st.form_submit_button("Submit completed task")
    if submit:
        final_annotations, errors = _human_corrections(annotation_text, allowed_categories)
        if errors:
            st.error(" ".join(errors))
            return
        if timer["active"]:
            timer["accumulated"] += time.monotonic() - timer["started_at"]
            timer["active"] = False
        append_jsonl(events_path, {
            "event_id": str(uuid.uuid4()),
            "event_schema_version": "human-study-event-v1",
            "event_type": plan["mode"],
            "plan_id": plan["plan_id"],
            "run_id": plan["run_id"],
            "manifest_sha256": plan["manifest_sha256"],
            "task_id": task_id,
            "image_id": image_id,
            "reviewer_id": reviewer,
            "condition": condition,
            "stratum": task.get("stratum"),
            "created_at": utc_now(),
            "active_seconds": timer["accumulated"],
            "final_annotations": final_annotations,
            "note": note,
            # Stored only after submission; these are never shown during a blind task.
            "source_model_output": prediction.get("raw_output"),
        })
        st.success("Task recorded in the append-only audit log.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Annotation QA Review", layout="wide")
    st.title("Annotation QA review and study")
    with st.sidebar:
        review_mode = st.radio("Activity", ["Benchmark review", "Blind manual audit", "Workflow time pilot"])
        manifest_text = st.text_input("Locked manifest", value="manifests/coco-val-sample.json")
        annotations_text = st.text_input("COCO annotations", value="data/annotations/instances_val2017.json")
        images_text = st.text_input("COCO images directory", value="data/val2017")
        run_text = st.text_input("Inference run directory", value="runs/evaluation-v1")
        events_text = st.text_input("Review event log", value="runs/review_events.jsonl")
        reviewer = st.text_input("Reviewer ID", value="reviewer-1")
        study_plan_text = st.text_input("Blind-audit / pilot plan", value="runs/studies/workflow-pilot-plan.json")

    if review_mode == "Benchmark review":
        st.caption("This benchmark screen shows both model drafts and COCO references. Reference disagreement is an offline evaluation trigger.")
    elif review_mode == "Blind manual audit":
        st.caption("The reviewer sees only the image until submission. The plan stratum, draft, and COCO reference stay hidden during annotation.")
    else:
        st.caption("The reviewer sees a draft only in the assisted condition. COCO references stay hidden in both conditions.")

    manifest_path = Path(manifest_text)
    annotations_path = Path(annotations_text)
    images_dir = Path(images_text)
    run_dir = Path(run_text)
    try:
        manifest = load_json(manifest_path)
        queue_doc = load_json(run_dir / "review_queue.json")
        metrics = load_json(run_dir / "metrics.json")
        run_metadata = load_json(run_dir / "run_metadata.json")
        records = read_jsonl(run_dir / "predictions.jsonl")
    except (OSError, ValueError, KeyError) as exc:
        st.info(f"Set the paths in the sidebar after creating an evaluation run. Details: {exc}")
        return

    try:
        manifest_hash = _cached_sha256(
            str(manifest_path.resolve()), manifest_path.stat().st_mtime_ns, manifest_path.stat().st_size
        )
        annotations_hash = _cached_sha256(
            str(annotations_path.resolve()), annotations_path.stat().st_mtime_ns, annotations_path.stat().st_size
        )
    except OSError as exc:
        st.error(f"Cannot verify benchmark inputs: {exc}")
        return
    if manifest_hash != metrics.get("manifest_sha256"):
        st.error("Manifest checksum differs from the evaluation run. Reopen the matching manifest.")
        return
    if annotations_hash != manifest.get("annotations_sha256"):
        st.error("COCO annotation checksum differs from the locked manifest.")
        return
    if run_metadata.get("run_id") != metrics.get("run_id"):
        st.error("Run metadata and evaluation report refer to different inference runs.")
        return
    if queue_doc.get("run_id") != metrics.get("run_id"):
        st.error("Review queue and evaluation report refer to different inference runs.")
        return

    evaluation_ids = {int(row["image_id"]) for row in manifest["images"] if row["split"] == "evaluation"}
    try:
        annotations = _cached_coco_dataset(
            str(annotations_path.resolve()),
            annotations_path.stat().st_mtime_ns,
            annotations_path.stat().st_size,
            tuple(sorted(evaluation_ids)),
        )
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load sampled COCO annotations: {exc}")
        return
    image_map = _manifest_image_map(manifest)
    reference_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in evaluation_ids:
            reference_by_image.setdefault(image_id, []).append(annotation)
    category_names = {int(row["id"]): str(row["name"]) for row in manifest["categories"]}
    allowed_categories = set(category_names.values())
    prediction_by_id = {}
    for row in records:
        image_id = int(row["image_id"])
        if row.get("run_id") != metrics.get("run_id") or image_id in prediction_by_id:
            st.error("Prediction records contain a mismatched run ID or duplicate image ID.")
            return
        prediction_by_id[image_id] = row
    if review_mode != "Benchmark review":
        try:
            study_plan = load_json(Path(study_plan_text))
        except (OSError, ValueError) as exc:
            st.info(f"Create a study plan first. Details: {exc}")
            return
        expected_mode = "blind_audit" if review_mode == "Blind manual audit" else "workflow_pilot"
        if study_plan.get("mode") != expected_mode:
            st.error(f"Selected activity requires a {expected_mode} plan.")
            return
        if study_plan.get("run_id") != metrics.get("run_id") or study_plan.get("manifest_sha256") != manifest_hash:
            st.error("Study plan does not match the selected inference run and manifest.")
            return
        _render_study(
            plan=study_plan,
            events_path=Path(events_text),
            reviewer=reviewer,
            images_dir=images_dir,
            image_map=image_map,
            category_names=category_names,
            allowed_categories=allowed_categories,
            prediction_by_id=prediction_by_id,
        )
        return
    queued = queue_doc.get("images", [])
    completed = {
        int(row["image_id"])
        for row in read_jsonl(Path(events_text))
        if row.get("event_type") == "benchmark_review"
        and row.get("run_id") == metrics.get("run_id")
        and row.get("action") != "defer"
    }
    pending = [row for row in queued if int(row["image_id"]) not in completed]
    if not pending:
        st.success(f"No pending review items. Reviewed {len(completed)} queued images.")
        st.stop()

    option_ids = [int(row["image_id"]) for row in pending]
    image_id = st.selectbox("Review image", option_ids, format_func=lambda value: f"{value} — {image_map[value]['file_name']}")
    queue_row = next(row for row in queued if int(row["image_id"]) == image_id)
    manifest_row = image_map[image_id]
    prediction = prediction_by_id.get(image_id)
    validation = prediction.get("validation", {}) if prediction else {}
    draft = validation.get("detections", []) if validation.get("status") == "valid" else []
    references = reference_by_image.get(image_id, [])
    image_path = (images_dir / str(manifest_row["file_name"])).resolve()
    if images_dir.resolve() not in image_path.parents or not image_path.is_file():
        st.error(f"Image file missing or outside the selected directory: {manifest_row['file_name']}")
        st.stop()
    if sha256_file(image_path) != manifest_row["sha256"]:
        st.error("Image checksum differs from the locked sample manifest.")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("Draft output")
        st.image(_draw_boxes(image_path, draft, [], category_names, show_model=True, show_reference=False), use_container_width=True)
    with right:
        st.subheader("COCO reference")
        st.image(_draw_boxes(image_path, [], references, category_names, show_model=False, show_reference=True), use_container_width=True)
    st.write({"status": queue_row.get("status"), "reasons": queue_row.get("reasons", [])})
    with st.expander("Original model response and validation details"):
        raw_output = prediction.get("raw_output") if prediction else None
        st.code(raw_output if isinstance(raw_output, str) else "No model output", language="json")
        st.json(validation)

    default_corrections = [
        {"category": row["category"], "bbox_xyxy_norm": row["bbox_xyxy_norm"]}
        for row in draft
    ]
    state_key = f"corrections_{image_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = json.dumps(default_corrections, indent=2)
    with st.form(f"review_form_{image_id}"):
        action = st.selectbox("Review decision", ["accept_draft", "correct", "reject_draft", "reference_issue", "defer"])
        correction_text = st.text_area(
            "Final annotations JSON (category plus normalized xyxy boxes)",
            key=state_key,
            height=240,
            help='Use [] for no objects; each item: {"category":"person","bbox_xyxy_norm":[0.1,0.2,0.4,0.8]}',
        )
        reason = st.text_input("Reason or correction note")
        save = st.form_submit_button("Save review event")
    if save:
        if action == "accept_draft" and validation.get("status") != "valid":
            st.error("An invalid draft cannot be accepted. Choose correct, reject_draft, or defer.")
            st.stop()
        final_annotations = None
        if action == "accept_draft":
            final_annotations = default_corrections
        elif action == "correct":
            final_annotations, errors = _human_corrections(correction_text, allowed_categories)
            if errors:
                st.error(" ".join(errors))
                st.stop()
        event = {
            "event_id": str(uuid.uuid4()),
            "event_schema_version": "benchmark-review-event-v1",
            "event_type": "benchmark_review",
            "created_at": utc_now(),
            "run_id": metrics.get("run_id"),
            "manifest_sha256": metrics.get("manifest_sha256"),
            "image_id": image_id,
            "reviewer_id": reviewer,
            "action": action,
            "reason": reason,
            "final_annotations": final_annotations,
            "raw_correction_json": (
                json.dumps(default_corrections, ensure_ascii=False)
                if action == "accept_draft"
                else correction_text if action == "correct" else None
            ),
            "model_status": validation.get("status"),
            "reference_annotation_ids": [int(row["id"]) for row in references],
            "source_model_output": prediction.get("raw_output") if prediction else None,
        }
        append_jsonl(Path(events_text), event)
        st.success("Review event appended to the audit log.")
        st.rerun()


if __name__ == "__main__":
    main()

