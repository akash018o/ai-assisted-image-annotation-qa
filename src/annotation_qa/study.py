"""Randomized workflow pilot and blind manual-audit planning/scoring."""

from __future__ import annotations

import contextlib
import io
import math
import platform
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from pycocotools.cocoeval import COCOeval

from .evaluate import (
    MAX_DETECTIONS,
    _bbox_iou_xywh,
    _count_metrics,
    _empty_detection_coco,
    _bootstrap_micro,
)
from .coco_data import coco_api_subset
from .io_utils import load_json, read_jsonl, sha256_file, utc_now, write_json
from .reporting import render_study_markdown


def _plan_task_id(image_id: int, reviewer_id: str, condition: str) -> str:
    safe_reviewer = "".join(char if char.isalnum() else "-" for char in reviewer_id)
    return f"{image_id}-{safe_reviewer}-{condition}"


def _stratum_for(image_id: int, queue: dict[int, dict[str, Any]], status: str) -> str:
    reasons = queue.get(image_id, {}).get("reasons", [])
    if "coco_reference_disagreement_at_iou_0_50" in reasons:
        return "disagreement"
    if status != "valid":
        return "invalid_output"
    return "agreement"


def _stratified_pick(groups: dict[str, list[int]], count: int, seed: int) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    available = {key: list(values) for key, values in groups.items() if values}
    for values in available.values():
        rng.shuffle(values)
    order = list(available)
    rng.shuffle(order)
    selected: list[tuple[int, str]] = []
    required_groups = [group for group in ("agreement", "disagreement") if available.get(group)]
    if count < len(required_groups):
        raise ValueError("Blind audit count must include at least one agreement and one disagreement case.")
    for group in required_groups:
        selected.append((available[group].pop(), group))
        if not available[group]:
            order.remove(group)
    while len(selected) < count and order:
        made_progress = False
        for group in list(order):
            if available[group] and len(selected) < count:
                selected.append((available[group].pop(), group))
                made_progress = True
            if not available[group]:
                order.remove(group)
        if not made_progress:
            break
    return selected


def create_study_plan(
    *,
    mode: str,
    manifest_path: Path,
    run_dir: Path,
    output_path: Path,
    reviewers: list[str],
    count: int = 24,
    seed: int = 20261117,
) -> dict[str, Any]:
    if mode not in {"workflow_pilot", "blind_audit"}:
        raise ValueError("mode must be workflow_pilot or blind_audit")
    if count < 2:
        raise ValueError("count must be at least 2")
    if not reviewers or len(set(reviewers)) != len(reviewers):
        raise ValueError("Provide one or two distinct reviewer IDs.")
    if len(reviewers) > 2:
        raise ValueError("This pilot supports at most two reviewers.")
    if mode == "blind_audit" and len(reviewers) != 1:
        raise ValueError("Blind audit planning currently supports one independent reviewer.")
    if output_path.exists():
        raise FileExistsError(f"Locked study plan already exists: {output_path}")

    manifest = load_json(manifest_path)
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("split") != "evaluation":
        raise ValueError("Studies must be planned from an evaluation-split run.")
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Inference run and study plan use different manifests.")
    eval_rows = [row for row in manifest["images"] if row["split"] == "evaluation"]
    records = {int(row["image_id"]): row for row in read_jsonl(run_dir / "predictions.jsonl")}
    queue_path = run_dir / "review_queue.json"
    if mode == "blind_audit" and not queue_path.is_file():
        raise FileNotFoundError("Run evaluation first so blind-audit sampling can include agreement and disagreement cases.")
    queue_doc = load_json(queue_path) if queue_path.is_file() else {"images": []}
    queue = {int(row["image_id"]): row for row in queue_doc.get("images", [])}
    run_id = metadata["run_id"]

    tasks: list[dict[str, Any]] = []
    if mode == "workflow_pilot":
        eligible = [
            row for row in eval_rows
            if records.get(int(row["image_id"]), {}).get("status") == "valid"
        ]
        if count > len(eligible):
            raise ValueError(
                f"Requested {count} workflow images, but only {len(eligible)} have valid drafts."
            )
        picked = random.Random(seed).sample(eligible, count)
        ids = [int(row["image_id"]) for row in picked]
        rng = random.Random(seed + 1)
        if len(reviewers) == 1:
            conditions = ["scratch"] * (count // 2) + ["assisted"] * (count - count // 2)
            rng.shuffle(conditions)
            for image_id, condition in zip(ids, conditions, strict=True):
                tasks.append({
                    "task_id": _plan_task_id(image_id, reviewers[0], condition),
                    "image_id": image_id,
                    "reviewer_id": reviewers[0],
                    "condition": condition,
                    "stratum": "valid_draft_random_sample",
                })
            design = "single_reviewer_randomized_personal_pilot"
        else:
            for image_id in ids:
                first_condition = rng.choice(["scratch", "assisted"])
                assignments = {
                    reviewers[0]: first_condition,
                    reviewers[1]: "assisted" if first_condition == "scratch" else "scratch",
                }
                for reviewer_id, condition in assignments.items():
                    tasks.append({
                        "task_id": _plan_task_id(image_id, reviewer_id, condition),
                        "image_id": image_id,
                        "reviewer_id": reviewer_id,
                        "condition": condition,
                        "stratum": "valid_draft_random_sample",
                    })
            design = "two_reviewer_counterbalanced_pilot"
    else:
        groups: dict[str, list[int]] = {"agreement": [], "disagreement": [], "invalid_output": []}
        for row in eval_rows:
            image_id = int(row["image_id"])
            status = records.get(image_id, {}).get("status", "inference_missing")
            groups[_stratum_for(image_id, queue, str(status))].append(image_id)
        picked = _stratified_pick(groups, min(count, len(eval_rows)), seed)
        if len(picked) < min(count, len(eval_rows)):
            raise ValueError("Unable to draw the requested blind-audit sample.")
        for image_id, stratum in picked:
            tasks.append({
                "task_id": _plan_task_id(image_id, reviewers[0], "blind_audit"),
                "image_id": image_id,
                "reviewer_id": reviewers[0],
                "condition": "blind_audit",
                "stratum": stratum,
            })
        design = "stratified_blind_manual_audit"

    rng = random.Random(seed + 2)
    rng.shuffle(tasks)
    plan = {
        "plan_version": 1,
        "plan_id": _plan_task_id(seed, reviewers[0], mode),
        "created_at": utc_now(),
        "python_version": platform.python_version(),
        "mode": mode,
        "design": design,
        "run_id": run_id,
        "manifest_sha256": metadata["manifest_sha256"],
        "seed": seed,
        "requested_image_count": count,
        "selected_image_count": len({int(row["image_id"]) for row in tasks}),
        "reviewers": reviewers,
        "reference_visibility_during_task": False,
        "tasks": tasks,
        "strata_counts": {
            name: sum(row.get("stratum") == name for row in tasks)
            for name in sorted({str(row.get("stratum")) for row in tasks})
        },
        "notes": (
            "Workflow pilot eligibility is limited to images with valid model drafts."
            if mode == "workflow_pilot"
            else "Strata use benchmark comparison only to select cases; the reviewer sees image only."
        ),
    }
    write_json(output_path, plan, exclusive=True)
    return plan


def _to_coco_results(
    image_id: int,
    image_row: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_by_name: dict[str, int],
    *,
    score: float,
) -> list[dict[str, Any]]:
    results = []
    width, height = float(image_row["width"]), float(image_row["height"])
    if len(annotations) > 100:
        raise ValueError(f"Image {image_id} has more than 100 annotations.")
    for annotation in annotations:
        category = str(annotation["category"])
        if category not in category_by_name:
            raise ValueError(f"Unsupported annotation category: {category}")
        category_id = category_by_name[category]
        box = annotation.get("bbox_xyxy_norm")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Each annotation must have four bbox_xyxy_norm coordinates.")
        x1, y1, x2, y2 = (float(value) for value in box)
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x1, y1, x2, y2)):
            raise ValueError("Annotation coordinates must be finite and within [0,1].")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Annotation boxes must have positive width and height.")
        results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height],
            "score": score,
        })
    return results


def _score_annotations(
    annotations_path: Path,
    image_ids: list[int],
    result_rows: list[dict[str, Any]],
    category_ids: list[int],
    seed: int,
    bootstrap_resamples: int,
) -> tuple[dict[str, Any], COCOeval]:
    coco_gt = coco_api_subset(annotations_path, image_ids)
    if result_rows:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(result_rows)
    else:
        coco_dt = _empty_detection_coco(coco_gt, image_ids)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = category_ids
    evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    threshold_metrics, image_counts, matched_ious = _count_metrics(evaluator, image_ids, category_ids)
    report = {
        "image_count": len(image_ids),
        "valid_annotation_count": len(result_rows),
        "max_detections_per_image": MAX_DETECTIONS,
        "score_handling": "constant score; AP intentionally omitted; rank scores are ignored for fixed-set P/R",
        "precision_recall_matching": (
            "Custom class-aware maximum-cardinality one-to-one matching at each IoU threshold. "
            "Crowd/ignore references are excluded from recall; unmatched detections overlapping them are excluded from FP."
        ),
        "precision_recall": threshold_metrics,
        "matched_box_iou_at_least_0_50": {
            "count": len(matched_ious["0.5"]),
            "mean": mean(matched_ious["0.5"]) if matched_ious["0.5"] else None,
            "median": median(matched_ious["0.5"]) if matched_ious["0.5"] else None,
        },
        "image_bootstrap_95_ci": _bootstrap_micro(
            image_ids, category_ids, image_counts, seed, bootstrap_resamples
        ),
    }
    return report, evaluator


def _simple_match_counts(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    edges: list[list[int]] = []
    for prediction in predictions:
        eligible = []
        for reference_index, reference in enumerate(references):
            if int(prediction["category_id"]) != int(reference["category_id"]):
                continue
            overlap = _bbox_iou_xywh(prediction["bbox"], reference["bbox"])
            if overlap >= threshold:
                eligible.append((overlap, reference_index))
        eligible.sort(key=lambda pair: (-pair[0], pair[1]))
        edges.append([reference_index for _, reference_index in eligible])

    reference_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for reference_index in edges[prediction_index]:
            if reference_index in visited:
                continue
            visited.add(reference_index)
            previous_prediction = reference_to_prediction.get(reference_index)
            if previous_prediction is None or augment(previous_prediction, visited):
                reference_to_prediction[reference_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        augment(prediction_index, set())
    matches = len(reference_to_prediction)
    return matches, len(predictions) - matches, len(references) - matches


def evaluate_study(
    *,
    annotations_path: Path,
    manifest_path: Path,
    run_dir: Path,
    plan_path: Path,
    events_path: Path,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    manifest = load_json(manifest_path)
    metadata = load_json(run_dir / "run_metadata.json")
    if plan.get("run_id") != metadata.get("run_id"):
        raise ValueError("Study plan and inference run IDs do not match.")
    if plan.get("manifest_sha256") != metadata.get("manifest_sha256"):
        raise ValueError("Study plan and inference run manifests do not match.")
    if sha256_file(manifest_path) != plan["manifest_sha256"]:
        raise ValueError("Current manifest checksum differs from the locked study plan.")
    if sha256_file(annotations_path) != manifest["annotations_sha256"]:
        raise ValueError("Current COCO annotations checksum differs from the locked manifest.")
    records = {int(row["image_id"]): row for row in read_jsonl(run_dir / "predictions.jsonl")}
    events = read_jsonl(events_path)
    plan_by_task = {row["task_id"]: row for row in plan["tasks"]}
    events_by_task: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != plan["mode"] or event.get("plan_id") != plan["plan_id"]:
            continue
        task_id = str(event["task_id"])
        if task_id in events_by_task:
            raise ValueError(f"Duplicate submitted study event for task {task_id}.")
        if task_id not in plan_by_task:
            raise ValueError(f"Study event refers to unplanned task {task_id}.")
        events_by_task[task_id] = event
    missing = sorted(set(plan_by_task) - set(events_by_task))
    if missing:
        raise ValueError(f"Study is incomplete; {len(missing)} planned tasks have no submitted event.")

    image_rows = {int(row["image_id"]): row for row in manifest["images"]}
    category_by_name = {str(row["name"]): int(row["id"]) for row in manifest["categories"]}
    category_ids = sorted(category_by_name.values())
    plan_rows = list(plan["tasks"])
    result: dict[str, Any] = {
        "report_version": 1,
        "created_at": utc_now(),
        "plan_id": plan["plan_id"],
        "mode": plan["mode"],
        "design": plan["design"],
        "selected_image_count": plan["selected_image_count"],
        "task_count": len(plan_rows),
        "limitations": [
            "This small pilot is descriptive, not a general estimate of annotator productivity.",
            "Reviewer-entered active time is a workflow measure and can include timing noise.",
            "COCO references may contain omissions or box disagreements.",
            "Fixed model digest and settings improve repeatability, but execution backend or hardware can affect exact outputs.",
        ],
    }

    def human_rows_for(task_rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        image_ids: list[int] = []
        coco_rows: list[dict[str, Any]] = []
        elapsed: list[dict[str, Any]] = []
        for task in task_rows:
            image_id = int(task["image_id"])
            event = events_by_task[task["task_id"]]
            image_ids.append(image_id)
            coco_rows.extend(_to_coco_results(
                image_id, image_rows[image_id], event["final_annotations"], category_by_name, score=1.0
            ))
            active_seconds = float(event["active_seconds"])
            if not math.isfinite(active_seconds) or active_seconds < 0:
                raise ValueError(f"Task {task['task_id']} has invalid active_seconds.")
            elapsed.append({
                "image_id": image_id,
                "reviewer_id": task["reviewer_id"],
                "condition": task["condition"],
                "active_seconds": active_seconds,
            })
        return image_ids, coco_rows, elapsed

    if plan["mode"] == "workflow_pilot":
        conditions: dict[str, Any] = {}
        condition_elapsed: dict[str, list[float]] = defaultdict(list)
        condition_results: dict[str, dict[str, Any]] = {}
        for condition in ("scratch", "assisted"):
            task_rows = [row for row in plan_rows if row["condition"] == condition]
            ids, rows, elapsed = human_rows_for(task_rows)
            report, _ = _score_annotations(
                annotations_path, ids, rows, category_ids, int(plan["seed"]), bootstrap_resamples
            )
            for item in elapsed:
                condition_elapsed[condition].append(item["active_seconds"])
            report["time_seconds"] = {
                "task_count": len(elapsed),
                "median": median(condition_elapsed[condition]) if condition_elapsed[condition] else None,
                "mean": mean(condition_elapsed[condition]) if condition_elapsed[condition] else None,
                "p95": (
                    float(np.quantile(condition_elapsed[condition], 0.95))
                    if condition_elapsed[condition] else None
                ),
            }
            conditions[condition] = report
            condition_results[condition] = {"tasks": task_rows, "elapsed": elapsed}
        assisted_tasks = condition_results["assisted"]["tasks"]
        changed_images = 0
        draft_box_count = 0
        unchanged_draft_boxes = 0
        added_final_boxes = 0
        for task in assisted_tasks:
            image_id = int(task["image_id"])
            prediction = records.get(image_id, {})
            draft_annotations = (
                [
                    {"category": detection["category"], "bbox_xyxy_norm": detection["bbox_xyxy_norm"]}
                    for detection in prediction.get("validation", {}).get("detections", [])
                ]
                if prediction.get("status") == "valid" else []
            )
            final_annotations = events_by_task[task["task_id"]]["final_annotations"]
            draft_coco = _to_coco_results(image_id, image_rows[image_id], draft_annotations, category_by_name, score=1.0)
            final_coco = _to_coco_results(image_id, image_rows[image_id], final_annotations, category_by_name, score=1.0)
            unchanged, changed_or_removed, added = _simple_match_counts(draft_coco, final_coco, threshold=0.95)
            draft_box_count += len(draft_coco)
            unchanged_draft_boxes += unchanged
            added_final_boxes += added
            if changed_or_removed or added:
                changed_images += 1
        result["draft_correction_rate"] = {
            "comparison": "same category and IoU >= 0.95 counts as unchanged",
            "assisted_images_reviewed": len(assisted_tasks),
            "images_with_any_change": changed_images,
            "image_change_rate": changed_images / len(assisted_tasks) if assisted_tasks else None,
            "draft_boxes_presented": draft_box_count,
            "draft_boxes_unchanged": unchanged_draft_boxes,
            "draft_boxes_changed_or_removed": draft_box_count - unchanged_draft_boxes,
            "draft_box_change_rate": (
                (draft_box_count - unchanged_draft_boxes) / draft_box_count if draft_box_count else None
            ),
            "final_boxes_added_or_replaced": added_final_boxes,
        }
        paired_time_differences = []
        scratch_times = {
            row["image_id"]: row
            for row in condition_results["scratch"]["elapsed"]
        }
        assisted_times = {
            row["image_id"]: row
            for row in condition_results["assisted"]["elapsed"]
        }
        for image_id, scratch_row in scratch_times.items():
            if image_id in assisted_times:
                assisted_row = assisted_times[image_id]
                paired_time_differences.append({
                    "image_id": image_id,
                    "scratch_reviewer_id": scratch_row["reviewer_id"],
                    "assisted_reviewer_id": assisted_row["reviewer_id"],
                    "assisted_minus_scratch_seconds": assisted_row["active_seconds"] - scratch_row["active_seconds"],
                })
        result["conditions"] = conditions
        result["paired_time_differences"] = paired_time_differences
        result["comparison_note"] = (
            "With one reviewer, conditions use different randomized images and are a personal pilot. "
            "With two reviewers, image assignments are counterbalanced."
        )
    else:
        task_rows = plan_rows
        ids, human_results, elapsed = human_rows_for(task_rows)
        human_report, _ = _score_annotations(
            annotations_path, ids, human_results, category_ids, int(plan["seed"]), bootstrap_resamples
        )
        by_image_human = {
            int(task["image_id"]): events_by_task[task["task_id"]]["final_annotations"]
            for task in task_rows
        }
        row_by_id = image_rows
        model_results: list[dict[str, Any]] = []
        for image_id in ids:
            record = records.get(image_id, {})
            if record.get("status") == "valid":
                model_results.extend(_to_coco_results(
                    image_id,
                    row_by_id[image_id],
                    [
                        {"category": detection["category"], "bbox_xyxy_norm": detection["bbox_xyxy_norm"]}
                        for detection in record["validation"]["detections"]
                    ],
                    category_by_name,
                    score=1.0,
                ))
        model_report, _ = _score_annotations(
            annotations_path, ids, model_results, category_ids, int(plan["seed"]), bootstrap_resamples
        )

        coco_gt = coco_api_subset(annotations_path, ids)
        references_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        audit_image_ids = set(ids)
        for annotation in coco_gt.dataset["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in audit_image_ids or annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
                continue
            references_by_image[image_id].append({
                "category_id": int(annotation["category_id"]),
                "bbox": [float(value) for value in annotation["bbox"]],
            })

        candidate_reference_issues = []
        strata_counts: dict[str, int] = defaultdict(int)
        for task in task_rows:
            image_id = int(task["image_id"])
            strata_counts[str(task["stratum"])] += 1
            human = _to_coco_results(
                image_id, image_rows[image_id], by_image_human[image_id], category_by_name, score=1.0
            )
            model_record = records.get(image_id, {})
            model = _to_coco_results(
                image_id,
                image_rows[image_id],
                [
                    {"category": detection["category"], "bbox_xyxy_norm": detection["bbox_xyxy_norm"]}
                    for detection in model_record.get("validation", {}).get("detections", [])
                ] if model_record.get("status") == "valid" else [],
                category_by_name,
                score=1.0,
            )
            reference = references_by_image.get(image_id, [])
            hm, _, _ = _simple_match_counts(human, model)
            human_gt, human_fp, human_fn = _simple_match_counts(human, reference)
            model_gt, model_fp, model_fn = _simple_match_counts(model, reference)
            if hm > 0 and hm == len(human) == len(model) and (human_fp or human_fn) and (model_fp or model_fn):
                candidate_reference_issues.append({
                    "image_id": image_id,
                    "stratum": task["stratum"],
                    "human_model_matches": hm,
                    "human_matches_to_reference": human_gt,
                    "model_matches_to_reference": model_gt,
                    "human_reference_unmatched": {"predictions": human_fp, "references": human_fn},
                    "model_reference_unmatched": {"predictions": model_fp, "references": model_fn},
                    "interpretation": "Candidate for adjudication only; agreement does not establish which annotation is correct.",
                })
        result["blind_audit"] = {
            "strata_counts": dict(strata_counts),
            "human_vs_coco": human_report,
            "model_vs_coco_on_same_images": model_report,
            "candidate_reference_issues": candidate_reference_issues,
            "candidate_reference_issue_count": len(candidate_reference_issues),
            "time_seconds_median": median([item["active_seconds"] for item in elapsed]) if elapsed else None,
        }

    output_path = plan_path.parent / "study_report.json"
    write_json(output_path, result)
    output_path.with_suffix(".md").write_text(render_study_markdown(result), encoding="utf-8")
    return result

