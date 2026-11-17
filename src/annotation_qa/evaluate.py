"""COCO-compatible evaluation, operational rates, image bootstrap, and review queue."""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.cocoeval import COCOeval

from .coco_data import coco_api_subset
from .io_utils import load_json, read_jsonl, sha256_file, utc_now, write_json
from .reporting import render_metrics_markdown

MAX_DETECTIONS = 100
IOU_THRESHOLDS = (0.50, 0.75)


def _empty_detection_coco(coco_gt, image_ids: list[int]):
    from pycocotools.coco import COCO

    coco_dt = COCO()
    coco_dt.dataset = {
        "images": [coco_gt.imgs[image_id] for image_id in image_ids],
        "categories": list(coco_gt.cats.values()),
        "annotations": [],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt.createIndex()
    return coco_dt


def _bbox_iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_w = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_h = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_w * intersection_h
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - intersection
    return intersection / union if union > 0 else 0.0


def _overlap_with_reference(prediction: dict[str, Any], reference: dict[str, Any], *, crowd: bool) -> float:
    pred_box = prediction["bbox"]
    ref_box = reference["bbox"]
    px, py, pw, ph = (float(value) for value in pred_box)
    rx, ry, rw, rh = (float(value) for value in ref_box)
    intersection = max(0.0, min(px + pw, rx + rw) - max(px, rx)) * max(
        0.0, min(py + ph, ry + rh) - max(py, ry)
    )
    if crowd:
        prediction_area = max(0.0, pw) * max(0.0, ph)
        return intersection / prediction_area if prediction_area else 0.0
    return _bbox_iou_xywh(pred_box, ref_box)


def _maximum_cardinality_matches(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int]]:
    edges: list[list[tuple[float, int]]] = []
    for prediction in predictions:
        eligible = [
            (_overlap_with_reference(prediction, reference, crowd=False), reference_index)
            for reference_index, reference in enumerate(references)
        ]
        edges.append(sorted(
            [(overlap, reference_index) for overlap, reference_index in eligible if overlap >= threshold],
            key=lambda pair: (-pair[0], int(references[pair[1]]["id"])),
        ))

    reference_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, visited: set[int]) -> bool:
        for _, reference_index in edges[prediction_index]:
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
    pairs = []
    for reference_index, prediction_index in sorted(reference_to_prediction.items(), key=lambda item: item[1]):
        overlap = _overlap_with_reference(
            predictions[prediction_index], references[reference_index], crowd=False
        )
        pairs.append((prediction_index, reference_index, overlap))
    matched_predictions = {prediction_index for prediction_index, _, _ in pairs}
    unmatched = [index for index in range(len(predictions)) if index not in matched_predictions]
    return pairs, unmatched


def _count_metrics(
    evaluator: COCOeval,
    image_ids: list[int],
    category_ids: list[int],
) -> tuple[dict[str, Any], dict[tuple[int, int, float], dict[str, int]], dict[str, list[float]]]:
    image_category_counts: dict[tuple[int, int, float], dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    matched_ious: dict[str, list[float]] = {str(threshold): [] for threshold in IOU_THRESHOLDS}
    result_by_threshold: dict[str, Any] = {}
    predictions_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    references_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    ignored_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in evaluator.cocoDt.dataset.get("annotations", []):
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        predictions_by_key[key].append(prediction)
    for reference in evaluator.cocoGt.dataset.get("annotations", []):
        key = (int(reference["image_id"]), int(reference["category_id"]))
        if reference.get("iscrowd", 0) or reference.get("ignore", 0):
            ignored_by_key[key].append(reference)
        else:
            references_by_key[key].append(reference)

    for threshold in IOU_THRESHOLDS:
        per_category = {
            category_id: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "ignored_gt": 0}
            for category_id in category_ids
        }
        for image_id in image_ids:
            for category_id in category_ids:
                key = (image_id, category_id)
                predictions = sorted(
                    predictions_by_key.get(key, []),
                    key=lambda row: (int(row.get("id", 0)), tuple(float(value) for value in row["bbox"])),
                )[:MAX_DETECTIONS]
                references = sorted(
                    references_by_key.get(key, []), key=lambda row: int(row.get("id", 0))
                )
                ignored_references = ignored_by_key.get(key, [])
                matches, unmatched_predictions = _maximum_cardinality_matches(
                    predictions, references, threshold
                )
                ignored_predictions = {
                    prediction_index
                    for prediction_index in unmatched_predictions
                    if any(
                        _overlap_with_reference(
                            predictions[prediction_index],
                            reference,
                            crowd=bool(reference.get("iscrowd", 0)),
                        ) >= threshold
                        for reference in ignored_references
                    )
                }
                tp_count = len(matches)
                fp_count = len(unmatched_predictions) - len(ignored_predictions)
                fn_count = len(references) - tp_count
                counts = per_category[category_id]
                counts["tp"] += tp_count
                counts["fp"] += fp_count
                counts["fn"] += fn_count
                counts["support"] += len(references)
                counts["ignored_gt"] += len(ignored_references)
                image_counts = image_category_counts[(image_id, category_id, threshold)]
                image_counts["tp"] += tp_count
                image_counts["fp"] += fp_count
                image_counts["fn"] += fn_count
                matched_ious[str(threshold)].extend(overlap for _, _, overlap in matches)

        total = {key: sum(values[key] for values in per_category.values()) for key in ("tp", "fp", "fn", "support", "ignored_gt")}
        total["precision"] = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else None
        total["recall"] = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else None
        total["iou_threshold"] = threshold
        per_class = {}
        for category_id, counts in per_category.items():
            denominator_p = counts["tp"] + counts["fp"]
            denominator_r = counts["tp"] + counts["fn"]
            per_class[str(category_id)] = {
                **counts,
                "precision": counts["tp"] / denominator_p if denominator_p else None,
                "recall": counts["tp"] / denominator_r if denominator_r else None,
            }
        result_by_threshold[str(threshold)] = {"micro": total, "per_class": per_class}

    return result_by_threshold, image_category_counts, matched_ious


def _bootstrap_micro(
    image_ids: list[int],
    category_ids: list[int],
    image_category_counts: dict[tuple[int, int, float], dict[str, int]],
    seed: int,
    resamples: int = 1000,
) -> dict[str, Any]:
    if not image_ids:
        raise ValueError("Cannot bootstrap an empty image sample.")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    per_image: dict[float, np.ndarray] = {}
    for threshold in IOU_THRESHOLDS:
        matrix = np.zeros((len(image_ids), 3), dtype=np.int64)
        for index, image_id in enumerate(image_ids):
            for category_id in category_ids:
                values = image_category_counts.get((image_id, category_id, threshold))
                if values:
                    matrix[index] += [values["tp"], values["fp"], values["fn"]]
        per_image[threshold] = matrix

    intervals: dict[str, Any] = {}
    for threshold, matrix in per_image.items():
        estimates = np.zeros((resamples, 2), dtype=float)
        for sample_index in range(resamples):
            draw = rng.integers(0, len(image_ids), size=len(image_ids))
            tp, fp, fn = matrix[draw].sum(axis=0)
            estimates[sample_index, 0] = tp / (tp + fp) if tp + fp else 0.0
            estimates[sample_index, 1] = tp / (tp + fn) if tp + fn else 0.0
        intervals[str(threshold)] = {
            "method": "percentile bootstrap resampling images with replacement",
            "resamples": resamples,
            "seed": seed,
            "micro_precision_95_ci": [float(value) for value in np.quantile(estimates[:, 0], [0.025, 0.975])],
            "micro_recall_95_ci": [float(value) for value in np.quantile(estimates[:, 1], [0.025, 0.975])],
        }
    return intervals


def evaluate_run(
    *,
    annotations_path: Path,
    manifest_path: Path,
    run_dir: Path,
    output_dir: Path | None = None,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("split") != "evaluation":
        raise ValueError("Only evaluation-split runs can be scored as final benchmark results.")
    if not metadata.get("completed_at"):
        raise ValueError("Inference run is incomplete; only completed evaluation runs can be scored.")
    if sha256_file(manifest_path) != metadata.get("manifest_sha256"):
        raise ValueError("Manifest checksum differs from the inference run.")
    if metadata.get("annotations_sha256") != manifest.get("annotations_sha256"):
        raise ValueError("Run and manifest refer to different COCO annotation files.")
    if sha256_file(annotations_path) != manifest["annotations_sha256"]:
        raise ValueError("COCO annotation file checksum differs from the locked manifest.")

    sample_rows = [row for row in manifest["images"] if row["split"] == "evaluation"]
    image_ids = [int(row["image_id"]) for row in sample_rows]
    row_by_id = {int(row["image_id"]): row for row in sample_rows}
    category_by_name = {str(row["name"]): int(row["id"]) for row in manifest["categories"]}
    category_by_id = {int(row["id"]): str(row["name"]) for row in manifest["categories"]}
    category_ids = sorted(category_by_id)
    records = read_jsonl(run_dir / "predictions.jsonl")
    records_by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        image_id = int(record["image_id"])
        if image_id in records_by_id:
            raise ValueError(f"Duplicate prediction record for image {image_id}.")
        if image_id not in row_by_id:
            raise ValueError(f"Prediction image {image_id} is outside the evaluation sample.")
        records_by_id[image_id] = record

    coco_results: list[dict[str, Any]] = []
    valid_detection_count = 0
    for image_id in image_ids:
        record = records_by_id.get(image_id)
        if not record or record.get("status") != "valid":
            continue
        image = row_by_id[image_id]
        width, height = float(image["width"]), float(image["height"])
        for detection in record["validation"]["detections"]:
            category_id = category_by_name.get(str(detection["category"]))
            if category_id is None:
                raise ValueError("A prediction marked valid has a category outside the locked COCO category list.")
            x1n, y1n, x2n, y2n = (float(value) for value in detection["bbox_xyxy_norm"])
            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1n * width, y1n * height, (x2n - x1n) * width, (y2n - y1n) * height],
                "score": float(detection["rank_score"]),
            })
            valid_detection_count += 1

    coco_gt = coco_api_subset(annotations_path, image_ids)
    if coco_results:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(coco_results)
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

    score_values = [float(row["score"]) for row in coco_results]
    distinct_scores = len(set(score_values))
    ap_available = len(score_values) >= 2 and distinct_scores >= 2
    coco_ap = None
    per_class_ap: dict[str, float | None] = {str(category_id): None for category_id in category_ids}
    if ap_available:
        stats = evaluator.stats
        coco_ap = {
            "ap": float(stats[0]),
            "ap50": float(stats[1]),
            "ap75": float(stats[2]),
            "ar100": float(stats[8]),
        }
        precision_tensor = evaluator.eval["precision"]
        for category_index, category_id in enumerate(evaluator.params.catIds):
            values = precision_tensor[:, :, category_index, 0, 2]
            valid = values[values > -1]
            per_class_ap[str(category_id)] = float(valid.mean()) if valid.size else None

    threshold_results, image_category_counts, matched_ious = _count_metrics(
        evaluator, image_ids, category_ids
    )
    seed = int(manifest["sampling"]["seed"])
    intervals = _bootstrap_micro(image_ids, category_ids, image_category_counts, seed, bootstrap_resamples)

    status_counts: dict[str, int] = defaultdict(int)
    review_queue: list[dict[str, Any]] = []
    for image_id in image_ids:
        record = records_by_id.get(image_id)
        status = str(record.get("status")) if record else "inference_missing"
        status_counts[status] += 1
        reasons: list[str] = []
        if status not in {"valid"}:
            reasons.append("invalid_or_missing_model_output")
        # Per-image FP/FN details are populated below from pycocotools matching.
        has_mismatch = False
        for category_id in category_ids:
            counts = image_category_counts.get((image_id, category_id, 0.5), {})
            if counts.get("fp", 0) or counts.get("fn", 0):
                has_mismatch = True
                break
        if has_mismatch and status == "valid":
            reasons.append("coco_reference_disagreement_at_iou_0_50")
        if reasons:
            review_queue.append({
                "image_id": image_id,
                "file_name": row_by_id[image_id]["file_name"],
                "status": status,
                "reasons": reasons,
                "record_present": record is not None,
            })

    per_class_support = {}
    for category_id, category_name in category_by_id.items():
        counts = threshold_results["0.5"]["per_class"][str(category_id)]
        per_class_support[str(category_id)] = {
            "name": category_name,
            "evaluable_reference_objects": counts["support"],
            "ignored_or_crowd_objects": counts["ignored_gt"],
            "ap": per_class_ap[str(category_id)],
            "iou_0_50": threshold_results["0.5"]["per_class"][str(category_id)],
            "iou_0_75": threshold_results["0.75"]["per_class"][str(category_id)],
        }

    completed_latencies = [
        float(row["elapsed_seconds"])
        for row in records
        if row.get("response") is not None and row.get("elapsed_seconds") is not None
    ]
    first_response = next((row for row in records if row.get("response") is not None), None)
    first_response_duration = (
        float(first_response["elapsed_seconds"])
        if first_response and first_response.get("elapsed_seconds") is not None else None
    )
    first_model_load = (
        float(first_response["ollama_load_duration_ns"]) / 1_000_000_000
        if first_response and first_response.get("ollama_load_duration_ns") is not None else None
    )
    output_failure_statuses = {"parse_failure", "schema_failure", "semantic_failure"}
    inference_failure_count = status_counts.get("inference_failure", 0) + status_counts.get("inference_missing", 0)
    invalid_output_count = sum(status_counts.get(status, 0) for status in output_failure_statuses)
    model_response_count = len(records) - status_counts.get("inference_failure", 0)

    output = output_dir or run_dir
    metrics = {
        "report_version": 1,
        "created_at": utc_now(),
        "run_id": metadata["run_id"],
        "model": metadata["model"],
        "manifest_sha256": metadata["manifest_sha256"],
        "annotations_sha256": manifest["annotations_sha256"],
        "evaluation_image_count": len(image_ids),
        "images_attempted": len(records),
        "coverage": {
            "sample_images": len(image_ids),
            "attempted_images": {"count": len(records), "denominator": len(image_ids)},
            "model_responses_received": {"count": model_response_count, "denominator": len(records)},
            "valid_outputs": {"count": status_counts.get("valid", 0), "denominator": model_response_count},
        },
        "images_with_valid_response": status_counts.get("valid", 0),
        "images_missing_response_record": len(image_ids) - len(records),
        "response_status_counts": dict(sorted(status_counts.items())),
        "invalid_output_rate": {
            "numerator": invalid_output_count,
            "denominator": model_response_count,
            "value": invalid_output_count / model_response_count if model_response_count else None,
        },
        "inference_failure_rate": {
            "numerator": inference_failure_count,
            "denominator": len(image_ids),
            "value": inference_failure_count / len(image_ids),
        },
        "precision_recall_method": (
            "Custom class-aware maximum-cardinality one-to-one matching at each IoU threshold; "
            "eligible pairs are visited by descending IoU; rank_score is ignored for these fixed-set metrics. "
            "At most 100 detections per image are considered. Crowd/ignore references are excluded from recall; "
            "unmatched detections overlapping ignored references are excluded from FP, using intersection over "
            "detection area for crowd boxes and ordinary IoU for other ignored boxes."
        ),
        "valid_detection_count": valid_detection_count,
        "inference_latency_seconds": {
            "successful_response_count": len(completed_latencies),
            "median": float(np.median(completed_latencies)) if completed_latencies else None,
            "p95": float(np.quantile(completed_latencies, 0.95)) if completed_latencies else None,
            "first_successful_request_total": first_response_duration,
            "first_successful_request_model_load": first_model_load,
            "first_request_note": "Model-load duration is a true cold-start measure only if Ollama had not already loaded the model.",
        },
        "max_detections_per_image": MAX_DETECTIONS,
        "coco_ap": {
            "available": ap_available,
            "reason_omitted": None if ap_available else "Fewer than two valid detection scores or all scores are constant.",
            "score_interpretation": "relative ranking only; not a calibrated probability",
            "metrics": coco_ap,
        },
        "threshold_precision_recall": threshold_results,
        "matched_box_iou_at_least_0_50": {
            "count": len(matched_ious["0.5"]),
            "mean": float(np.mean(matched_ious["0.5"])) if matched_ious["0.5"] else None,
            "median": float(np.median(matched_ious["0.5"])) if matched_ious["0.5"] else None,
        },
        "image_bootstrap_95_ci": intervals,
        "per_class": per_class_support,
        "review_queue": {
            "image_count": len(review_queue),
            "reason_counts": {
                "invalid_or_missing_model_output": sum("invalid_or_missing_model_output" in row["reasons"] for row in review_queue),
                "coco_reference_disagreement_at_iou_0_50": sum("coco_reference_disagreement_at_iou_0_50" in row["reasons"] for row in review_queue),
            },
        },
        "limitations": [
            "COCO annotations are reference labels, not perfect ground truth.",
            "Results apply only to this locked random sample and local model revision.",
            "Fixed model digest and settings improve repeatability, but execution backend or hardware can affect exact outputs.",
            "Public COCO images may overlap with the model's training data.",
            "AP is omitted when rank scores do not provide a non-constant ranking.",
        ],
    }
    write_json(output / "metrics.json", metrics)
    (output / "metrics.md").write_text(render_metrics_markdown(metrics), encoding="utf-8")
    write_json(output / "review_queue.json", {"run_id": metadata["run_id"], "images": review_queue})
    write_json(output / "coco_results.json", coco_results)
    return metrics

