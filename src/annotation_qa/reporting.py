"""Human-readable Markdown views of the machine-readable result files."""

from __future__ import annotations

from typing import Any


def _number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def render_metrics_markdown(metrics: dict[str, Any]) -> str:
    ap = metrics["coco_ap"]
    ap_line = "Omitted: rank scores did not meet the predeclared variation rule."
    if ap["available"]:
        values = ap["metrics"]
        ap_line = (
            f"AP **{_number(values['ap'])}**, AP50 **{_number(values['ap50'])}**, "
            f"AP75 **{_number(values['ap75'])}**, AR100 **{_number(values['ar100'])}**."
        )
    statuses = metrics["response_status_counts"]
    queue = metrics["review_queue"]
    coverage = metrics["coverage"]
    latency = metrics["inference_latency_seconds"]
    lines = [
        "# COCO Annotation QA Evaluation",
        "",
        f"Run ID: `{metrics['run_id']}`  ",
        f"Evaluation images: **{metrics['evaluation_image_count']}**  ",
        f"Valid detections: **{metrics['valid_detection_count']}**  ",
        f"Model: `{metrics['model'].get('model')}` ({metrics['model'].get('digest')})",
        "",
        "## Coverage and output validity",
        "",
        f"- Attempted: {coverage['attempted_images']['count']} / {coverage['attempted_images']['denominator']} images.",
        f"- Model responses received: {coverage['model_responses_received']['count']}.",
        f"- Valid outputs: {coverage['valid_outputs']['count']} / {coverage['valid_outputs']['denominator']} responses.",
        f"- Invalid-output rate: {_percent(metrics['invalid_output_rate']['value'])} "
        f"({metrics['invalid_output_rate']['numerator']} / {metrics['invalid_output_rate']['denominator']} responses).",
        f"- Inference-failure rate: {_percent(metrics['inference_failure_rate']['value'])} "
        f"({metrics['inference_failure_rate']['numerator']} / {metrics['inference_failure_rate']['denominator']} sample images).",
        f"- Output statuses: {', '.join(f'{key}={value}' for key, value in sorted(statuses.items())) or 'none'}.",
        "",
        "## Detection quality",
        "",
        f"- COCO bbox metrics: {ap_line}",
        f"- Matched-box IoU at IoU ≥ 0.50: mean {_number(metrics['matched_box_iou_at_least_0_50']['mean'])}, "
        f"median {_number(metrics['matched_box_iou_at_least_0_50']['median'])} "
        f"(n={metrics['matched_box_iou_at_least_0_50']['count']}).",
        "- Fixed-set precision/recall uses the custom deterministic matching method described below.",
        "",
        "| IoU threshold | Micro precision | 95% image-bootstrap CI | Micro recall | 95% image-bootstrap CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in ("0.5", "0.75"):
        micro = metrics["threshold_precision_recall"][threshold]["micro"]
        interval = metrics["image_bootstrap_95_ci"][threshold]
        p_ci = interval["micro_precision_95_ci"]
        r_ci = interval["micro_recall_95_ci"]
        lines.append(
            f"| {threshold} | {_percent(micro['precision'])} | {_percent(p_ci[0])}–{_percent(p_ci[1])} "
            f"| {_percent(micro['recall'])} | {_percent(r_ci[0])}–{_percent(r_ci[1])} |"
        )
    lines.extend([
        "",
        "## Per-class results",
        "",
        "Precision and recall are micro counts within each class. Sparse classes are descriptive; support is shown.",
        "",
        "| COCO ID | Class | Evaluable refs | Ignored/crowd refs | AP | P@0.50 | R@0.50 | P@0.75 | R@0.75 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category_id, result in sorted(metrics["per_class"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {category_id} | {result['name']} | {result['evaluable_reference_objects']} "
            f"| {result['ignored_or_crowd_objects']} | {_number(result['ap'])} "
            f"| {_percent(result['iou_0_50']['precision'])} | {_percent(result['iou_0_50']['recall'])} "
            f"| {_percent(result['iou_0_75']['precision'])} | {_percent(result['iou_0_75']['recall'])} |"
        )
    lines.extend([
        "",
        "## Review workload and runtime",
        "",
        f"- Unique images routed to review: **{queue['image_count']} / {metrics['evaluation_image_count']}**.",
        f"- Invalid/missing-output cases: {queue['reason_counts']['invalid_or_missing_model_output']}.",
        f"- Reference disagreements at IoU 0.50: {queue['reason_counts']['coco_reference_disagreement_at_iou_0_50']}.",
        f"- Inference latency: median {_number(latency['median'], 2)} s, p95 {_number(latency['p95'], 2)} s "
        f"(n={latency['successful_response_count']} responses).",
        f"- First successful request: {_number(latency['first_successful_request_total'], 2)} s; "
        f"reported model-load component {_number(latency['first_successful_request_model_load'], 2)} s.",
        "",
        "## Matching and interpretation",
        "",
        metrics["precision_recall_method"],
        "",
        "COCO AP uses the official COCO evaluator and relative detection ranking scores; those scores are not calibrated probabilities. "
        "Bootstrap intervals resample images and cover micro precision/recall, not AP. COCO references can contain omissions or box disagreements. "
        "This small natural-image sample does not establish performance on other annotation modalities.",
        "",
        "### Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in metrics["limitations"])
    return "\n".join(lines) + "\n"


def render_study_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Human Review Study",
        "",
        f"Plan ID: `{report['plan_id']}`  ",
        f"Mode: **{report['mode']}**  ",
        f"Design: `{report['design']}`  ",
        f"Images: **{report['selected_image_count']}**; tasks: **{report['task_count']}**.",
        "",
        "## Results",
        "",
    ]
    if report["mode"] == "workflow_pilot":
        lines.extend([
            "| Condition | Tasks | Median seconds | Mean seconds | p95 seconds | P@0.50 | R@0.50 | P@0.75 | R@0.75 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for condition in ("scratch", "assisted"):
            details = report["conditions"][condition]
            timing = details["time_seconds"]
            p50 = details["precision_recall"]["0.5"]["micro"]
            p75 = details["precision_recall"]["0.75"]["micro"]
            lines.append(
                f"| {condition} | {timing['task_count']} | {_number(timing['median'], 1)} "
                f"| {_number(timing['mean'], 1)} | {_number(timing['p95'], 1)} "
                f"| {_percent(p50['precision'])} | {_percent(p50['recall'])} "
                f"| {_percent(p75['precision'])} | {_percent(p75['recall'])} |"
            )
        correction = report["draft_correction_rate"]
        lines.extend([
            "",
            f"- Assisted images with any draft change: {_percent(correction['image_change_rate'])} "
            f"({correction['images_with_any_change']} / {correction['assisted_images_reviewed']}).",
            f"- Draft boxes changed or removed: {_percent(correction['draft_box_change_rate'])} "
            f"({correction['draft_boxes_changed_or_removed']} / {correction['draft_boxes_presented']}).",
            f"- Final boxes added or replacing unmatched draft boxes: {correction['final_boxes_added_or_replaced']}.",
            "",
            report["comparison_note"],
        ])
    else:
        audit = report["blind_audit"]
        human = audit["human_vs_coco"]["precision_recall"]
        model = audit["model_vs_coco_on_same_images"]["precision_recall"]
        lines.extend([
            f"Strata counts: {audit['strata_counts']}.",
            "",
            "| Annotation source | P@0.50 | R@0.50 | P@0.75 | R@0.75 |",
            "|---|---:|---:|---:|---:|",
            f"| Independent human | {_percent(human['0.5']['micro']['precision'])} "
            f"| {_percent(human['0.5']['micro']['recall'])} | {_percent(human['0.75']['micro']['precision'])} "
            f"| {_percent(human['0.75']['micro']['recall'])} |",
            f"| Model draft | {_percent(model['0.5']['micro']['precision'])} "
            f"| {_percent(model['0.5']['micro']['recall'])} | {_percent(model['0.75']['micro']['precision'])} "
            f"| {_percent(model['0.75']['micro']['recall'])} |",
            "",
            f"Potential reference cases for adjudication: **{audit['candidate_reference_issue_count']}**. "
            "These indicate agreement between human and model that differs from COCO; they do not prove the reference is wrong.",
        ])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"

