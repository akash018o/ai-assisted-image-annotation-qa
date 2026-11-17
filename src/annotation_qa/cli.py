"""Command-line entry points for sampling, local inference, evaluation, and review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .evaluate import evaluate_run
from .manifest import create_manifest
from .runner import run_inference
from .study import create_study_plan, evaluate_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation-qa",
        description="Reproducible local Qwen3-VL annotation benchmark for COCO.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="Create a locked, seeded COCO sample manifest.")
    sample.add_argument("--annotations", type=Path, required=True, help="instances_val2017.json")
    sample.add_argument("--images", type=Path, required=True, help="COCO val2017 image directory")
    sample.add_argument("--output", type=Path, required=True, help="New manifest JSON path")
    sample.add_argument("--seed", type=int, default=20261117)
    sample.add_argument("--evaluation-count", type=int, default=300)
    sample.add_argument("--development-count", type=int, default=30)

    infer = commands.add_parser("infer", help="Run local Ollama inference on one locked sample split.")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--images", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True, help="New directory for this run")
    infer.add_argument("--model", default="qwen3-vl:8b-instruct")
    infer.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    infer.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    infer.add_argument("--timeout-seconds", type=int, default=600)
    infer.add_argument("--transport-retries", type=int, default=1)
    infer.add_argument("--prompt-file", type=Path, help="Optional prompt template with <<categories>>, <<width>>, <<height>>")
    infer.add_argument("--prompt-version", default="draft-v1")
    infer.add_argument("--generation-seed", type=int, default=20261117)

    evaluate = commands.add_parser("evaluate", help="Score an evaluation-split run with COCOeval.")
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--bootstrap-resamples", type=int, default=1000)

    review = commands.add_parser("review", help="Open the local Streamlit benchmark review screen.")
    review.add_argument("--server-port", type=int, default=8501)

    plan = commands.add_parser("study-plan", help="Create a locked blind-audit or workflow-pilot plan.")
    plan.add_argument("--mode", choices=("blind_audit", "workflow_pilot"), required=True)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--run-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--reviewers", default="reviewer-1", help="One or two comma-separated reviewer IDs")
    plan.add_argument("--count", type=int, default=24)
    plan.add_argument("--seed", type=int, default=20261117)

    study_eval = commands.add_parser("study-evaluate", help="Score completed blind-audit/workflow-pilot events.")
    study_eval.add_argument("--annotations", type=Path, required=True)
    study_eval.add_argument("--manifest", type=Path, required=True)
    study_eval.add_argument("--run-dir", type=Path, required=True)
    study_eval.add_argument("--plan", type=Path, required=True)
    study_eval.add_argument("--events", type=Path, required=True)
    study_eval.add_argument("--bootstrap-resamples", type=int, default=1000)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "sample":
        manifest = create_manifest(
            args.annotations,
            args.images,
            args.output,
            seed=args.seed,
            evaluation_count=args.evaluation_count,
            development_count=args.development_count,
        )
        summary = {
            "manifest": str(args.output.resolve()),
            "evaluation_images": manifest["sampling"]["evaluation_count"],
            "development_images": manifest["sampling"]["development_count"],
            "seed": manifest["sampling"]["seed"],
            "annotations_sha256": manifest["annotations_sha256"],
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "infer":
        metadata = run_inference(
            manifest_path=args.manifest,
            images_dir=args.images,
            output_dir=args.output_dir,
            model_name=args.model,
            split=args.split,
            ollama_host=args.ollama_host,
            timeout_seconds=args.timeout_seconds,
            transport_retries=args.transport_retries,
            prompt_file=args.prompt_file,
            prompt_version=args.prompt_version,
            generation_seed=args.generation_seed,
        )
        print(json.dumps({"run_id": metadata["run_id"], "output_dir": str(args.output_dir.resolve())}, indent=2))
    elif args.command == "evaluate":
        metrics = evaluate_run(
            annotations_path=args.annotations,
            manifest_path=args.manifest,
            run_dir=args.run_dir,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        print(json.dumps({
            "run_id": metrics["run_id"],
            "evaluation_images": metrics["evaluation_image_count"],
            "review_queue_images": metrics["review_queue"]["image_count"],
            "metrics_file": str((args.run_dir / "metrics.json").resolve()),
        }, indent=2))
    elif args.command == "review":
        if importlib_util_find_spec("streamlit") is None:
            raise SystemExit("Install the review extra first: pip install -e '.[review]'")
        app_path = Path(__file__).with_name("review_app.py")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(args.server_port),
        ]
        raise SystemExit(subprocess.call(command))
    elif args.command == "study-plan":
        reviewers = [value.strip() for value in args.reviewers.split(",") if value.strip()]
        plan = create_study_plan(
            mode=args.mode,
            manifest_path=args.manifest,
            run_dir=args.run_dir,
            output_path=args.output,
            reviewers=reviewers,
            count=args.count,
            seed=args.seed,
        )
        print(json.dumps({
            "plan": str(args.output.resolve()),
            "mode": plan["mode"],
            "images": plan["selected_image_count"],
            "tasks": len(plan["tasks"]),
            "strata": plan["strata_counts"],
        }, indent=2))
    elif args.command == "study-evaluate":
        report = evaluate_study(
            annotations_path=args.annotations,
            manifest_path=args.manifest,
            run_dir=args.run_dir,
            plan_path=args.plan,
            events_path=args.events,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        print(json.dumps({
            "report": str((args.plan.parent / "study_report.json").resolve()),
            "mode": report["mode"],
            "tasks": report["task_count"],
        }, indent=2))


def importlib_util_find_spec(name: str):
    # Kept local so --help and non-review commands do not import Streamlit.
    import importlib.util

    return importlib.util.find_spec(name)


if __name__ == "__main__":
    main()

