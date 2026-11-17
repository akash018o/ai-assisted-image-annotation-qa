# Local Vision Model Annotation QA

I built this project to see how well a local vision model can draft object-detection labels, and to make the checks and human review around those labels measurable. I use a small, fixed sample of COCO 2017 validation images as the reference set.

The model suggests categories and boxes. The pipeline checks the output, keeps invalid results visible, compares valid predictions with COCO annotations, and puts cases that need attention into a review workflow. It also records the model and prompt version, raw response, validation outcome, and review events so a run can be traced later.

This is a local prototype and a portfolio project. I have not included benchmark results here; those should be added only after running the evaluation and reviewing a sample of images. COCO is useful for a repeatable object-detection study, but it does not establish performance on other kinds of annotation work, such as LiDAR or video.

## What I implemented

- A seeded COCO sampler that creates separate development and evaluation splits and records image and annotation checksums.
- Local inference through Ollama, using Qwen3-VL and a versioned built-in prompt. I can optionally supply a different prompt from a local file.
- Strict checks for category names, required fields, normalized coordinates, and boxes that fall outside the image. Invalid model output is recorded and flagged rather than silently fixed.
- COCO detection metrics, class-level precision and recall, box-overlap summaries, invalid-output and inference-failure rates, and a review queue.
- A Streamlit interface for inspecting predictions against the reference labels, conducting a blind human audit, and piloting draft-assisted annotation.
- JSONL records for inference attempts and reviewer actions.

## What I measure

The evaluation reports COCO AP metrics when the model provides usable ranking scores, along with fixed-threshold precision and recall, class support, matched-box IoU, invalid-output rate, inference failures, and the share of images sent for review. Bootstrap intervals are included for micro precision and recall. The blind-audit and workflow-pilot reports are separate from the model benchmark; a small one-reviewer pilot is personal workflow evidence, not a general productivity claim.

COCO annotations are reference labels rather than perfect ground truth. I plan to inspect a small set of images manually and treat model/reference disagreements as cases to investigate, not automatic proof that either side is correct.

## What I need

- Python 3.10 or newer (Python 3.11 recommended).
- Ollama installed locally with `qwen3-vl:8b-instruct` available.
- COCO 2017 validation images and the `instances_val2017.json` annotation file, downloaded separately.

The project does not redistribute COCO images. COCO annotation files are under CC BY 4.0; image rights remain with their respective owners. See the [COCO terms of use](https://cocodataset.org/dataset/termsofuse.htm) and [COCO detection dataset page](https://cocodataset.org/dataset/detection-2017.htm).

## Set it up

Open PowerShell in this repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[review]"
```

I keep COCO files locally in this layout:

```text
data/
  annotations/instances_val2017.json
  val2017/000000000139.jpg
manifests/
runs/
```

The dataset, manifests, model files, prompts, and run outputs are local and excluded from this repository. Ollama must be running on the same computer before inference. The default model is `qwen3-vl:8b-instruct`; if my local Ollama tag differs, I pass it with `--model`.

## Create a sample manifest

The default sample has 30 development images and 300 evaluation images. The sample command records the split and checksums and refuses to overwrite an existing manifest.

```powershell
annotation-qa sample `
  --annotations data\annotations\instances_val2017.json `
  --images data\val2017 `
  --output manifests\coco-val-sample.json `
  --seed 20261117 `
  --evaluation-count 300 `
  --development-count 30
```

## Run the model

I use the development split while adjusting settings. Once I choose the model and prompt version, I keep them fixed for the evaluation split. The prompt is built into the package, so these commands do not need a prompt file.

```powershell
annotation-qa infer `
  --manifest manifests\coco-val-sample.json `
  --images data\val2017 `
  --output-dir runs\development-v1 `
  --model qwen3-vl:8b-instruct `
  --split development `
  --prompt-version draft-v1

annotation-qa infer `
  --manifest manifests\coco-val-sample.json `
  --images data\val2017 `
  --output-dir runs\evaluation-v1 `
  --model qwen3-vl:8b-instruct `
  --split evaluation `
  --prompt-version draft-v1
```

Each run uses a new output directory. The pipeline saves the raw responses and records each attempt, including transport failures. Invalid model content is retained for review and is not retried or repaired. The prompt hash, model details, and generation settings are saved with the run.

## Evaluate and review

```powershell
annotation-qa evaluate `
  --annotations data\annotations\instances_val2017.json `
  --manifest manifests\coco-val-sample.json `
  --run-dir runs\evaluation-v1

annotation-qa review
```

The evaluation writes `metrics.json` and `metrics.md` into the run folder. The review screen lets me inspect model/reference disagreements and invalid cases. Reference boxes are visible in benchmark review; the blind-audit activity hides both sets of labels until the reviewer submits an independent annotation.

## Optional human studies

To create a blind-audit plan after inference, I run:

```powershell
annotation-qa study-plan `
  --mode blind_audit `
  --manifest manifests\coco-val-sample.json `
  --run-dir runs\evaluation-v1 `
  --output runs\studies\blind-audit-plan.json `
  --reviewers reviewer-1 `
  --count 30
```

For a small comparison of manual annotation time with draft-assisted correction, I use `--mode workflow_pilot` and a separate plan. The reviewer completes the assigned tasks in the review screen, then I score the resulting event log with `annotation-qa study-evaluate`. This pilot can help me understand my own workflow; it is too small to support broad productivity claims.

## Current limitations

- A 300-image evaluation sample may contain little support for some of COCO's 80 categories, so I report class support alongside class metrics.
- COCO references can contain annotation errors. I manually inspect a sample before interpreting disagreements.
- Public benchmark images may have appeared in model training data.
- Natural-image COCO results do not demonstrate quality on LiDAR, video, or a vendor's production data.
- I have not added numerical results to this README because they need to come from a completed run.
