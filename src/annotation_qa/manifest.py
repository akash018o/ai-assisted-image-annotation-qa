"""Create a locked, reproducible COCO development/evaluation sample manifest."""

from __future__ import annotations

import platform
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .coco_data import annotation_counts, read_catalog
from .io_utils import sha256_file, utc_now, write_json

ANNOTATIONS_LICENSE = "CC BY 4.0 (COCO Consortium)"
ANNOTATIONS_LICENSE_URL = "https://cocodataset.org/dataset/termsofuse.htm"
IMAGE_TERMS = "Flickr Terms of Use apply; COCO does not own image copyrights."
IMAGE_TERMS_URL = "https://www.flickr.com/help/terms"


def create_manifest(
    annotations_path: Path,
    images_dir: Path,
    output_path: Path,
    *,
    seed: int = 20261117,
    evaluation_count: int = 300,
    development_count: int = 30,
) -> dict[str, Any]:
    if evaluation_count < 1 or development_count < 1:
        raise ValueError("Evaluation and development counts must both be positive.")
    if output_path.exists():
        raise FileExistsError(f"Locked manifest already exists: {output_path}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotations_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {images_dir}")

    images, categories = read_catalog(annotations_path)
    requested = evaluation_count + development_count
    if requested > len(images):
        raise ValueError(f"Requested {requested} images, but annotation file has {len(images)}.")

    rng = random.Random(seed)
    sampled = rng.sample(images, requested)
    development = sampled[:development_count]
    evaluation = sampled[development_count:]
    sampled_ids = {int(row["id"]) for row in sampled}
    counts_by_image = annotation_counts(annotations_path, sampled_ids)

    manifest_images: list[dict[str, Any]] = []
    for split, rows in (("development", development), ("evaluation", evaluation)):
        for row in rows:
            image_id = int(row["id"])
            filename = str(row["file_name"])
            image_path = (images_dir / filename).resolve()
            if images_dir.resolve() not in image_path.parents:
                raise ValueError(f"Image path escapes image directory: {filename}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image {image_id} is missing: {image_path}")
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
            if actual_width != int(row["width"]) or actual_height != int(row["height"]):
                raise ValueError(
                    f"Image {image_id} dimensions differ from COCO metadata: "
                    f"file={actual_width}x{actual_height}, json={row['width']}x{row['height']}"
                )
            annotation_counts_for_image = counts_by_image[image_id]
            manifest_images.append({
                "image_id": image_id,
                "split": split,
                "file_name": filename,
                "width": actual_width,
                "height": actual_height,
                "sha256": sha256_file(image_path),
                "reference_annotation_count": annotation_counts_for_image["all"],
                "ignored_crowd_count": annotation_counts_for_image["crowd"],
            })

    manifest = {
        "manifest_version": 1,
        "created_at": utc_now(),
        "python_version": platform.python_version(),
        "dataset": "COCO 2017 object detection validation split",
        "annotations_file_name": annotations_path.name,
        "annotations_sha256": sha256_file(annotations_path),
        "annotations_license": ANNOTATIONS_LICENSE,
        "annotations_license_url": ANNOTATIONS_LICENSE_URL,
        "image_terms_note": IMAGE_TERMS,
        "image_terms_url": IMAGE_TERMS_URL,
        "sampling": {
            "method": "uniform random sample from sorted COCO image IDs",
            "seed": seed,
            "evaluation_count": evaluation_count,
            "development_count": development_count,
            "category_labels_used_for_selection": False,
        },
        "categories": categories,
        "images": manifest_images,
    }
    write_json(output_path, manifest, exclusive=True)
    return manifest


def split_images(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"development", "evaluation"}:
        raise ValueError("split must be 'development' or 'evaluation'")
    return [image for image in manifest["images"] if image["split"] == split]

