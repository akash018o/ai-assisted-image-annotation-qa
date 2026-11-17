"""Stream only the COCO records needed for the locked sample."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Iterator

import ijson
from pycocotools.coco import COCO


def iter_coco_items(path: Path, prefix: str) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def read_catalog(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images = list(iter_coco_items(path, "images.item"))
    categories = list(iter_coco_items(path, "categories.item"))
    images.sort(key=lambda row: int(row["id"]))
    categories.sort(key=lambda row: int(row["id"]))
    if not images or not categories:
        raise ValueError(f"Not a valid COCO instances annotation file: {path}")
    return images, categories


def annotation_counts(path: Path, image_ids: set[int]) -> dict[int, dict[str, int]]:
    counts = {image_id: {"all": 0, "crowd": 0} for image_id in image_ids}
    for annotation in iter_coco_items(path, "annotations.item"):
        image_id = int(annotation["image_id"])
        if image_id not in counts:
            continue
        counts[image_id]["all"] += 1
        counts[image_id]["crowd"] += int(bool(annotation.get("iscrowd", 0)))
    return counts


def coco_api_subset(path: Path, image_ids: list[int] | set[int]) -> COCO:
    wanted = {int(image_id) for image_id in image_ids}
    image_rows = [row for row in iter_coco_items(path, "images.item") if int(row["id"]) in wanted]
    found = {int(row["id"]) for row in image_rows}
    if found != wanted:
        raise ValueError(f"COCO annotation file is missing image IDs: {sorted(wanted - found)[:10]}")
    categories = list(iter_coco_items(path, "categories.item"))
    annotations = [
        row for row in iter_coco_items(path, "annotations.item")
        if int(row["image_id"]) in wanted
    ]
    coco = COCO()
    coco.dataset = {"images": image_rows, "categories": categories, "annotations": annotations}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco

