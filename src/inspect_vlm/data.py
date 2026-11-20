"""MVTec AD images for a few industrial parts, with one fixed evaluation split for every method.

MVTec AD (Bergmann et al., CVPR 2019) has, per part, a training set of defect-free images and a
test set of defect-free and defective ones. Every method here is scored on the same images:

* ``train_good``: defect-free training images. The anomaly detector learns only from these.
* ``shots``: a few labeled test images (``k`` defective + ``k`` defect-free per part), held back
  for methods that need labeled defects (supervised fine-tuning) or a reference image (VLM).
* ``eval``: the remaining test images. No method ever trains on or looks at them before scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = "Voxel51/mvtec-ad"  # mirror of MVTec AD on the Hugging Face Hub (CC BY-NC-SA 4.0)
PARTS = ("metal_nut", "screw", "transistor")
# (defect-free training images, test images) per part in MVTec AD; checked at load time.
EXPECTED_COUNTS = {"metal_nut": (220, 115), "screw": (320, 160), "transistor": (213, 100)}


@dataclass(frozen=True)
class Image:
    path: str  # relative path inside the dataset repository
    part: str
    defect: str  # "good" or the defect type, e.g. "scratch"
    split: str  # "train" or "test" in MVTec terms

    @property
    def is_defective(self) -> int:
        return int(self.defect != "good")


@dataclass(frozen=True)
class PartSplit:
    part: str
    train_good: list[Image]
    shots: list[Image]
    eval: list[Image]


def parse_samples(samples: list[dict], parts: tuple[str, ...] = PARTS) -> list[Image]:
    return [
        Image(s["filepath"], s["category"]["label"], s["defect"]["label"], s["split"])
        for s in samples
        if s["category"]["label"] in parts
    ]


def split_part(images: list[Image], part: str, k: int, seed: int) -> PartSplit:
    """Hold back ``k`` defective and ``k`` defect-free test images of ``part`` as labeled shots."""
    rng = np.random.default_rng(seed)
    mine = sorted((im for im in images if im.part == part), key=lambda im: im.path)
    train_good = [im for im in mine if im.split == "train"]
    test = [im for im in mine if im.split == "test"]
    shots: list[Image] = []
    for defective in (0, 1):
        pool = [im for im in test if im.is_defective == defective]
        shots += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
    held = {im.path for im in shots}
    return PartSplit(part, train_good, shots, [im for im in test if im.path not in held])


def load(root: Path, parts: tuple[str, ...] = PARTS, k: int = 5, seed: int = 0) -> list[PartSplit]:
    """Download the metadata and the needed images (about 1,500 PNGs) and split each part."""
    from huggingface_hub import hf_hub_download, snapshot_download

    meta = hf_hub_download(REPO, "samples.json", repo_type="dataset", local_dir=root)
    images = parse_samples(json.loads(Path(meta).read_text())["samples"], parts)
    snapshot_download(
        REPO, repo_type="dataset", local_dir=root, allow_patterns=[im.path for im in images]
    )
    return [split_part(images, part, k, seed) for part in parts]
