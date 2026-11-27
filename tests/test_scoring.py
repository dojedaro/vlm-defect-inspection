"""Tests for the code behind the headline numbers: the PatchCore scorer and run.py metrics."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image as PILImage
from torch import nn

from inspect_vlm import methods
from inspect_vlm.run import _method_entry, calibrated_threshold, recall_at_false_alarm


def test_recall_at_false_alarm_matches_hand_count():
    # 10 good parts scored 0.0..0.9 and 4 defective ones. Allowing 1 of 10 good parts to be flagged
    # (the one at 0.9) lets the threshold drop to 0.85, which catches 2 of 4 defects.
    labels = np.array([0] * 10 + [1] * 4)
    scores = np.array([i / 10 for i in range(10)] + [0.95, 0.85, 0.5, 0.05])
    assert recall_at_false_alarm(labels, scores, 0.10) == 0.5
    assert recall_at_false_alarm(labels, scores, 0.0) == 0.25


def test_auroc_by_defect_type_compares_each_type_only_with_good_parts():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.9, 0.8, 0.05, 0.15])
    defects = pd.Series(["good", "good", "good", "scratch", "scratch", "flip", "flip"])
    entry = _method_entry(labels, scores, defects)
    assert entry["auroc_by_defect_type"] == {"flip": 0.5, "scratch": 1.0}
    assert entry["recall_at_10pct_false_alarms"] == recall_at_false_alarm(labels, scores, 0.10)


def test_calibrated_threshold_separates_labeled_shots():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.30, 0.31, 0.33, 0.35, 0.40, 0.41, 0.44, 0.45, 0.47, 0.49])
    threshold = calibrated_threshold(labels, scores)
    flagged = scores >= threshold
    assert (flagged == labels.astype(bool)).all()
    assert threshold < 0.5  # the VLM case: P(Yes) never reaches 0.5, yet the shots still separate


class _PixelPatches(nn.Module):
    """Stand-in for the WideResNet features: each 28x28 block's mean colour is one patch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = nn.functional.avg_pool2d(x, 28)  # (images, 3, 8, 8)
        return pooled.permute(0, 2, 3, 1).flatten(1, 2)  # (images, 64 patches, 3 channels)


def _part(defect: bool, shade: int) -> PILImage.Image:
    pixels = np.full((224, 224, 3), shade, dtype=np.uint8)
    if defect:
        pixels[:28, :28] = 255  # one bright block, like a scratch in a corner
    return PILImage.fromarray(pixels)


@pytest.fixture
def stub_features(monkeypatch):
    monkeypatch.setattr(methods, "_PatchFeatures", _PixelPatches)
    monkeypatch.setattr(methods, "_device", lambda: torch.device("cpu"))


def test_patchcore_scores_a_defect_above_every_good_part(stub_features):
    bank = [_part(False, shade) for shade in (118, 122, 126, 130, 134)]
    good = [_part(False, shade) for shade in (120, 124, 132)]
    defective = [_part(True, 124)]
    scores = methods.patchcore_scores(bank, good + defective, batch=2)
    assert scores[-1] > scores[:-1].max()


def test_patchcore_scores_a_part_already_in_the_bank_as_zero(stub_features):
    bank = [_part(False, shade) for shade in (118, 126, 134)]
    scores = methods.patchcore_scores(bank, [bank[1]])
    assert scores[0] == pytest.approx(0.0, abs=1e-5)
