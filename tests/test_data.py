from inspect_vlm.data import Image, parse_samples, split_part
from inspect_vlm.methods import vlm_prompt
from inspect_vlm.run import auroc_with_ci


def _images():
    images = [Image(f"t/{i}.png", "screw", "good", "train") for i in range(10)]
    images += [Image(f"g/{i}.png", "screw", "good", "test") for i in range(8)]
    images += [Image(f"d/{i}.png", "screw", "scratch", "test") for i in range(12)]
    images += [Image("other.png", "bottle", "good", "test")]
    return images


def test_split_holds_back_k_per_class_and_never_evaluates_on_them():
    split = split_part(_images(), "screw", k=3, seed=0)
    assert len(split.train_good) == 10
    assert sum(im.is_defective for im in split.shots) == 3 and len(split.shots) == 6
    assert not {im.path for im in split.shots} & {im.path for im in split.eval}
    assert len(split.eval) == 8 + 12 - 6
    assert all(im.part == "screw" and im.split == "test" for im in split.eval)


def test_split_is_deterministic():
    a = split_part(_images(), "screw", k=3, seed=1)
    b = split_part(_images(), "screw", k=3, seed=1)
    assert [im.path for im in a.shots] == [im.path for im in b.shots]


def test_parse_samples_keeps_requested_parts():
    samples = [
        {
            "filepath": "a.png",
            "category": {"label": "screw"},
            "defect": {"label": "good"},
            "split": "train",
        },
        {
            "filepath": "b.png",
            "category": {"label": "bottle"},
            "defect": {"label": "good"},
            "split": "train",
        },
    ]
    assert [im.path for im in parse_samples(samples, ("screw",))] == ["a.png"]


def test_prompts_name_the_part_and_ask_yes_no():
    assert "metal nut" in vlm_prompt("metal_nut", with_reference=False)
    assert vlm_prompt("screw", with_reference=True).startswith("The first image shows")
    assert vlm_prompt("screw", with_reference=False).endswith("Answer Yes or No.")


def test_auroc_interval_brackets_the_estimate():
    import numpy as np

    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.9, 0.7])
    result = auroc_with_ci(labels, scores, rounds=200)
    assert result["ci95"][0] <= result["auroc"] <= result["ci95"][1]


def test_calibrated_threshold_uses_only_the_given_scores():
    import numpy as np

    from inspect_vlm.run import calibrated_threshold

    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    threshold = calibrated_threshold(labels, scores)
    assert 0.3 < threshold <= 0.6


def test_recall_at_false_alarm_and_decisions():
    import numpy as np

    from inspect_vlm.run import decision_metrics, recall_at_false_alarm

    labels = np.array([0] * 20 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.5, 20), np.linspace(0.3, 1.0, 10)])
    assert recall_at_false_alarm(labels, scores, 0.0) <= recall_at_false_alarm(labels, scores, 0.10)
    metrics = decision_metrics(labels, scores > 0.45)
    assert set(metrics) == {"accuracy", "defect_recall", "false_alarm_rate"}
