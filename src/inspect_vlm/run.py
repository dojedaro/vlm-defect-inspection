"""Score every method on the same held-out MVTec AD images and write results/metrics.json.

python -m inspect_vlm.run --data data --out results            # all methods (GPU for the VLM)
python -m inspect_vlm.run --data data --out results --no-vlm   # CNN methods only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from inspect_vlm.data import load
from inspect_vlm.methods import VLM_ID, finetune_scores, load_rgb, patchcore_scores

METHODS = ("patchcore", "finetuned_resnet18", "vlm_zero_shot", "vlm_with_reference")


def auroc_with_ci(labels: np.ndarray, scores: np.ndarray, rounds: int = 2000, seed: int = 0):
    """Image-level AUROC and a 95% bootstrap interval (resampling images within each class)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    boot = []
    for _ in range(rounds):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boot.append(roc_auc_score(labels[idx], scores[idx]))
    return {
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
    }


def run(data: Path, out: Path, k: int, seed: int, use_vlm: bool, extra_good: int = 40) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    splits = load(data, k=k, seed=seed)
    inspector = None
    if use_vlm:
        from inspect_vlm.methods import VLMInspector

        inspector = VLMInspector()
    rows, results = [], {"k_labeled_per_class": k, "seed": seed, "parts": {}}
    rng = np.random.default_rng(seed)
    for split in splits:
        part = split.part
        train_good = [load_rgb(data, im.path) for im in split.train_good]
        evals = [load_rgb(data, im.path) for im in split.eval]
        labels = np.array([im.is_defective for im in split.eval])
        frame = pd.DataFrame(
            {
                "part": part,
                "image": [im.path for im in split.eval],
                "defect": [im.defect for im in split.eval],
                "label": labels,
            }
        )
        timing = {}

        started = time.perf_counter()
        frame["patchcore"] = patchcore_scores(train_good, evals, seed=seed)
        timing["patchcore"] = time.perf_counter() - started

        # Supervised transfer learning sees the labeled shots plus some defect-free train images.
        good_idx = rng.choice(len(train_good), min(extra_good, len(train_good)), replace=False)
        sup_images = [load_rgb(data, im.path) for im in split.shots] + [
            train_good[i] for i in good_idx
        ]
        sup_labels = [im.is_defective for im in split.shots] + [0] * len(good_idx)
        started = time.perf_counter()
        frame["finetuned_resnet18"] = finetune_scores(sup_images, sup_labels, evals, seed=seed)
        timing["finetuned_resnet18"] = time.perf_counter() - started

        if inspector is not None:
            reference = train_good[int(rng.integers(len(train_good)))]
            started = time.perf_counter()
            frame["vlm_zero_shot"] = [inspector.score(im, part) for im in evals]
            timing["vlm_zero_shot"] = time.perf_counter() - started
            started = time.perf_counter()
            frame["vlm_with_reference"] = [inspector.score(im, part, reference) for im in evals]
            timing["vlm_with_reference"] = time.perf_counter() - started

        results["parts"][part] = {
            "n_eval": int(len(labels)),
            "n_defective": int(labels.sum()),
            "n_train_good": len(train_good),
            "methods": {
                m: auroc_with_ci(labels, frame[m].to_numpy()) for m in METHODS if m in frame
            },
            "seconds_per_image": {m: round(s / len(evals), 3) for m, s in timing.items()},
        }
        if "vlm_zero_shot" in frame:
            # How often does a plain "P(Yes) > 0.5" decision get it right?
            for m in ("vlm_zero_shot", "vlm_with_reference"):
                predicted = frame[m].to_numpy() > 0.5
                results["parts"][part][f"{m}_at_0.5"] = {
                    "accuracy": round(float(np.mean(predicted == labels)), 4),
                    "defect_recall": round(float(predicted[labels == 1].mean()), 4),
                    "false_alarm_rate": round(float(predicted[labels == 0].mean()), 4),
                }
        rows.append(frame)
        print(part, json.dumps(results["parts"][part]["methods"]), flush=True)

    results["mean_auroc"] = {
        m: round(float(np.mean([p["methods"][m]["auroc"] for p in results["parts"].values()])), 4)
        for m in METHODS
        if all(m in p["methods"] for p in results["parts"].values())
    }
    results["vlm"] = VLM_ID if use_vlm else None
    pd.concat(rows).to_csv(out / "scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=5, help="labeled shots per class per part")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()
    results = run(args.data, args.out, args.k, args.seed, not args.no_vlm)
    print(json.dumps(results["mean_auroc"], indent=2))


if __name__ == "__main__":
    main()
