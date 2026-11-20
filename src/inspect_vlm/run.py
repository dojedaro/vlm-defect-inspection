"""Score every method on the same held-out MVTec AD images and write results/metrics.json.

    python -m inspect_vlm.run --data data --out results            # all methods (GPU for the VLM)
    python -m inspect_vlm.run --data data --out results --no-vlm   # CNN methods only

Three seeds. Each seed draws a different set of labeled shots (k defective + k defect-free test
images per part); those shots are removed from that seed's evaluation set for every method.
Scores that do not depend on the shots (PatchCore, zero-shot VLM) are computed once for all test
images and then read off for each seed's evaluation subset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from inspect_vlm.data import EXPECTED_COUNTS, load
from inspect_vlm.methods import VLM_ID, finetune_scores, load_rgb, patchcore_scores

METHODS = ("patchcore", "finetuned_resnet18", "vlm_zero_shot", "vlm_with_reference")
FALSE_ALARM_RATES = (0.05, 0.10)


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


def recall_at_false_alarm(labels: np.ndarray, scores: np.ndarray, rate: float) -> float:
    """Share of defective parts caught when at most ``rate`` of good parts are flagged."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return round(float(tpr[fpr <= rate].max()), 4)


def calibrated_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on the labeled shots (never on evaluation images)."""
    candidates = np.unique(scores)
    balanced = [
        (np.mean(scores[labels == 1] >= c) + np.mean(scores[labels == 0] < c)) / 2
        for c in candidates
    ]
    return float(candidates[int(np.argmax(balanced))])


def decision_metrics(labels: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(np.mean(flagged == labels)), 4),
        "defect_recall": round(float(flagged[labels == 1].mean()), 4),
        "false_alarm_rate": round(float(flagged[labels == 0].mean()), 4),
    }


def _method_entry(labels: np.ndarray, scores: np.ndarray, defects: pd.Series) -> dict:
    entry = auroc_with_ci(labels, scores)
    for rate in FALSE_ALARM_RATES:
        key = f"recall_at_{int(rate * 100)}pct_false_alarms"
        entry[key] = recall_at_false_alarm(labels, scores, rate)
    entry["auroc_by_defect_type"] = {}
    for defect in sorted(set(defects) - {"good"}):
        mask = ((defects == defect) | (labels == 0)).to_numpy()
        entry["auroc_by_defect_type"][defect] = round(
            float(roc_auc_score(labels[mask], scores[mask])), 4
        )
    return entry


def run(data: Path, out: Path, k: int, seeds: list[int], use_vlm: bool, extra_good: int = 40):
    out.mkdir(parents=True, exist_ok=True)
    per_seed = {seed: load(data, k=k, seed=seed) for seed in seeds}
    first = per_seed[seeds[0]]
    parts = [split.part for split in first]
    for split in first:
        n_test = len(split.shots) + len(split.eval)
        assert (len(split.train_good), n_test) == EXPECTED_COUNTS[split.part], split.part

    # Every test image of every part, scored once by the shot-independent methods.
    test_paths = {s.part: sorted(im.path for im in s.shots + s.eval) for s in first}
    cache = {path: load_rgb(data, path) for paths in test_paths.values() for path in paths}
    shared: dict[str, dict[str, float]] = {part: {} for part in parts}
    timing: dict[str, dict[str, float]] = {part: {} for part in parts}
    for split in first:
        part = split.part
        train_good = [load_rgb(data, im.path) for im in split.train_good]
        started = time.perf_counter()
        scores = patchcore_scores(train_good, [cache[p] for p in test_paths[part]])
        timing[part]["patchcore_total"] = time.perf_counter() - started
        for path, value in zip(test_paths[part], scores, strict=True):
            shared[part][f"patchcore|{path}"] = float(value)

    finetuned: dict[tuple[int, str], dict[str, float]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for split in per_seed[seed]:
            good_idx = rng.choice(len(split.train_good), extra_good, replace=False)
            sup = [load_rgb(data, im.path) for im in split.shots]
            sup += [load_rgb(data, split.train_good[i].path) for i in good_idx]
            labels = [im.is_defective for im in split.shots] + [0] * len(good_idx)
            targets = [im.path for im in split.shots + split.eval]
            started = time.perf_counter()
            scores = finetune_scores(sup, labels, [cache[p] for p in targets], seed=seed)
            timing[split.part][f"finetune_seed{seed}"] = time.perf_counter() - started
            finetuned[(seed, split.part)] = dict(zip(targets, map(float, scores), strict=True))

    vlm_ref: dict[tuple[int, str], dict[str, float]] = {}
    references: dict[str, str] = {}
    if use_vlm:
        import torch

        from inspect_vlm.methods import VLMInspector

        torch.cuda.empty_cache()
        inspector = VLMInspector()
        for part in parts:
            inspector.score(cache[test_paths[part][0]], part)  # warm-up, not timed
            started = time.perf_counter()
            for path in test_paths[part]:
                p_yes, mass = inspector.score(cache[path], part)
                shared[part][f"vlm_zero_shot|{path}"] = p_yes
                shared[part][f"vlm_mass|{path}"] = mass
            timing[part]["vlm_per_image"] = (time.perf_counter() - started) / len(test_paths[part])
        for seed in seeds:
            for split in per_seed[seed]:
                rng = np.random.default_rng(1000 + seed)
                ref = split.train_good[int(rng.integers(len(split.train_good)))]
                references[f"{split.part}_seed{seed}"] = ref.path
                reference = load_rgb(data, ref.path)
                vlm_ref[(seed, split.part)] = {
                    path: inspector.score(cache[path], split.part, reference)[0]
                    for path in test_paths[split.part]
                }

    rows, results = [], {"k_labeled_per_class": k, "seeds": seeds, "parts": {}}
    for part in parts:
        per_method: dict[str, list[dict]] = {m: [] for m in METHODS}
        decisions: dict[str, list[dict]] = {}
        for seed in seeds:
            split = next(s for s in per_seed[seed] if s.part == part)
            eval_paths = [im.path for im in split.eval]
            labels = np.array([im.is_defective for im in split.eval])
            frame = pd.DataFrame(
                {
                    "seed": seed,
                    "part": part,
                    "image": eval_paths,
                    "defect": [im.defect for im in split.eval],
                    "label": labels,
                    "patchcore": [shared[part][f"patchcore|{p}"] for p in eval_paths],
                    "finetuned_resnet18": [finetuned[(seed, part)][p] for p in eval_paths],
                }
            )
            if use_vlm:
                frame["vlm_zero_shot"] = [shared[part][f"vlm_zero_shot|{p}"] for p in eval_paths]
                frame["vlm_answer_mass"] = [shared[part][f"vlm_mass|{p}"] for p in eval_paths]
                frame["vlm_with_reference"] = [vlm_ref[(seed, part)][p] for p in eval_paths]
                shot_labels = np.array([im.is_defective for im in split.shots])
                shot_scores = {
                    "vlm_zero_shot": [
                        shared[part][f"vlm_zero_shot|{im.path}"] for im in split.shots
                    ],
                    "vlm_with_reference": [vlm_ref[(seed, part)][im.path] for im in split.shots],
                }
                for method, values in shot_scores.items():
                    threshold = calibrated_threshold(shot_labels, np.array(values))
                    scores = frame[method].to_numpy()
                    decisions.setdefault(f"{method}_at_0.5", []).append(
                        decision_metrics(labels, scores > 0.5)
                    )
                    decisions.setdefault(f"{method}_shot_calibrated", []).append(
                        decision_metrics(labels, scores >= threshold)
                    )
            for method in METHODS:
                if method in frame:
                    per_method[method].append(
                        _method_entry(labels, frame[method].to_numpy(), frame["defect"])
                    )
            rows.append(frame)

        summary = {}
        for method, entries in per_method.items():
            if not entries:
                continue
            aurocs = [e["auroc"] for e in entries]
            summary[method] = {
                "auroc_mean": round(float(np.mean(aurocs)), 4),
                "auroc_std": round(float(np.std(aurocs, ddof=1)), 4) if len(aurocs) > 1 else 0.0,
                "per_seed": entries,
            }
            for rate in FALSE_ALARM_RATES:
                key = f"recall_at_{int(rate * 100)}pct_false_alarms"
                summary[method][f"{key}_mean"] = round(float(np.mean([e[key] for e in entries])), 4)
        results["parts"][part] = {
            "n_eval_per_seed": len(split.eval),
            "n_defective_eval": int(sum(im.is_defective for im in split.eval)),
            "n_train_good": len(split.train_good),
            "methods": summary,
            "decisions": {
                name: {m: round(float(np.mean([d[m] for d in values])), 4) for m in values[0]}
                for name, values in decisions.items()
            },
            "timing_seconds": {key: round(value, 3) for key, value in timing[part].items()},
        }
        print(part, json.dumps({m: s["auroc_mean"] for m, s in summary.items()}), flush=True)

    results["mean_auroc_over_parts"] = {
        m: round(
            float(np.mean([results["parts"][p]["methods"][m]["auroc_mean"] for p in parts])), 4
        )
        for m in METHODS
        if all(m in results["parts"][p]["methods"] for p in parts)
    }
    results["vlm"] = VLM_ID if use_vlm else None
    results["vlm_reference_images"] = references
    pd.concat(rows).to_csv(out / "scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=5, help="labeled shots per class per part")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()
    results = run(args.data, args.out, args.k, args.seeds, not args.no_vlm)
    print(json.dumps(results["mean_auroc_over_parts"], indent=2))


if __name__ == "__main__":
    main()
