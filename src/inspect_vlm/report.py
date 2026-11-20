"""Chart for the README from results/metrics.json.

python -m inspect_vlm.report --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "patchcore": "PatchCore-style\n(good parts only)",
    "finetuned_resnet18": "ResNet-18 fine-tuned\n(10 labeled images)",
    "vlm_zero_shot": "Qwen2.5-VL-3B\nzero-shot",
    "vlm_with_reference": "Qwen2.5-VL-3B\n+ good reference",
}
COLORS = {"patchcore": "#2563eb"}


def auroc_chart(metrics: dict, path: Path) -> None:
    parts = list(metrics["parts"])
    methods = [m for m in LABELS if m in metrics["parts"][parts[0]]["methods"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(parts)
    shades = ["#1e3a8a", "#3b82f6", "#93c5fd"]
    x = np.arange(len(methods))
    for i, part in enumerate(parts):
        stats = metrics["parts"][part]["methods"]
        means = [stats[m]["auroc_mean"] for m in methods]
        stds = [stats[m]["auroc_std"] for m in methods]
        ax.bar(
            x + (i - (len(parts) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=shades[i % len(shades)],
            label=part.replace("_", " "),
        )
    ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=1)
    ax.text(len(methods) - 0.5, 0.51, "chance", color="#6b7280", fontsize=8, ha="right")
    ax.set_xticks(x, [LABELS[m] for m in methods], fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Image-level AUROC")
    ax.set_title(
        "Defect detection on held-out MVTec AD images (mean ± std over 3 seeds)",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = json.loads((args.results / "metrics.json").read_text())
    auroc_chart(metrics, args.results / "auroc_by_method.png")


if __name__ == "__main__":
    main()
