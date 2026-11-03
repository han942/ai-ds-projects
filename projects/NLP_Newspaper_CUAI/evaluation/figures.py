"""Static figures for the Markdown evaluation report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sari_baselines(summary: pd.DataFrame, target: Path) -> None:
    scores = summary.set_index(["system", "metric"])
    systems = ["model", "copy", "shuffled"]
    labels = ["Gemma 3-1B", "Copy source", "Shuffled output"]
    colors = ["#3266a8", "#747b85", "#c77824"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, metric, title in zip(axes, ["sari_ws", "sari_morph"], ["Whitespace / 13a", "Korean morphemes"]):
        rows = scores.loc[[(s, metric) for s in systems]]
        means = rows["mean"].to_numpy()
        err = np.array([means - rows.lo.to_numpy(), rows.hi.to_numpy() - means])
        ax.bar(labels, means, color=colors, width=0.6)
        ax.errorbar(labels, means, yerr=err, fmt="none", ecolor="#24292f", capsize=5, lw=1.3)
        for x, (_, r) in enumerate(rows.iterrows()):
            ax.text(x, r.hi + 3, f"{r['mean']:.2f}", ha="center", fontsize=10)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#e4e7eb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("SARI (0–100)")
    n = int(scores.loc[("model", "sari_ws"), "n"])
    fig.suptitle(f"Model outputs and control baselines · {n:,} samples", fontsize=15)
    fig.text(0.5, 0.02, "Error bars: 95% row-bootstrap intervals (1,000 resamples). One synthetic reference per source.",
             ha="center", fontsize=9, color="#4f5761")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)
