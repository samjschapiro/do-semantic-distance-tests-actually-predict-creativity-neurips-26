"""Camera-ready figure: DAT score distribution for LLMs vs. a greedy
GloVe algorithm, both averaged across GloVe / FastText / SBERT.

LLMs: all per-trial DAT scores at T=1.0 across our 55-model pool
(n≈2100), rescored under all three embeddings and averaged per trial.
Greedy: 120 word lists produced by argmax-mean-cosine-distance over
GloVe nouns, rescored under all three embeddings and averaged.

Output: outputs/figures/fig_greedy_baseline.pdf

    python -m src.dat_eval.scripts.make_greedy_figure
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ROOT

GREEDY_PATH = ROOT / "data/greedy_baseline/trials_multi_embed.json"
LLM_PATH = ROOT / "data/greedy_baseline/llm_trials_multi_embed.json"
OUT_PATH = ROOT / "outputs/figures/fig_greedy_baseline.pdf"

# Olson 2021 Study 1A (GloVe only, n=141).
HUMAN_MEAN = 78.38
HUMAN_SD = 6.35
HUMAN_N = 141


def main():
    greedy_all = json.load(open(GREEDY_PATH))
    greedy = np.asarray([x["mean_across_embeds"] for x in greedy_all])

    llm_all = json.load(open(LLM_PATH))
    llm = np.asarray([x["mean_across_embeds"] for x in llm_all])

    rng = np.random.default_rng(0)
    human = rng.normal(HUMAN_MEAN, HUMAN_SD, size=HUMAN_N)

    fig, ax = plt.subplots(figsize=(3.5, 2.1), dpi=300)

    bins = np.arange(60, 101, 1.5)
    alpha = 0.6
    ax.hist(
        human, bins=bins, density=True, color="#4c72b0", alpha=alpha,
        edgecolor="black", linewidth=0.4,
        label=f"Humans (GloVe only, Olson 2021, $n={HUMAN_N}$)",
    )
    ax.hist(
        llm, bins=bins, density=True, color="#55a868", alpha=alpha,
        edgecolor="black", linewidth=0.4,
        label=f"LLMs (55 models, $n={len(llm)}$)",
    )
    ax.hist(
        greedy, bins=bins, density=True, color="#c44e52", alpha=alpha,
        edgecolor="black", linewidth=0.4,
        label=f"Greedy algorithm ($n={len(greedy)}$)",
    )

    ax.set_xlabel("DAT score", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_xlim(60, 100)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for _mean, color in [
        (HUMAN_MEAN, "#4c72b0"),
        (llm.mean(), "#55a868"),
        (greedy.mean(), "#c44e52"),
    ]:
        ax.axvline(_mean, color=color, linewidth=0.8, linestyle="--", alpha=0.8)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")
    print(f"human:  M={human.mean():.2f}, SD={human.std(ddof=1):.2f}, n={len(human)} (GloVe only)")
    print(f"llm:    M={llm.mean():.2f}, SD={llm.std(ddof=1):.2f}, n={len(llm)} (averaged across 3 embeddings)")
    print(f"greedy: M={greedy.mean():.2f}, SD={greedy.std(ddof=1):.2f}, n={len(greedy)} (averaged across 3 embeddings)")


if __name__ == "__main__":
    main()
