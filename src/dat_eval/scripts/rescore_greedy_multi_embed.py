"""Rescore greedy and LLM DAT responses under all 3 embeddings (GloVe,
FastText, SBERT), then average across embeddings to produce a single
per-trial score. Feeds the 1-panel greedy-vs-LLM figure.

Inputs:
  data/greedy_baseline/trials.json               (120 greedy trials)
  data/responses/<model>/dat_responses_t1-0.json (LLM trials at T = 1.0)

Output (new files; the shipped copies in data/greedy_baseline/ are not touched):
  outputs/rescored/greedy_baseline/trials_multi_embed.json
  outputs/rescored/greedy_baseline/llm_trials_multi_embed.json

The paper figure used the 55 models that had T = 1.0 DAT responses when it was made (2026-04);
by default the LLM side is restricted to the models listed in the shipped
llm_trials_multi_embed.json. Pass --all-models to use every model in data/responses/.

Usage (needs all three embeddings; see README):
    python -m src.dat_eval.scripts.rescore_greedy_multi_embed [--all-models]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from src.dat_eval.dat import GloVeEmbeddings
from src.dat_eval.scripts.rescore_multi_embed import (
    FastTextEmbedder,
    GloVeEmbedder,
    SBERTEmbedder,
    score_dat_trial,
)
from src.utils import GLOVE_PATH, ROOT

GREEDY_PATH = ROOT / "data" / "greedy_baseline" / "trials.json"
RUN_DIR = ROOT / "data" / "responses"
OUT_DIR = ROOT / "outputs" / "rescored" / "greedy_baseline"


def load_llm_trials_t10(models: set[str] | None = None) -> list[tuple[str, list[str]]]:
    """Return (model_key, words) tuples for every DAT trial at T=1.0."""
    out: list[tuple[str, list[str]]] = []
    for model_dir in sorted(RUN_DIR.iterdir()):
        if not model_dir.is_dir() or (models is not None and model_dir.name not in models):
            continue
        f = model_dir / "dat_responses_t1-0.json"
        if not f.exists():
            continue
        for trial in json.load(open(f)):
            if trial.get("words"):
                out.append((model_dir.name, trial["words"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-models", action="store_true")
    all_models = ap.parse_args().all_models
    paper_models = None if all_models else {
        x["model"] for x in json.load(open(ROOT / "data/greedy_baseline/llm_trials_multi_embed.json"))}
    print("Loading GloVe (for vocab validation + scoring)...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("trials_multi_embed.json", "llm_trials_multi_embed.json"):
        if (OUT_DIR / f).exists():
            raise SystemExit(f"{OUT_DIR / f} already exists; move it away first")
    glove_vocab = GloVeEmbeddings(GLOVE_PATH)
    glove = GloVeEmbedder(glove_vocab)
    fasttext = FastTextEmbedder()
    sbert = SBERTEmbedder()

    embedders = [glove, fasttext, sbert]
    names = ["glove", "fasttext", "sbert"]

    # -- Greedy --
    greedy_trials = json.load(open(GREEDY_PATH))
    print(f"Rescoring {len(greedy_trials)} greedy trials...", flush=True)
    greedy_out = []
    for t in greedy_trials:
        words = t["words"]
        scores = {
            name: score_dat_trial(words, emb, glove_vocab)
            for name, emb in zip(names, embedders)
        }
        avg = float(np.nanmean(list(scores.values())))
        greedy_out.append({
            "trial": t["trial"],
            "words": words,
            "scores_per_embed": scores,
            "mean_across_embeds": avg,
        })
    with open(OUT_DIR / "trials_multi_embed.json", "w") as f:
        json.dump(greedy_out, f, indent=2)

    g_avgs = np.array([x["mean_across_embeds"] for x in greedy_out])
    print(f"  greedy mean-across-embeds: M={g_avgs.mean():.2f}, SD={g_avgs.std(ddof=1):.2f}, n={len(g_avgs)}")
    for name in names:
        per = np.array([x["scores_per_embed"][name] for x in greedy_out])
        print(f"    {name}: M={np.nanmean(per):.2f}, SD={np.nanstd(per, ddof=1):.2f}")

    # -- LLMs --
    llm = load_llm_trials_t10(paper_models)
    print(f"Rescoring {len(llm)} LLM trials @ T=1.0...", flush=True)
    llm_out = []
    for i, (model, words) in enumerate(llm):
        scores = {
            name: score_dat_trial(words, emb, glove_vocab)
            for name, emb in zip(names, embedders)
        }
        vals = [v for v in scores.values() if not math.isnan(v)]
        if not vals:
            continue
        avg = float(np.mean(vals))
        llm_out.append({
            "model": model,
            "words": words,
            "scores_per_embed": scores,
            "mean_across_embeds": avg,
        })
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(llm)}", flush=True)
    with open(OUT_DIR / "llm_trials_multi_embed.json", "w") as f:
        json.dump(llm_out, f, indent=2)

    l_avgs = np.array([x["mean_across_embeds"] for x in llm_out])
    print(f"  LLM mean-across-embeds: M={l_avgs.mean():.2f}, SD={l_avgs.std(ddof=1):.2f}, n={len(l_avgs)}")
    for name in names:
        per = np.array([x["scores_per_embed"][name] for x in llm_out])
        print(f"    {name}: M={np.nanmean(per):.2f}, SD={np.nanstd(per, ddof=1):.2f}")


if __name__ == "__main__":
    main()
