"""Implement Nakajima et al.'s CDAT appropriateness gate and emit gated scores.

For each model and each temperature:

  1. Compute mean per-cue appropriateness (already done in cdat_responses).
  2. Compare the model's distribution of per-cue appropriateness values against
     a task-agnostic random baseline (random 7-noun draws from a common-noun
     pool, scored against the same cue) via two-sided Welch's t-test.
  3. FDR-correct (Benjamini-Hochberg) across all models within each temperature.
  4. A model is *retained* at a given temperature iff its FDR-adjusted
     p-value is < alpha (default 0.001) AND its mean appropriateness exceeds
     the random baseline mean.

The model-level gated CDAT score is the mean novelty across all temperatures
the model passes; if the model fails every temperature, the score is None.

We compute this under all three embeddings (GloVe / FastText / SBERT) using
the same word lists, so downstream multi-embedding analysis can use the
gated score per embedding too.

Output (a new file; the shipped data/scores/cdat_gated_scores.json is never touched):
  outputs/rescored/cdat_gated_scores.json
  Structure: {emb_name: {model_key: {"cdat": float|null,
                                     "passed_temps": [...],
                                     "gate_details": {temp: {...}}}}}

Usage:
  python -m src.dat_eval.scripts.cdat_gate [--out outputs/rescored/cdat_gated_scores.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance

from src.utils import ROOT

RUN_DIR = ROOT / "data" / "responses"
OUT = ROOT / "outputs" / "rescored" / "cdat_gated_scores.json"

# Hardcoded random-noun pool: 300 common English nouns sampled across concrete
# and abstract categories, deduplicated, lowercase. Used as the task-agnostic
# baseline word draw for the appropriateness gate.
NOUN_POOL = [
    # body
    "head", "hand", "foot", "eye", "ear", "mouth", "tooth", "tongue", "neck",
    "arm", "leg", "knee", "finger", "thumb", "skin", "bone", "blood", "heart",
    "brain", "lung", "stomach", "spine", "chest", "shoulder",
    # nature
    "tree", "leaf", "flower", "grass", "river", "lake", "ocean", "mountain",
    "valley", "forest", "desert", "field", "stone", "rock", "sand", "soil",
    "cloud", "rain", "snow", "wind", "storm", "fog", "ice", "fire",
    # animals
    "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "bird", "fish",
    "snake", "bear", "wolf", "fox", "deer", "rabbit", "mouse", "elephant",
    "lion", "tiger", "monkey", "eagle", "owl", "spider", "ant", "bee",
    "butterfly", "whale", "shark", "frog", "turtle",
    # food
    "bread", "rice", "meat", "fish", "egg", "milk", "cheese", "butter",
    "salt", "pepper", "sugar", "honey", "fruit", "apple", "orange", "banana",
    "potato", "tomato", "onion", "carrot", "soup", "wine", "water", "tea",
    "coffee", "juice",
    # household / artifact
    "house", "room", "door", "window", "wall", "floor", "ceiling", "roof",
    "bed", "chair", "table", "lamp", "book", "paper", "pen", "knife",
    "fork", "spoon", "cup", "plate", "bowl", "bottle", "glass", "mirror",
    "clock", "phone", "key", "lock", "candle", "rope", "needle", "wheel",
    "hammer", "nail", "saw", "engine", "machine", "tool", "vehicle", "car",
    "boat", "train", "plane", "bicycle",
    # clothing
    "shirt", "coat", "hat", "shoe", "boot", "sock", "glove", "belt", "ring",
    "crown", "veil", "thread", "silk", "cotton", "leather",
    # urban / built environment
    "city", "town", "village", "street", "road", "bridge", "tower", "wall",
    "garden", "park", "square", "market", "school", "church", "temple",
    "hospital", "library", "museum", "theater", "factory", "office", "shop",
    "prison", "fence", "gate",
    # social / human
    "man", "woman", "child", "baby", "father", "mother", "brother", "sister",
    "friend", "neighbor", "stranger", "king", "queen", "soldier", "doctor",
    "teacher", "student", "farmer", "merchant",
    # abstract
    "love", "hate", "fear", "joy", "anger", "grief", "shame", "pride",
    "hope", "trust", "doubt", "truth", "lie", "dream", "memory", "thought",
    "idea", "mind", "soul", "spirit", "ghost", "death", "life", "birth",
    "time", "moment", "past", "future", "history", "fate", "luck", "chance",
    "freedom", "justice", "peace", "war", "law", "rule", "order", "chaos",
    # phenomena / natural
    "light", "shadow", "dust", "smoke", "ash", "flame", "spark", "bubble",
    "wave", "tide", "current", "echo", "noise", "silence", "music", "song",
    "rhythm", "melody", "voice", "whisper", "shout", "laughter",
    # time / season
    "morning", "noon", "evening", "night", "dawn", "dusk", "midnight",
    "summer", "winter", "spring", "autumn", "year", "month", "week", "day",
    "hour", "minute", "decay", "growth",
    # measurement / structure
    "number", "letter", "word", "name", "story", "sentence", "verse",
    "chapter", "title", "page", "line", "color", "shape", "size", "weight",
    "length", "depth", "height",
]
NOUN_POOL = sorted(set(w.lower() for w in NOUN_POOL))


# ---------------------------------------------------------------------------
# Embedders (re-uses the canonical implementations + multi_embed wrappers)
# ---------------------------------------------------------------------------
from src.dat_eval.scripts.rescore_multi_embed import (  # noqa: E402
    GloVeEmbedder, FastTextEmbedder, SBERTEmbedder, score_cdat_trial,
)


def appropriateness_for(words: list[str], cue: str, emb) -> float:
    """Mean cosine similarity between cue and each word in `words`,
    on the [0, 200] shifted scale used in our cdat scoring."""
    if not words:
        return math.nan
    cue_vec = emb.encode(cue.lower())
    if np.linalg.norm(cue_vec) == 0:
        return math.nan
    sims = []
    for w in words:
        v = emb.encode(w.lower())
        if np.linalg.norm(v) == 0:
            continue
        d = cosine_distance(cue_vec, v)
        if math.isfinite(d):
            sims.append(1.0 - d)
    if not sims:
        return math.nan
    return float(np.mean([100.0 * (1.0 + s) for s in sims]))


def random_baseline(cues: list[str], emb, n_samples: int = 100, seed: int = 42,
                    n_per_sample: int = 7) -> list[float]:
    """Generate per-cue random appropriateness scores under one embedding.
    Returns one flat list of n_samples × len(cues) appropriateness values."""
    rng = np.random.default_rng(seed)
    pool = NOUN_POOL
    out = []
    for cue in cues:
        for _ in range(n_samples):
            words = list(rng.choice(pool, size=n_per_sample, replace=False))
            a = appropriateness_for(words, cue, emb)
            if math.isfinite(a):
                out.append(a)
    return out


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Return BH-adjusted p-values for a list of p-values.

    NaN-robust: NaN inputs (e.g. a degenerate Welch t-test) are excluded from
    the correction and returned as NaN, rather than being sorted to the end and
    poisoning every adjusted p-value via the monotonicity accumulate. The FDR
    is computed over the valid (non-NaN) p-values only.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return [float(x) for x in out]
    order = np.argsort(pv)
    sorted_p = pv[order]
    sorted_adj = sorted_p * m / np.arange(1, m + 1)
    sorted_adj_mono = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(sorted_adj_mono, 1.0)
    out[valid] = adj
    return [float(x) for x in out]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_per_cue(model_dir: Path, emb, temp_pat: str = "cdat_responses_t*.json"
                    ) -> dict[str, dict[str, list[tuple[str, float, float]]]]:
    """For one model, scoring under one embedding, return:
        {temp_label: [(cue, appropriateness, novelty), ...]}
    """
    out: dict[str, list[tuple[str, float, float]]] = {}
    for path in sorted(model_dir.glob(temp_pat)):
        # File names look like cdat_responses_t1-0.json -> extract "1.0"
        temp = path.stem.replace("cdat_responses_t", "").replace("-", ".")
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        rows = []
        for cue, entry in data.items():
            words = entry.get("words") or []
            if not words:
                continue
            # Appropriateness under this embedder
            app = appropriateness_for(words, cue, emb)
            if math.isnan(app):
                continue
            # Novelty: use existing scoring from multi_embed
            nov, _ = score_cdat_trial(cue, words, emb)
            if math.isnan(nov):
                continue
            rows.append((cue, app, nov))
        if rows:
            out[temp] = rows
    return out


def gate_one_embedding(emb, emb_name: str, model_dirs: list[Path], cues: list[str],
                       alpha: float = 0.001):
    """Compute gated CDAT under one embedding. Returns dict keyed by model_key."""
    print(f"\n=== {emb_name}: random baseline ===", flush=True)
    random_scores = random_baseline(cues, emb)
    rmean = float(np.mean(random_scores))
    rsd = float(np.std(random_scores))
    print(f"Random baseline: n={len(random_scores)}, mean={rmean:.3f}, sd={rsd:.3f}",
          flush=True)

    # Collect per-model per-temp data
    per_model: dict[str, dict[str, list[tuple[str, float, float]]]] = {}
    for mdir in model_dirs:
        per_model[mdir.name] = collect_per_cue(mdir, emb)

    # Determine all temperatures
    all_temps = sorted({t for d in per_model.values() for t in d})

    # Per-temperature: collect per-model (mean_app, t_stat, raw_p)
    # Then BH-adjust p across models within the temperature.
    per_temp_results: dict[str, dict[str, dict]] = {}  # temp -> model -> {pass, ...}
    for temp in all_temps:
        models_with_data = [m for m in per_model if temp in per_model[m]]
        ts, ps, means = [], [], []
        for m in models_with_data:
            apps = [a for _, a, _ in per_model[m][temp]]
            t, p = stats.ttest_ind(apps, random_scores, equal_var=False)
            ts.append(float(t))
            ps.append(float(p))
            means.append(float(np.mean(apps)))
        adj_ps = benjamini_hochberg(ps)
        per_temp_results[temp] = {}
        for m, t, raw_p, adj_p, mu in zip(models_with_data, ts, ps, adj_ps, means):
            passed = (adj_p < alpha) and (mu > rmean)
            per_temp_results[temp][m] = {
                "t_stat": t, "raw_p": raw_p, "adj_p": adj_p,
                "mean_app": mu, "random_mean": rmean,
                "passed": bool(passed),
            }

    # Per-model gated CDAT score: mean novelty over passing temps.
    out: dict[str, dict] = {}
    for m, by_temp in per_model.items():
        passing_temps = [t for t in by_temp
                         if t in per_temp_results
                         and per_temp_results[t].get(m, {}).get("passed")]
        if not passing_temps:
            out[m] = {"cdat": None, "passed_temps": [], "gate": {}}
            continue
        novs = []
        for t in passing_temps:
            for _, _, nov in by_temp[t]:
                novs.append(nov)
        gated = float(np.mean(novs)) if novs else None
        out[m] = {
            "cdat": gated,
            "passed_temps": passing_temps,
            "gate": {t: per_temp_results[t][m] for t in by_temp
                     if t in per_temp_results and m in per_temp_results[t]},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out_path = Path(ap.parse_args().out)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; pass a new --out path")
    model_dirs = sorted(d for d in RUN_DIR.iterdir() if d.is_dir())
    print(f"{len(model_dirs)} model directories.")

    # Collect all unique cues from any model's CDAT responses
    cues: set[str] = set()
    for mdir in model_dirs:
        for path in mdir.glob("cdat_responses_t*.json"):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            cues.update(data.keys())
    cues_sorted = sorted(cues)
    print(f"{len(cues_sorted)} unique cues across all models.")

    embedders = [
        ("glove", lambda: GloVeEmbedder()),
        ("fasttext", lambda: FastTextEmbedder()),
        ("sbert", lambda: SBERTEmbedder("all-mpnet-base-v2")),
    ]

    all_out: dict[str, dict] = {}
    for emb_name, factory in embedders:
        emb = factory()
        all_out[emb_name] = gate_one_embedding(emb, emb_name, model_dirs, cues_sorted)
        del emb
        # Brief summary
        passed = sum(1 for v in all_out[emb_name].values() if v["cdat"] is not None)
        print(f"{emb_name}: {passed}/{len(all_out[emb_name])} models passed gate "
              f"on at least one temperature.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved gated scores to {out_path}")


if __name__ == "__main__":
    main()
