"""Rescore DAT, CDAT and PACE responses under three embedding models.

Every model's saved responses (data/responses/<model>/) are rescored under

  - GloVe 840B 300d (the original DAT embedding)
  - FastText crawl-300d-2M (the original PACE embedding)
  - Sentence-BERT all-mpnet-base-v2 (the original CDAT embedding)

and the gated CDAT score from the appropriateness gate (cdat_gate.py) is attached per embedding.
The result has the same layout as data/scores/multi_embed_scores.json, which is the file the paper
tables and figures read: {embedding: {model: {dat, cdat, cdat_novelty, cdat_appropriateness, pace}}}.

Usage (writes a new file; the shipped data/ is never touched):
    python -m src.dat_eval.scripts.rescore_multi_embed \
        [--gate data/scores/cdat_gated_scores.json] [--out outputs/rescored/multi_embed_scores.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from src.dat_eval.dat import GloVeEmbeddings, validate_words
from src.utils import FASTTEXT_PATH, GLOVE_PATH, ROOT

RUN_DIR = ROOT / "data" / "responses"


# ---------------------------------------------------------------------------
# Unified embedder interface
# ---------------------------------------------------------------------------
class Embedder:
    """Simple protocol: encode(word) -> np.ndarray; returns zero vector on OOV.
    has(word) -> bool tells you whether this word is natively in-vocab.
    """
    name: str

    def has(self, word: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def encode(self, word: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GloVeEmbedder(Embedder):
    name = "glove"

    def __init__(self, glove: GloVeEmbeddings | None = None):
        self._g = glove if glove is not None else GloVeEmbeddings(GLOVE_PATH)

    def has(self, word: str) -> bool:
        return word in self._g

    def encode(self, word: str) -> np.ndarray:
        if word in self._g:
            return self._g[word]
        return np.zeros(300, dtype=np.float32)


class FastTextEmbedder(Embedder):
    name = "fasttext"

    def __init__(self):
        self._v: dict[str, np.ndarray] = {}
        print(f"Loading FastText from {FASTTEXT_PATH}...", flush=True)
        with open(FASTTEXT_PATH, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip().split(" ")
                self._v[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        print(f"  FastText loaded: {len(self._v)} vectors", flush=True)

    def has(self, word: str) -> bool:
        return word in self._v or word.lower() in self._v

    def encode(self, word: str) -> np.ndarray:
        if word in self._v:
            return self._v[word]
        if word.lower() in self._v:
            return self._v[word.lower()]
        return np.zeros(300, dtype=np.float32)


class SBERTEmbedder(Embedder):
    name = "sbert"

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading SBERT ({model_name})...", flush=True)
        self._m = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def has(self, word: str) -> bool:
        return True

    def encode(self, word: str) -> np.ndarray:
        if word not in self._cache:
            self._cache[word] = self._m.encode(word, normalize_embeddings=True)
        return self._cache[word]


# ---------------------------------------------------------------------------
# Generic scoring functions (embedder-agnostic)
# ---------------------------------------------------------------------------
def cosine_d(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return float("nan")
    d = cosine_distance(a, b)
    return float(d) if np.isfinite(d) else float("nan")


def score_dat_trial(words: list[str], emb: Embedder, glove_vocab: GloVeEmbeddings) -> float:
    """Original DAT: first 7 valid GloVe words, mean pairwise cosine * 100.
    We keep GloVe's vocab check so the word set is identical across embedders;
    only the distance computation changes."""
    valid = validate_words(words, glove_vocab)
    if len(valid) < 7:
        return math.nan
    vecs = [emb.encode(w) for w in valid[:7]]
    ds = []
    for i in range(7):
        for j in range(i + 1, 7):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
    return float(np.mean(ds)) * 100 if ds else math.nan


def score_cdat_trial(cue: str, words: list[str], emb: Embedder) -> tuple[float, float]:
    """CDAT under arbitrary embedder: novelty = pairwise distance among the
    first 7 valid words (matching the paper and cdat.py); appropriateness =
    mean cos-sim to the cue on the shifted [0, 200] scale (matching
    cdat.py and cdat_gate.py)."""
    if not words:
        return math.nan, math.nan
    # Keep only valid words (non-empty embedding) and truncate to first 7
    valid = []
    for w in words:
        v = emb.encode(w)
        if np.linalg.norm(v) > 0:
            valid.append((w, v))
        if len(valid) == 7:
            break
    if len(valid) < 2:
        return math.nan, math.nan
    vecs = [v for _, v in valid]
    # Novelty: mean pairwise cosine distance of first 7 valid words × 100
    ds = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
    nov = float(np.mean(ds)) * 100 if ds else math.nan
    # Appropriateness: mean (1 + cos_sim) × 100, shifted to [0, 200]
    cue_v = emb.encode(cue)
    if np.linalg.norm(cue_v) == 0:
        app = math.nan
    else:
        sims = []
        for v in vecs:
            d = cosine_d(cue_v, v)
            if not math.isnan(d):
                sims.append(1.0 - d)
        app = float(np.mean([100.0 * (1.0 + s) for s in sims])) if sims else math.nan
    return nov, app


def score_pace_chain(chain: list[str], emb: Embedder) -> float:
    """PACE chain score = mean-over-positions of mean-cosine-distance to all
    earlier positions. Matches Qiu & Hu; embedder pluggable."""
    vecs = [emb.encode(w.lower()) for w in chain]
    vecs = [v for v in vecs if np.linalg.norm(v) > 0]
    if len(vecs) < 2:
        return math.nan
    per_pos = []
    for i in range(1, len(vecs)):
        ds = []
        for j in range(i):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
        if ds:
            per_pos.append(float(np.mean(ds)))
    return float(np.mean(per_pos)) if per_pos else math.nan


# ---------------------------------------------------------------------------
# Per-model rescore driver
# ---------------------------------------------------------------------------
def rescore_model(model_dir: Path, emb: Embedder, glove_vocab: GloVeEmbeddings) -> dict:
    out = {"dat": math.nan, "cdat_novelty": math.nan,
           "cdat_appropriateness": math.nan, "pace": math.nan}

    # --- DAT: pooled across temperatures ---
    dat_trials = []
    for tf in sorted(model_dir.glob("dat_responses_t*.json")):
        try:
            data = json.loads(tf.read_text())
        except Exception:
            continue
        for tr in data:
            words = tr.get("words") or []
            if not words:
                continue
            s = score_dat_trial(words, emb, glove_vocab)
            if not math.isnan(s):
                dat_trials.append(s)
    if dat_trials:
        out["dat"] = float(np.mean(dat_trials))

    # --- CDAT: pooled across temperatures and cues ---
    cdat_nov, cdat_app = [], []
    for tf in sorted(model_dir.glob("cdat_responses_t*.json")):
        try:
            data = json.loads(tf.read_text())
        except Exception:
            continue
        # data is dict cue -> {words, ...}
        for cue, entry in data.items():
            words = entry.get("words") or []
            if not words:
                continue
            nov, app = score_cdat_trial(cue, words, emb)
            if not math.isnan(nov):
                cdat_nov.append(nov)
            if not math.isnan(app):
                cdat_app.append(app)
    if cdat_nov:
        out["cdat_novelty"] = float(np.mean(cdat_nov))
    if cdat_app:
        out["cdat_appropriateness"] = float(np.mean(cdat_app))

    # --- PACE: pooled across seeds and chains ---
    pace_file = model_dir / "pace_responses_t0-0.json"
    if pace_file.exists():
        try:
            data = json.loads(pace_file.read_text())
        except Exception:
            data = {}
        # Match the original PACE pipeline's filtering: only chains with >=3
        # words, then drop chain scores <= 0 and seed scores <= 0 before
        # taking means. This keeps multi-embed numbers comparable to the
        # main-pipeline PACE values in all_scores.json (within ~1e-4).
        seed_scores = []
        for seed, entry in data.items():
            chains = entry.get("chains") or []
            chain_scores = []
            for chain_obj in chains:
                words = chain_obj.get("chain") if isinstance(chain_obj, dict) else None
                if not words or len(words) < 3:
                    continue
                s = score_pace_chain(list(words), emb)
                if not math.isnan(s) and s > 0:
                    chain_scores.append(s)
            if chain_scores:
                seed_score = float(np.mean(chain_scores))
                if seed_score > 0:
                    seed_scores.append(seed_score)
        if seed_scores:
            out["pace"] = float(np.mean(seed_scores))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=str(ROOT / "data/scores/cdat_gated_scores.json"),
                    help="gated CDAT scores from cdat_gate.py")
    ap.add_argument("--out", default=str(ROOT / "outputs/rescored/multi_embed_scores.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; pass a new --out path")

    model_dirs = sorted(d for d in RUN_DIR.iterdir() if d.is_dir())
    print(f"{len(model_dirs)} model directories to rescore.")

    # Shared GloVe vocab used for DAT validation across all embedders.
    glove_vocab = GloVeEmbeddings(GLOVE_PATH)

    embedders = [
        ("glove", lambda: GloVeEmbedder(glove_vocab)),  # reuse the vocab copy: one GloVe load
        ("fasttext", lambda: FastTextEmbedder()),
        ("sbert", lambda: SBERTEmbedder("all-mpnet-base-v2")),
    ]

    all_scores: dict[str, dict[str, dict[str, float]]] = {}
    for emb_name, factory in embedders:
        print(f"\n=== Embedder: {emb_name} ===", flush=True)
        emb = factory()
        by_model = {}
        for mdir in model_dirs:
            scores = rescore_model(mdir, emb, glove_vocab)
            by_model[mdir.name] = scores
            nonan = {k: v for k, v in scores.items() if not math.isnan(v)}
            if nonan:
                print(f"  {mdir.name}: {nonan}", flush=True)
        all_scores[emb_name] = by_model
        del emb

    # Attach the gated CDAT score per embedding (NaN when the model failed the gate everywhere).
    gated = json.loads(Path(args.gate).read_text())
    for emb_name in all_scores:
        for model_key, scores in all_scores[emb_name].items():
            g = gated.get(emb_name, {}).get(model_key, {}).get("cdat")
            scores["cdat"] = g if g is not None else float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_scores, indent=2))
    print(f"\nSaved scores to {out_path}")


if __name__ == "__main__":
    main()
