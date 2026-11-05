"""Greedy-embedding baseline for the DAT.

Demonstrates that the DAT is trivially solvable by an algorithm with
direct access to the scoring embedding: pick a random first noun, then
iteratively pick the noun that maximises mean cosine distance to the
already-chosen set. No creativity required.

Usage (needs GloVe 840B; see README):
    python -m src.dat_eval.scripts.run_greedy_baseline configs/greedy_baseline.yaml
"""

import argparse
import json
import time

import numpy as np

from src.utils import GLOVE_PATH, init_directory, load_config, save_config
from src.dat_eval.dat import GloVeEmbeddings, score_dat


def load_noun_vocab(glove: GloVeEmbeddings) -> list[str]:
    """WordNet noun lemmas ∩ GloVe vocab, lowercase alpha, length 3–19."""
    import nltk
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    nouns: set[str] = set()
    for syn in wn.all_synsets("n"):
        for lemma in syn.lemma_names():
            w = lemma.lower()
            if w.isalpha() and 2 < len(w) < 20:
                nouns.add(w)
    glove_vocab = glove.vocab
    return sorted(w for w in nouns if w in glove_vocab)


def greedy_pick(embedding_matrix: np.ndarray, seed: int, n_words: int) -> list[int]:
    """Return the indices chosen by the greedy argmax-distance algorithm.

    First word is uniform random. Each subsequent word minimises mean
    cosine similarity to the already-chosen set (= maximises mean
    cosine distance).
    """
    rng = np.random.default_rng(seed)
    n_vocab = embedding_matrix.shape[0]
    first = int(rng.integers(n_vocab))
    chosen = [first]
    sum_sim = embedding_matrix @ embedding_matrix[first]
    sum_sim[first] = np.inf
    for _ in range(n_words - 1):
        nxt = int(np.argmin(sum_sim))
        chosen.append(nxt)
        sim_nxt = embedding_matrix @ embedding_matrix[nxt]
        sum_sim = sum_sim + sim_nxt
        sum_sim[nxt] = np.inf
    return chosen


def main(config_path: str, debug: bool = False):
    config = load_config(config_path)
    output_dir = init_directory(config["output_dir"])
    save_config(config, output_dir)

    glove_path = config.get("glove_path") or GLOVE_PATH
    n_trials = config.get("n_trials", 120)
    n_words = config.get("n_words", 10)
    base_seed = config.get("base_seed", 0)

    if debug:
        n_trials = 3

    glove = GloVeEmbeddings(glove_path)
    print("Loading / filtering noun vocab...")
    vocab = load_noun_vocab(glove)
    print(f"Filtered noun vocab: {len(vocab)}")

    # Row-normalised embedding matrix.
    mat = np.stack([glove[w] for w in vocab]).astype(np.float32)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    trials = []
    t0 = time.time()
    for i in range(n_trials):
        seed = base_seed + i
        idxs = greedy_pick(mat, seed=seed, n_words=n_words)
        words = [vocab[j] for j in idxs]
        result = score_dat(words, glove)
        trials.append({
            "trial": i,
            "seed": seed,
            "words": words,
            "valid_words": result["valid_words"],
            "score": result["score"],
            "sufficient": result["sufficient"],
        })
        if (i + 1) % 20 == 0 or i + 1 == n_trials:
            elapsed = time.time() - t0
            print(f"  trial {i+1}/{n_trials}  ({elapsed:.1f}s, score={result['score']:.2f})")

    scores = np.array([t["score"] for t in trials if t["sufficient"]])
    summary = {
        "n_trials": n_trials,
        "n_sufficient": int(len(scores)),
        "n_vocab": len(vocab),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "median": float(np.median(scores)),
    }

    with open(output_dir / "trials.json", "w") as f:
        json.dump(trials, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Greedy DAT over {len(vocab)} nouns, {n_trials} trials:")
    print(f"  mean   = {summary['mean']:.2f}")
    print(f"  std    = {summary['std']:.2f}")
    print(f"  range  = [{summary['min']:.2f}, {summary['max']:.2f}]")
    print(f"  median = {summary['median']:.2f}")
    print()
    print("Example trial 0:")
    print(f"  words          : {trials[0]['words']}")
    print(f"  scored 7       : {trials[0]['valid_words']}")
    print(f"  DAT score      : {trials[0]['score']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.debug)
