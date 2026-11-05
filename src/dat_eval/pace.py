"""PACE (Parallel Association Chain Evaluation) implementation.

Implements the PACE metric from Qiu & Hu (2025, EMNLP): given a seed word,
the model produces 3 first-associations, then builds a 20-word chain from
each. Scored by average semantic distance between each word and all preceding
words in the chain, using FastText embeddings.

Higher association distance = higher creativity.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


class FastTextEmbeddings:
    """Lazy-loaded FastText embeddings for PACE scoring.

    Loads from a .vec text file (same format as GloVe). Download crawl-300d-2M.vec
    from https://fasttext.cc/docs/en/english-vectors.html
    """

    def __init__(self, vec_path: str):
        """Initialize FastText embeddings.

        Args:
            vec_path: Path to FastText .vec text file (e.g., crawl-300d-2M.vec).
        """
        self.vec_path = vec_path
        self._vectors: dict[str, np.ndarray] | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self._vectors is not None:
            return
        from pathlib import Path
        path = Path(self.vec_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FastText vectors not found: {path}\n"
                "Download from: https://fasttext.cc/docs/en/english-vectors.html\n"
                "File: crawl-300d-2M.vec.zip"
            )
        print(f"Loading FastText embeddings from {path}...")
        self._vectors = {}
        with open(path, "r", encoding="utf-8") as f:
            # First line is header: "2000000 300"
            header = f.readline()
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self._vectors[word] = vec
        print(f"Loaded {len(self._vectors)} FastText vectors")

    def encode(self, word: str) -> np.ndarray:
        """Get FastText embedding for a word."""
        if word not in self._cache:
            self._load()
            if word in self._vectors:
                self._cache[word] = self._vectors[word]
            elif word.lower() in self._vectors:
                self._cache[word] = self._vectors[word.lower()]
            else:
                self._cache[word] = np.zeros(300, dtype=np.float32)
        return self._cache[word]


def score_chain(
    chain: list[str],
    embeddings: FastTextEmbeddings,
) -> dict:
    """Score a single association chain using the PACE metric.

    For each position n, compute average cosine distance from position n to
    ALL preceding positions. Then average across all positions.

    A_chain = (1/(n-1)) * sum_{i=2}^{n} [ (1/(i-1)) * sum_{j=1}^{i-1} D(i,j) ]

    Args:
        chain: List of words in the chain (including seed and first-association).
        embeddings: FastText embedding model.

    Returns:
        Dict with chain_score, per_position_scores, and word list.
    """
    if len(chain) < 2:
        return {"chain_score": 0.0, "per_position_scores": [], "words": chain}

    # Encode; drop OOV words that produce zero vectors (cosine distance would be NaN)
    valid_pairs = [(w, embeddings.encode(w.lower())) for w in chain]
    valid_pairs = [(w, v) for w, v in valid_pairs if np.linalg.norm(v) > 0]

    if len(valid_pairs) < 2:
        return {"chain_score": 0.0, "per_position_scores": [],
                "words": [w for w, _ in valid_pairs], "n_oov": len(chain) - len(valid_pairs)}

    vectors = [v for _, v in valid_pairs]
    kept_words = [w for w, _ in valid_pairs]
    n = len(vectors)

    per_position = []
    for i in range(1, n):
        dists = []
        for j in range(i):
            d = cosine_distance(vectors[i], vectors[j])
            if np.isfinite(d):
                dists.append(float(d))
        if dists:
            per_position.append(float(np.mean(dists)))

    chain_score = float(np.mean(per_position)) if per_position else 0.0

    return {
        "chain_score": chain_score,
        "per_position_scores": per_position,
        "words": kept_words,
        "n_oov": len(chain) - len(kept_words),
    }


def score_seed(
    chains: list[list[str]],
    embeddings: FastTextEmbeddings,
) -> dict:
    """Score all chains for a single seed word.

    A_seed = (1/3) * sum of chain scores.

    Args:
        chains: List of 3 chains, each a list of 20 words.
        embeddings: FastText model.

    Returns:
        Dict with seed_score and per-chain breakdowns.
    """
    chain_results = [score_chain(c, embeddings) for c in chains]
    chain_scores = [r["chain_score"] for r in chain_results if np.isfinite(r["chain_score"]) and r["chain_score"] > 0]
    seed_score = float(np.mean(chain_scores)) if chain_scores else 0.0

    return {
        "seed_score": seed_score,
        "chain_results": chain_results,
    }


def score_model(
    all_chains: dict[str, list[list[str]]],
    embeddings: FastTextEmbeddings,
) -> dict:
    """Score all seed words for a single model.

    A_model = (1/S) * sum of seed scores.

    Args:
        all_chains: Dict mapping seed words to list of 3 chains each.
        embeddings: FastText model.

    Returns:
        Dict with model_score and per-seed breakdowns.
    """
    seed_results = {}
    seed_scores = []

    for seed_word, chains in all_chains.items():
        result = score_seed(chains, embeddings)
        seed_results[seed_word] = result
        if np.isfinite(result["seed_score"]) and result["seed_score"] > 0:
            seed_scores.append(result["seed_score"])

    model_score = float(np.mean(seed_scores)) if seed_scores else 0.0

    return {
        "model_score": model_score,
        "seed_results": seed_results,
        "n_seeds": len(seed_scores),
    }


# --- Prompting ---

def pace_stage1_prompt(seed: str) -> str:
    """Generate the PACE first-stage prompt (3 first-associations)."""
    return (
        f'Starting with the word "{seed}", generate three different words that '
        f"directly associate with this initial word only (not with each other). "
        f"Please put down only single words, and do not use proper nouns "
        f"(such as names, brands, etc.). "
        f"For each word, provide a brief explanation of its connection to "
        f'"{seed}". Return in JSON format:\n'
        f'{{"results": [{{"word": "", "reason": ""}}, '
        f'{{"word": "", "reason": ""}}, '
        f'{{"word": "", "reason": ""}}]}}'
    )


def pace_stage2_prompt(seed: str, second_word: str, reason: str) -> str:
    """Generate the PACE second-stage prompt (20-word chain)."""
    return (
        f'Starting with the word pair "{seed}" -> "{second_word}", generate a '
        f"chain of 20 words where each new word should be associated with ONLY "
        f"the word immediately before it. Generate the third word based on "
        f'"{second_word}", then generate the fourth word based on your third '
        f"word, and so on. Please put down only single words, and do not use "
        f"proper nouns (such as names, brands, etc.). For each word, provide "
        f"a brief explanation of its connection to the previous word. "
        f"Return in JSON format with exactly 20 entries:\n"
        f'{{"results": [{{"word": "{second_word}", "reason": "{reason}"}}, '
        f'{{"word": "", "reason": ""}}, ... ]}}'
    )


# --- Seed words ---

# 50 seed words spanning IDS semantic domains, balanced by frequency.
# Following PACE's methodology but with a smaller set for efficiency.
DEFAULT_SEEDS = [
    # Physical world
    "rock", "cloud", "dust", "rainbow", "volcano",
    # Kinship
    "mother", "child", "widow", "brother", "ancestor",
    # Animals
    "eagle", "worm", "dove", "spider", "whale",
    # Body
    "bone", "lung", "tooth", "thumb", "skull",
    # Food & drink
    "bread", "pepper", "honey", "feast", "grain",
    # Clothing
    "silk", "boot", "crown", "thread", "veil",
    # House
    "roof", "chimney", "pillow", "cellar", "fence",
    # Tools
    "hammer", "needle", "wheel", "rope", "anchor",
    # Motion
    "flight", "crawl", "drift", "leap", "chase",
    # Emotions
    "grief", "pride", "shame", "bliss", "rage",
]
