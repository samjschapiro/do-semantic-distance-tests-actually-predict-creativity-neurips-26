"""Divergent Association Task (DAT) implementation.

Implements the DAT from Olson et al. (2021, PNAS): participant generates 10
words as different from each other as possible, scored by average pairwise
cosine distance using GloVe 840B embeddings.

Score = mean(21 pairwise cosine distances of first 7 valid words) × 100
"""

import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


class GloVeEmbeddings:
    """Lazy-loaded GloVe 840B 300d embeddings."""

    def __init__(self, glove_path: str | Path):
        self.glove_path = Path(glove_path)
        if not self.glove_path.exists():
            raise FileNotFoundError(
                f"GloVe file not found: {self.glove_path}\n"
                "Download from: https://nlp.stanford.edu/data/glove.840B.300d.zip"
            )
        self._vectors: dict[str, np.ndarray] | None = None

    def _load(self) -> None:
        """Load GloVe vectors from file."""
        if self._vectors is not None:
            return
        print(f"Loading GloVe embeddings from {self.glove_path}...")
        self._vectors = {}
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                self._vectors[word] = vec
        print(f"Loaded {len(self._vectors)} GloVe vectors")

    def __contains__(self, word: str) -> bool:
        self._load()
        return word in self._vectors

    def __getitem__(self, word: str) -> np.ndarray:
        self._load()
        return self._vectors[word]

    @property
    def vocab(self) -> set[str]:
        self._load()
        return set(self._vectors.keys())


def validate_words(words: list[str], embeddings: GloVeEmbeddings) -> list[str]:
    """Validate and filter words following DAT protocol.

    - Lowercase and strip non-alpha (except hyphens)
    - Must be > 1 character
    - Must exist in GloVe vocabulary
    - Remove duplicates (preserving order)

    Returns:
        List of valid, unique words.
    """
    valid = []
    seen = set()

    for word in words:
        # Lowercase and strip
        word = word.lower().strip()
        # Remove non-alphabetic characters except hyphens
        word = re.sub(r"[^a-z\-]", "", word)

        if len(word) <= 1:
            continue
        if word in seen:
            continue
        if word not in embeddings:
            # Try hyphenated variant
            hyphenated = word.replace(" ", "-")
            if hyphenated not in embeddings:
                continue
            word = hyphenated

        seen.add(word)
        valid.append(word)

    return valid


def score_dat(words: list[str], embeddings: GloVeEmbeddings) -> dict:
    """Score a set of words using the DAT metric.

    Takes the first 7 valid words, computes all 21 pairwise cosine distances,
    returns the mean × 100.

    Args:
        words: Raw list of words from the model.
        embeddings: GloVe embedding lookup.

    Returns:
        Dict with:
            - score: DAT score (float)
            - valid_words: the 7 words used for scoring
            - n_valid: number of valid words found
            - pairwise_distances: all 21 distances
            - sufficient: whether 7+ valid words were found
    """
    valid = validate_words(words, embeddings)

    if len(valid) < 7:
        return {
            "score": 0.0,
            "valid_words": valid,
            "n_valid": len(valid),
            "pairwise_distances": [],
            "sufficient": False,
        }

    # Take first 7
    scored_words = valid[:7]

    # Compute all 21 pairwise cosine distances
    vectors = [embeddings[w] for w in scored_words]
    distances = []
    for i in range(7):
        for j in range(i + 1, 7):
            d = cosine_distance(vectors[i], vectors[j])
            distances.append(float(d))

    score = float(np.mean(distances)) * 100

    return {
        "score": score,
        "valid_words": scored_words,
        "n_valid": len(valid),
        "pairwise_distances": distances,
        "sufficient": True,
    }


# --- Prompting ---

DAT_PROMPT = (
    "Please enter 10 words that are as different from each other as possible, "
    "in all meanings and uses of the words. Only use single nouns. "
    "Do not use proper nouns (names, places, brands). "
    "Do not use variations of the same word (e.g., don't use both 'run' and 'running').\n\n"
    "Respond with ONLY a JSON array of exactly 10 words, like: "
    '["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"]'
)
