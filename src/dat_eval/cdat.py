"""Conditional Divergent Association Task (CDAT) implementation.

Implements the CDAT from Nakajima et al. (2026, Findings of EACL): like DAT
but with a cue word. Words must be diverse from each other (novelty) AND
associated with the cue (appropriateness).

Novelty = mean pairwise cosine distance × 100 (same as DAT)
Appropriateness = mean cosine similarity to cue × 100 (shifted to [0, 200])
CDAT score = Novelty, conditional on passing appropriateness gate
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from src.dat_eval.dat import validate_words, GloVeEmbeddings


class SBERTEmbeddings:
    """Lazy-loaded Sentence-BERT embeddings for CDAT scoring."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"Loading SBERT model: {self.model_name}...")
        self._model = SentenceTransformer(self.model_name)

    def encode(self, word: str) -> np.ndarray:
        """Get SBERT embedding for a word."""
        if word not in self._cache:
            self._load()
            self._cache[word] = self._model.encode(word, normalize_embeddings=True)
        return self._cache[word]

    def __contains__(self, word: str) -> bool:
        """SBERT can encode any string, so always returns True."""
        return True

    def encode_batch(self, words: list[str]) -> list[np.ndarray]:
        """Batch encode multiple words."""
        self._load()
        uncached = [w for w in words if w not in self._cache]
        if uncached:
            vecs = self._model.encode(uncached, normalize_embeddings=True)
            for w, v in zip(uncached, vecs):
                self._cache[w] = v
        return [self._cache[w] for w in words]


def validate_words_sbert(words: list[str]) -> list[str]:
    """Validate words for CDAT (less strict than DAT since SBERT handles anything).

    - Lowercase and strip
    - Must be > 1 character
    - Must be alphabetic
    - Remove duplicates

    Returns:
        List of valid, unique words.
    """
    valid = []
    seen = set()
    for word in words:
        word = word.lower().strip()
        if len(word) <= 1:
            continue
        if not word.replace("-", "").isalpha():
            continue
        if word in seen:
            continue
        seen.add(word)
        valid.append(word)
    return valid


def score_cdat(
    words: list[str],
    cue: str,
    embeddings: SBERTEmbeddings,
) -> dict:
    """Score a set of words using the CDAT metric.

    Args:
        words: Raw list of words from the model.
        cue: The cue word that was given.
        embeddings: SBERT embedding model.

    Returns:
        Dict with:
            - novelty: mean pairwise distance × 100
            - appropriateness: mean (1 + cos_sim(cue, word)) × 100
            - valid_words: the 7 words used
            - sufficient: whether 7+ valid words found
    """
    valid = validate_words_sbert(words)

    if len(valid) < 7:
        return {
            "novelty": 0.0,
            "appropriateness": 0.0,
            "valid_words": valid,
            "n_valid": len(valid),
            "sufficient": False,
        }

    scored_words = valid[:7]

    # Encode all words + cue
    all_to_encode = [cue] + scored_words
    vectors = embeddings.encode_batch(all_to_encode)
    cue_vec = vectors[0]
    word_vecs = vectors[1:]

    # Novelty: mean pairwise cosine distance × 100
    distances = []
    for i in range(7):
        for j in range(i + 1, 7):
            d = cosine_distance(word_vecs[i], word_vecs[j])
            distances.append(float(d))
    novelty = float(np.mean(distances)) * 100

    # Appropriateness: mean (1 + cos_sim(cue, word)) × 100
    similarities = []
    for i in range(7):
        cos_sim = 1.0 - cosine_distance(cue_vec, word_vecs[i])
        similarities.append(float(cos_sim))
    appropriateness = float(np.mean([100 * (1 + s) for s in similarities]))

    return {
        "novelty": novelty,
        "appropriateness": appropriateness,
        "valid_words": scored_words,
        "n_valid": len(valid),
        "sufficient": True,
        "per_word_similarity": similarities,
    }


def appropriateness_gate(
    model_scores: list[float],
    random_scores: list[float],
    alpha: float = 0.001,
) -> dict:
    """Test whether a model passes the CDAT appropriateness gate.

    Two-sided Welch's t-test: model appropriateness vs random baseline.
    Model must have higher mean AND p < alpha.

    Args:
        model_scores: Appropriateness scores across cues for this model.
        random_scores: Appropriateness scores for random baseline.
        alpha: Significance threshold (FDR-adjusted).

    Returns:
        Dict with passed (bool), t_stat, p_value, model_mean, random_mean.
    """
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(model_scores, random_scores, equal_var=False)
    model_mean = float(np.mean(model_scores))
    random_mean = float(np.mean(random_scores))

    passed = (p_value < alpha) and (model_mean > random_mean)

    return {
        "passed": passed,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "model_mean": model_mean,
        "random_mean": random_mean,
    }


def generate_random_baseline(
    n_words: int,
    n_samples: int,
    cues: list[str],
    word_pool: list[str],
    embeddings: SBERTEmbeddings,
    seed: int = 42,
) -> list[float]:
    """Generate random baseline appropriateness scores.

    For each cue, samples n_words random nouns and computes appropriateness.

    Args:
        n_words: Words per sample (7).
        n_samples: Number of random samples per cue.
        cues: List of cue words.
        word_pool: Pool of nouns to sample from.
        embeddings: SBERT model.
        seed: Random seed.

    Returns:
        List of appropriateness scores across all cues × samples.
    """
    rng = np.random.default_rng(seed)
    scores = []

    for cue in cues:
        for _ in range(n_samples):
            words = list(rng.choice(word_pool, size=n_words, replace=False))
            result = score_cdat(words, cue, embeddings)
            if result["sufficient"]:
                scores.append(result["appropriateness"])

    return scores


# --- Prompting ---

def cdat_prompt(cue: str) -> str:
    """Generate the CDAT prompt for a given cue word."""
    return (
        f'Please enter 10 words that are as different from each other as possible, '
        f'in all meanings and uses of the words, yet semantically associated with '
        f'the following cue word: "{cue}".\n\n'
        f'Only use single nouns. Do not use proper nouns. '
        f'Do not use the cue word itself or variations of it.\n\n'
        f'Respond with ONLY a JSON array of exactly 10 words, like: '
        f'["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"]'
    )


# --- Cue words ---

# Subset of cue words spanning diverse semantic domains (from IDS chapters).
# Full set from Nakajima et al. uses 539 cues; we use a representative sample.
DEFAULT_CUES = [
    # Physical world
    "rock", "rain", "fire", "ocean", "shadow",
    # Animals
    "eagle", "spider", "whale", "butterfly", "wolf",
    # Food & drink
    "bread", "pepper", "honey", "salt", "wine",
    # Body
    "heart", "bone", "blood", "skin", "eye",
    # Emotions
    "anger", "joy", "fear", "grief", "hope",
    # Time
    "dawn", "winter", "midnight", "decay", "rhythm",
    # Social
    "crown", "prison", "market", "temple", "bridge",
    # Abstract
    "silence", "chaos", "truth", "dream", "memory",
    # Tools & artifacts
    "needle", "mirror", "wheel", "candle", "bell",
    # Nature
    "forest", "river", "storm", "dust", "crystal",
]
