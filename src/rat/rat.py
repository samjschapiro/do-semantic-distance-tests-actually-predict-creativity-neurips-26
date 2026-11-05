"""Compound Remote Associates Test (RAT) for LLMs.

Standard psychometric instrument from Mednick (1962), refined into the
Compound Remote Associates form by Bowden & Jung-Beeman (2003). Each item
presents three remote stimulus words; the canonical answer is the single
word that combines with all three to form compound words or common phrases.

Example: cottage / swiss / cake -> cheese
         (cottage cheese, swiss cheese, cheesecake)

Pilot uses a 30-item hand-curated subset of the Bowden & Jung-Beeman norms,
spanning easy/medium/hard difficulty levels.

Scoring: exact-match accuracy. The literature convention is strict match
on the canonical answer; this module also exposes a lenient matcher that
accepts plurals and simple morphological variants for sensitivity checks.
"""

import re

# 30-item subset from Bowden & Jung-Beeman 2003 norms.
# Each item: stems (3-tuple) and the canonical single-word answer (lowercase).
# Optional `aliases` lists alternates accepted in the original norms.
ITEM_BANK_V1 = [
    # --- Easy (high human accuracy, fast solution time) ---
    {"stems": ("cottage", "swiss", "cake"),         "answer": "cheese"},
    {"stems": ("cream", "skate", "water"),          "answer": "ice"},
    {"stems": ("show", "life", "row"),              "answer": "boat"},
    {"stems": ("night", "wrist", "stop"),           "answer": "watch"},
    {"stems": ("duck", "fold", "dollar"),           "answer": "bill"},
    {"stems": ("rocking", "wheel", "high"),         "answer": "chair"},
    {"stems": ("dew", "comb", "bee"),               "answer": "honey"},
    {"stems": ("fountain", "baking", "pop"),        "answer": "soda"},
    {"stems": ("preserve", "ranger", "tropical"),   "answer": "forest"},
    {"stems": ("aid", "rubber", "wagon"),           "answer": "band"},

    # --- Medium ---
    {"stems": ("flake", "mobile", "cone"),          "answer": "snow"},
    {"stems": ("safety", "cushion", "point"),       "answer": "pin"},
    {"stems": ("cane", "daddy", "plum"),            "answer": "sugar"},
    {"stems": ("dream", "break", "light"),          "answer": "day"},
    {"stems": ("fish", "mine", "rush"),             "answer": "gold"},
    {"stems": ("political", "surprise", "line"),    "answer": "party"},
    {"stems": ("measure", "worm", "video"),         "answer": "tape"},
    {"stems": ("sense", "courtesy", "place"),       "answer": "common"},
    {"stems": ("worm", "shelf", "end"),             "answer": "book"},
    {"stems": ("piece", "mind", "dating"),          "answer": "game"},

    # --- Hard (lower human accuracy / longer solution time) ---
    {"stems": ("sandwich", "golf", "foot"),         "answer": "club"},
    {"stems": ("river", "note", "account"),         "answer": "bank"},
    {"stems": ("print", "berry", "bird"),           "answer": "blue"},
    {"stems": ("fly", "clip", "wall"),              "answer": "paper"},
    {"stems": ("food", "forward", "break"),         "answer": "fast"},
    {"stems": ("cracker", "fly", "fighter"),        "answer": "fire"},
    {"stems": ("dust", "cereal", "fish"),           "answer": "bowl"},
    {"stems": ("shock", "shave", "taste"),          "answer": "after"},
    {"stems": ("cross", "rain", "tie"),             "answer": "bow"},
    {"stems": ("broken", "clear", "eye"),           "answer": "glass"},
]


def rat_prompt(stems: tuple[str, str, str] | list[str]) -> str:
    """Generate the RAT prompt for an item.

    Fixed template, no in-context example. Asks for a single-word answer
    in lowercase, no explanation.
    """
    a, b, c = stems
    return (
        f'What single word can be combined with each of "{a}", "{b}", and "{c}" '
        f'to form a compound word or common phrase?\n\n'
        f'Respond with ONLY the single answer word in lowercase. No explanation.'
    )


# --- Response parsing and scoring ---

_WORD_RE = re.compile(r"[A-Za-z]+")


def parse_response(raw: str | None) -> str | None:
    """Extract the model's answer word from a raw response.

    Strategy:
      1. Strip whitespace / surrounding punctuation.
      2. Take the FIRST alphabetic token (handles cases where the model
         emits a longer string despite the prompt — common with smaller
         models that ignore "no explanation").
      3. Lowercase.

    Returns None if no alphabetic token is present.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    m = _WORD_RE.search(raw)
    if m is None:
        return None
    return m.group(0).lower()


def score_response_strict(response: str | None, answer: str) -> bool:
    """Strict exact match — the literature convention."""
    parsed = parse_response(response)
    return parsed is not None and parsed == answer.lower()


def score_response_lenient(response: str | None, answer: str) -> bool:
    """Lenient match — accepts simple morphological variants.

    Treats the response as correct if the parsed token equals the canonical
    answer OR is a plural/-ing/-ed variant. Reported as a sensitivity check
    only; the headline number uses strict match.
    """
    parsed = parse_response(response)
    if parsed is None:
        return False
    a = answer.lower()
    if parsed == a:
        return True
    # Drop trailing 's', 'es', 'ing', 'ed' — and check both directions
    variants = {parsed, parsed.rstrip("s"), parsed.rstrip("es"),
                parsed[:-3] if parsed.endswith("ing") else parsed,
                parsed[:-2] if parsed.endswith("ed") else parsed,
                a, a.rstrip("s"), a.rstrip("es"),
                a[:-3] if a.endswith("ing") else a,
                a[:-2] if a.endswith("ed") else a}
    # Strip empties
    variants = {v for v in variants if v}
    # Match if any variant of parsed equals any variant of answer
    return parsed in variants and a in variants and (
        parsed == a
        or (parsed.startswith(a) and len(parsed) - len(a) <= 3)
        or (a.startswith(parsed) and len(a) - len(parsed) <= 3)
    )


def score_item(response: str | None, answer: str) -> dict:
    """Score one (response, answer) pair under both strict and lenient modes."""
    parsed = parse_response(response)
    return {
        "parsed": parsed,
        "answer": answer.lower(),
        "correct_strict": score_response_strict(response, answer),
        "correct_lenient": score_response_lenient(response, answer),
    }


def score_bok(responses: list[str | None], answer: str) -> dict:
    """Score a best-of-K sample list. Item is correct if ANY sample matches.

    Returns dict with parsed list, answer, correct_strict_any, correct_lenient_any.
    """
    parsed = [parse_response(r) for r in responses]
    return {
        "parsed": parsed,
        "answer": answer.lower(),
        "correct_strict_any": any(score_response_strict(r, answer) for r in responses),
        "correct_lenient_any": any(score_response_lenient(r, answer) for r in responses),
    }
