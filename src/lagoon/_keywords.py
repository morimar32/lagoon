"""YAKE-based keyword extraction for document segments.

Extracts statistically prominent single-word keywords from text using YAKE,
filtered with a size-relative threshold (k / sqrt(n_tokens)) that adapts to
segment length.  Returns plain keyword lists — scores are an internal detail.

This provides a corpus-independent baseline for keyword extraction.  Future
versions may reconcile YAKE keywords with reef/island signals for richer
context-aware extraction.
"""

from __future__ import annotations

import math
import re

from ._stop_words import STOP_WORDS

_WORD_RE = re.compile(r"[a-z]{3,}")

# Threshold constant: max_yake = YAKE_K / sqrt(n_tokens).
# YAKE scores scale roughly with 1/sqrt(n), so this gives a length-adaptive
# quality cutoff.  Empirically tuned: 1.5 works well across chapter-sized
# sections down to ~50-token paragraphs.
YAKE_K = 1.5

# Additional low-signal words beyond lagoon's STOP_WORDS.
# These are common in documents but carry no topical signal.
_EXTRA_STOPS: frozenset[str] = frozenset({
    # verbs / auxiliaries not in lagoon STOP_WORDS
    "get", "got", "gets", "getting",
    "let", "make", "made", "makes", "making",
    "go", "goes", "went", "gone", "going",
    "come", "came", "comes", "coming",
    "take", "took", "taken", "takes", "taking",
    "give", "gave", "given", "gives", "giving",
    "say", "said", "says", "saying",
    "see", "saw", "seen", "sees", "seeing",
    "put", "set", "run", "ran",
    "know", "knew", "known", "knows",
    "think", "thought", "thinks",
    "use", "used", "uses", "using",
    "find", "found", "finds",
    "tell", "told", "tells",
    "ask", "asked", "asks",
    "seem", "seemed", "seems",
    "feel", "felt", "feels",
    "try", "tried", "tries",
    "leave", "left", "leaves",
    "call", "called", "calls",
    "keep", "kept", "keeps",
    "begin", "began", "begun", "begins",
    "show", "showed", "shown", "shows",
    "hear", "heard", "hears",
    "play", "played", "plays",
    "move", "moved", "moves",
    "live", "lived", "lives",
    "believe", "believed",
    "bring", "brought",
    "happen", "happened", "happens",
    "write", "wrote", "written", "writes",
    "provide", "provided", "provides",
    "sit", "sat", "stand", "stood",
    "lose", "lost",
    "pay", "paid",
    "meet", "met",
    "include", "included", "includes", "including",
    "continue", "continued", "continues",
    "become", "became", "becomes",
    "remain", "remained", "remains",
    "looked", "look", "looks", "looking",
    "need", "needed", "needs",
    # negation fragments
    "dont", "didnt", "wont", "wouldnt", "cant", "couldnt",
    "shouldnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "doesnt", "mustnt",
    # adverbs / adjectives / filler
    "very", "really", "quite", "rather", "much", "many",
    "new", "old", "good", "great", "big", "small", "large",
    "long", "high", "low", "first", "last", "early", "late", "later",
    "able", "certain", "particular", "different", "various",
    "several", "either", "per", "yet",
    "often", "always", "never", "sometimes", "usually",
    "already", "ever", "perhaps", "probably",
    "however", "therefore", "thus", "hence", "otherwise",
    "like", "also", "although", "though", "whether",
    "because", "since", "while",
    # common document / temporal noise
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million", "billion",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "time", "times", "year", "years", "day", "days", "way", "ways",
    "part", "parts", "number", "numbers", "case", "point",
    "thing", "things", "place", "form", "forms",
    "end", "area", "work", "world", "people", "state",
    "based",
    # wikipedia / reference artifacts
    "isbn", "retrieved", "archived", "accessed", "cite", "ref",
    "references", "edit", "page", "pages", "vol",
    "published", "press", "university", "journal",
})

_ALL_STOPS = STOP_WORDS | _EXTRA_STOPS

# Minimum tokens for keyword extraction to be meaningful.
_MIN_TOKENS = 30


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alpha tokens of length >= 3."""
    return _WORD_RE.findall(text.lower())


def extract_segment_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from a segment of text using YAKE.

    Uses a size-relative threshold (YAKE_K / sqrt(n_tokens)) to filter noise,
    so most segments will get fewer keywords than max_keywords.

    Args:
        text: The segment text to extract keywords from.
        max_keywords: Upper bound on returned keywords.

    Returns:
        List of lowercase keyword strings, ordered by relevance (best first).
    """
    tokens = _tokenize(text)
    n_tokens = len(tokens)

    if n_tokens < _MIN_TOKENS:
        return []

    import yake

    # Extract more candidates than needed for headroom after filtering.
    kw = yake.KeywordExtractor(n=1, top=max_keywords * 3, windowSize=2)
    raw = kw.extract_keywords(text)

    # Size-relative threshold
    max_yake = YAKE_K / math.sqrt(n_tokens)

    candidates: list[str] = []
    seen: set[str] = set()
    for term, yake_score in raw:
        t = term.lower()
        if (
            len(t) < 3
            or t in _ALL_STOPS
            or not t.isalpha()
            or t in seen
        ):
            continue
        seen.add(t)
        if yake_score <= max_yake:
            candidates.append(t)

    return candidates[:max_keywords]
