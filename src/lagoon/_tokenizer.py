"""Compound scan (Aho-Corasick) and tokenize/normalize pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import Stemmer

from ._hash import fnv1a_u64

if TYPE_CHECKING:
    import ahocorasick

    from ._types import WordInfo

_WORD_RE = re.compile(r"[a-z]+")


class Tokenizer:
    __slots__ = (
        "_word_lookup", "_compound_ac", "_compound_word_ids",
        "_compound_strings", "_stemmer",
    )

    def __init__(
        self,
        word_lookup: dict[int, WordInfo],
        compound_ac: ahocorasick.Automaton,
        compound_word_ids: list[int],
        compound_strings: list[str],
    ) -> None:
        self._word_lookup = word_lookup
        self._compound_ac = compound_ac
        self._compound_word_ids = compound_word_ids
        self._compound_strings = compound_strings
        self._stemmer = Stemmer.Stemmer("english")

    def scan_compounds(
        self, text_lower: str
    ) -> tuple[set[int], list[tuple[int, int]]]:
        """Phase 1: Aho-Corasick compound scan.

        Returns (matched_word_ids, consumed_spans) with leftmost-longest
        non-overlapping matches.
        """
        # Collect all matches
        raw_matches: list[tuple[int, int, int]] = []  # (start, end, idx)
        for end_inclusive, idx in self._compound_ac.iter(text_lower):
            end = end_inclusive + 1
            start = end - len(self._compound_strings[idx])
            raw_matches.append((start, end, idx))

        # Sort by start position, then by length descending (longest first)
        raw_matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))

        # Greedy leftmost-longest non-overlapping selection
        matched_word_ids: set[int] = set()
        consumed: list[tuple[int, int]] = []
        last_end = -1
        for start, end, idx in raw_matches:
            if start >= last_end:
                matched_word_ids.add(self._compound_word_ids[idx])
                consumed.append((start, end))
                last_end = end

        return matched_word_ids, consumed

    def tokenize(
        self, text: str, consumed_spans: list[tuple[int, int]]
    ) -> tuple[set[int], list[str]]:
        """Phase 2: Tokenize unconsumed text, hash-lookup, stem fallback.

        Returns (matched_word_ids, unknown_words).
        """
        text_lower = text.lower()
        matched: set[int] = set()
        unknown: list[str] = []

        for m in _WORD_RE.finditer(text_lower):
            tok_start, tok_end = m.start(), m.end()

            # Skip tokens inside consumed compound spans
            in_compound = False
            for cs, ce in consumed_spans:
                if tok_start >= cs and tok_end <= ce:
                    in_compound = True
                    break
            if in_compound:
                continue

            token = m.group()
            h = fnv1a_u64(token)
            info = self._word_lookup.get(h)
            if info is not None:
                matched.add(info.word_id)
            else:
                stem = self._stemmer.stemWord(token)
                if stem != token:
                    sh = fnv1a_u64(stem)
                    info = self._word_lookup.get(sh)
                    if info is not None:
                        matched.add(info.word_id)
                        continue
                unknown.append(token)

        return matched, unknown

    def process(self, text: str) -> tuple[set[int], list[str]]:
        """Run full tokenization pipeline (Phase 1 + Phase 2).

        Returns (all_word_ids, unknown_words).
        """
        text_lower = text.lower()
        compound_ids, consumed = self.scan_compounds(text_lower)
        token_ids, unknown = self.tokenize(text, consumed)
        return compound_ids | token_ids, unknown
