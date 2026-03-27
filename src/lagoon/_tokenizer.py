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
        "_compound_strings", "_stemmer", "_equivalences",
    )

    def __init__(
        self,
        word_lookup: dict[int, WordInfo],
        compound_ac: ahocorasick.Automaton,
        compound_word_ids: list[int],
        compound_strings: list[str],
        equivalences: dict[int, list[int]] | None = None,
    ) -> None:
        self._word_lookup = word_lookup
        self._compound_ac = compound_ac
        self._compound_word_ids = compound_word_ids
        self._compound_strings = compound_strings
        self._stemmer = Stemmer.Stemmer("english")
        self._equivalences = equivalences or {}

    def scan_compounds(
        self, text_lower: str
    ) -> tuple[set[int], list[tuple[int, int]]]:
        """Phase 1: Aho-Corasick compound scan.

        Returns (matched_word_ids, consumed_spans) with leftmost-longest
        non-overlapping matches.
        """
        # Collect all matches
        raw_matches: list[tuple[int, int, int]] = []  # (start, end, idx)
        if not self._compound_strings:
            return set(), []
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
                # Equivalence fallback: variant_hash → word_id(s)
                equiv_wids = self._equivalences.get(h)
                if equiv_wids is not None:
                    matched.update(equiv_wids)
                else:
                    unknown.append(token)

        return matched, unknown

    def _scan_compounds_with_spans(
        self, text_lower: str
    ) -> list[tuple[int, int, int]]:
        """Compound scan returning (start, end, word_id) triples, sorted by start."""
        raw_matches: list[tuple[int, int, int]] = []
        if not self._compound_strings:
            return []
        for end_inclusive, idx in self._compound_ac.iter(text_lower):
            end = end_inclusive + 1
            start = end - len(self._compound_strings[idx])
            raw_matches.append((start, end, idx))

        raw_matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))

        # Greedy leftmost-longest non-overlapping
        result: list[tuple[int, int, int]] = []
        last_end = -1
        for start, end, idx in raw_matches:
            if start >= last_end:
                result.append((start, end, self._compound_word_ids[idx]))
                last_end = end
        return result

    def process_ordered(
        self, text: str
    ) -> tuple[list[int | None], set[int], list[str], int]:
        """Tokenize with order preservation.

        Returns:
            word_ids_ordered: word_id per position (None = stop/unknown),
                              compounds = one entry at their text position
            word_ids_set: deduplicated set of matched word_ids
            unknown_words: unrecognized tokens
            total_tokens: count of all tokens seen (including stop/unknown)
        """
        text_lower = text.lower()
        compound_spans = self._scan_compounds_with_spans(text_lower)

        ordered: list[int | None] = []
        matched_set: set[int] = set()
        unknown: list[str] = []
        total_tokens = 0

        # Track which compound spans have been emitted
        emitted_compounds: set[int] = set()  # index into compound_spans

        for m in _WORD_RE.finditer(text_lower):
            tok_start, tok_end = m.start(), m.end()

            # Check if this token is inside a compound span
            compound_match = -1
            for ci, (cs, ce, cid) in enumerate(compound_spans):
                if tok_start >= cs and tok_end <= ce:
                    compound_match = ci
                    break

            if compound_match >= 0:
                # Inside a compound — emit once at first constituent token
                if compound_match not in emitted_compounds:
                    emitted_compounds.add(compound_match)
                    cid = compound_spans[compound_match][2]
                    ordered.append(cid)
                    matched_set.add(cid)
                    total_tokens += 1
                continue

            total_tokens += 1
            token = m.group()
            h = fnv1a_u64(token)
            info = self._word_lookup.get(h)
            if info is not None:
                ordered.append(info.word_id)
                matched_set.add(info.word_id)
            else:
                stem = self._stemmer.stemWord(token)
                if stem != token:
                    sh = fnv1a_u64(stem)
                    info = self._word_lookup.get(sh)
                    if info is not None:
                        ordered.append(info.word_id)
                        matched_set.add(info.word_id)
                        continue
                # Equivalence fallback: variant_hash → word_id(s)
                equiv_wids = self._equivalences.get(h)
                if equiv_wids is not None:
                    # Use first word_id for ordered position, add all to set
                    ordered.append(equiv_wids[0])
                    matched_set.update(equiv_wids)
                else:
                    unknown.append(token)
                    ordered.append(None)

        return ordered, matched_set, unknown, total_tokens

    def process(self, text: str) -> tuple[set[int], list[str]]:
        """Run full tokenization pipeline (Phase 1 + Phase 2).

        Returns (all_word_ids, unknown_words).
        """
        text_lower = text.lower()
        compound_ids, consumed = self.scan_compounds(text_lower)
        token_ids, unknown = self.tokenize(text, consumed)
        return compound_ids | token_ids, unknown
