"""ReefScorer: 5-phase scoring engine with vocabulary extension support."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ._document import analyze_document
from ._stop_words import STOP_WORDS
from ._tokenizer import Tokenizer
from ._types import (
    DocumentAnalysis,
    ScoredIsland,
    ScoredReef,
    TopicResult,
    WordInfo,
)

if TYPE_CHECKING:
    import ahocorasick

    from ._types import IslandMeta, ReefMeta


class ReefScorer:
    """Main scoring engine. Holds all loaded data and exposes the public API."""

    __slots__ = (
        "_word_lookup", "_word_reefs", "_reef_meta", "_island_meta",
        "_bg_mean", "_bg_std", "_tokenizer", "_n_reefs",
        "_reef_total_dims", "_reef_n_words", "_avg_reef_words",
    )

    def __init__(
        self,
        word_lookup: dict[int, WordInfo],
        word_reefs: list[list[tuple[int, int]]],
        reef_meta: list[ReefMeta],
        island_meta: list[IslandMeta],
        bg_mean: list[float],
        bg_std: list[float],
        compound_ac: ahocorasick.Automaton,
        compound_word_ids: list[int],
        compound_strings: list[str],
        constants: dict,
    ) -> None:
        self._word_lookup = word_lookup
        self._word_reefs = word_reefs
        self._reef_meta = reef_meta
        self._island_meta = island_meta
        self._bg_mean = bg_mean
        self._bg_std = bg_std
        self._n_reefs = len(reef_meta)
        self._reef_total_dims = constants["reef_total_dims"]
        self._reef_n_words = constants["reef_n_words"]
        self._avg_reef_words = constants["avg_reef_words"]

        self._tokenizer = Tokenizer(
            word_lookup, compound_ac, compound_word_ids, compound_strings,
        )

    # -- Public scoring API --

    def score(self, text: str, top_k: int = 10) -> TopicResult:
        """Score a single text string through the full 5-phase pipeline."""
        word_ids, unknown = self._tokenizer.process(text)
        return self._score_from_ids(word_ids, unknown, top_k)

    def score_batch(
        self, texts: list[str], top_k: int = 10
    ) -> list[TopicResult]:
        """Score multiple texts."""
        return [self.score(t, top_k) for t in texts]

    def lookup_word(self, word: str) -> WordInfo | None:
        """Look up a single word with the same normalization as scoring."""
        from ._hash import fnv1a_u64

        import Stemmer

        token = word.lower()
        h = fnv1a_u64(token)
        info = self._word_lookup.get(h)
        if info is not None:
            return info
        stemmer = Stemmer.Stemmer("english")
        stem = stemmer.stemWord(token)
        if stem != token:
            sh = fnv1a_u64(stem)
            info = self._word_lookup.get(sh)
            if info is not None:
                return info
        return None

    def score_raw(self, text: str) -> list[float]:
        """Score text and return the full 207-element z-score vector.

        Used internally by document analysis.
        """
        word_ids, _ = self._tokenizer.process(text)
        if not word_ids:
            return [0.0] * self._n_reefs
        scores_q, _ = self._accumulate_bm25(word_ids)
        raw = [sq / 8192.0 for sq in scores_q]
        return self._subtract_background(raw)

    def analyze(
        self,
        text: str | list[str],
        *,
        sensitivity: float = 1.0,
        smooth_window: int = 2,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 30,
    ) -> DocumentAnalysis:
        """Document-level topic segmentation with chunk size constraints.

        Args:
            text: Raw text or pre-split list of sentences.
            sensitivity: Boundary detection threshold. Lower = more boundaries.
            smooth_window: Sliding window size for z-score vector smoothing.
            min_chunk_sentences: Minimum sentences per chunk (merge small segments).
            max_chunk_sentences: Maximum sentences per chunk (split large segments).
                Set to 0 to disable maximum size enforcement.
        """
        return analyze_document(
            self, text,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
        )

    # -- Vocabulary extension API --

    @property
    def next_word_id(self) -> int:
        """Return the next available word_id for custom words."""
        return len(self._word_reefs)

    def filter_unknown(self, words: list[str]) -> list[str]:
        """Return words not in the dictionary and not stop words.

        Applies the same normalization as scoring: lowercase, hash lookup,
        Snowball stemmer fallback. Stop words are excluded even if they
        are not in the dictionary.
        """
        from ._hash import fnv1a_u64

        import Stemmer

        stemmer = Stemmer.Stemmer("english")
        result: list[str] = []

        for word in words:
            token = word.lower()
            if token in STOP_WORDS:
                continue
            h = fnv1a_u64(token)
            if self._word_lookup.get(h) is not None:
                continue
            stem = stemmer.stemWord(token)
            if stem != token:
                sh = fnv1a_u64(stem)
                if self._word_lookup.get(sh) is not None:
                    continue
            result.append(word)

        return result

    def compute_custom_word_scores(
        self,
        n_associated_reefs: int,
        associations: list[tuple[int, float]],
    ) -> tuple[int, list[tuple[int, int]]]:
        """Compute IDF and BM25 scores for a custom word from reef associations.

        Uses lagoon's BM25 formula (k1=1.2, b=0.75) with the association
        strength as a synthetic tf proxy. Produces quantized scores compatible
        with the base vocabulary (u8 IDF scale 51, u16 BM25 scale 8192).

        Args:
            n_associated_reefs: Number of reefs the word is associated with.
                Used for IDF computation.
            associations: List of (reef_id, strength) pairs where strength
                is 0.0-1.0 normalized association strength.

        Returns:
            (idf_q, reef_scores) where idf_q is u8-quantized IDF and
            reef_scores is a list of (reef_id, bm25_q) pairs.

        Raises:
            ValueError: If reef_id is out of range or strength is invalid.
        """
        n_total = self._n_reefs  # 207

        if n_associated_reefs < 1:
            raise ValueError("n_associated_reefs must be >= 1")
        if n_associated_reefs > n_total:
            raise ValueError(
                f"n_associated_reefs {n_associated_reefs} exceeds "
                f"total reefs ({n_total})"
            )

        # IDF using lagoon's formula: ln((N - n + 0.5) / (n + 0.5) + 1)
        idf = math.log(
            (n_total - n_associated_reefs + 0.5)
            / (n_associated_reefs + 0.5)
            + 1
        )
        idf_q = max(0, min(255, round(idf * 51)))

        # BM25 parameters (same as lagoon's base vocabulary)
        k1 = 1.2
        b = 0.75
        avg_reef_words = self._avg_reef_words

        reef_scores: list[tuple[int, int]] = []
        for reef_id, strength in associations:
            if reef_id < 0 or reef_id >= n_total:
                raise ValueError(
                    f"reef_id {reef_id} out of range [0, {n_total - 1}]"
                )
            if not (0.0 <= strength <= 1.0):
                raise ValueError(
                    f"strength must be in [0.0, 1.0], got {strength}"
                )

            tf_proxy = strength
            reef_n_words = self._reef_n_words[reef_id]

            numerator = idf * (tf_proxy * (k1 + 1))
            denominator = tf_proxy + k1 * (
                1 - b + b * reef_n_words / avg_reef_words
            )
            bm25 = numerator / denominator if denominator > 0 else 0.0
            bm25_q = max(0, min(65535, round(bm25 * 8192)))

            reef_scores.append((reef_id, bm25_q))

        return idf_q, reef_scores

    def add_custom_word(
        self,
        word: str,
        reef_associations: list[tuple[int, float]],
        *,
        specificity: int = 2,
    ) -> WordInfo:
        """Add a custom word to the vocabulary.

        Computes IDF and BM25 scores from reef associations, validates all
        fields, and injects the word into the scorer's vocabulary. The word
        becomes immediately available for scoring.

        Args:
            word: The word to add (will be lowercased and whitespace-normalized).
                For compounds, use space-separated tokens (e.g., "machine learning").
            reef_associations: List of (reef_id, strength) pairs where strength
                is 0.0-1.0 normalized association strength.
            specificity: Sigma band (-2 to +2). Defaults to 2 (highly specific).

        Returns:
            The WordInfo that was injected.

        Raises:
            ValueError: If the word already exists, fields are out of range,
                or reef_associations is empty.
        """
        from ._hash import fnv1a_u64

        # Normalize
        normalized = " ".join(word.lower().split())
        if not normalized:
            raise ValueError("word must not be empty")

        # Hash
        word_hash = fnv1a_u64(normalized)

        # Check for duplicates
        if word_hash in self._word_lookup:
            raise ValueError(f"word '{normalized}' already exists in vocabulary")

        # Validate specificity
        if specificity not in (-2, -1, 0, 1, 2):
            raise ValueError(
                f"specificity must be in [-2, -1, 0, 1, 2], got {specificity}"
            )

        # Validate reef associations
        if not reef_associations:
            raise ValueError("reef_associations must not be empty")

        for reef_id, strength in reef_associations:
            if reef_id < 0 or reef_id >= self._n_reefs:
                raise ValueError(
                    f"reef_id {reef_id} out of range [0, {self._n_reefs - 1}]"
                )
            if not (0.0 <= strength <= 1.0):
                raise ValueError(
                    f"strength must be in [0.0, 1.0], got {strength}"
                )

        # Compute scores (lagoon owns the BM25 formula)
        n_associated = len(reef_associations)
        idf_q, reef_scores = self.compute_custom_word_scores(
            n_associated, reef_associations,
        )

        # Allocate word_id
        word_id = self.next_word_id

        # Create WordInfo
        info = WordInfo(
            word_hash=word_hash,
            word_id=word_id,
            specificity=specificity,
            idf_q=idf_q,
        )

        # Inject into scorer
        self._word_lookup[word_hash] = info
        while len(self._word_reefs) <= word_id:
            self._word_reefs.append([])
        self._word_reefs[word_id] = reef_scores

        return info

    def rebuild_compounds(
        self, additional_compounds: list[tuple[str, int]]
    ) -> None:
        """Add custom compounds and rebuild the Aho-Corasick automaton.

        Merges the additional compounds with the existing base compounds
        and rebuilds the automaton. The scorer immediately begins matching
        the new compounds during scoring.

        Args:
            additional_compounds: List of (compound_string, word_id) pairs.
                Compound strings should be lowercase and space-separated
                (e.g., "machine learning").
        """
        import ahocorasick

        # Collect existing base compounds
        base = list(zip(
            self._tokenizer._compound_strings,
            self._tokenizer._compound_word_ids,
        ))
        all_compounds = base + additional_compounds

        # Build new automaton
        ac = ahocorasick.Automaton()
        compound_word_ids: list[int] = []
        compound_strings: list[str] = []
        for idx, (compound_str, word_id) in enumerate(all_compounds):
            ac.add_word(compound_str, idx)
            compound_word_ids.append(word_id)
            compound_strings.append(compound_str)
        ac.make_automaton()

        # Replace on tokenizer
        self._tokenizer._compound_ac = ac
        self._tokenizer._compound_word_ids = compound_word_ids
        self._tokenizer._compound_strings = compound_strings

    # -- Internal methods --

    def _score_full(
        self, text: str, top_k: int = 10
    ) -> tuple[list[float], TopicResult]:
        """Score and return both the z-score vector and TopicResult.

        Used by analyze() to avoid double tokenization when both the
        z-score vector (for boundary detection) and the TopicResult
        (for per-sentence results) are needed.
        """
        word_ids, unknown = self._tokenizer.process(text)

        if not word_ids:
            z = [0.0] * self._n_reefs
            filtered = [w for w in unknown if w not in STOP_WORDS]
            total = len(unknown)
            tr = TopicResult(
                top_reefs=[],
                top_islands=[],
                arch_scores=[0.0] * 4,
                confidence=0.0,
                coverage=0.0 if total == 0 else 0.0,
                matched_words=0,
                unknown_words=filtered,
                matched_word_ids=frozenset(),
            )
            return z, tr

        scores_q, word_counts = self._accumulate_bm25(word_ids)
        raw = [sq / 8192.0 for sq in scores_q]
        z = self._subtract_background(raw)
        tr = self._extract_results(z, raw, word_counts, word_ids, unknown, top_k)
        return z, tr

    def _score_from_ids(
        self, word_ids: set[int], unknown: list[str], top_k: int
    ) -> TopicResult:
        """Phases 3-5: accumulate, subtract background, extract results."""
        n = self._n_reefs

        if not word_ids:
            total_words = len(unknown)
            filtered = [w for w in unknown if w not in STOP_WORDS]
            return TopicResult(
                top_reefs=[],
                top_islands=[],
                arch_scores=[0.0] * 4,
                confidence=0.0,
                coverage=0.0 if total_words == 0 else 0.0,
                matched_words=0,
                unknown_words=filtered,
                matched_word_ids=frozenset(),
            )

        # Phase 3: BM25 accumulation (quantized integer accumulation)
        scores_q, word_counts = self._accumulate_bm25(word_ids)

        # Dequantize once
        raw_scores = [sq / 8192.0 for sq in scores_q]

        # Phase 4: Background subtraction
        z_scores = self._subtract_background(raw_scores)

        # Phase 5: Result extraction
        return self._extract_results(
            z_scores, raw_scores, word_counts,
            word_ids, unknown, top_k,
        )

    def _accumulate_bm25(
        self, word_ids: set[int]
    ) -> tuple[list[int], list[int]]:
        """Phase 3: Accumulate quantized BM25 scores."""
        n = self._n_reefs
        scores_q = [0] * n
        word_counts = [0] * n
        word_reefs = self._word_reefs

        for word_id in word_ids:
            for reef_id, bm25_q in word_reefs[word_id]:
                scores_q[reef_id] += bm25_q
                word_counts[reef_id] += 1

        return scores_q, word_counts

    def _subtract_background(self, raw_scores: list[float]) -> list[float]:
        """Phase 4: Convert raw BM25 to z-scores."""
        n = self._n_reefs
        bg_mean = self._bg_mean
        bg_std = self._bg_std
        z_scores = [0.0] * n

        for i in range(n):
            std = bg_std[i]
            if std > 0.0:
                z_scores[i] = (raw_scores[i] - bg_mean[i]) / std
            else:
                z_scores[i] = 0.0

        return z_scores

    def _extract_results(
        self,
        z_scores: list[float],
        raw_scores: list[float],
        word_counts: list[int],
        matched: set[int],
        unknown: list[str],
        top_k: int,
    ) -> TopicResult:
        """Phase 5: Extract top-K reefs, islands, archipelago rollup."""
        # Sort all 207 by z-score descending
        indexed = sorted(
            range(self._n_reefs), key=lambda i: z_scores[i], reverse=True,
        )

        reef_meta = self._reef_meta
        top_reefs = [
            ScoredReef(
                reef_id=i,
                z_score=z_scores[i],
                raw_bm25=raw_scores[i],
                n_contributing_words=word_counts[i],
                name=reef_meta[i].name,
            )
            for i in indexed[:top_k]
        ]

        # Confidence: gap between #1 and #2
        confidence = 0.0
        if len(indexed) >= 2:
            confidence = z_scores[indexed[0]] - z_scores[indexed[1]]

        # Coverage
        total_words = len(matched) + len(unknown)
        coverage = len(matched) / total_words if total_words > 0 else 0.0

        # Filter stop words from unknown list
        filtered_unknown = [w for w in unknown if w not in STOP_WORDS]

        # Island rollup from ALL reefs (not just top-K)
        island_agg: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
        for i in indexed[:top_k]:
            iid = reef_meta[i].island_id
            island_agg[iid][0] += z_scores[i]
            island_agg[iid][1] += 1

        island_meta = self._island_meta
        top_islands = sorted(
            [
                ScoredIsland(
                    island_id=iid,
                    aggregate_z=agg[0],  # type: ignore[arg-type]
                    n_contributing_reefs=int(agg[1]),
                    name=island_meta[iid].name,
                )
                for iid, agg in island_agg.items()
            ],
            key=lambda x: x.aggregate_z,
            reverse=True,
        )

        # Archipelago rollup
        arch_scores = [0.0] * 4
        for island in top_islands:
            aid = island_meta[island.island_id].arch_id
            arch_scores[aid] += island.aggregate_z

        return TopicResult(
            top_reefs=top_reefs,
            top_islands=top_islands,
            arch_scores=arch_scores,
            confidence=confidence,
            coverage=coverage,
            matched_words=len(matched),
            unknown_words=filtered_unknown,
            matched_word_ids=frozenset(matched),
        )
