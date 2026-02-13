"""ReefScorer: 5-phase scoring engine."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ._document import analyze_document
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
        min_segment_sentences: int = 1,
    ) -> DocumentAnalysis:
        """Document-level topic segmentation."""
        return analyze_document(
            self, text,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_segment_sentences=min_segment_sentences,
        )

    # -- Internal methods --

    def _score_from_ids(
        self, word_ids: set[int], unknown: list[str], top_k: int
    ) -> TopicResult:
        """Phases 3-5: accumulate, subtract background, extract results."""
        n = self._n_reefs

        if not word_ids:
            total_words = len(unknown)
            coverage = 0.0 if total_words == 0 else 0.0
            return TopicResult(
                top_reefs=[],
                top_islands=[],
                arch_scores=[0.0] * 4,
                confidence=0.0,
                coverage=coverage,
                matched_words=0,
                unknown_words=unknown,
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
            unknown_words=unknown,
        )
