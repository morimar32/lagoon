"""ReefScorer: 5-phase scoring engine with vocabulary extension support."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ._document import analyze_document
from ._stop_words import STOP_WORDS
from ._text_result import build_text_result
from ._tokenizer import Tokenizer
from ._types import (
    DocumentAnalysis,
    ScoredIsland,
    ScoredReef,
    SubReefMeta,
    TextResult,
    TopicResult,
    WordInfo,
)

_BG_STD_FLOOR = 0.01  # Minimum credible bg_std; anything below is epsilon noise
_BG_ALPHA_THRESHOLD = 6  # matched words at which full bg subtraction kicks in
_SUB_REEF_SENTINEL = 0xFFFF

# 4-pass contextual scoring constants
_NOISE_SCORE_FLOOR = 48      # ~19th percentile on u8 scale; hits below are pruned
_CONTEXT_RAMP_WORDS = 3      # words before full context influence
_CONTEXT_GAMMA = 0.3         # max boost multiplier for island coherence
_CONTEXT_THRESHOLD = 1.0     # island activation level for "established"
_CONTEXTUAL_MIN_WORDS = 4    # minimum matched words to engage contextual path

if TYPE_CHECKING:
    import ahocorasick

    from ._types import IslandMeta, ReefMeta


class ReefScorer:
    """Main scoring engine. Holds all loaded data and exposes the public API."""

    __slots__ = (
        "_word_lookup", "_word_reefs", "_reef_meta", "_island_meta",
        "_bg_mean", "_bg_std", "_tokenizer", "_n_reefs", "_n_archs",
        "_reef_total_dims", "_reef_n_words", "_avg_reef_words",
        "_reef_edges", "_word_tags", "_weight_scale", "_reef_weight_p75",
        "_word_reef_detail", "_sub_reef_meta", "_domainless_word_ids",
    )

    def __init__(
        self,
        word_lookup: dict[int, WordInfo],
        word_reefs: list[list[tuple[int, int, int]]],
        reef_meta: list[ReefMeta],
        island_meta: list[IslandMeta],
        bg_mean: list[float],
        bg_std: list[float],
        compound_ac: ahocorasick.Automaton,
        compound_word_ids: list[int],
        compound_strings: list[str],
        constants: dict,
        reef_edges: list[tuple[int, int, float]] | None = None,
        word_reef_detail: list[list[tuple[int, int, int]]] | None = None,
        sub_reef_meta: list[SubReefMeta] | None = None,
        domainless_word_ids: frozenset[int] | None = None,
    ) -> None:
        self._word_lookup = word_lookup
        self._word_reefs = word_reefs
        self._reef_meta = reef_meta
        self._island_meta = island_meta
        self._bg_mean = bg_mean
        self._bg_std = bg_std
        self._n_reefs = len(reef_meta)
        self._n_archs = constants["N_ARCHS"]
        self._reef_total_dims = constants["reef_total_dims"]
        self._reef_n_words = constants["reef_n_words"]
        self._avg_reef_words = constants["avg_reef_words"]
        self._weight_scale = float(constants.get("WEIGHT_SCALE", 100.0))
        self._reef_edges = reef_edges if reef_edges else []
        self._word_reef_detail = word_reef_detail if word_reef_detail else []
        self._sub_reef_meta = sub_reef_meta if sub_reef_meta else []
        self._domainless_word_ids = domainless_word_ids or frozenset()
        self._word_tags: dict[int, int] = {}
        self._reef_weight_p75 = self._compute_reef_weight_percentiles(word_reefs)

        self._tokenizer = Tokenizer(
            word_lookup, compound_ac, compound_word_ids, compound_strings,
        )

    # -- Public properties --

    @property
    def reef_word_counts(self) -> list[int]:
        """Number of words per reef (read-only)."""
        return self._reef_n_words

    @property
    def avg_reef_words(self) -> float:
        """Average words per reef (read-only)."""
        return self._avg_reef_words

    @property
    def n_reefs(self) -> int:
        """Total number of reefs (read-only)."""
        return self._n_reefs

    # -- Public scoring API --

    def score(self, text: str, top_k: int = 10, *, min_reef_z: float | None = None) -> TopicResult:
        """Score text. Uses 4-pass contextual path for longer text, simple path for short."""
        ordered, word_ids, unknown, total_tokens = self._tokenizer.process_ordered(text)
        n_effective = len(word_ids - self._domainless_word_ids) if self._domainless_word_ids else len(word_ids)

        if n_effective < _CONTEXTUAL_MIN_WORDS:
            return self._score_from_ids(word_ids, unknown, top_k, min_reef_z)

        return self._score_contextual(
            ordered, word_ids, unknown, total_tokens, n_effective, top_k, min_reef_z,
        )

    def score_batch(
        self, texts: list[str], top_k: int = 10, *, min_reef_z: float | None = None,
    ) -> list[TopicResult]:
        """Score multiple texts."""
        return [self.score(t, top_k, min_reef_z=min_reef_z) for t in texts]

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
        """Score text and return the full z-score vector (one entry per reef).

        Used internally by document analysis.
        """
        ordered, word_ids, unknown, total_tokens = self._tokenizer.process_ordered(text)
        if not word_ids:
            return [0.0] * self._n_reefs

        n_effective = len(word_ids - self._domainless_word_ids) if self._domainless_word_ids else len(word_ids)

        if n_effective >= _CONTEXTUAL_MIN_WORDS:
            tr = build_text_result(
                ordered, unknown, total_tokens,
                self._word_reefs, self._reef_meta, self._domainless_word_ids,
            )
            self._filter_noise(tr)
            raw, _ = self._evaluate_contextual(tr)
            raw = self._propagate(raw)
            return self._subtract_background(raw, n_matched=n_effective)

        scores_q, _ = self._accumulate_weights(word_ids)
        ws = self._weight_scale
        raw = [sq / ws for sq in scores_q]
        raw = self._propagate(raw)
        return self._subtract_background(raw, n_matched=n_effective)

    def analyze(
        self,
        text: str | list[str],
        *,
        sensitivity: float = 1.0,
        smooth_window: int = 2,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 30,
        min_reef_z: float = 2.0,
    ) -> DocumentAnalysis:
        """Document-level topic segmentation with chunk size constraints.

        Args:
            text: Raw text or pre-split list of sentences.
            sensitivity: Boundary detection threshold. Lower = more boundaries.
            smooth_window: Sliding window size for z-score vector smoothing.
            min_chunk_sentences: Minimum sentences per chunk (merge small segments).
            max_chunk_sentences: Maximum sentences per chunk (split large segments).
                Set to 0 to disable maximum size enforcement.
            min_reef_z: Minimum z-score threshold for reef inclusion. Only reefs
                with z >= min_reef_z are returned, replacing fixed top-k.
        """
        return analyze_document(
            self, text,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
            min_reef_z=min_reef_z,
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

    def _compute_reef_weight_percentiles(
        self, word_reefs: list[list[tuple[int, int, int]]],
    ) -> list[int]:
        """Compute 75th-percentile weight_q for each reef from base vocabulary.

        Single pass over all word_reefs entries, collecting weight_q per reef.
        Reefs with zero words get a fallback equal to the global median of
        all per-reef p75 values.

        Returns a list indexed by reef_id.
        """
        n = self._n_reefs
        reef_weights: list[list[int]] = [[] for _ in range(n)]
        for entries in word_reefs:
            for reef_id, weight_q, _sub in entries:
                if reef_id < n:
                    reef_weights[reef_id].append(weight_q)

        p75s: list[int] = []
        populated: list[int] = []
        for rid in range(n):
            ws = reef_weights[rid]
            if ws:
                ws.sort()
                idx = min(len(ws) - 1, (3 * len(ws)) // 4)
                p = ws[idx]
                p75s.append(p)
                populated.append(p)
            else:
                p75s.append(0)  # placeholder, will be replaced by fallback

        # Fallback for empty reefs: median of populated p75s
        if populated:
            populated.sort()
            fallback = populated[len(populated) // 2]
        else:
            fallback = 500  # safe default when no data at all

        for rid in range(n):
            if not reef_weights[rid]:
                p75s[rid] = fallback

        return p75s

    def calc_custom_weight(self, reef_id: int, strength: float) -> int:
        """Compute a per-reef calibrated weight_q for a custom word.

        Returns round(p75[reef_id] * strength), clamped to u8.
        """
        if reef_id < 0 or reef_id >= self._n_reefs:
            raise ValueError(
                f"reef_id {reef_id} out of range [0, {self._n_reefs - 1}]"
            )
        raw = round(self._reef_weight_p75[reef_id] * strength)
        return max(0, min(255, raw))

    def calc_custom_idf(self, n_associated_reefs: int) -> int:
        """Compute quantized IDF (u8) for a custom word.

        Uses lagoon's IDF formula: ln((N - n + 0.5) / (n + 0.5) + 1) * 51.
        """
        n_total = self._n_reefs
        if n_associated_reefs < 1:
            raise ValueError("n_associated_reefs must be >= 1")
        if n_associated_reefs > n_total:
            raise ValueError(
                f"n_associated_reefs {n_associated_reefs} exceeds "
                f"total reefs ({n_total})"
            )
        idf = math.log(
            (n_total - n_associated_reefs + 0.5)
            / (n_associated_reefs + 0.5)
            + 1
        )
        return max(0, min(255, round(idf * 51)))

    def add_custom_word(
        self,
        word: str,
        reef_weights: list[tuple[int, int]],
        *,
        idf_q: int,
        specificity: int = 2,
        tag: int = 0,
    ) -> WordInfo:
        """Add a custom word to the vocabulary with pre-computed weights.

        Accepts pre-computed weight_q values and IDF. Use calc_custom_weight()
        and calc_custom_idf() to compute these before calling.

        Args:
            word: The word to add (will be lowercased and whitespace-normalized).
            reef_weights: List of (reef_id, weight_q) pairs with pre-computed weights.
            idf_q: Pre-computed quantized IDF (u8, 0-255).
            specificity: Sigma band (-2 to +2). Defaults to 2.
            tag: Custom tag for tracking. Defaults to 0.

        Returns:
            The WordInfo that was injected.

        Raises:
            ValueError: If the word already exists, fields are out of range,
                or reef_weights is empty.
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

        # Validate reef_weights
        if not reef_weights:
            raise ValueError("reef_weights must not be empty")

        # Validate idf_q
        if not (0 <= idf_q <= 255):
            raise ValueError(f"idf_q must be in [0, 255], got {idf_q}")

        for reef_id, weight_q in reef_weights:
            if reef_id < 0 or reef_id >= self._n_reefs:
                raise ValueError(
                    f"reef_id {reef_id} out of range [0, {self._n_reefs - 1}]"
                )
            if not (0 <= weight_q <= 255):
                raise ValueError(
                    f"weight_q must be in [0, 255], got {weight_q}"
                )

        # Allocate word_id
        word_id = self.next_word_id

        # Create WordInfo
        info = WordInfo(
            word_hash=word_hash,
            word_id=word_id,
            specificity=specificity,
            idf_q=idf_q,
            tag=tag,
        )
        if tag != 0:
            self._word_tags[word_id] = tag

        # Inject into scorer
        self._word_lookup[word_hash] = info
        while len(self._word_reefs) <= word_id:
            self._word_reefs.append([])
        # Store as 3-element tuples with sentinel sub_reef_id
        self._word_reefs[word_id] = [
            (rid, wq, _SUB_REEF_SENTINEL) for rid, wq in reef_weights
        ]

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

    def get_word_tags(self, word_ids: frozenset[int] | set[int]) -> dict[int, int]:
        """Return non-zero tags for the given word_ids.

        Only entries whose tag != 0 are included. Base vocabulary words
        (tag 0) are omitted, so callers can distinguish custom-injected
        words from base words by membership alone.
        """
        tags = self._word_tags
        return {wid: tags[wid] for wid in word_ids if wid in tags}

    # -- Internal methods --

    def _score_full(
        self, text: str, top_k: int = 10,
        min_reef_z: float | None = None,
    ) -> tuple[list[float], TopicResult]:
        """Score and return both the z-score vector and TopicResult.

        Used by analyze() to avoid double tokenization when both the
        z-score vector (for boundary detection) and the TopicResult
        (for per-sentence results) are needed.
        """
        ordered, word_ids, unknown, total_tokens = self._tokenizer.process_ordered(text)

        if not word_ids:
            z = [0.0] * self._n_reefs
            filtered = [w for w in unknown if w not in STOP_WORDS]
            total = len(unknown)
            tr = TopicResult(
                top_reefs=[],
                top_islands=[],
                arch_scores=[0.0] * self._n_archs,
                confidence=0.0,
                coverage=0.0 if total == 0 else 0.0,
                matched_words=0,
                unknown_words=filtered,
                matched_word_ids=frozenset(),
                valence_signal=0.0,
            )
            return z, tr

        n_effective = len(word_ids - self._domainless_word_ids) if self._domainless_word_ids else len(word_ids)

        if n_effective >= _CONTEXTUAL_MIN_WORDS:
            text_result = build_text_result(
                ordered, unknown, total_tokens,
                self._word_reefs, self._reef_meta, self._domainless_word_ids,
            )
            self._filter_noise(text_result)
            raw, word_counts = self._evaluate_contextual(text_result)
            raw = self._propagate(raw)
            z = self._subtract_background(raw, n_matched=n_effective)
        else:
            scores_q, word_counts = self._accumulate_weights(word_ids)
            ws = self._weight_scale
            raw = [sq / ws for sq in scores_q]
            raw = self._propagate(raw)
            z = self._subtract_background(raw, n_matched=n_effective)

        tr = self._extract_results(z, raw, word_counts, word_ids, unknown, top_k, min_reef_z)
        return z, tr

    def _score_from_ids(
        self, word_ids: set[int], unknown: list[str], top_k: int,
        min_reef_z: float | None = None,
    ) -> TopicResult:
        """Phases 3-5: accumulate, subtract background, extract results."""
        n = self._n_reefs

        if not word_ids:
            total_words = len(unknown)
            filtered = [w for w in unknown if w not in STOP_WORDS]
            return TopicResult(
                top_reefs=[],
                top_islands=[],
                arch_scores=[0.0] * self._n_archs,
                confidence=0.0,
                coverage=0.0 if total_words == 0 else 0.0,
                matched_words=0,
                unknown_words=filtered,
                matched_word_ids=frozenset(),
                valence_signal=0.0,
            )

        # Phase 3: Weight accumulation (quantized integer accumulation)
        scores_q, word_counts = self._accumulate_weights(word_ids)

        # Dequantize once
        ws = self._weight_scale
        raw_scores = [sq / ws for sq in scores_q]

        # Propagate raw scores through reef edges first, then normalize.
        # Propagation on raw scores (always >= 0) spreads only positive
        # signal, which preserves discriminative power for custom words.
        raw_scores = self._propagate(raw_scores)

        # Phase 4: Background subtraction (scaled by effective matched words)
        n_effective = len(word_ids)
        if self._domainless_word_ids:
            n_effective = len(word_ids - self._domainless_word_ids)
        z_scores = self._subtract_background(raw_scores, n_matched=n_effective)

        # Phase 5: Result extraction
        return self._extract_results(
            z_scores, raw_scores, word_counts,
            word_ids, unknown, top_k, min_reef_z,
        )

    def _score_contextual(
        self,
        ordered: list[int | None],
        word_ids: set[int],
        unknown: list[str],
        total_tokens: int,
        n_effective: int,
        top_k: int,
        min_reef_z: float | None,
    ) -> TopicResult:
        """4-pass contextual scoring for longer texts."""
        # Build TextResult structure
        tr = build_text_result(
            ordered, unknown, total_tokens,
            self._word_reefs, self._reef_meta, self._domainless_word_ids,
        )

        # Pass 3: Noise filtering (in-place)
        self._filter_noise(tr)

        # Pass 4: Contextual evaluation
        raw_scores, word_counts = self._evaluate_contextual(tr)

        # Propagate through reef edges
        raw_scores = self._propagate(raw_scores)

        # Background subtraction
        z_scores = self._subtract_background(raw_scores, n_matched=n_effective)

        # Result extraction
        return self._extract_results(
            z_scores, raw_scores, word_counts,
            word_ids, unknown, top_k, min_reef_z,
        )

    def _filter_noise(self, tr: TextResult) -> None:
        """In-place noise filtering. Zeroes out low-confidence reef hits.

        Two filters:
        1. Score floor: hits below NOISE_SCORE_FLOOR are zeroed out
        2. Island corroboration: for texts with 8+ matched words, islands
           with only 1 contributing word get their hits dampened (halved)
        """
        if tr.matched_words < _CONTEXTUAL_MIN_WORDS:
            return

        # Filter 1: Score floor — zero out weak hits
        for hit in tr.reef_hits:
            if hit.score > 0 and hit.score < _NOISE_SCORE_FLOOR:
                hit.score = 0

        # Filter 2: Island corroboration — dampen single-word islands
        if tr.matched_words >= 6:
            for island in tr.islands:
                if island.word_count == 1:
                    for hit_idx in island.reef_hit_indices:
                        hit = tr.reef_hits[hit_idx]
                        if hit.score > 0:
                            hit.score = hit.score // 2

    def _evaluate_contextual(
        self, tr: TextResult,
    ) -> tuple[list[float], list[int]]:
        """Walk word_order sequentially with island-coherence boosting.

        Returns (raw_scores[n_reefs], word_counts[n_reefs]).
        """
        n = self._n_reefs
        raw_scores = [0.0] * n
        word_counts = [0] * n
        ws = self._weight_scale
        n_islands = len(tr.islands)
        island_activation = [0.0] * n_islands

        words_processed = 0

        for head_idx in tr.word_order:
            if head_idx == -1:
                continue

            # Context ramp: no boost for first word, full boost after N words
            context_alpha = min(1.0, words_processed / _CONTEXT_RAMP_WORDS) if _CONTEXT_RAMP_WORDS > 0 else 1.0

            # Accumulate scores from this word's reef chain
            word_contributions: list[tuple[int, float]] = []  # (island_idx, contribution)
            reefs_hit: set[int] = set()

            hit_idx = head_idx
            while hit_idx != -1:
                hit = tr.reef_hits[hit_idx]
                if hit.score > 0:
                    base = hit.score / ws

                    # Boost if this reef's island is already active
                    boost = 0.0
                    if hit.island_idx >= 0:
                        activation = island_activation[hit.island_idx]
                        boost = context_alpha * _CONTEXT_GAMMA * min(1.0, activation / _CONTEXT_THRESHOLD)

                    contribution = base * (1.0 + boost)
                    raw_scores[hit.reef_id] += contribution

                    if hit.reef_id not in reefs_hit:
                        word_counts[hit.reef_id] += 1
                        reefs_hit.add(hit.reef_id)

                    if hit.island_idx >= 0:
                        word_contributions.append((hit.island_idx, contribution))

                hit_idx = hit.next_idx

            # Update island activations from this word's contributions
            for island_idx, contrib in word_contributions:
                island_activation[island_idx] += contrib

            words_processed += 1

        # Reef-level corroboration: for longer texts, penalize reefs where
        # only a small fraction of matched words contribute.  This suppresses
        # noise reefs that get signal from a few generic words (e.g. "provide",
        # "main") while leaving well-corroborated reefs untouched.
        if tr.matched_words >= 6:
            min_contrib = max(2, tr.matched_words // 3)  # ~33% of matched words
            for reef_id in range(n):
                wc = word_counts[reef_id]
                if 0 < wc < min_contrib:
                    raw_scores[reef_id] *= wc / min_contrib

        return raw_scores, word_counts

    def _accumulate_weights(
        self, word_ids: set[int], score_floor: int = 0,
    ) -> tuple[list[int], list[int]]:
        """Phase 3: Accumulate quantized weight scores."""
        n = self._n_reefs
        scores_q = [0] * n
        word_counts = [0] * n
        word_reefs = self._word_reefs

        for word_id in word_ids:
            for reef_id, weight_q, _sub_reef_id in word_reefs[word_id]:
                if weight_q < score_floor:
                    continue
                scores_q[reef_id] += weight_q
                word_counts[reef_id] += 1

        return scores_q, word_counts

    def _propagate(self, raw_scores: list[float]) -> list[float]:
        """Propagate scores through reef edges (single-hop, additive)."""
        if not self._reef_edges:
            return raw_scores
        propagated = list(raw_scores)
        for src, tgt, w in self._reef_edges:
            propagated[tgt] += raw_scores[src] * w
        return propagated

    def _subtract_background(
        self, raw_scores: list[float], n_matched: int = 0,
    ) -> list[float]:
        """Phase 4: Background subtraction scaled by matched word count.

        Alpha (0→1) controls how much of bg_mean to subtract:
        - 1 matched word  → alpha=0: raw/bg_std (no mean subtraction).
          A single word activates only a few reefs; there is no accumulated
          common-reef noise to remove, and raw BM25 correctly identifies
          the reef (67% accuracy vs 27% with full subtraction).
        - 6+ matched words → alpha=1: full z-score (raw−bg_mean)/bg_std.
          Many words accumulate signal on common "style" reefs (dialect,
          arithmetic, logic) that must be subtracted for discrimination.
        - In between: linear ramp.

        Dividing by bg_std at all alpha values keeps scores on a
        comparable scale and acts as reef-level IDF (specific reefs
        with low bg_std get amplified).
        """
        bg_mean = self._bg_mean
        bg_std = self._bg_std
        n = self._n_reefs

        # Linear ramp: 0 at n_matched<=1, 1.0 at n_matched>=threshold
        threshold = _BG_ALPHA_THRESHOLD
        if n_matched <= 1:
            alpha = 0.0
        elif n_matched >= threshold:
            alpha = 1.0
        else:
            alpha = (n_matched - 1) / (threshold - 1)

        z_scores = [0.0] * n
        for i in range(n):
            std = bg_std[i]
            if std < _BG_STD_FLOOR:
                std = _BG_STD_FLOOR
            z_scores[i] = (raw_scores[i] - alpha * bg_mean[i]) / std
        return z_scores

    def _resolve_sub_reefs(
        self, top_reef_ids: list[int], matched_word_ids: set[int],
    ) -> dict[int, int | None]:
        """For each top island, resolve which gen-2 sub-reef best matches."""
        if not self._sub_reef_meta:
            return {}

        results: dict[int, int | None] = {}
        word_reefs = self._word_reefs
        word_reef_detail = self._word_reef_detail

        for island_id in top_reef_ids:
            votes: dict[int, float] = defaultdict(float)
            for word_id in matched_word_ids:
                if word_id >= len(word_reefs):
                    continue
                for rid, wq, sub_rid in word_reefs[word_id]:
                    if rid != island_id:
                        continue
                    if sub_rid != _SUB_REEF_SENTINEL:
                        votes[sub_rid] += wq
                    else:
                        # Multi-reef: look up detail
                        if word_id < len(word_reef_detail):
                            for d_iid, d_srid, d_wq in word_reef_detail[word_id]:
                                if d_iid == island_id:
                                    votes[d_srid] += d_wq
            results[island_id] = max(votes, key=votes.get) if votes else None
        return results

    def _extract_results(
        self,
        z_scores: list[float],
        raw_scores: list[float],
        word_counts: list[int],
        matched: set[int],
        unknown: list[str],
        top_k: int,
        min_reef_z: float | None = None,
    ) -> TopicResult:
        """Phase 5: Extract top-K reefs, islands, archipelago rollup."""
        # Sort all reefs by z_score descending (specificity baked into bg_std)
        indexed = sorted(
            range(self._n_reefs), key=lambda i: z_scores[i], reverse=True,
        )

        if min_reef_z is not None:
            selected = [i for i in indexed if z_scores[i] >= min_reef_z]
        else:
            selected = indexed[:top_k]

        reef_meta = self._reef_meta
        sub_reef_map = self._resolve_sub_reefs(selected, matched)
        sub_reef_meta = self._sub_reef_meta
        top_reefs = [
            ScoredReef(
                reef_id=i,
                z_score=z_scores[i],
                raw_score=raw_scores[i],
                n_contributing_words=word_counts[i],
                name=reef_meta[i].name,
                quality_score=z_scores[i],
                valence=reef_meta[i].valence,
                avg_specificity=reef_meta[i].avg_specificity,
                resolved_sub_reef_id=sub_reef_map.get(i),
                resolved_sub_reef_name=(
                    sub_reef_meta[sub_reef_map[i]].name
                    if sub_reef_map.get(i) is not None
                    and sub_reef_map[i] < len(sub_reef_meta)
                    else None
                ),
            )
            for i in selected
        ]

        # Confidence: strength of the strongest signal above noise.
        # Using the top z-score (clamped to 0) rather than the gap between
        # #1 and #2 — the gap penalises queries that correctly activate
        # multiple related reefs, producing artificially low confidence
        # as the number of reefs increases.
        confidence = max(0.0, z_scores[indexed[0]]) if indexed else 0.0

        # Coverage
        total_words = len(matched) + len(unknown)
        coverage = len(matched) / total_words if total_words > 0 else 0.0

        # Filter stop words from unknown list
        filtered_unknown = [w for w in unknown if w not in STOP_WORDS]

        # Valence signal: z-score-weighted mean of reef valences
        valence_signal = 0.0
        if selected:
            qs_sum = 0.0
            val_weighted = 0.0
            for i in selected:
                z = z_scores[i]
                if z > 0.0:
                    val_weighted += z * reef_meta[i].valence
                    qs_sum += z
            if qs_sum > 0.0:
                valence_signal = val_weighted / qs_sum

        # Island rollup from selected reefs
        island_agg: dict[int, list[float | int]] = defaultdict(lambda: [0.0, 0])
        for i in selected:
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
        arch_scores = [0.0] * self._n_archs
        for island in top_islands:
            aid = island_meta[island.island_id].arch_id
            arch_scores[aid] += island.aggregate_z

        # Count domainless matches: words recognized but not domain-specific
        n_domainless = len(matched & self._domainless_word_ids) if self._domainless_word_ids else 0

        return TopicResult(
            top_reefs=top_reefs,
            top_islands=top_islands,
            arch_scores=arch_scores,
            confidence=confidence,
            coverage=coverage,
            matched_words=len(matched),
            unknown_words=filtered_unknown,
            matched_word_ids=frozenset(matched),
            n_domainless=n_domainless,
            valence_signal=valence_signal,
        )
