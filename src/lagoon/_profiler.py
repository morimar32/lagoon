"""Multi-lens text profiler with cross-lens reconciliation.

Runs multiple ReefScorer instances (one per lens) against the same text,
then reconciles results to produce cross-lens signals like register
detection, polysemy disambiguation, and contextual modulation.

The Profiler is the coordination layer above individual scorers.
Each scorer is data-agnostic — it scores word→town associations.
The Profiler is lens-aware — it knows what each scorer represents
and derives meaning from their interactions.

Usage:
    profiler = Profiler(
        lenses={
            "domain": lagoon.load("exports/"),
            "human": lagoon.load("exports_human/"),
        },
    )
    profile = profiler.score("My father was diagnosed with cancer")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import (
    DocumentProfile,
    LookupData,
    POVSignal,
    ProfiledSegment,
    RegisterSignal,
    TextProfile,
    TopicResult,
)

if TYPE_CHECKING:
    from ._scorer import ReefScorer

# Pronoun sets for POV detection (lowercase)
_FIRST_PERSON = frozenset({
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
})
_SECOND_PERSON = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
})

# Minimum pronoun tokens required to emit a POV signal
_POV_MIN_EVIDENCE = 2


class Profiler:
    """Multi-lens text profiler.

    Coordinates N named scorers, runs them independently, then
    reconciles results into a TextProfile with cross-lens signals.

    Optionally accepts LookupData for name detection and word tag
    analysis. These are additive signals — the profiler works fine
    without them.
    """

    __slots__ = ("_lenses", "_lookup")

    def __init__(
        self,
        lenses: dict[str, ReefScorer],
        lookup: LookupData | None = None,
    ) -> None:
        if not lenses:
            raise ValueError("Profiler requires at least one lens")
        self._lenses = lenses
        self._lookup = lookup

    @property
    def lens_names(self) -> list[str]:
        return list(self._lenses.keys())

    def scorer(self, lens_name: str) -> ReefScorer:
        """Access an individual scorer by lens name."""
        return self._lenses[lens_name]

    def score(
        self,
        text: str,
        top_k: int = 10,
        *,
        min_reef_z: float | None = None,
    ) -> TextProfile:
        """Score text against all lenses and reconcile results."""
        # Run each lens independently
        results: dict[str, TopicResult] = {}
        for name, scorer in self._lenses.items():
            results[name] = scorer.score(text, top_k=top_k, min_reef_z=min_reef_z)

        # Compute cross-lens signals
        register = self._compute_register(results)
        pov = self._detect_pov(text)
        names = self._detect_names(text) if self._lookup else []

        return TextProfile(
            lenses=results,
            register=register,
            pov=pov,
            names_detected=names,
        )

    def score_batch(
        self,
        texts: list[str],
        top_k: int = 10,
        *,
        min_reef_z: float | None = None,
    ) -> list[TextProfile]:
        """Score a batch of texts against all lenses."""
        # Run each lens in batch
        batch_results: dict[str, list[TopicResult]] = {}
        for name, scorer in self._lenses.items():
            batch_results[name] = scorer.score_batch(
                texts, top_k=top_k, min_reef_z=min_reef_z
            )

        # Reconcile per-text
        profiles = []
        for i in range(len(texts)):
            results = {name: batch_results[name][i] for name in self._lenses}
            register = self._compute_register(results)
            pov = self._detect_pov(texts[i])
            names = self._detect_names(texts[i]) if self._lookup else []
            profiles.append(TextProfile(
                lenses=results, register=register, pov=pov, names_detected=names,
            ))

        return profiles

    def analyze(
        self,
        text: str | list[str],
        *,
        segmentation_lens: str = "domain",
        sensitivity: float = 1.0,
        smooth_window: int = 2,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 30,
        min_reef_z: float = 2.0,
    ) -> DocumentProfile:
        """Document-level analysis with multi-lens profiling per segment.

        Uses one lens (default: "domain") for segmentation — it drives
        boundary detection via z-score vector similarity. Then scores
        each segment against ALL lenses and reconciles cross-lens signals.

        Args:
            text: Raw text or pre-split list of sentences.
            segmentation_lens: Which lens drives boundary detection.
            sensitivity: Boundary detection threshold. Lower = more boundaries.
            smooth_window: Sliding window size for z-score vector smoothing.
            min_chunk_sentences: Minimum sentences per chunk.
            max_chunk_sentences: Maximum sentences per chunk (0 to disable).
            min_reef_z: Minimum z-score for reef inclusion.

        Returns:
            DocumentProfile with per-segment TextProfiles.
        """
        if segmentation_lens not in self._lenses:
            raise ValueError(
                f"segmentation_lens {segmentation_lens!r} not in lenses: "
                f"{list(self._lenses.keys())}"
            )

        # Step 1: Use the segmentation lens to determine boundaries
        seg_scorer = self._lenses[segmentation_lens]
        doc = seg_scorer.analyze(
            text,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
            min_reef_z=min_reef_z,
        )

        if not doc.segments:
            return DocumentProfile(
                segments=[], n_sentences=0, n_segments=0, boundaries=[],
            )

        # Step 2: Score each segment against ALL lenses
        profiled: list[ProfiledSegment] = []
        for seg in doc.segments:
            combined = " ".join(seg.sentences)

            # Score all lenses
            results: dict[str, TopicResult] = {}
            for name, scorer in self._lenses.items():
                if name == segmentation_lens:
                    # Reuse the already-computed result
                    results[name] = seg.topic
                else:
                    results[name] = scorer.score(
                        combined, min_reef_z=min_reef_z,
                    )

            # Cross-lens signals
            register = self._compute_register(results)
            pov = self._detect_pov(combined)
            names = self._detect_names(combined) if self._lookup else []

            profile = TextProfile(
                lenses=results,
                register=register,
                pov=pov,
                names_detected=names,
            )

            profiled.append(ProfiledSegment(
                sentences=seg.sentences,
                start_idx=seg.start_idx,
                end_idx=seg.end_idx,
                profile=profile,
                keywords=seg.keywords,
            ))

        return DocumentProfile(
            segments=profiled,
            n_sentences=doc.n_sentences,
            n_segments=doc.n_segments,
            boundaries=doc.boundaries,
        )

    # -----------------------------------------------------------------
    # Register detection
    # -----------------------------------------------------------------

    def _compute_register(self, results: dict[str, TopicResult]) -> RegisterSignal:
        """Derive register signal from cross-lens coverage.

        The ratio of domain vs human coverage reveals the text's register:
        - High domain, low human → technical/academic
        - Low domain, high human → casual/personal narrative
        - Both high → academic discussion of human topics (e.g. psychology paper)
        - Both low → short/ambiguous text or out-of-vocabulary
        """
        domain = results.get("domain")
        human = results.get("human")

        # If we don't have both lenses, return a neutral signal
        if domain is None or human is None:
            only = domain or human
            cov = only.coverage if only else 0.0
            return RegisterSignal(
                label="unknown",
                domain_coverage=domain.coverage if domain else 0.0,
                human_coverage=human.coverage if human else 0.0,
                overlap_fraction=0.0,
                score=0.5,
            )

        d_cov = domain.coverage
        h_cov = human.coverage

        # Overlap: words matched by BOTH lenses
        if domain.matched_word_ids and human.matched_word_ids:
            overlap_ids = domain.matched_word_ids & human.matched_word_ids
            total_matched = len(domain.matched_word_ids | human.matched_word_ids)
            overlap_frac = len(overlap_ids) / total_matched if total_matched > 0 else 0.0
        else:
            overlap_frac = 0.0

        # Register score: 0 = casual, 1 = technical
        # Based on domain/human coverage ratio
        total = d_cov + h_cov
        if total > 0:
            score = d_cov / total
        else:
            score = 0.5  # ambiguous

        # Label assignment
        if total < 0.05:
            label = "unknown"
        elif score >= 0.75:
            label = "technical"
        elif score <= 0.25:
            label = "casual"
        elif overlap_frac >= 0.5:
            label = "mixed"
        elif h_cov > d_cov:
            label = "narrative"
        else:
            label = "mixed"

        return RegisterSignal(
            label=label,
            domain_coverage=d_cov,
            human_coverage=h_cov,
            overlap_fraction=overlap_frac,
            score=score,
        )

    # -----------------------------------------------------------------
    # Point-of-view detection
    # -----------------------------------------------------------------

    @staticmethod
    def _detect_pov(text: str) -> POVSignal | None:
        """Detect dominant point of view from pronoun distribution.

        Returns None when fewer than _POV_MIN_EVIDENCE pronoun tokens
        are found — insufficient signal to make a determination.
        """
        tokens = text.lower().split()
        if not tokens:
            return None

        first = 0
        second = 0
        for token in tokens:
            # Strip common trailing punctuation for matching
            clean = token.rstrip(".,!?;:\"')-]}")
            if clean in _FIRST_PERSON:
                first += 1
            elif clean in _SECOND_PERSON:
                second += 1

        total_pronouns = first + second
        if total_pronouns < _POV_MIN_EVIDENCE:
            return None

        n = len(tokens)
        first_ratio = first / n
        second_ratio = second / n

        # Determine dominant POV
        if first > 0 and second > 0:
            # Both present — check if one dominates (3:1 ratio)
            if first >= second * 3:
                dominant = "first"
            elif second >= first * 3:
                dominant = "second"
            else:
                dominant = "mixed"
        elif first > 0:
            dominant = "first"
        elif second > 0:
            dominant = "second"
        else:
            dominant = "impersonal"

        return POVSignal(
            dominant=dominant,
            first_ratio=first_ratio,
            second_ratio=second_ratio,
            evidence=total_pronouns,
        )

    # -----------------------------------------------------------------
    # Name detection
    # -----------------------------------------------------------------

    def _detect_names(self, text: str) -> list[str]:
        """Detect first names from the lookup DB's names set.

        Returns deduplicated list of detected names (original casing from
        text, lowercased for matching). Only emits names that appear with
        an initial capital letter to reduce false positives on common words
        that happen to also be names (e.g., "will", "mark", "grace").
        """
        if not self._lookup or not self._lookup.names:
            return []

        names_set = self._lookup.names
        seen: set[str] = set()
        result: list[str] = []

        for token in text.split():
            # Only consider tokens that start with a capital letter
            if not token or not token[0].isupper():
                continue
            clean = token.strip(".,!?;:\"'()-[]{}").lower()
            if clean in names_set and clean not in seen:
                seen.add(clean)
                result.append(clean)

        return result
