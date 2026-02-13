"""Document-level topic segmentation using cosine similarity on z-score vectors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._sentence import split_sentences
from ._types import DocumentAnalysis, TopicSegment

if TYPE_CHECKING:
    from ._scorer import ReefScorer


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _smooth_vectors(
    vectors: list[list[float]], window: int
) -> list[list[float]]:
    """Sliding window average on z-score vectors."""
    if window <= 0:
        return vectors
    n = len(vectors)
    if n == 0:
        return []
    dim = len(vectors[0])
    smoothed: list[list[float]] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        count = hi - lo + 1
        avg = [0.0] * dim
        for j in range(lo, hi + 1):
            for d in range(dim):
                avg[d] += vectors[j][d]
        for d in range(dim):
            avg[d] /= count
        smoothed.append(avg)
    return smoothed


def _detect_boundaries(
    similarities: list[float], sensitivity: float
) -> list[int]:
    """Find valleys in the similarity curve.

    A boundary is placed between sentence i and i+1 when similarities[i]
    is below mean - sensitivity * std AND is a local minimum.
    """
    if not similarities:
        return []

    n = len(similarities)
    mean = sum(similarities) / n
    variance = sum((s - mean) ** 2 for s in similarities) / n
    std = math.sqrt(variance)
    threshold = mean - sensitivity * std

    boundaries: list[int] = []
    for i in range(n):
        if similarities[i] >= threshold:
            continue
        # Check local minimum
        left = similarities[i - 1] if i > 0 else float("inf")
        right = similarities[i + 1] if i < n - 1 else float("inf")
        if similarities[i] <= left and similarities[i] <= right:
            # Boundary after sentence i+1 (0-indexed):
            # similarities[i] is between sentence i and i+1
            boundaries.append(i + 1)

    return boundaries


def analyze_document(
    scorer: ReefScorer,
    text: str | list[str],
    *,
    sensitivity: float = 1.0,
    smooth_window: int = 2,
    min_segment_sentences: int = 1,
) -> DocumentAnalysis:
    """Segment a document by topic shifts."""
    # Step 1: Sentence segmentation
    if isinstance(text, list):
        sentences = text
    else:
        sentences = split_sentences(text)

    if not sentences:
        return DocumentAnalysis(
            segments=[], n_sentences=0, n_segments=0, boundaries=[],
        )

    n = len(sentences)

    if n == 1:
        topic = scorer.score(sentences[0])
        seg = TopicSegment(
            sentences=sentences, start_idx=0, end_idx=0, topic=topic,
        )
        return DocumentAnalysis(
            segments=[seg], n_sentences=1, n_segments=1, boundaries=[],
        )

    # Step 2: Per-sentence z-score vectors
    vectors = [scorer.score_raw(s) for s in sentences]

    # Step 3: Smoothing
    smoothed = _smooth_vectors(vectors, smooth_window)

    # Step 4: Similarity curve between adjacent sentences
    similarities = [
        _cosine_similarity(smoothed[i], smoothed[i + 1])
        for i in range(n - 1)
    ]

    # Step 5: Boundary detection
    boundaries = _detect_boundaries(similarities, sensitivity)

    # Step 6: Assemble segments
    # Boundaries are sentence indices where a new segment starts
    seg_starts = [0] + sorted(boundaries)
    # Remove duplicates and ensure within range
    seg_starts = sorted(set(s for s in seg_starts if 0 <= s < n))
    if not seg_starts or seg_starts[0] != 0:
        seg_starts = [0] + seg_starts

    # Merge tiny segments if needed
    if min_segment_sentences > 1:
        merged = [seg_starts[0]]
        for i in range(1, len(seg_starts)):
            prev_start = merged[-1]
            if seg_starts[i] - prev_start >= min_segment_sentences:
                merged.append(seg_starts[i])
        seg_starts = merged

    # Step 7: Build TopicSegments by re-scoring concatenated text
    segments: list[TopicSegment] = []
    for i, start in enumerate(seg_starts):
        end = seg_starts[i + 1] - 1 if i + 1 < len(seg_starts) else n - 1
        seg_sentences = sentences[start : end + 1]
        combined = " ".join(seg_sentences)
        topic = scorer.score(combined)
        segments.append(TopicSegment(
            sentences=seg_sentences,
            start_idx=start,
            end_idx=end,
            topic=topic,
        ))

    # Actual boundaries are the starts of segments after the first
    actual_boundaries = [s for s in seg_starts[1:]]

    return DocumentAnalysis(
        segments=segments,
        n_sentences=n,
        n_segments=len(segments),
        boundaries=actual_boundaries,
    )
