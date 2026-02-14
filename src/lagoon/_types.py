"""Data structures for lagoon."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WordInfo:
    word_hash: int   # u64 FNV-1a hash
    word_id: int     # u32 index into word_reefs
    specificity: int # i8 sigma band (-2 to +2)
    idf_q: int       # u8 quantized IDF


@dataclass(slots=True)
class ReefMeta:
    reef_id: int          # positional index
    hierarchy_addr: int   # u16 bit-packed
    n_words: int
    name: str
    island_id: int        # derived: (addr >> 8) & 0x3F
    arch_id: int          # derived: (addr >> 14) & 0x03


@dataclass(slots=True)
class IslandMeta:
    island_id: int
    arch_id: int
    name: str


@dataclass(slots=True, frozen=True)
class ScoredReef:
    reef_id: int
    z_score: float
    raw_bm25: float
    n_contributing_words: int
    name: str


@dataclass(slots=True, frozen=True)
class ScoredIsland:
    island_id: int
    aggregate_z: float
    n_contributing_reefs: int
    name: str


@dataclass(slots=True, frozen=True)
class TopicResult:
    top_reefs: list[ScoredReef]
    top_islands: list[ScoredIsland]
    arch_scores: list[float]     # len=4
    confidence: float
    coverage: float
    matched_words: int
    unknown_words: list[str]
    matched_word_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(slots=True, frozen=True)
class TopicSegment:
    sentences: list[str]
    start_idx: int
    end_idx: int    # inclusive
    topic: TopicResult
    sentence_results: list[TopicResult] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class DocumentAnalysis:
    segments: list[TopicSegment]
    n_sentences: int
    n_segments: int
    boundaries: list[int]   # sentence indices where topic shifts
