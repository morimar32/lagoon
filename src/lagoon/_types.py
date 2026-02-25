"""Data structures for lagoon."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WordInfo:
    word_hash: int   # u64 FNV-1a hash
    word_id: int     # u32 index into word_reefs
    specificity: int # i8 sigma band (-2 to +2)
    idf_q: int       # u8 quantized IDF
    tag: int = 0     # opaque consumer metadata, not interpreted by lagoon


@dataclass(slots=True)
class ReefMeta:
    reef_id: int          # positional index
    hierarchy_addr: int   # u32 bit-packed: arch(16)|island(16)
    n_words: int
    name: str
    island_id: int        # derived: (addr >> 16) & 0xFF
    arch_id: int          # derived: (addr >> 24) & 0xFF
    valence: float = 0.0
    avg_specificity: float = 0.0
    noun_frac: float = 0.0
    verb_frac: float = 0.0
    adj_frac: float = 0.0
    adv_frac: float = 0.0


@dataclass(slots=True)
class IslandMeta:
    island_id: int
    arch_id: int
    name: str


@dataclass(slots=True)
class SubReefMeta:
    sub_reef_id: int
    parent_island_id: int
    n_words: int
    name: str


@dataclass(slots=True)
class ReefHit:
    reef_id: int            # index into reef_meta
    score: int              # u8 percentile from export (0-255)
    island_idx: int         # index into TextResult.islands (-1 if none)
    next_idx: int           # next ReefHit for same word (-1 = end of chain)


@dataclass(slots=True)
class IslandEntry:
    island_id: int               # index into island_meta
    reef_hit_indices: list[int]  # indices into TextResult.reef_hits
    word_count: int              # distinct words hitting reefs in this island


@dataclass(slots=True)
class TextResult:
    reef_hits: list[ReefHit]      # Array 1: flat, linked-list threaded per word
    word_order: list[int]         # Array 2: text-order indices into reef_hits (-1 = no reef)
    islands: list[IslandEntry]    # Array 3: island summaries with backrefs to reef_hits
    total_words: int
    matched_words: int
    domainless_words: int
    dropped_words: int            # stop words + unknown


@dataclass(slots=True, frozen=True)
class ScoredReef:
    reef_id: int
    z_score: float
    raw_score: float
    n_contributing_words: int
    name: str
    quality_score: float = 0.0
    valence: float = 0.0
    avg_specificity: float = 0.0
    resolved_sub_reef_id: int | None = None
    resolved_sub_reef_name: str | None = None


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
    arch_scores: list[float]     # len=N_ARCHS
    confidence: float
    coverage: float
    matched_words: int
    unknown_words: list[str]
    matched_word_ids: frozenset[int] = field(default_factory=frozenset)
    n_domainless: int = 0
    valence_signal: float = 0.0


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
