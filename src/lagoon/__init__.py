"""Lagoon: text-scoring library mapping input text to semantic reefs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import LagoonChecksumError, LagoonError, LagoonVersionError
from ._sentence import split_sentences
from ._stop_words import STOP_WORDS
from ._profiler import Profiler
from ._types import (
    DocumentAnalysis,
    DocumentProfile,
    IslandMeta,
    LookupData,
    POVSignal,
    ProfiledSegment,
    ReefMeta,
    RegisterSignal,
    ScoredIsland,
    ScoredReef,
    ScoredTown,
    SubReefMeta,
    TextProfile,
    TopicResult,
    TopicSegment,
    TownMeta,
    V3ReefMeta,
    WordInfo,
)

if TYPE_CHECKING:
    from pathlib import Path

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load",
    "load_lookup",
    "DocumentAnalysis",
    "DocumentProfile",
    "IslandMeta",
    "LagoonChecksumError",
    "LagoonError",
    "LagoonVersionError",
    "LookupData",
    "POVSignal",
    "ProfiledSegment",
    "Profiler",
    "ReefMeta",
    "ReefScorer",
    "RegisterSignal",
    "ScoredIsland",
    "ScoredReef",
    "ScoredTown",
    "STOP_WORDS",
    "SubReefMeta",
    "TextProfile",
    "TopicResult",
    "TopicSegment",
    "TownMeta",
    "V3ReefMeta",
    "WordInfo",
    "split_sentences",
]


def load(
    data_dir: Path | str | None = None,
    lookup: LookupData | None = None,
) -> "ReefScorer":
    """Load data and return a ready-to-use ReefScorer.

    Args:
        data_dir: Path to data directory. If None, uses bundled package data.
        lookup: Optional LookupData for equivalence resolution. If provided,
            the tokenizer will use equivalences as a fallback after stemming.
    """
    from ._loader import load_data
    from ._scorer import ReefScorer

    data = load_data(data_dir)
    return ReefScorer(
        word_lookup=data["word_lookup"],
        word_reefs=data["word_reefs"],
        reef_meta=data["reef_meta"],
        island_meta=data["island_meta"],
        bg_mean=data["bg_mean"],
        bg_std=data["bg_std"],
        compound_ac=data["compound_ac"],
        compound_word_ids=data["compound_word_ids"],
        compound_strings=data["compound_strings"],
        constants=data["constants"],
        reef_edges=data["reef_edges"],
        word_reef_detail=data["word_reef_detail"],
        sub_reef_meta=data["sub_reef_meta"],
        v3_reef_meta=data["v3_reef_meta"],
        domainless_word_ids=data["domainless_word_ids"],
        equivalences=lookup.equivalences if lookup else None,
    )


def load_lookup(data_dir: Path | str | None = None) -> LookupData:
    """Load optional lookup reference data (equivalences, word tags, names).

    Returns a LookupData with empty collections if data is not available.

    Args:
        data_dir: Path to lookup data directory. If None, uses bundled package data.
    """
    from ._loader import load_lookup as _load_lookup

    return _load_lookup(data_dir)


# Deferred import so ReefScorer is available as lagoon.ReefScorer
# without circular import issues at module load time.
def __getattr__(name: str):
    if name == "ReefScorer":
        from ._scorer import ReefScorer
        return ReefScorer
    raise AttributeError(f"module 'lagoon' has no attribute {name!r}")
