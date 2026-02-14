"""Lagoon: text-scoring library mapping input text to 207 semantic reefs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import LagoonChecksumError, LagoonError, LagoonVersionError
from ._sentence import split_sentences
from ._stop_words import STOP_WORDS
from ._types import (
    DocumentAnalysis,
    IslandMeta,
    ReefMeta,
    ScoredIsland,
    ScoredReef,
    TopicResult,
    TopicSegment,
    WordInfo,
)

if TYPE_CHECKING:
    from pathlib import Path

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load",
    "DocumentAnalysis",
    "IslandMeta",
    "LagoonChecksumError",
    "LagoonError",
    "LagoonVersionError",
    "ReefMeta",
    "ReefScorer",
    "ScoredIsland",
    "ScoredReef",
    "STOP_WORDS",
    "TopicResult",
    "TopicSegment",
    "WordInfo",
    "split_sentences",
]


def load(data_dir: Path | str | None = None) -> "ReefScorer":
    """Load data and return a ready-to-use ReefScorer.

    Args:
        data_dir: Path to data directory. If None, uses bundled package data.
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
    )


# Deferred import so ReefScorer is available as lagoon.ReefScorer
# without circular import issues at module load time.
def __getattr__(name: str):
    if name == "ReefScorer":
        from ._scorer import ReefScorer
        return ReefScorer
    raise AttributeError(f"module 'lagoon' has no attribute {name!r}")
