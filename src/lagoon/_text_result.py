"""Build TextResult from ordered tokenizer output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import IslandEntry, ReefHit, TextResult

if TYPE_CHECKING:
    from ._types import ReefMeta


def build_text_result(
    word_ids_ordered: list[int | None],
    unknown_words: list[str],
    total_tokens: int,
    word_reefs: list[list[tuple[int, int, int]]],
    reef_meta: list[ReefMeta],
    domainless_word_ids: frozenset[int],
) -> TextResult:
    """Build a TextResult from ordered tokenizer output.

    For each position in word_ids_ordered:
    - None → dropped (stop/unknown)
    - In domainless_word_ids → domainless
    - Else → look up word_reefs, build linked-list chain sorted by weight desc

    Args:
        word_ids_ordered: word_id per text position (None = stop/unknown)
        unknown_words: unrecognized tokens
        total_tokens: total token count from tokenizer
        word_reefs: word_id → list[(reef_id, weight_q, sub_reef_id)]
        reef_meta: reef metadata for island_id lookup
        domainless_word_ids: words in vocab but without domain scores
    """
    reef_hits: list[ReefHit] = []
    word_order: list[int] = []
    island_map: dict[int, int] = {}  # island_id → index into islands list
    islands: list[IslandEntry] = []
    island_words: dict[int, set[int]] = {}  # island_id → set of word positions

    matched_words = 0
    domainless_count = 0
    dropped_count = 0

    for pos, wid in enumerate(word_ids_ordered):
        if wid is None:
            word_order.append(-1)
            dropped_count += 1
            continue

        if wid in domainless_word_ids:
            word_order.append(-1)
            domainless_count += 1
            continue

        # Look up reef entries for this word
        if wid >= len(word_reefs) or not word_reefs[wid]:
            word_order.append(-1)
            dropped_count += 1
            continue

        entries = word_reefs[wid]
        # Sort by weight_q descending (head = best reef)
        sorted_entries = sorted(entries, key=lambda e: e[1], reverse=True)

        matched_words += 1
        head_idx = len(reef_hits)
        word_order.append(head_idx)

        for i, (reef_id, weight_q, _sub_reef_id) in enumerate(sorted_entries):
            # Resolve island from reef_meta
            island_id = reef_meta[reef_id].island_id if reef_id < len(reef_meta) else -1

            # Get or create island entry
            island_idx = -1
            if island_id >= 0:
                if island_id not in island_map:
                    island_map[island_id] = len(islands)
                    islands.append(IslandEntry(
                        island_id=island_id,
                        reef_hit_indices=[],
                        word_count=0,
                    ))
                    island_words[island_id] = set()
                island_idx = island_map[island_id]

            hit_idx = len(reef_hits)
            # next_idx: points to next entry in chain, -1 for last
            next_idx = hit_idx + 1 if i < len(sorted_entries) - 1 else -1

            reef_hits.append(ReefHit(
                reef_id=reef_id,
                score=weight_q,
                island_idx=island_idx,
                next_idx=next_idx,
            ))

            # Track reef_hit in island
            if island_idx >= 0:
                islands[island_idx].reef_hit_indices.append(hit_idx)
                island_words[island_id].add(pos)

    # Finalize island word counts
    for island_id, idx in island_map.items():
        islands[idx].word_count = len(island_words[island_id])

    return TextResult(
        reef_hits=reef_hits,
        word_order=word_order,
        islands=islands,
        total_words=total_tokens,
        matched_words=matched_words,
        domainless_words=domainless_count,
        dropped_words=dropped_count,
    )
