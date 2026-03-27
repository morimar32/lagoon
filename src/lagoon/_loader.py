"""Data loading, manifest validation, and SHA-256 checksum verification."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from statistics import mean
from typing import Any

import ahocorasick
import msgpack

from ._errors import LagoonChecksumError, LagoonError, LagoonVersionError
from ._types import IslandMeta, LookupData, ReefMeta, SubReefMeta, TownMeta, V3ReefMeta, WordInfo

_EXPECTED_VERSION_PREFIX = "3.1"

_REQUIRED_FILES = (
    "word_lookup.bin",
    "word_islands.bin",
    "word_towns.bin",
    "island_meta.bin",
    "background.bin",
    "constants.bin",
    "compounds.bin",
    "word_detail.bin",
    "town_meta.bin",
)

_OPTIONAL_FILES = (
    "reef_meta.bin",
    "bucket_words.bin",
)


def _default_data_dir() -> Path:
    return Path(str(resources.files("lagoon") / "data"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise LagoonError(f"manifest.json not found in {data_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def _validate_manifest(manifest: dict[str, Any], data_dir: Path) -> None:
    version = manifest.get("version", "")
    if not version.startswith(_EXPECTED_VERSION_PREFIX):
        raise LagoonVersionError(
            f"Expected data version starting with {_EXPECTED_VERSION_PREFIX!r}, "
            f"got {version!r}"
        )
    checksums = manifest.get("files", {})
    for filename in _REQUIRED_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            raise LagoonError(f"Missing data file: {filepath}")
        expected = checksums.get(filename)
        if expected is None:
            raise LagoonError(f"No checksum in manifest for {filename}")
        actual = _sha256(filepath)
        if actual != expected:
            raise LagoonChecksumError(
                f"Checksum mismatch for {filename}: "
                f"expected {expected[:16]}..., got {actual[:16]}..."
            )
    # Validate optional files if present
    for filename in _OPTIONAL_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            expected = checksums.get(filename)
            if expected is not None:
                actual = _sha256(filepath)
                if actual != expected:
                    raise LagoonChecksumError(
                        f"Checksum mismatch for {filename}: "
                        f"expected {expected[:16]}..., got {actual[:16]}..."
                    )


def _load_msgpack(path: Path, **kwargs: Any) -> Any:
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, **kwargs)


def load_data(data_dir: Path | str | None = None) -> dict[str, Any]:
    """Load and validate all data files, returning a dict of parsed structures."""
    if data_dir is None:
        data_dir = _default_data_dir()
    else:
        data_dir = Path(data_dir)

    manifest = _read_manifest(data_dir)
    _validate_manifest(manifest, data_dir)

    # word_lookup: dict[u64_hash -> WordInfo]
    raw_lookup = _load_msgpack(
        data_dir / "word_lookup.bin", strict_map_key=False
    )
    word_lookup: dict[int, WordInfo] = {}
    for key, val in raw_lookup.items():
        word_lookup[key] = WordInfo(
            word_hash=val[0], word_id=val[1],
            specificity=val[2], idf_q=val[3],
        )

    # word_reefs: list[list[tuple[int,int,int]]]  (town_id, weight_q, sentinel)
    # v3.1 word_towns.bin has 2-element entries [town_id, weight]; pad to 3-element
    # with sentinel sub_reef_id for compat with _accumulate_weights, add_custom_word, etc.
    raw_towns = _load_msgpack(data_dir / "word_towns.bin")
    word_reefs: list[list[tuple[int, int, int]]] = [
        [(e[0], e[1], 0xFFFF) for e in entries]
        for entries in raw_towns
    ]

    # island_meta.bin → island_meta (40 islands for rollup)
    raw_island_meta = _load_msgpack(data_dir / "island_meta.bin")
    island_meta: list[IslandMeta] = [
        IslandMeta(island_id=i, arch_id=im["arch_id"], name=im["name"])
        for i, im in enumerate(raw_island_meta)
    ]

    # Build arch_id lookup from island_meta
    arch_for_island = [im["arch_id"] for im in raw_island_meta]

    # reef_meta: 298 towns become the scoring reefs
    raw_town_meta = _load_msgpack(data_dir / "town_meta.bin")
    reef_meta: list[ReefMeta] = []
    for i, tm in enumerate(raw_town_meta):
        island_id = tm["island_id"]
        arch_id = arch_for_island[island_id]
        reef_meta.append(ReefMeta(
            reef_id=i,
            hierarchy_addr=(arch_id << 16) | i,
            n_words=tm["n_words"],
            name=tm["name"],
            island_id=island_id,
            arch_id=arch_id,
            valence=0.0,
            avg_specificity=tm.get("avg_specificity", 0.0),
            noun_frac=0.0, verb_frac=0.0, adj_frac=0.0, adv_frac=0.0,
        ))

    # background: town-level bg_mean, bg_std (298 entries)
    bg = _load_msgpack(data_dir / "background.bin")
    bg_mean: list[float] = bg["town_bg_mean"]
    bg_std: list[float] = bg["town_bg_std"]

    # compounds: build Aho-Corasick automaton
    raw_compounds = _load_msgpack(data_dir / "compounds.bin")
    ac = ahocorasick.Automaton()
    compound_word_ids: list[int] = []
    compound_strings: list[str] = []
    for idx, (compound_str, word_id) in enumerate(raw_compounds):
        ac.add_word(compound_str, idx)
        compound_word_ids.append(word_id)
        compound_strings.append(compound_str)
    if raw_compounds:
        ac.make_automaton()

    # constants — synthesize v7-compat keys using town-level data
    constants = _load_msgpack(data_dir / "constants.bin")
    town_n_words = constants["town_n_words"]
    constants["reef_n_words"] = town_n_words
    constants["reef_total_dims"] = [0] * len(town_n_words)
    constants["avg_reef_words"] = mean(town_n_words) if town_n_words else 0
    # IQF placeholder per town (no IQF column yet)
    constants["reef_iqf"] = [tm.get("tqf", 128) for tm in raw_town_meta]

    # domainless word_ids — town-level (words not in any town, larger set)
    domainless_word_ids = frozenset(constants.get("town_domainless_word_ids", []))

    # bucket_only_word_ids
    bucket_only_word_ids = frozenset(constants.get("bucket_only_word_ids", []))

    # reef_edges: empty in v3
    reef_edges: list[tuple[int, int, float]] = []

    # word_reef_detail: v3 word_detail.bin has 5-element entries
    # (island_id, town_id, reef_id, weight, level)
    raw_detail = _load_msgpack(data_dir / "word_detail.bin")
    word_reef_detail: list[list[tuple]] = [
        [tuple(e) for e in entries]
        for entries in raw_detail
    ]

    # sub_reef_meta: empty — towns ARE the reefs now, no deeper resolution
    sub_reef_meta: list[SubReefMeta] = []

    # town_meta: full TownMeta objects (same data used for reef_meta above)
    town_meta: list[TownMeta] = [
        TownMeta(
            town_id=i,
            island_id=tm["island_id"],
            name=tm["name"],
            n_words=tm["n_words"],
            tqf=tm.get("tqf", 128),
            avg_specificity=tm.get("avg_specificity", 0.0),
        )
        for i, tm in enumerate(raw_town_meta)
    ]

    # v3_reef_meta: full V3ReefMeta objects from reef_meta.bin (optional)
    reef_meta_path = data_dir / "reef_meta.bin"
    if reef_meta_path.exists():
        raw_reef_meta_v3 = _load_msgpack(reef_meta_path)
        v3_reef_meta: list[V3ReefMeta] = [
            V3ReefMeta(
                reef_id=i,
                town_id=rm["town_id"],
                name=rm.get("name", ""),
                n_words=rm["n_words"],
                avg_specificity=rm.get("avg_specificity", 0.0),
            )
            for i, rm in enumerate(raw_reef_meta_v3)
        ]
    else:
        v3_reef_meta = []

    # bucket_words: sparse list indexed by word_id (optional)
    bucket_path = data_dir / "bucket_words.bin"
    if bucket_path.exists():
        raw_bucket = _load_msgpack(bucket_path)
        bucket_words: list[list[tuple[int, int]]] = [
            [tuple(e) for e in entries]  # type: ignore[misc]
            for entries in raw_bucket
        ]
    else:
        bucket_words = []

    return {
        "word_lookup": word_lookup,
        "word_reefs": word_reefs,
        "reef_meta": reef_meta,
        "island_meta": island_meta,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "compound_ac": ac,
        "compound_word_ids": compound_word_ids,
        "compound_strings": compound_strings,
        "constants": constants,
        "reef_edges": reef_edges,
        "word_reef_detail": word_reef_detail,
        "sub_reef_meta": sub_reef_meta,
        "domainless_word_ids": domainless_word_ids,
        "town_meta": town_meta,
        "v3_reef_meta": v3_reef_meta,
        "bucket_words": bucket_words,
        "bucket_only_word_ids": bucket_only_word_ids,
    }


_MASK64 = 0xFFFFFFFFFFFFFFFF


def _i64_to_u64(signed: int) -> int:
    """Convert a signed i64 (from SQLite) to unsigned u64 (Lagoon hash space)."""
    if signed < 0:
        return signed + (1 << 64)
    return signed


def _default_lookup_dir() -> Path:
    return Path(str(resources.files("lagoon") / "data_lookup"))


def load_lookup(data_dir: Path | str | None = None) -> LookupData:
    """Load optional lookup reference data (equivalences, word tags, names).

    Returns a LookupData with empty collections if the directory or any
    individual file is missing. Never raises on absent data.
    """
    if data_dir is None:
        data_dir = _default_lookup_dir()
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return LookupData(equivalences={}, word_tags={}, names=frozenset())

    # Equivalences: sorted [variant_hash_i64, word_id] pairs → dict[u64, list[int]]
    equiv_path = data_dir / "equivalences.bin"
    equivalences: dict[int, list[int]] = {}
    if equiv_path.exists():
        raw_equiv = _load_msgpack(equiv_path)
        for pair in raw_equiv:
            h_u64 = _i64_to_u64(pair[0])
            wid = pair[1]
            if h_u64 in equivalences:
                equivalences[h_u64].append(wid)
            else:
                equivalences[h_u64] = [wid]

    # Word tags: {word: [[tag_type, tag_value], ...]}
    tags_path = data_dir / "word_tags.bin"
    if tags_path.exists():
        word_tags = _load_msgpack(tags_path)
    else:
        word_tags = {}

    # Names: [[name, type], ...] → frozenset of lowercased names
    names_path = data_dir / "names.bin"
    if names_path.exists():
        raw_names = _load_msgpack(names_path)
        names = frozenset(entry[0].lower() for entry in raw_names)
    else:
        names = frozenset()

    return LookupData(equivalences=equivalences, word_tags=word_tags, names=names)
