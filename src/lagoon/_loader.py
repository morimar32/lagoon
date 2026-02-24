"""Data loading, manifest validation, and SHA-256 checksum verification."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import ahocorasick
import msgpack

from ._errors import LagoonChecksumError, LagoonError, LagoonVersionError
from ._types import IslandMeta, ReefMeta, SubReefMeta, WordInfo

_EXPECTED_VERSION = "6.0"

_DATA_FILES = (
    "word_lookup.bin",
    "word_reefs.bin",
    "reef_meta.bin",
    "island_meta.bin",
    "background.bin",
    "compounds.bin",
    "constants.bin",
    "reef_edges.bin",
    "word_reef_detail.bin",
    "sub_reef_meta.bin",
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
    version = manifest.get("version")
    if version != _EXPECTED_VERSION:
        raise LagoonVersionError(
            f"Expected data version {_EXPECTED_VERSION!r}, got {version!r}"
        )
    checksums = manifest.get("files", {})
    for filename in _DATA_FILES:
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

    # word_reefs: list[list[tuple[int,int,int]]]  (reef_id, weight_q, sub_reef_id)
    raw_reefs = _load_msgpack(data_dir / "word_reefs.bin")
    word_reefs: list[list[tuple[int, int, int]]] = [
        [tuple(entry) for entry in word_entries]  # type: ignore[misc]
        for word_entries in raw_reefs
    ]

    # reef_meta: list[ReefMeta]
    # v6 hierarchy_addr: arch(16)|island(16), island_id = reef_id (1:1)
    raw_reef_meta = _load_msgpack(data_dir / "reef_meta.bin")
    reef_meta: list[ReefMeta] = []
    for i, rm in enumerate(raw_reef_meta):
        addr = rm["hierarchy_addr"]
        reef_meta.append(ReefMeta(
            reef_id=i,
            hierarchy_addr=addr,
            n_words=rm["n_words"],
            name=rm["name"],
            island_id=addr & 0xFFFF,
            arch_id=(addr >> 16) & 0xFFFF,
            valence=rm.get("valence", 0.0),
            avg_specificity=rm.get("avg_specificity", 0.0),
            noun_frac=rm.get("noun_frac", 0.0),
            verb_frac=rm.get("verb_frac", 0.0),
            adj_frac=rm.get("adj_frac", 0.0),
            adv_frac=rm.get("adv_frac", 0.0),
        ))

    # island_meta: list[IslandMeta]
    raw_island_meta = _load_msgpack(data_dir / "island_meta.bin")
    island_meta: list[IslandMeta] = [
        IslandMeta(island_id=i, arch_id=im["arch_id"], name=im["name"])
        for i, im in enumerate(raw_island_meta)
    ]

    # background: bg_mean, bg_std
    bg = _load_msgpack(data_dir / "background.bin")
    bg_mean: list[float] = bg["bg_mean"]
    bg_std: list[float] = bg["bg_std"]

    # compounds: build Aho-Corasick automaton
    raw_compounds = _load_msgpack(data_dir / "compounds.bin")
    ac = ahocorasick.Automaton()
    compound_word_ids: list[int] = []
    compound_strings: list[str] = []
    for idx, (compound_str, word_id) in enumerate(raw_compounds):
        ac.add_word(compound_str, idx)
        compound_word_ids.append(word_id)
        compound_strings.append(compound_str)
    ac.make_automaton()

    # constants
    constants = _load_msgpack(data_dir / "constants.bin")

    # domainless word_ids (words recognized but not domain-specific)
    domainless_word_ids = frozenset(constants.get("domainless_word_ids", []))

    # reef_edges: list of [src, tgt, weight] triplets
    raw_edges = _load_msgpack(data_dir / "reef_edges.bin")
    reef_edges: list[tuple[int, int, float]] = [
        (e[0], e[1], e[2]) for e in raw_edges
    ]

    # word_reef_detail: list[list[tuple[int,int,int]]] (island_id, sub_reef_id, weight_q)
    raw_detail = _load_msgpack(data_dir / "word_reef_detail.bin")
    word_reef_detail: list[list[tuple[int, int, int]]] = [
        [tuple(e) for e in entries]  # type: ignore[misc]
        for entries in raw_detail
    ]

    # sub_reef_meta: list[SubReefMeta]
    raw_sub = _load_msgpack(data_dir / "sub_reef_meta.bin")
    sub_reef_meta: list[SubReefMeta] = [
        SubReefMeta(
            sub_reef_id=i,
            parent_island_id=sm["parent_island_id"],
            n_words=sm["n_words"],
            name=sm["name"],
        )
        for i, sm in enumerate(raw_sub)
    ]

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
    }
