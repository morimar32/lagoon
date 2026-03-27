"""Tests for data loading and validation."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import lagoon
from lagoon._errors import LagoonChecksumError, LagoonError, LagoonVersionError
from lagoon._loader import _default_data_dir, load_data


def test_load_default():
    """Loading with default path should succeed."""
    data = load_data()
    assert "word_lookup" in data
    assert "word_reefs" in data
    assert "reef_meta" in data
    assert "island_meta" in data
    assert "bg_mean" in data
    assert "bg_std" in data
    assert "compound_ac" in data
    assert "constants" in data
    assert "reef_edges" in data
    assert "word_reef_detail" in data
    assert "sub_reef_meta" in data
    assert "town_meta" in data
    assert "v3_reef_meta" in data
    assert "bucket_words" in data
    assert "bucket_only_word_ids" in data


def test_load_explicit_path():
    """Loading with explicit path to bundled data should succeed."""
    data = load_data(_default_data_dir())
    n_reefs = len(data["reef_meta"])
    # v3.1: 298 topical towns = lagoon reefs
    assert n_reefs == 298
    assert len(data["island_meta"]) == 40  # islands for rollup
    assert len(data["bg_mean"]) == n_reefs
    assert len(data["bg_std"]) == n_reefs


def test_word_lookup_size():
    data = load_data()
    assert len(data["word_lookup"]) > 140000


def test_word_reefs_size():
    data = load_data()
    assert len(data["word_reefs"]) > 140000


def test_reef_meta_fields():
    data = load_data()
    rm = data["reef_meta"][0]
    assert hasattr(rm, "reef_id")
    assert hasattr(rm, "hierarchy_addr")
    assert hasattr(rm, "n_words")
    assert hasattr(rm, "name")
    assert hasattr(rm, "island_id")
    assert hasattr(rm, "arch_id")
    assert rm.reef_id == 0


def test_island_meta_fields():
    data = load_data()
    im = data["island_meta"][0]
    assert hasattr(im, "island_id")
    assert hasattr(im, "arch_id")
    assert hasattr(im, "name")
    assert im.island_id == 0


def test_reef_edges():
    """reef_edges should be empty (v3 has no edge file)."""
    data = load_data()
    edges = data["reef_edges"]
    assert len(edges) == 0


def test_word_reefs_3_element():
    """word_reefs entries should be 3-element tuples (island_id, weight_q, sentinel)."""
    data = load_data()
    # Find a word with at least one reef entry
    for entries in data["word_reefs"]:
        if entries:
            assert len(entries[0]) == 3, (
                f"Expected 3-element tuples, got {len(entries[0])}-element: {entries[0]}"
            )
            break


def test_sub_reef_meta_empty():
    """sub_reef_meta should be empty — towns ARE the reefs now."""
    data = load_data()
    assert len(data["sub_reef_meta"]) == 0


def test_word_reef_detail_loaded():
    """word_reef_detail should be loaded with v3 5-element structure."""
    data = load_data()
    assert len(data["word_reef_detail"]) == len(data["word_reefs"])
    # At least some entries should have detail
    non_empty = sum(1 for entries in data["word_reef_detail"] if entries)
    assert non_empty > 0, "No words found in word_reef_detail"
    # Check structure of a non-empty entry
    for entries in data["word_reef_detail"]:
        if entries:
            assert len(entries[0]) == 5, (
                f"Expected 5-element tuples (island_id, town_id, reef_id, weight, level), "
                f"got {len(entries[0])}-element"
            )
            break


def test_town_meta_loaded():
    """town_meta should be loaded with TownMeta objects."""
    data = load_data()
    assert len(data["town_meta"]) == 298
    tm = data["town_meta"][0]
    assert hasattr(tm, "town_id")
    assert hasattr(tm, "island_id")
    assert hasattr(tm, "name")
    assert hasattr(tm, "n_words")
    assert hasattr(tm, "tqf")
    assert hasattr(tm, "avg_specificity")
    assert tm.town_id == 0


def test_v3_reef_meta_loaded():
    """v3_reef_meta should be loaded with V3ReefMeta objects."""
    data = load_data()
    assert len(data["v3_reef_meta"]) > 3000  # 3885 reefs
    rm = data["v3_reef_meta"][0]
    assert hasattr(rm, "reef_id")
    assert hasattr(rm, "town_id")
    assert hasattr(rm, "name")
    assert hasattr(rm, "n_words")
    assert hasattr(rm, "avg_specificity")
    assert rm.reef_id == 0


def test_bucket_words_loaded():
    """bucket_words should be loaded as a sparse list."""
    data = load_data()
    assert len(data["bucket_words"]) > 0
    # At least some entries should have bucket data
    non_empty = sum(1 for entries in data["bucket_words"] if entries)
    assert non_empty > 0, "No words found in bucket_words"


def test_missing_directory():
    with pytest.raises(LagoonError, match="manifest.json not found"):
        load_data("/nonexistent/path")


def test_version_mismatch():
    """Tampered version should raise LagoonVersionError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy all data files
        src = _default_data_dir()
        for f in src.iterdir():
            shutil.copy2(f, tmpdir)
        # Tamper manifest version
        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["version"] = "99.0"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        with pytest.raises(LagoonVersionError):
            load_data(tmpdir)


def test_checksum_mismatch():
    """Tampered file should raise LagoonChecksumError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = _default_data_dir()
        for f in src.iterdir():
            shutil.copy2(f, tmpdir)
        # Tamper a data file
        bad_file = Path(tmpdir) / "background.bin"
        with open(bad_file, "ab") as f:
            f.write(b"tampered")
        with pytest.raises(LagoonChecksumError):
            load_data(tmpdir)
