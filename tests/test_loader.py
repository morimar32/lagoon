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


def test_load_explicit_path():
    """Loading with explicit path to bundled data should succeed."""
    data = load_data(_default_data_dir())
    n_reefs = len(data["reef_meta"])
    # After coral island promotion, gen-1 count should exceed the original 68
    assert n_reefs > 68, f"Expected more than 68 reefs after coral promotion, got {n_reefs}"
    assert len(data["island_meta"]) == n_reefs
    assert len(data["bg_mean"]) == n_reefs
    assert len(data["bg_std"]) == n_reefs


def test_word_lookup_size():
    data = load_data()
    assert len(data["word_lookup"]) > 160000


def test_word_reefs_size():
    data = load_data()
    assert len(data["word_reefs"]) > 146000


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
    """reef_edges should be empty (island-level export skips edge propagation)."""
    data = load_data()
    edges = data["reef_edges"]
    assert len(edges) == 0


def test_word_reefs_3_element():
    """word_reefs entries should be 3-element tuples (reef_id, weight_q, sub_reef_id)."""
    data = load_data()
    # Find a word with at least one reef entry
    for entries in data["word_reefs"]:
        if entries:
            assert len(entries[0]) == 3, (
                f"Expected 3-element tuples, got {len(entries[0])}-element: {entries[0]}"
            )
            break


def test_sub_reef_meta_loaded():
    """sub_reef_meta should be loaded and non-empty."""
    data = load_data()
    assert len(data["sub_reef_meta"]) > 0
    sm = data["sub_reef_meta"][0]
    assert hasattr(sm, "sub_reef_id")
    assert hasattr(sm, "parent_island_id")
    assert hasattr(sm, "n_words")
    assert hasattr(sm, "name")
    assert sm.sub_reef_id == 0


def test_word_reef_detail_loaded():
    """word_reef_detail should be loaded with correct structure."""
    data = load_data()
    assert len(data["word_reef_detail"]) == len(data["word_reefs"])
    # At least some entries should have detail (multi-reef words)
    non_empty = sum(1 for entries in data["word_reef_detail"] if entries)
    assert non_empty > 0, "No multi-reef words found in word_reef_detail"
    # Check structure of a non-empty entry
    for entries in data["word_reef_detail"]:
        if entries:
            assert len(entries[0]) == 3, (
                f"Expected 3-element tuples (island_id, sub_reef_id, weight_q), "
                f"got {len(entries[0])}-element"
            )
            break


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
