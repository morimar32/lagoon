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


def test_load_explicit_path():
    """Loading with explicit path to bundled data should succeed."""
    data = load_data(_default_data_dir())
    assert len(data["reef_meta"]) == 207
    assert len(data["island_meta"]) == 52
    assert len(data["bg_mean"]) == 207
    assert len(data["bg_std"]) == 207


def test_word_lookup_size():
    data = load_data()
    assert len(data["word_lookup"]) > 170000


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
    """reef_edges should have 3925 valid (src, tgt, weight) entries."""
    data = load_data()
    edges = data["reef_edges"]
    assert len(edges) == 3925
    n_reefs = len(data["reef_meta"])
    for src, tgt, weight in edges:
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert isinstance(weight, float)
        assert 0 <= src < n_reefs
        assert 0 <= tgt < n_reefs
        assert weight > 0.0


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
