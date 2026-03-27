"""Tests for lookup data loading and integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import msgpack
import pytest

from lagoon._hash import fnv1a_u64
from lagoon._loader import _i64_to_u64, load_lookup
from lagoon._profiler import Profiler
from lagoon._tokenizer import Tokenizer
from lagoon._types import LookupData, TextProfile, WordInfo


# -- i64/u64 conversion ---------------------------------------------------

def test_i64_to_u64_positive():
    assert _i64_to_u64(42) == 42

def test_i64_to_u64_zero():
    assert _i64_to_u64(0) == 0

def test_i64_to_u64_negative():
    # Signed -1 → unsigned 2^64 - 1
    assert _i64_to_u64(-1) == (1 << 64) - 1

def test_i64_to_u64_min_i64():
    # Signed -2^63 → unsigned 2^63
    assert _i64_to_u64(-(1 << 63)) == 1 << 63


# -- load_lookup with missing directory ------------------------------------

def test_load_lookup_missing_dir():
    """Missing directory returns empty LookupData, no error."""
    data = load_lookup("/tmp/nonexistent_lagoon_lookup_dir_xyz")
    assert data.equivalences == {}
    assert data.word_tags == {}
    assert data.names == frozenset()


# -- load_lookup with real files -------------------------------------------

def _write_lookup_files(tmpdir: Path) -> None:
    """Write minimal lookup files for testing."""
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Equivalences: "geese" (signed hash) → word_id 42 (goose)
    # Use a signed i64 value to test conversion
    equiv_pairs = [[-123456789, 42], [987654321, 99]]
    with open(tmpdir / "equivalences.bin", "wb") as f:
        f.write(msgpack.packb(equiv_pairs, use_bin_type=True))

    # Word tags: word → [[tag_type, tag_value], ...]
    tags = {"gonna": [["register", "informal"]], "colour": [["regional", "british"]]}
    with open(tmpdir / "word_tags.bin", "wb") as f:
        f.write(msgpack.packb(tags, use_bin_type=True))

    # Names: [[name, type], ...]
    names = [["John", "male"], ["Alice", "female"], ["Pat", "neutral"]]
    with open(tmpdir / "names.bin", "wb") as f:
        f.write(msgpack.packb(names, use_bin_type=True))

    # Manifest (not validated by load_lookup, but let's be complete)
    import json
    with open(tmpdir / "manifest.json", "w") as f:
        json.dump({"format": "windowsill-lookup", "version": "1.0"}, f)


def test_load_lookup_equivalences():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "lookup"
        _write_lookup_files(p)
        data = load_lookup(p)

        # Negative i64 should be converted to u64
        u64_key = _i64_to_u64(-123456789)
        assert u64_key in data.equivalences
        assert data.equivalences[u64_key] == [42]

        # Positive value should be unchanged
        assert 987654321 in data.equivalences
        assert data.equivalences[987654321] == [99]


def test_load_lookup_multi_target():
    """Multiple equivalences with same hash produce a list of word_ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "lookup"
        p.mkdir(parents=True)

        # Two pairs with same hash → different word_ids
        pairs = [[100, 5], [100, 10]]
        with open(p / "equivalences.bin", "wb") as f:
            f.write(msgpack.packb(pairs, use_bin_type=True))

        data = load_lookup(p)
        assert data.equivalences[100] == [5, 10]


def test_load_lookup_word_tags():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "lookup"
        _write_lookup_files(p)
        data = load_lookup(p)
        assert "gonna" in data.word_tags
        assert data.word_tags["gonna"] == [["register", "informal"]]


def test_load_lookup_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "lookup"
        _write_lookup_files(p)
        data = load_lookup(p)
        assert "john" in data.names  # lowercased
        assert "alice" in data.names
        assert "pat" in data.names


# -- Tokenizer equivalence fallback ---------------------------------------

def _make_tokenizer_with_equiv(
    vocab: dict[str, int],
    equivalences: dict[int, list[int]],
) -> Tokenizer:
    """Create a minimal tokenizer with a vocabulary and equivalences."""
    import ahocorasick

    word_lookup: dict[int, WordInfo] = {}
    for word, wid in vocab.items():
        h = fnv1a_u64(word)
        word_lookup[h] = WordInfo(word_hash=h, word_id=wid, specificity=0, idf_q=128)

    ac = ahocorasick.Automaton()
    return Tokenizer(word_lookup, ac, [], [], equivalences=equivalences)


def test_tokenizer_equiv_fallback():
    """Equivalences resolve unknown inflections to known word_ids."""
    goose_hash = fnv1a_u64("goose")
    geese_hash = fnv1a_u64("geese")

    tok = _make_tokenizer_with_equiv(
        vocab={"goose": 1},
        equivalences={geese_hash: [1]},  # geese → goose (word_id=1)
    )

    matched, unknown = tok.process("geese fly south")
    assert 1 in matched  # resolved via equivalence
    assert "geese" not in unknown


def test_tokenizer_equiv_multi_target():
    """Multi-target equivalence adds all word_ids to matched set."""
    leaves_hash = fnv1a_u64("leaves")

    tok = _make_tokenizer_with_equiv(
        vocab={"leaf": 1, "leave": 2},
        equivalences={leaves_hash: [1, 2]},  # leaves → leaf AND leave
    )

    matched, unknown = tok.process("leaves fall")
    assert 1 in matched  # leaf
    assert 2 in matched  # leave
    assert "leaves" not in unknown


def test_tokenizer_equiv_ordered():
    """process_ordered uses first equiv word_id for position."""
    leaves_hash = fnv1a_u64("leaves")

    tok = _make_tokenizer_with_equiv(
        vocab={"leaf": 1, "leave": 2},
        equivalences={leaves_hash: [1, 2]},
    )

    ordered, matched_set, unknown, total = tok.process_ordered("leaves fall")
    # First word_id in equiv list used for ordered position
    assert ordered[0] == 1
    # Both word_ids in matched set
    assert matched_set == {1, 2}
    assert total == 2


def test_tokenizer_direct_lookup_takes_priority():
    """Direct hash match is preferred over equivalence."""
    word_hash = fnv1a_u64("test")

    tok = _make_tokenizer_with_equiv(
        vocab={"test": 5},
        equivalences={word_hash: [99]},  # should NOT be used
    )

    matched, unknown = tok.process("test")
    assert 5 in matched
    assert 99 not in matched


# -- Profiler name detection -----------------------------------------------

def test_profiler_name_detection():
    """Names detected from capitalized tokens in text."""
    lookup = LookupData(
        equivalences={},
        word_tags={},
        names=frozenset({"john", "alice", "will"}),
    )

    # "Will" is a name but also a common word — capitalization required
    text = "John met Alice at the park but will not stay"
    profiler = _make_minimal_profiler(lookup)
    names = profiler._detect_names(text)

    assert "john" in names
    assert "alice" in names
    # "will" is lowercase in text, should NOT be detected
    assert "will" not in names


def test_profiler_name_detection_dedup():
    """Same name mentioned twice is only returned once."""
    lookup = LookupData(
        equivalences={},
        word_tags={},
        names=frozenset({"john"}),
    )
    profiler = _make_minimal_profiler(lookup)
    names = profiler._detect_names("John met John again")
    assert names.count("john") == 1


def test_profiler_no_lookup():
    """Profiler works fine without lookup data."""
    profiler = _make_minimal_profiler(None)
    names = profiler._detect_names("John met Alice")
    assert names == []


def _make_minimal_profiler(lookup: LookupData | None) -> Profiler:
    """Create a minimal profiler for name detection tests."""
    # We need at least one scorer, but name detection doesn't use it.
    # Use the real loaded scorer.
    import lagoon
    scorer = lagoon.load()
    return Profiler(lenses={"domain": scorer}, lookup=lookup)


# -- Profiler.analyze() ---------------------------------------------------

def test_profiler_analyze_basic():
    """Profiler.analyze() produces DocumentProfile with TextProfiles per segment."""
    import lagoon

    scorer = lagoon.load()
    profiler = Profiler(lenses={"domain": scorer})

    text = (
        "Quantum computing uses qubits to process information. "
        "Superposition allows parallel computation in quantum systems. "
        "The heart pumps blood through arteries and veins. "
        "Cardiac surgery requires careful monitoring of vital signs."
    )
    doc = profiler.analyze(text)

    assert doc.n_sentences > 0
    assert doc.n_segments > 0
    assert len(doc.segments) == doc.n_segments

    for seg in doc.segments:
        # Each segment has a TextProfile, not a bare TopicResult
        assert isinstance(seg.profile, TextProfile)
        assert "domain" in seg.profile.lenses
        assert seg.profile.register is not None
        assert isinstance(seg.sentences, list)
        assert seg.start_idx >= 0
        assert seg.end_idx >= seg.start_idx


def test_profiler_analyze_multi_lens():
    """Profiler.analyze() scores segments against all lenses."""
    import lagoon
    from pathlib import Path

    domain = lagoon.load()
    human_dir = Path(lagoon.__file__).parent / "data_human"
    if not human_dir.exists():
        pytest.skip("human lens data not available")

    human = lagoon.load(str(human_dir))
    profiler = Profiler(lenses={"domain": domain, "human": human})

    text = (
        "The doctor examined the patient carefully. "
        "Blood pressure was elevated and heart rate was irregular."
    )
    doc = profiler.analyze(text)
    assert doc.n_segments >= 1

    seg = doc.segments[0]
    assert "domain" in seg.profile.lenses
    assert "human" in seg.profile.lenses


def test_profiler_analyze_invalid_lens():
    """Requesting segmentation from a non-existent lens raises ValueError."""
    import lagoon

    scorer = lagoon.load()
    profiler = Profiler(lenses={"domain": scorer})

    with pytest.raises(ValueError, match="segmentation_lens"):
        profiler.analyze("test", segmentation_lens="nonexistent")


def test_profiler_analyze_empty_text():
    """Empty text produces empty DocumentProfile."""
    import lagoon

    scorer = lagoon.load()
    profiler = Profiler(lenses={"domain": scorer})

    doc = profiler.analyze("")
    assert doc.n_sentences == 0
    assert doc.n_segments == 0
    assert doc.segments == []
