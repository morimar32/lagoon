"""Tests for tokenizer: compound scan + word tokenization."""


def test_single_word_lookup(scorer):
    """A known word should be found via hash lookup."""
    info = scorer.lookup_word("brain")
    assert info is not None
    assert info.word_id > 0
    assert info.specificity == -1  # universal word


def test_stemmer_fallback(scorer):
    """An inflected form should resolve via Snowball stemmer."""
    info = scorer.lookup_word("neurons")
    assert info is not None
    # Should resolve to same word_id as "neuron"
    base = scorer.lookup_word("neuron")
    assert base is not None
    assert info.word_id == base.word_id


def test_unknown_word(scorer):
    """A nonsense word should not be found."""
    info = scorer.lookup_word("xyzzyplugh")
    assert info is None


def test_compound_detection(scorer):
    """Text with a known compound should match it as a unit."""
    result = scorer.score("heart attack")
    # Should match the compound and have at least 1 matched word
    assert result.matched_words >= 1


def test_deduplication(scorer):
    """Repeated words should not increase scores (binary occurrence)."""
    single = scorer.score("cortex")
    repeated = scorer.score("cortex cortex cortex cortex")
    # Same matched word set -> same top reef results
    assert single.top_reefs[0].reef_id == repeated.top_reefs[0].reef_id
    assert abs(single.top_reefs[0].z_score - repeated.top_reefs[0].z_score) < 0.01


def test_case_insensitive(scorer):
    """Scoring should be case-insensitive."""
    lower = scorer.score("cortex")
    upper = scorer.score("CORTEX")
    mixed = scorer.score("Cortex")
    assert lower.top_reefs[0].reef_id == upper.top_reefs[0].reef_id
    assert lower.top_reefs[0].reef_id == mixed.top_reefs[0].reef_id
