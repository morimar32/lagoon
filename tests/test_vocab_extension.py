"""Tests for vocabulary extension API: add_custom_word, filter_unknown, etc."""

import pytest

import lagoon
from lagoon._types import WordInfo


def test_next_word_id(scorer):
    """next_word_id should return the current length of _word_reefs."""
    initial = scorer.next_word_id
    assert initial > 146000  # base vocabulary


def test_filter_unknown_known_words(scorer):
    """Known words should be filtered out."""
    result = scorer.filter_unknown(["brain", "cortex", "neuron"])
    assert result == []


def test_filter_unknown_unknown_words(scorer):
    """Unknown words should be returned."""
    result = scorer.filter_unknown(["xyzzy", "plugh", "kubernetes"])
    assert "xyzzy" in result
    assert "plugh" in result
    assert "kubernetes" in result


def test_filter_unknown_stop_words(scorer):
    """Stop words should be excluded even if not in dictionary."""
    result = scorer.filter_unknown(["the", "xyzzy", "and", "kubernetes", "a"])
    assert "the" not in result
    assert "and" not in result
    assert "a" not in result
    assert "xyzzy" in result
    assert "kubernetes" in result


def test_filter_unknown_stemmed_words(scorer):
    """Stemmed forms of known words should be filtered out."""
    result = scorer.filter_unknown(["neurons", "brains"])
    assert result == []  # both resolve via Snowball stemmer


def test_compute_custom_word_scores(scorer):
    """Compute BM25 scores for a custom word with 3 reef associations."""
    idf_q, reef_scores = scorer.compute_custom_word_scores(
        n_associated_reefs=3,
        associations=[(42, 1.0), (17, 0.6), (103, 0.3)],
    )
    # IDF should be in valid u8 range
    assert 0 <= idf_q <= 255
    # With only 3 reefs, IDF should be high (specific word)
    assert idf_q > 150

    # Should have 3 reef scores
    assert len(reef_scores) == 3

    # BM25 scores should be in valid u16 range
    for reef_id, bm25_q in reef_scores:
        assert 0 <= bm25_q <= 65535
        assert reef_id in (42, 17, 103)

    # Strongest association should have highest BM25
    scores_by_reef = {r: b for r, b in reef_scores}
    assert scores_by_reef[42] >= scores_by_reef[17]
    assert scores_by_reef[17] >= scores_by_reef[103]


def test_compute_custom_word_scores_validation(scorer):
    """Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError, match="n_associated_reefs must be >= 1"):
        scorer.compute_custom_word_scores(0, [])

    with pytest.raises(ValueError, match="reef_id.*out of range"):
        scorer.compute_custom_word_scores(1, [(999, 0.5)])

    with pytest.raises(ValueError, match="strength must be in"):
        scorer.compute_custom_word_scores(1, [(42, 1.5)])


def test_add_custom_word(scorer):
    """Adding a custom word should make it immediately scorable."""
    initial_id = scorer.next_word_id

    info = scorer.add_custom_word(
        "kubernetes",
        reef_associations=[(42, 0.9), (17, 0.5)],
    )

    assert isinstance(info, WordInfo)
    assert info.word_id == initial_id
    assert info.specificity == 2  # default
    assert 0 <= info.idf_q <= 255

    # Word should now be findable
    found = scorer.lookup_word("kubernetes")
    assert found is not None
    assert found.word_id == info.word_id

    # next_word_id should have advanced
    assert scorer.next_word_id == initial_id + 1

    # Scoring text with the custom word should match it
    result = scorer.score("kubernetes")
    assert result.matched_words == 1
    assert result.coverage == 1.0
    assert "kubernetes" not in result.unknown_words


def test_add_custom_word_case_normalization(scorer):
    """Custom words should be normalized to lowercase."""
    info = scorer.add_custom_word(
        "Terraform",
        reef_associations=[(10, 0.8)],
    )
    # Should be findable as lowercase
    found = scorer.lookup_word("terraform")
    assert found is not None
    assert found.word_id == info.word_id

    # Should also be findable with original casing
    found2 = scorer.lookup_word("Terraform")
    assert found2 is not None
    assert found2.word_id == info.word_id


def test_add_custom_word_duplicate_rejected(scorer):
    """Adding a word that already exists should raise ValueError."""
    # "brain" is in the base vocabulary
    with pytest.raises(ValueError, match="already exists"):
        scorer.add_custom_word("brain", reef_associations=[(42, 0.9)])


def test_add_custom_word_empty_rejected(scorer):
    """Empty word should raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        scorer.add_custom_word("", reef_associations=[(42, 0.9)])


def test_add_custom_word_bad_specificity(scorer):
    """Invalid specificity should raise ValueError."""
    with pytest.raises(ValueError, match="specificity"):
        scorer.add_custom_word(
            "testword123",
            reef_associations=[(42, 0.9)],
            specificity=5,
        )


def test_add_custom_word_empty_associations(scorer):
    """Empty reef associations should raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        scorer.add_custom_word("testword456", reef_associations=[])


def test_add_custom_word_bad_reef_id(scorer):
    """Out-of-range reef_id should raise ValueError."""
    with pytest.raises(ValueError, match="reef_id.*out of range"):
        scorer.add_custom_word("testword789", reef_associations=[(999, 0.5)])


def test_add_custom_word_bad_strength(scorer):
    """Out-of-range strength should raise ValueError."""
    with pytest.raises(ValueError, match="strength must be in"):
        scorer.add_custom_word("testwordabc", reef_associations=[(42, 2.0)])


def test_rebuild_compounds(scorer):
    """Rebuilding compounds should allow new compound matching."""
    # Add the individual words first
    info1 = scorer.add_custom_word(
        "zorblax", reef_associations=[(42, 0.8)],
    )
    info2 = scorer.add_custom_word(
        "frimble", reef_associations=[(17, 0.7)],
    )

    # Add a compound for the pair
    compound_word = "zorblax frimble"
    compound_info = scorer.add_custom_word(
        compound_word, reef_associations=[(42, 0.95), (17, 0.85)],
    )

    # Rebuild automaton with the new compound
    scorer.rebuild_compounds([(compound_word, compound_info.word_id)])

    # Scoring the compound should match it as a single unit
    result = scorer.score("zorblax frimble")
    # Should match at least the compound
    assert result.matched_words >= 1


def test_stop_words_exported():
    """STOP_WORDS should be importable from lagoon."""
    from lagoon import STOP_WORDS
    assert isinstance(STOP_WORDS, frozenset)
    assert "the" in STOP_WORDS
    assert "and" in STOP_WORDS
    assert "a" in STOP_WORDS
    # Nonsense words should not be in stop words
    assert "kubernetes" not in STOP_WORDS
    assert "neuron" not in STOP_WORDS


def test_split_sentences_exported():
    """split_sentences should be importable from lagoon."""
    from lagoon import split_sentences
    result = split_sentences("Hello world. This is a test.")
    assert len(result) == 2


def test_reef_scorer_importable():
    """ReefScorer should be importable from lagoon."""
    # Via __getattr__
    assert lagoon.ReefScorer is not None


# -- Tag field tests --


def test_wordinfo_default_tag():
    """Default tag should be 0 on plain WordInfo construction."""
    info = WordInfo(word_hash=123, word_id=0, specificity=1, idf_q=100)
    assert info.tag == 0


def test_tag_roundtrip_add_custom_word(scorer):
    """Tag should round-trip through add_custom_word()."""
    info = scorer.add_custom_word(
        "tagword_rt",
        reef_associations=[(42, 0.8)],
        tag=7,
    )
    assert info.tag == 7


def test_tag_accessible_via_lookup(scorer):
    """Tag should be accessible via lookup_word() after add_custom_word()."""
    scorer.add_custom_word(
        "tagword_lu",
        reef_associations=[(10, 0.6)],
        tag=42,
    )
    found = scorer.lookup_word("tagword_lu")
    assert found is not None
    assert found.tag == 42


def test_get_word_tags_returns_nonzero(scorer):
    """get_word_tags() should return only non-zero tags."""
    info_tagged = scorer.add_custom_word(
        "tagword_nz1",
        reef_associations=[(42, 0.8)],
        tag=5,
    )
    info_untagged = scorer.add_custom_word(
        "tagword_nz2",
        reef_associations=[(17, 0.7)],
        tag=0,
    )
    result = scorer.get_word_tags({info_tagged.word_id, info_untagged.word_id})
    assert result == {info_tagged.word_id: 5}


def test_get_word_tags_empty_for_base_only(scorer):
    """get_word_tags() should return empty dict for base-vocabulary-only queries."""
    # Look up a known base word
    base_info = scorer.lookup_word("brain")
    assert base_info is not None
    result = scorer.get_word_tags({base_info.word_id})
    assert result == {}


def test_base_vocab_tag_is_zero(scorer):
    """Base vocabulary words should have tag 0."""
    info = scorer.lookup_word("cortex")
    assert info is not None
    assert info.tag == 0


def test_tag_does_not_affect_scoring(scorer):
    """Scoring results should be identical regardless of tag value."""
    info_no_tag = scorer.add_custom_word(
        "tagword_sc1",
        reef_associations=[(42, 0.9), (17, 0.5)],
        tag=0,
    )
    result_no_tag = scorer.score("tagword_sc1")

    info_with_tag = scorer.add_custom_word(
        "tagword_sc2",
        reef_associations=[(42, 0.9), (17, 0.5)],
        tag=99,
    )
    result_with_tag = scorer.score("tagword_sc2")

    # Same associations → same scoring behavior
    assert result_no_tag.matched_words == result_with_tag.matched_words
    assert result_no_tag.coverage == result_with_tag.coverage
    for r1, r2 in zip(result_no_tag.top_reefs, result_with_tag.top_reefs):
        assert r1.z_score == pytest.approx(r2.z_score)
