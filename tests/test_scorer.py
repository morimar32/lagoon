"""Tests for the scoring engine, including canonical test cases from README Section 12."""

import pytest


def test_empty_string(scorer):
    """Empty input should return zero-confidence result."""
    result = scorer.score("")
    assert result.confidence == 0.0
    assert result.matched_words == 0
    assert result.top_reefs == []
    assert result.matched_word_ids == frozenset()


def test_all_unknown_words(scorer):
    """All-unknown input should return zero-confidence result."""
    result = scorer.score("xyzzy plugh qwerty asdfgh")
    assert result.confidence == 0.0
    assert result.matched_words == 0
    assert len(result.unknown_words) == 4
    assert result.coverage == 0.0
    assert result.matched_word_ids == frozenset()


def test_single_word(scorer):
    """A single known word should produce a result with matched_words=1."""
    result = scorer.score("cortex")
    assert result.matched_words == 1
    assert result.coverage == 1.0
    assert len(result.top_reefs) > 0
    assert len(result.matched_word_ids) == 1


def test_result_structure(scorer):
    """TopicResult should have correct structure."""
    result = scorer.score("neuron cortex brain")
    assert len(result.arch_scores) >= 5
    assert isinstance(result.confidence, float)
    assert isinstance(result.coverage, float)
    assert isinstance(result.matched_words, int)
    assert isinstance(result.unknown_words, list)
    assert isinstance(result.matched_word_ids, frozenset)
    assert len(result.matched_word_ids) > 0
    assert isinstance(result.valence_signal, float)
    for reef in result.top_reefs:
        assert hasattr(reef, "reef_id")
        assert hasattr(reef, "z_score")
        assert hasattr(reef, "raw_score")
        assert hasattr(reef, "n_contributing_words")
        assert hasattr(reef, "name")
        assert isinstance(reef.name, str)
        assert isinstance(reef.quality_score, float)
        assert isinstance(reef.valence, float)
        assert isinstance(reef.avg_specificity, float)
    for island in result.top_islands:
        assert hasattr(island, "island_id")
        assert hasattr(island, "aggregate_z")
        assert hasattr(island, "n_contributing_reefs")
        assert hasattr(island, "name")


def test_score_batch(scorer):
    """score_batch should return one result per input."""
    texts = ["neuron cortex", "ocean waves", "market stock"]
    results = scorer.score_batch(texts)
    assert len(results) == 3
    for r in results:
        assert hasattr(r, "top_reefs")
        assert hasattr(r, "matched_word_ids")


def test_top_k(scorer):
    """top_k parameter should limit number of returned reefs."""
    result = scorer.score("neuron synapse axon dendrite cortex brain", top_k=5)
    assert len(result.top_reefs) == 5


# --- Canonical test cases (README Section 12) ---

def test_canonical_generic_text(scorer):
    """Test 1: Generic text should produce a valid result with positive coverage."""
    result = scorer.score(
        "now is the time for all good men to come to the aid of the country"
    )
    assert result.coverage > 0.0
    assert result.matched_words > 0
    assert len(result.top_reefs) > 0


def test_canonical_huck_finn(scorer):
    """Test 2: Huck Finn passage - high confidence, strong z-scores."""
    huck = (
        'It was after sun-up now, but we went right on and didn\'t tie up. '
        'The king and the duke turned out by-and-by looking pretty rusty; '
        "but after they'd jumped overboard and took a swim it chippered them "
        'up a good deal. After breakfast the king he took a seat on the corner '
        'of the raft, and pulled off his boots and rolled up his britches, and '
        'let his legs dangle in the water, so as to be comfortable, and lit '
        'his pipe, and went to getting his Romeo and Juliet by heart. When he '
        'had got it pretty good him and the duke begun to practice it together. '
        'The duke had to learn him over and over again how to say every speech; '
        'and he made him sigh, and put his hand on his heart, and after a while '
        'he said he done it pretty well; "only," he says, "you mustn\'t bellow '
        'out Romeo! that way, like a bull -- you must say it soft and sick and '
        'languishy, so -- R-o-o-meo! that is the idea; for Juliet\'s a dear '
        'sweet mere child of a girl, you know, and she doesn\'t bray like a '
        'jackass."'
    )
    result = scorer.score(huck)
    # This rich literary passage should produce high confidence and strong z-scores
    assert result.confidence > 3.0
    assert result.top_reefs[0].z_score > 5.0


def test_canonical_neuroscience(scorer):
    """Test 4: Neuroscience words - full coverage, neurological reef in top results."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    # All words should match (100% coverage)
    assert result.coverage == 1.0
    assert result.matched_words == 8
    # A biology/neuroscience-related reef should appear in top results
    bio_keywords = {"biolog", "organism", "anatom", "neural", "microscop", "life science", "neuro"}
    neuro = next(
        (r for r in result.top_reefs
         if any(kw in r.name.lower() for kw in bio_keywords)),
        None,
    )
    assert neuro is not None, (
        f"No biology-related reef in top results: "
        f"{[r.name for r in result.top_reefs]}"
    )


def test_canonical_topic_shift(scorer):
    """Test 5: Two halves of Huck Finn should produce divergent z-score distributions."""
    huck = (
        'It was after sun-up now, but we went right on and didn\'t tie up. '
        'The king and the duke turned out by-and-by looking pretty rusty; '
        "but after they'd jumped overboard and took a swim it chippered them "
        'up a good deal. After breakfast the king he took a seat on the corner '
        'of the raft, and pulled off his boots and rolled up his britches, and '
        'let his legs dangle in the water, so as to be comfortable, and lit '
        'his pipe, and went to getting his Romeo and Juliet by heart. When he '
        'had got it pretty good him and the duke begun to practice it together. '
        'The duke had to learn him over and over again how to say every speech; '
        'and he made him sigh, and put his hand on his heart, and after a while '
        'he said he done it pretty well; "only," he says, "you mustn\'t bellow '
        'out Romeo! that way, like a bull -- you must say it soft and sick and '
        'languishy, so -- R-o-o-meo! that is the idea; for Juliet\'s a dear '
        'sweet mere child of a girl, you know, and she doesn\'t bray like a '
        'jackass."'
    )
    words = huck.split()
    mid = len(words) // 2
    first_half = " ".join(words[:mid])
    second_half = " ".join(words[mid:])

    r1 = scorer.score(first_half)
    r2 = scorer.score(second_half)

    # The two halves should have different top-5 reef rankings
    top5_1 = [r.reef_id for r in r1.top_reefs[:5]]
    top5_2 = [r.reef_id for r in r2.top_reefs[:5]]
    assert top5_1 != top5_2, "Two halves should have different top-5 reef rankings"


def test_single_word_no_bg_subtraction(scorer):
    """Single matched word should use alpha=0 (no mean subtraction)."""
    result = scorer.score("cortex")
    # With 1 matched word, alpha=0: z = raw / bg_std (no bg_mean subtraction)
    # Top reefs should have positive z_scores from the word's reef weights
    assert len(result.top_reefs) > 0
    assert result.top_reefs[0].z_score > 0


# --- Stop word filtering ---

def test_stop_words_filtered_from_unknown(scorer):
    """Stop words should not appear in unknown_words even if not in dictionary."""
    # Mix of unknown nonsense + stop words that might not be in dictionary
    result = scorer.score("xyzzy the plugh and qwerty")
    # "the" and "and" are stop words — should not be in unknown_words
    for w in result.unknown_words:
        assert w not in ("the", "and", "a", "is")


# --- matched_word_ids ---

def test_matched_word_ids_consistency(scorer):
    """matched_word_ids should have exactly matched_words unique IDs."""
    result = scorer.score("neuron cortex brain")
    assert len(result.matched_word_ids) == result.matched_words


# --- min_reef_z tests ---

def test_min_reef_z_overrides_top_k(scorer):
    """All returned reefs should have z >= threshold when min_reef_z is set."""
    threshold = 2.0
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        min_reef_z=threshold,
    )
    for reef in result.top_reefs:
        assert reef.z_score >= threshold


def test_min_reef_z_empty_result(scorer):
    """Very high threshold should produce empty top_reefs."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        min_reef_z=999.0,
    )
    assert result.top_reefs == []


def test_min_reef_z_none_preserves_top_k(scorer):
    """Default None keeps top-k behavior."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
    )
    assert len(result.top_reefs) == 10


def test_min_reef_z_reefs_sorted_descending(scorer):
    """Results with min_reef_z should remain sorted by quality_score descending."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        min_reef_z=1.0,
    )
    for i in range(len(result.top_reefs) - 1):
        assert result.top_reefs[i].quality_score >= result.top_reefs[i + 1].quality_score


def test_min_reef_z_island_rollup_matches_selected(scorer):
    """Island contributing count should equal len(top_reefs)."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        min_reef_z=2.0,
    )
    total_contributing = sum(isl.n_contributing_reefs for isl in result.top_islands)
    assert total_contributing == len(result.top_reefs)


def test_score_batch_with_min_reef_z(scorer):
    """Batch should pass through min_reef_z threshold."""
    threshold = 3.0
    results = scorer.score_batch(
        ["neuron synapse cortex brain", "ocean waves tide current"],
        min_reef_z=threshold,
    )
    for result in results:
        for reef in result.top_reefs:
            assert reef.z_score >= threshold


def test_min_reef_z_backward_compat(scorer):
    """score() without min_reef_z should still return exactly 10 reefs."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    assert len(result.top_reefs) == 10


def test_tiny_reefs_no_absurd_z_scores(scorer):
    """Tiny reefs (n_words < 100) must not produce absurd z-scores.

    Small reefs have unreliable background estimates (bg_std near zero
    because random samples almost never hit them).  The scorer inflates
    bg_std for tiny reefs proportionally to the size gap so that z-scores
    stay in a sane range relative to normal-sized reefs.
    """
    # Use a broad query with common words that might hit tiny reefs
    result = scorer.score(
        "good best better most well great very really",
        top_k=124,
    )
    # Tiny reefs (n_words < 100) should not produce z > 110 — this guards
    # against the background model mismatch that previously caused extreme
    # z-scores for tiny reefs.  After noise cleanup, bg_std is smaller for
    # niche reefs, so legitimate queries that perfectly match a small reef
    # (e.g. "degree and intensity") can reach z ~ 103.
    for reef in result.top_reefs:
        nw = scorer._reef_n_words[reef.reef_id]
        if nw < 100:
            assert reef.z_score < 110.0, (
                f"reef {reef.reef_id} ({reef.name}): z_score={reef.z_score:.1f} "
                f"exceeds sane maximum for tiny reef (n_words={nw})"
            )


# --- Quality score tests ---

def test_quality_score_present(scorer):
    """quality_score should be populated on ScoredReef."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    for reef in result.top_reefs:
        assert isinstance(reef.quality_score, float)
        # Top reefs with positive z should have positive quality_score
        if reef.z_score > 0:
            assert reef.quality_score > 0


def test_quality_score_ranking(scorer):
    """top_reefs should be sorted by quality_score descending."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    for i in range(len(result.top_reefs) - 1):
        assert result.top_reefs[i].quality_score >= result.top_reefs[i + 1].quality_score


def test_quality_score_equals_z_score(scorer):
    """quality_score should equal z_score (specificity baked into background)."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        top_k=20,
    )
    for r in result.top_reefs:
        assert r.quality_score == r.z_score


def test_valence_signal_present(scorer):
    """valence_signal should be a float on TopicResult."""
    result = scorer.score("neuron synapse axon dendrite cortex brain")
    assert isinstance(result.valence_signal, float)


def test_valence_signal_meaningful(scorer):
    """valence_signal should be a numeric value (may be 0.0 if valence data not populated)."""
    huck = (
        'It was after sun-up now, but we went right on and didn\'t tie up. '
        'The king and the duke turned out by-and-by looking pretty rusty; '
        "but after they'd jumped overboard and took a swim it chippered them "
        'up a good deal.'
    )
    medical = (
        "The patient presented with acute myocardial infarction and was "
        "treated with thrombolytic therapy and anticoagulant medication."
    )
    r_huck = scorer.score(huck)
    r_med = scorer.score(medical)
    # Both should produce numeric valence signals
    assert isinstance(r_huck.valence_signal, float)
    assert isinstance(r_med.valence_signal, float)


def test_quality_score_equals_z_for_all_reefs(scorer):
    """quality_score should equal z_score for all reefs (specificity baked into bg_std)."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        top_k=124,  # get all reefs
    )
    for reef in result.top_reefs:
        assert reef.quality_score == reef.z_score, (
            f"reef {reef.reef_id}: quality_score={reef.quality_score} "
            f"should equal z_score={reef.z_score}"
        )
