"""Tests for the scoring engine, including canonical test cases from README Section 12."""


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
    assert len(result.arch_scores) == 4
    assert isinstance(result.confidence, float)
    assert isinstance(result.coverage, float)
    assert isinstance(result.matched_words, int)
    assert isinstance(result.unknown_words, list)
    assert isinstance(result.matched_word_ids, frozenset)
    assert len(result.matched_word_ids) > 0
    for reef in result.top_reefs:
        assert hasattr(reef, "reef_id")
        assert hasattr(reef, "z_score")
        assert hasattr(reef, "raw_bm25")
        assert hasattr(reef, "n_contributing_words")
        assert hasattr(reef, "name")
        assert isinstance(reef.name, str)
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
    """Test 1: Generic text should produce low confidence."""
    result = scorer.score(
        "now is the time for all good men to come to the aid of the country"
    )
    assert result.confidence < 1.5


def test_canonical_huck_finn(scorer):
    """Test 2: Huck Finn passage - literary/cultural reefs in top results."""
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
    reef_names = [r.name for r in result.top_reefs]
    # poetic religious texts should be in top 10 (literary passage with
    # Romeo & Juliet references; propagation spreads to related literary reefs)
    assert "poetic religious texts" in reef_names
    # All z-scores should be positive and high for this rich passage
    assert result.top_reefs[0].z_score > 10.0


def test_canonical_neuroscience(scorer):
    """Test 4: Neuroscience words - raw BM25 for 'neural and structural' ~7.96."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    # All words should match (100% coverage)
    assert result.coverage == 1.0
    assert result.matched_words == 8
    # Find neural and structural in results
    neural = next(
        (r for r in result.top_reefs if r.name == "neural and structural"), None
    )
    assert neural is not None
    # Raw BM25 should be approximately 7.96 (direct + propagated signal)
    assert abs(neural.raw_bm25 - 7.96) < 0.1


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

    # The two halves should have different top-3 reef rankings
    top3_1 = [r.reef_id for r in r1.top_reefs[:3]]
    top3_2 = [r.reef_id for r in r2.top_reefs[:3]]
    assert top3_1 != top3_2, "Two halves should have different top-3 reef rankings"


def test_background_subtraction_effect(scorer):
    """Background subtraction should change rankings vs raw BM25."""
    result = scorer.score("neuron synapse axon dendrite cortex brain neural hippocampus")
    # Get raw and z-score rankings
    raw_ranking = sorted(result.top_reefs, key=lambda r: r.raw_bm25, reverse=True)
    z_ranking = result.top_reefs  # already sorted by z-score

    # Rankings should differ (background subtraction reorders)
    raw_top = raw_ranking[0].reef_id
    z_top = z_ranking[0].reef_id
    # At minimum, the raw top should differ from z-score top
    # (demonstrating background subtraction has an effect)
    raw_order = [r.reef_id for r in raw_ranking]
    z_order = [r.reef_id for r in z_ranking]
    assert raw_order != z_order


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
    """Results with min_reef_z should remain sorted by z-score descending."""
    result = scorer.score(
        "neuron synapse axon dendrite cortex brain neural hippocampus",
        min_reef_z=1.0,
    )
    for i in range(len(result.top_reefs) - 1):
        assert result.top_reefs[i].z_score >= result.top_reefs[i + 1].z_score


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
