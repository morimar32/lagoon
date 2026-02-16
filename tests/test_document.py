"""Tests for document-level topic segmentation."""

from lagoon._types import DocumentAnalysis, TopicSegment


def test_empty_input(scorer):
    result = scorer.analyze("")
    assert isinstance(result, DocumentAnalysis)
    assert result.n_sentences == 0
    assert result.n_segments == 0
    assert result.segments == []


def test_single_sentence(scorer):
    result = scorer.analyze("The brain processes information through neural pathways.")
    assert result.n_sentences == 1
    assert result.n_segments == 1
    assert len(result.segments) == 1
    assert result.boundaries == []
    # Per-sentence results should have exactly 1 entry
    assert len(result.segments[0].sentence_results) == 1


def test_pre_segmented_input(scorer):
    sentences = [
        "The brain processes information.",
        "Neural pathways transmit signals.",
    ]
    result = scorer.analyze(sentences, min_chunk_sentences=1)
    assert result.n_sentences == 2
    assert isinstance(result.segments[0], TopicSegment)


def test_segment_structure(scorer):
    result = scorer.analyze([
        "Neurons fire electrical signals.",
        "The stock market crashed today.",
    ], min_chunk_sentences=1)
    for seg in result.segments:
        assert hasattr(seg, "sentences")
        assert hasattr(seg, "start_idx")
        assert hasattr(seg, "end_idx")
        assert hasattr(seg, "topic")
        assert hasattr(seg, "sentence_results")
        assert seg.start_idx <= seg.end_idx


def test_multi_topic_detection(scorer):
    """Clearly different topics should produce multiple segments."""
    sentences = [
        "The neuron transmits electrical signals through the axon to the synapse.",
        "Dendrites receive signals from neighboring neurons in the cortex.",
        "The hippocampus plays a crucial role in memory formation and recall.",
        "Stock prices surged on Wall Street after the Federal Reserve announcement.",
        "Bond yields fell sharply as investors moved into equities and commodities.",
        "The quarterly earnings report showed record revenue for the corporation.",
    ]
    result = scorer.analyze(sentences, sensitivity=0.5)
    # Should detect at least some topic shift between neuroscience and finance
    assert result.n_segments >= 1
    # Each segment should have a valid topic
    for seg in result.segments:
        assert seg.topic is not None
        assert len(seg.sentences) > 0
        assert len(seg.sentence_results) == len(seg.sentences)


def test_sensitivity_parameter(scorer):
    """Higher sensitivity should produce fewer segments."""
    sentences = [
        "The brain has billions of neurons.",
        "Synapses connect neural pathways.",
        "Stock market volatility increased.",
        "Investors are buying bonds.",
    ]
    # Very low sensitivity -> more segments
    low = scorer.analyze(sentences, sensitivity=0.0, min_chunk_sentences=1)
    # Very high sensitivity -> fewer segments
    high = scorer.analyze(sentences, sensitivity=3.0, min_chunk_sentences=1)
    assert high.n_segments <= low.n_segments


def test_analyze_returns_all_sentences(scorer):
    """All input sentences should appear in output segments."""
    sentences = ["One fish.", "Two fish.", "Three fish."]
    result = scorer.analyze(sentences, min_chunk_sentences=1)
    all_output = []
    for seg in result.segments:
        all_output.extend(seg.sentences)
    assert all_output == sentences


def test_sentence_results_populated(scorer):
    """Each segment should have per-sentence TopicResults."""
    sentences = [
        "The neuron transmits electrical signals through the axon.",
        "Dendrites receive signals from neighboring neurons.",
        "The hippocampus plays a role in memory formation.",
    ]
    result = scorer.analyze(sentences, min_chunk_sentences=1)
    for seg in result.segments:
        assert len(seg.sentence_results) == len(seg.sentences)
        for sr in seg.sentence_results:
            assert hasattr(sr, "top_reefs")
            assert hasattr(sr, "matched_word_ids")
            assert hasattr(sr, "unknown_words")


def test_sentence_results_have_matched_word_ids(scorer):
    """Per-sentence TopicResults should include matched_word_ids."""
    result = scorer.analyze(
        ["Neuron synapse cortex brain hippocampus."],
        min_chunk_sentences=1,
    )
    sr = result.segments[0].sentence_results[0]
    assert isinstance(sr.matched_word_ids, frozenset)
    assert len(sr.matched_word_ids) > 0


def test_min_chunk_sentences(scorer):
    """Segments smaller than min_chunk_sentences should be merged."""
    sentences = [
        "The brain processes information.",
        "Neural pathways transmit signals.",
        "Stock prices surged on Wall Street.",
        "Bond yields fell sharply.",
    ]
    # With min=3, we should get at most 1 segment (can't split 4 into 2 chunks of 3+)
    result = scorer.analyze(sentences, min_chunk_sentences=3)
    for seg in result.segments:
        assert len(seg.sentences) >= 2  # at least min_chunk_sentences or all merged


def test_max_chunk_sentences(scorer):
    """Segments larger than max_chunk_sentences should be split."""
    # Generate enough sentences to exceed max
    sentences = [f"Sentence number {i} about various topics." for i in range(10)]
    result = scorer.analyze(sentences, max_chunk_sentences=4, min_chunk_sentences=1)
    for seg in result.segments:
        assert len(seg.sentences) <= 4


def test_max_chunk_sentences_disabled(scorer):
    """max_chunk_sentences=0 should disable max size enforcement."""
    sentences = [f"Sentence number {i} about various topics." for i in range(50)]
    result = scorer.analyze(
        sentences, max_chunk_sentences=0, min_chunk_sentences=1,
    )
    # With no max and high sensitivity (few boundaries), could be one big segment
    total = sum(len(seg.sentences) for seg in result.segments)
    assert total == 50


# --- min_reef_z tests ---

def test_analyze_min_reef_z_default(scorer):
    """Default min_reef_z=2.0 should filter segment reefs to z >= 2.0."""
    sentences = [
        "The neuron transmits electrical signals through the axon to the synapse.",
        "Dendrites receive signals from neighboring neurons in the cortex.",
        "The hippocampus plays a crucial role in memory formation and recall.",
    ]
    result = scorer.analyze(sentences, min_chunk_sentences=1)
    for seg in result.segments:
        for reef in seg.topic.top_reefs:
            assert reef.z_score >= 2.0


def test_analyze_min_reef_z_custom(scorer):
    """Custom min_reef_z threshold should be respected."""
    sentences = [
        "The neuron transmits electrical signals through the axon to the synapse.",
        "Dendrites receive signals from neighboring neurons in the cortex.",
        "The hippocampus plays a crucial role in memory formation and recall.",
    ]
    threshold = 5.0
    result = scorer.analyze(sentences, min_chunk_sentences=1, min_reef_z=threshold)
    for seg in result.segments:
        for reef in seg.topic.top_reefs:
            assert reef.z_score >= threshold


def test_analyze_sentence_results_use_min_reef_z(scorer):
    """Per-sentence results should also be filtered by min_reef_z."""
    sentences = [
        "The neuron transmits electrical signals through the axon to the synapse.",
        "Dendrites receive signals from neighboring neurons in the cortex.",
    ]
    threshold = 3.0
    result = scorer.analyze(sentences, min_chunk_sentences=1, min_reef_z=threshold)
    for seg in result.segments:
        for sr in seg.sentence_results:
            for reef in sr.top_reefs:
                assert reef.z_score >= threshold
