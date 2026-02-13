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


def test_pre_segmented_input(scorer):
    sentences = [
        "The brain processes information.",
        "Neural pathways transmit signals.",
    ]
    result = scorer.analyze(sentences)
    assert result.n_sentences == 2
    assert isinstance(result.segments[0], TopicSegment)


def test_segment_structure(scorer):
    result = scorer.analyze([
        "Neurons fire electrical signals.",
        "The stock market crashed today.",
    ])
    for seg in result.segments:
        assert hasattr(seg, "sentences")
        assert hasattr(seg, "start_idx")
        assert hasattr(seg, "end_idx")
        assert hasattr(seg, "topic")
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


def test_sensitivity_parameter(scorer):
    """Higher sensitivity should produce fewer segments."""
    sentences = [
        "The brain has billions of neurons.",
        "Synapses connect neural pathways.",
        "Stock market volatility increased.",
        "Investors are buying bonds.",
    ]
    # Very low sensitivity -> more segments
    low = scorer.analyze(sentences, sensitivity=0.0)
    # Very high sensitivity -> fewer segments
    high = scorer.analyze(sentences, sensitivity=3.0)
    assert high.n_segments <= low.n_segments


def test_analyze_returns_all_sentences(scorer):
    """All input sentences should appear in output segments."""
    sentences = ["One.", "Two.", "Three."]
    result = scorer.analyze(sentences)
    all_output = []
    for seg in result.segments:
        all_output.extend(seg.sentences)
    assert all_output == sentences
