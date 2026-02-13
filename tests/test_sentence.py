"""Tests for sentence splitter."""

from lagoon._sentence import split_sentences


def test_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence():
    result = split_sentences("Hello world.")
    assert len(result) == 1


def test_two_sentences():
    result = split_sentences("Hello world. This is a test.")
    assert len(result) == 2


def test_question_mark():
    result = split_sentences("What is this? It is a test.")
    assert len(result) == 2


def test_exclamation():
    result = split_sentences("Stop! That is enough.")
    assert len(result) == 2


def test_abbreviation_mr():
    """Mr. should not trigger a split."""
    result = split_sentences("Mr. Smith went to Washington. He was happy.")
    assert len(result) == 2
    assert "Mr" in result[0]


def test_abbreviation_dr():
    """Dr. should not trigger a split."""
    result = split_sentences("Dr. Jones is here. She is busy.")
    assert len(result) == 2


def test_no_capital_after_period():
    """Period followed by lowercase should not split."""
    result = split_sentences("The value is 3.14 approximately.")
    assert len(result) == 1


def test_multiline():
    text = """The neural pathways connect through axons.
    Synaptic transmission occurs at junctions. The market surged today."""
    result = split_sentences(text)
    assert len(result) >= 2
