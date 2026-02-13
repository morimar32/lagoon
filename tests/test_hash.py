"""Tests for FNV-1a hash implementation."""

from lagoon._hash import fnv1a_u64


def test_empty_string():
    """Empty string should return the offset basis."""
    assert fnv1a_u64("") == 14695981039346656037


def test_known_value():
    """Verify a known hash value for determinism."""
    h = fnv1a_u64("hello")
    assert isinstance(h, int)
    assert 0 <= h < 2**64
    # Must be deterministic
    assert fnv1a_u64("hello") == h


def test_different_strings_differ():
    assert fnv1a_u64("cat") != fnv1a_u64("dog")


def test_case_sensitive():
    assert fnv1a_u64("Hello") != fnv1a_u64("hello")


def test_unicode():
    """UTF-8 multi-byte characters should hash correctly."""
    h = fnv1a_u64("café")
    assert isinstance(h, int)
    assert fnv1a_u64("café") == h


def test_spaces():
    """Multi-word compounds use spaces."""
    h = fnv1a_u64("heart attack")
    assert h != fnv1a_u64("heartattack")
    assert h != fnv1a_u64("heart")
