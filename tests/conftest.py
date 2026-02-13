"""Shared fixtures for lagoon tests."""

import pytest

import lagoon


@pytest.fixture(scope="session")
def scorer():
    """Load the scorer once for all tests."""
    return lagoon.load()
