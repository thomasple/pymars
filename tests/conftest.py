"""
Pytest configuration and shared fixtures for molfrag tests.
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture that provides the path to the test data directory."""
    return Path(__file__).parent