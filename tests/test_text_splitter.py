"""Unit tests for TextSplitter – no external deps required."""
import pytest
from utils.text_splitter import TextSplitter


def test_short_text_not_split():
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    result = splitter.split("Hello world")
    assert result == ["Hello world"]


def test_splits_long_text():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    long = "sentence one. sentence two. sentence three. sentence four. sentence five."
    chunks = splitter.split(long)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50 + 10 + 5  # slight tolerance


def test_empty_text_returns_empty():
    splitter = TextSplitter()
    assert splitter.split("") == []
    assert splitter.split("   ") == []


def test_overlap_present():
    splitter = TextSplitter(chunk_size=30, chunk_overlap=10)
    text = "abcdefghij " * 20
    chunks = splitter.split(text)
    if len(chunks) > 1:
        # The tail of chunk[0] should appear in the start of chunk[1]
        tail = chunks[0][-10:]
        assert tail in chunks[1]
