"""
test_tokenizer.py
-----------------
Unit tests for the tokenizer module.
Run with: pytest tests/
"""

import pytest
from llm_cost.tokenizer import (
    count_tokens,
    get_token_ids,
    decode_tokens,
    get_token_strings,
)


class TestCountTokens:

    def test_simple_string(self):
        # "Hello, world!" should tokenize to exactly 4 tokens in cl100k_base
        result = count_tokens("Hello, world!", "cl100k_base")
        assert result == 4

    def test_empty_string(self):
        result = count_tokens("", "cl100k_base")
        assert result == 0

    def test_different_encodings_same_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        count_cl100k = count_tokens(text, "cl100k_base")
        count_o200k  = count_tokens(text, "o200k_base")
        # Both should be reasonable (not zero, not absurdly high)
        assert 5 < count_cl100k < 20
        assert 5 < count_o200k  < 20

    def test_longer_text_has_more_tokens(self):
        short = "Hello"
        long  = "Hello " * 100
        assert count_tokens(long, "cl100k_base") > count_tokens(short, "cl100k_base")


class TestRoundTrip:

    def test_encode_decode_roundtrip(self):
        text = "Tokenization is fascinating."
        encoding = "cl100k_base"
        token_ids = get_token_ids(text, encoding)
        decoded   = decode_tokens(token_ids, encoding)
        assert decoded == text

    def test_token_strings_rejoin(self):
        text   = "Hello world"
        pieces = get_token_strings(text, "cl100k_base")
        # Joining the pieces should reconstruct the original text
        assert "".join(pieces) == text


class TestTokenStrings:

    def test_returns_list(self):
        result = get_token_strings("test", "cl100k_base")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_non_english_uses_more_tokens(self):
        # Non-Latin scripts typically need many more tokens than equivalent English
        english = "Hello, how are you?"
        arabic  = "مرحبا، كيف حالك؟"
        eng_count = count_tokens(english, "cl100k_base")
        ara_count = count_tokens(arabic,  "cl100k_base")
        # Arabic should use more tokens per character due to BPE vocabulary bias
        eng_chars_per_token = len(english) / eng_count
        ara_chars_per_token = len(arabic)  / ara_count
        assert ara_chars_per_token < eng_chars_per_token