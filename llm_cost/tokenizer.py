"""
tokenizer.py
------------
Wraps tiktoken to count tokens for any given text and model encoding.
tiktoken uses BPE (Byte Pair Encoding) under the hood — the same
tokenizer OpenAI uses in production.
"""

import tiktoken


# Maps each encoding name to the tiktoken encoder object.
# We cache these so we don't reload them on every call.
_encoder_cache: dict = {}


def get_encoder(encoding_name: str) -> tiktoken.Encoding:
    """
    Load and cache a tiktoken encoder by encoding name.
    Example encoding names: 'cl100k_base', 'o200k_base', 'p50k_base'
    """
    if encoding_name not in _encoder_cache:
        _encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _encoder_cache[encoding_name]


def count_tokens(text: str, encoding_name: str) -> int:
    """
    Count the number of tokens in `text` using the given encoding.

    Args:
        text:          The input string to tokenize.
        encoding_name: The BPE encoding to use (e.g. 'cl100k_base').

    Returns:
        Integer token count.

    Example:
        >>> count_tokens("Hello, world!", "cl100k_base")
        4
    """
    encoder = get_encoder(encoding_name)
    tokens = encoder.encode(text)
    return len(tokens)


def get_token_ids(text: str, encoding_name: str) -> list[int]:
    """
    Return the actual token ID list for `text`.
    Useful for debugging and visualization.

    Example:
        >>> get_token_ids("Hello!", "cl100k_base")
        [9906, 0]
    """
    encoder = get_encoder(encoding_name)
    return encoder.encode(text)


def decode_tokens(token_ids: list[int], encoding_name: str) -> str:
    """
    Decode a list of token IDs back into a string.
    Useful for verifying round-trip tokenization.
    """
    encoder = get_encoder(encoding_name)
    return encoder.decode(token_ids)


def get_token_strings(text: str, encoding_name: str) -> list[str]:
    """
    Return each token as a human-readable string.
    Great for showing users exactly how their text is split.

    Example:
        >>> get_token_strings("tokenization", "cl100k_base")
        ['token', 'ization']
    """
    encoder = get_encoder(encoding_name)
    token_ids = encoder.encode(text)
    # Decode each token individually to show the split
    return [encoder.decode([tid]) for tid in token_ids]