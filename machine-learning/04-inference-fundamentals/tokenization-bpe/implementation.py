"""
Byte Pair Encoding (BPE) Tokenizer -- From-scratch Python implementation.

Implements the subword tokenization algorithm used by GPT-2, GPT-4, LLaMA, and
most modern LLMs. Starting from a base vocabulary of 256 byte tokens, BPE
iteratively merges the most frequent adjacent pairs to build a vocabulary of
subword units. The byte-level base ensures any UTF-8 text can be encoded without
an unknown token -- rare characters simply fall back to their raw byte
representation. This module provides training (learning merge rules from a
corpus), encoding (text to token IDs), decoding (token IDs back to text),
special token support, and JSON serialization.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# GPT-2 style pre-tokenization pattern using standard re syntax.
# Matches contractions, optional-space + letters, optional-space + digits,
# optional-space + punctuation, trailing whitespace, and other whitespace.
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z\u00C0-\u024F\u0370-\u03FF"""
    r"""\u0400-\u04FF\u0500-\u052F\u0600-\u06FF\u0900-\u097F"""
    r"""\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF"""
    r"""\uAC00-\uD7AF\u1100-\u11FF\uA000-\uA4CF\u1000-\u109F"""
    r"""\u10A0-\u10FF\u0530-\u058F\u0590-\u05FF\u0700-\u074F"""
    r"""\u0780-\u07BF\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"""
    r"""\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"""
    r"""\u0D00-\u0D7F\u0E00-\u0E7F\u0E80-\u0EFF\u1780-\u17FF"""
    r"""\uF900-\uFAFF]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


Token = bytes


def _get_pair_counts(
    word_freqs: Dict[Tuple[Token, ...], int],
) -> Dict[Tuple[Token, Token], int]:
    """Count frequencies of all adjacent symbol pairs across the corpus."""
    counts: Dict[Tuple[Token, Token], int] = {}
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def _apply_merge(
    word_freqs: Dict[Tuple[Token, ...], int],
    pair: Tuple[Token, Token],
) -> Dict[Tuple[Token, ...], int]:
    """Replace all occurrences of pair (a, b) with merged token ab."""
    new_word_freqs: Dict[Tuple[Token, ...], int] = {}
    merged = pair[0] + pair[1]
    for symbols, freq in word_freqs.items():
        new_symbols: List[Token] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_freqs[tuple(new_symbols)] = freq
    return new_word_freqs


class BPETrainer:
    """Trains a BPE vocabulary from a text corpus."""

    def train(
        self,
        corpus: List[str],
        vocab_size: int,
        pattern: str = GPT2_SPLIT_PATTERN,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 1,
    ) -> "BPETokenizer":
        """
        Learn BPE merge rules from corpus.

        Args:
            corpus: List of text strings to train on
            vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens))
            pattern: Regex pattern for pre-tokenization
            special_tokens: List of special token strings (e.g. ["<|endoftext|>"])
            min_frequency: Minimum pair frequency to consider for merging

        Returns:
            Trained BPETokenizer instance
        """
        if not corpus or all(len(s) == 0 for s in corpus):
            raise ValueError("Corpus must contain at least one non-empty string")

        special_tokens = special_tokens or []
        base_vocab_size = 256 + len(special_tokens)
        if vocab_size < base_vocab_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= 256 + {len(special_tokens)} special tokens"
            )

        compiled_pattern = re.compile(pattern)

        word_freqs: Dict[Tuple[Token, ...], int] = {}
        for text in corpus:
            words = compiled_pattern.findall(text)
            for word in words:
                byte_seq = tuple(bytes([b]) for b in word.encode("utf-8"))
                word_freqs[byte_seq] = word_freqs.get(byte_seq, 0) + 1

        merges: List[Tuple[Token, Token]] = []
        vocab: Dict[Token | str, int] = {}

        for i in range(256):
            vocab[bytes([i])] = i

        next_id = 256
        for st in special_tokens:
            vocab[st] = next_id
            next_id += 1

        num_merges = vocab_size - base_vocab_size
        for _ in range(num_merges):
            pair_counts = _get_pair_counts(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < min_frequency:
                break

            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            vocab[merged_token] = next_id
            next_id += 1

            word_freqs = _apply_merge(word_freqs, best_pair)

        return BPETokenizer(
            vocab=vocab,
            merges=merges,
            pattern=pattern,
            special_tokens=special_tokens,
        )


class BPETokenizer:
    """BPE tokenizer with encode/decode, special tokens, and serialization."""

    def __init__(
        self,
        vocab: Dict[Token | str, int],
        merges: List[Tuple[Token, Token]],
        pattern: str = GPT2_SPLIT_PATTERN,
        special_tokens: Optional[List[str]] = None,
    ):
        self._vocab = vocab
        self._merges = merges
        self._pattern = pattern
        self._compiled_pattern = re.compile(pattern)
        self._special_tokens = special_tokens or []

        self._id_to_token: Dict[int, Token | str] = {v: k for k, v in self._vocab.items()}

        self._merge_priority: Dict[Tuple[Token, Token], int] = {
            pair: i for i, pair in enumerate(self._merges)
        }

        if self._special_tokens:
            escaped = [re.escape(t) for t in sorted(self._special_tokens, key=len, reverse=True)]
            self._special_pattern = re.compile("|".join(escaped))
        else:
            self._special_pattern = None

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary as {token_string: id} mapping."""
        result: Dict[str, int] = {}
        for token, tid in self._vocab.items():
            if isinstance(token, bytes):
                try:
                    result[token.decode("utf-8")] = tid
                except UnicodeDecodeError:
                    result[repr(token)] = tid
            else:
                result[token] = tid
        return result

    def token_to_bytes(self, token_id: int) -> bytes:
        """Return the raw bytes for a token ID."""
        token = self._id_to_token.get(token_id)
        if token is None:
            raise ValueError(f"Unknown token ID: {token_id}")
        if isinstance(token, bytes):
            return token
        return token.encode("utf-8")

    def _apply_merges_to_word(self, symbols: List[Token]) -> List[Token]:
        """Apply learned merges to a list of byte symbols in priority order."""
        if len(symbols) <= 1:
            return symbols

        while True:
            best_pair: Optional[Tuple[Token, Token]] = None
            best_priority = float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                priority = self._merge_priority.get(pair)
                if priority is not None and priority < best_priority:
                    best_priority = priority
                    best_pair = pair

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_symbols: List[Token] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def _encode_chunk(self, chunk: str) -> List[int]:
        """Encode a text chunk (no special tokens) into token IDs."""
        token_ids: List[int] = []
        pos = 0

        for match in self._compiled_pattern.finditer(chunk):
            start, end = match.start(), match.end()

            # Encode any gap characters as individual bytes (byte fallback)
            if start > pos:
                for b in chunk[pos:start].encode("utf-8"):
                    token_ids.append(b)
            pos = end

            word = match.group()
            byte_seq: List[Token] = [bytes([b]) for b in word.encode("utf-8")]
            merged = self._apply_merges_to_word(byte_seq)
            for token in merged:
                token_ids.append(self._vocab[token])

        # Encode any trailing unmatched characters
        if pos < len(chunk):
            for b in chunk[pos:].encode("utf-8"):
                token_ids.append(b)

        return token_ids

    def encode(self, text: str) -> List[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text: Input string (any valid UTF-8)

        Returns:
            List of integer token IDs
        """
        if not text:
            return []

        chunks: List[str] = []
        if self._special_pattern is not None:
            parts = self._special_pattern.split(text)
            specials = self._special_pattern.findall(text)
            for i, part in enumerate(parts):
                if part:
                    chunks.append(part)
                if i < len(specials):
                    chunks.append(specials[i])
        else:
            chunks = [text]

        token_ids: List[int] = []

        for chunk in chunks:
            if chunk in self._special_tokens:
                token_ids.append(self._vocab[chunk])
                continue
            token_ids.extend(self._encode_chunk(chunk))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs

        Returns:
            Decoded string
        """
        if not token_ids:
            return ""

        raw_bytes = bytearray()
        parts: List[str | bytes] = []

        for tid in token_ids:
            token = self._id_to_token.get(tid)
            if token is None:
                raise ValueError(f"Unknown token ID: {tid}")

            if isinstance(token, str) and token in self._special_tokens:
                if raw_bytes:
                    parts.append(bytes(raw_bytes))
                    raw_bytes = bytearray()
                parts.append(token)
            else:
                if isinstance(token, bytes):
                    raw_bytes.extend(token)
                else:
                    raw_bytes.extend(token.encode("utf-8"))

        if raw_bytes:
            parts.append(bytes(raw_bytes))

        result_parts: List[str] = []
        for part in parts:
            if isinstance(part, bytes):
                result_parts.append(part.decode("utf-8"))
            else:
                result_parts.append(part)

        return "".join(result_parts)

    def save(self, path: str) -> None:
        """
        Save tokenizer to a JSON file.

        Args:
            path: File path (should end in .json)
        """
        def _token_to_serializable(t: Token | str) -> dict:
            if isinstance(t, bytes):
                return {"type": "bytes", "value": list(t)}
            return {"type": "str", "value": t}

        vocab_serialized = []
        for token, tid in self._vocab.items():
            vocab_serialized.append({
                "token": _token_to_serializable(token),
                "id": tid,
            })

        merges_serialized = [
            [_token_to_serializable(a), _token_to_serializable(b)]
            for a, b in self._merges
        ]

        data = {
            "vocab": vocab_serialized,
            "merges": merges_serialized,
            "pattern": self._pattern,
            "special_tokens": self._special_tokens,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """
        Load tokenizer from a JSON file.

        Args:
            path: File path to load from

        Returns:
            BPETokenizer instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _deserialize_token(obj: dict) -> Token | str:
            if obj["type"] == "bytes":
                return bytes(obj["value"])
            return obj["value"]

        vocab: Dict[Token | str, int] = {}
        for entry in data["vocab"]:
            token = _deserialize_token(entry["token"])
            vocab[token] = entry["id"]

        merges: List[Tuple[Token, Token]] = [
            (_deserialize_token(pair[0]), _deserialize_token(pair[1]))
            for pair in data["merges"]
        ]

        return cls(
            vocab=vocab,
            merges=merges,
            pattern=data["pattern"],
            special_tokens=data.get("special_tokens", []),
        )

    def compression_ratio(self, text: str) -> float:
        """Ratio of original bytes to number of tokens (higher = better compression)."""
        tokens = self.encode(text)
        if not tokens:
            return 0.0
        return len(text.encode("utf-8")) / len(tokens)

    def tokens_per_character(self, text: str) -> float:
        """Average number of tokens per character (lower = better compression)."""
        if not text:
            return 0.0
        tokens = self.encode(text)
        return len(tokens) / len(text)

    def characters_per_token(self, text: str) -> float:
        """Average number of characters per token (higher = better compression)."""
        tokens = self.encode(text)
        if not tokens:
            return 0.0
        return len(text) / len(tokens)
