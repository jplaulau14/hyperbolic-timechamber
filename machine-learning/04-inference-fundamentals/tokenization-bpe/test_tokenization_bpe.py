"""Tests for BPE tokenization."""

import json
import os
import tempfile
import unittest

from implementation import BPETrainer, BPETokenizer


def _make_tokenizer(
    corpus: list[str],
    vocab_size: int = 300,
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """Helper to train a tokenizer on a corpus."""
    trainer = BPETrainer()
    return trainer.train(corpus, vocab_size=vocab_size, special_tokens=special_tokens)


class TestRoundtrip(unittest.TestCase):
    """decode(encode(text)) == text for all inputs."""

    def setUp(self):
        self.corpus = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "Hello world! This is a test of the tokenizer. " * 20,
            "Machine learning is fascinating and powerful. " * 20,
            "The the the the the. " * 50,
        ]
        self.tokenizer = _make_tokenizer(self.corpus, vocab_size=400)

    def test_ascii_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_unicode_emoji(self):
        text = "Hello world! \U0001f600\U0001f680\U0001f30d"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_cjk_characters(self):
        text = "\u4f60\u597d\u4e16\u754c\uff01\u6d4b\u8bd5"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_arabic_text(self):
        text = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("")), "")

    def test_whitespace_only(self):
        for text in [" ", "  ", "\t", "\n", "  \n\t  "]:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_mixed_scripts(self):
        text = "Hello \u4f60\u597d \u0645\u0631\u062d\u0628\u0627 \U0001f600"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_special_characters(self):
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_newlines_and_tabs(self):
        text = "line1\nline2\n\tindented\n\n"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_long_text(self):
        text = "abcdefghij" * 200
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_roundtrip_with_special_tokens(self):
        specials = ["<|endoftext|>", "<|pad|>"]
        tokenizer = _make_tokenizer(self.corpus, vocab_size=400, special_tokens=specials)
        text = "Hello<|endoftext|>world<|pad|>end"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class TestVocabSizeControl(unittest.TestCase):
    """Training produces correct vocabulary sizes."""

    def setUp(self):
        self.corpus = [
            "the cat sat on the mat. " * 50,
            "a quick brown fox jumps over the lazy dog. " * 50,
            "machine learning is fascinating and powerful for everyone. " * 50,
            "natural language processing enables understanding text data. " * 50,
        ]

    def test_byte_only_vocab(self):
        tokenizer = _make_tokenizer(self.corpus, vocab_size=256)
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_expected_merge_count(self):
        tokenizer = _make_tokenizer(self.corpus, vocab_size=300)
        self.assertEqual(tokenizer.vocab_size, 300)

    def test_larger_vocab_shorter_sequences(self):
        text = "the cat sat on the mat."
        tok_small = _make_tokenizer(self.corpus, vocab_size=270)
        tok_large = _make_tokenizer(self.corpus, vocab_size=350)
        ids_small = tok_small.encode(text)
        ids_large = tok_large.encode(text)
        self.assertGreaterEqual(len(ids_small), len(ids_large))

    def test_vocab_size_matches_get_vocab(self):
        tokenizer = _make_tokenizer(self.corpus, vocab_size=300)
        self.assertEqual(tokenizer.vocab_size, len(tokenizer.get_vocab()))

    def test_special_tokens_in_vocab_size(self):
        specials = ["<|endoftext|>", "<|pad|>"]
        tokenizer = _make_tokenizer(self.corpus, vocab_size=300, special_tokens=specials)
        self.assertEqual(tokenizer.vocab_size, 300)


class TestMergeOrder(unittest.TestCase):
    """Merge rules are applied in correct priority order."""

    def test_aaaa_corpus(self):
        corpus = ["aaaa " * 100]
        tokenizer = _make_tokenizer(corpus, vocab_size=260)

        self.assertEqual(tokenizer._merges[0], (b"a", b"a"))
        self.assertEqual(tokenizer._merges[1], (b"aa", b"aa"))

        ids = tokenizer.encode("aaaa")
        self.assertEqual(len(ids), 1)

    def test_deterministic_merges(self):
        corpus = ["abab " * 100]
        tok1 = _make_tokenizer(corpus, vocab_size=270)
        tok2 = _make_tokenizer(corpus, vocab_size=270)
        text = "abababab"
        self.assertEqual(tok1.encode(text), tok2.encode(text))

    def test_earlier_merges_higher_priority(self):
        corpus = ["xyxy " * 100, "zzz " * 10]
        tokenizer = _make_tokenizer(corpus, vocab_size=270)
        text = "xyxy"
        ids = tokenizer.encode(text)
        self.assertTrue(len(ids) <= 2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_empty_corpus_raises(self):
        trainer = BPETrainer()
        with self.assertRaises(ValueError):
            trainer.train([], vocab_size=300)

    def test_empty_strings_corpus_raises(self):
        trainer = BPETrainer()
        with self.assertRaises(ValueError):
            trainer.train(["", ""], vocab_size=300)

    def test_single_character_corpus(self):
        tokenizer = _make_tokenizer(["a" * 100], vocab_size=256)
        ids = tokenizer.encode("a")
        self.assertEqual(len(ids), 1)
        self.assertEqual(tokenizer.decode(ids), "a")

    def test_single_unique_word(self):
        tokenizer = _make_tokenizer(["hello " * 100], vocab_size=270)
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_very_long_word(self):
        long_word = "x" * 1500
        corpus = [long_word + " " + "hello world " * 50]
        tokenizer = _make_tokenizer(corpus, vocab_size=270)
        self.assertEqual(tokenizer.decode(tokenizer.encode(long_word)), long_word)

    def test_null_bytes(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=270)
        text = "hello\x00world"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_decode_invalid_token_id(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=270)
        with self.assertRaises(ValueError):
            tokenizer.decode([999999])


class TestUnknownCharacters(unittest.TestCase):
    """Byte fallback handles characters not seen in training."""

    def setUp(self):
        self.tokenizer = _make_tokenizer(
            ["simple english text only " * 50],
            vocab_size=300,
        )

    def test_unseen_characters_encode(self):
        text = "\u4f60\u597d"
        ids = self.tokenizer.encode(text)
        self.assertTrue(len(ids) > 0)

    def test_unseen_characters_roundtrip(self):
        text = "\u4f60\u597d\u4e16\u754c"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_emoji_roundtrip(self):
        text = "\U0001f600\U0001f680\U0001f30d"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_rare_unicode_roundtrip(self):
        text = "\u10e6\u10d4\u10da\u10d0"  # Georgian script
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)


class TestSpecialTokens(unittest.TestCase):
    """Special tokens encode as single IDs and survive merges."""

    def setUp(self):
        self.specials = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        self.tokenizer = _make_tokenizer(
            ["hello world this is a test " * 50],
            vocab_size=300,
            special_tokens=self.specials,
        )

    def test_special_token_single_id(self):
        for st in self.specials:
            ids = self.tokenizer.encode(st)
            self.assertEqual(len(ids), 1, f"Special token '{st}' should encode to single ID")

    def test_special_tokens_not_split(self):
        text = "hello<|endoftext|>world"
        ids = self.tokenizer.encode(text)
        eot_id = self.tokenizer.encode("<|endoftext|>")[0]
        self.assertIn(eot_id, ids)

    def test_multiple_special_tokens(self):
        text = "<|endoftext|><|pad|><|unk|>"
        ids = self.tokenizer.encode(text)
        self.assertEqual(len(ids), 3)

    def test_special_token_at_boundaries(self):
        for text in [
            "<|endoftext|>hello",
            "hello<|endoftext|>",
            "hello<|endoftext|>world",
        ]:
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_special_token_roundtrip(self):
        text = "Hello<|endoftext|>World<|pad|>End"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)


class TestSerialization(unittest.TestCase):
    """Save/load produces identical tokenizer."""

    def test_save_and_load_roundtrip(self):
        corpus = ["the quick brown fox " * 50, "jumps over the lazy dog " * 50]
        specials = ["<|endoftext|>", "<|pad|>"]
        tokenizer = _make_tokenizer(corpus, vocab_size=320, special_tokens=specials)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)

            self.assertEqual(tokenizer.vocab_size, loaded.vocab_size)

            test_texts = [
                "the quick brown fox",
                "Hello \U0001f600 world!",
                "<|endoftext|>test<|pad|>",
                "",
                "\u4f60\u597d",
            ]
            for text in test_texts:
                self.assertEqual(
                    tokenizer.encode(text),
                    loaded.encode(text),
                    f"Encoding mismatch for: {text!r}",
                )
        finally:
            os.unlink(path)

    def test_saved_file_is_valid_json(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=270)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tokenizer.save(path)
            with open(path, "r") as f:
                data = json.load(f)
            self.assertIn("vocab", data)
            self.assertIn("merges", data)
            self.assertIn("pattern", data)
        finally:
            os.unlink(path)


class TestDeterminism(unittest.TestCase):
    """Same inputs produce identical outputs."""

    def test_same_corpus_same_tokenizer(self):
        corpus = ["deterministic test data " * 50]
        tok1 = _make_tokenizer(corpus, vocab_size=280)
        tok2 = _make_tokenizer(corpus, vocab_size=280)
        text = "deterministic test data is important"
        self.assertEqual(tok1.encode(text), tok2.encode(text))

    def test_repeated_encode_same_result(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=280)
        text = "hello world test"
        ids1 = tokenizer.encode(text)
        ids2 = tokenizer.encode(text)
        self.assertEqual(ids1, ids2)


class TestCompressionEfficiency(unittest.TestCase):
    """Compression improves with vocabulary size and varies by content type."""

    def test_english_compression(self):
        corpus = [
            "The quick brown fox jumps over the lazy dog. " * 100,
            "Machine learning is a subset of artificial intelligence. " * 100,
            "Natural language processing enables computers to understand text. " * 100,
        ]
        tokenizer = _make_tokenizer(corpus, vocab_size=500)
        text = "The quick brown fox jumps over the lazy dog."
        cpt = tokenizer.characters_per_token(text)
        self.assertGreater(cpt, 2.0)

    def test_random_chars_low_compression(self):
        import random
        random.seed(42)
        random_text = "".join(chr(random.randint(33, 126)) for _ in range(500))
        corpus = ["hello world " * 100]
        tokenizer = _make_tokenizer(corpus, vocab_size=280)
        cpt = tokenizer.characters_per_token(random_text)
        self.assertLess(cpt, 3.0)

    def test_larger_vocab_better_compression(self):
        corpus = ["the quick brown fox jumps over the lazy dog " * 200]
        tok_small = _make_tokenizer(corpus, vocab_size=280)
        tok_large = _make_tokenizer(corpus, vocab_size=500)
        text = "the quick brown fox jumps over the lazy dog"
        ratio_small = tok_small.compression_ratio(text)
        ratio_large = tok_large.compression_ratio(text)
        self.assertGreaterEqual(ratio_large, ratio_small)


class TestNumericalCorrectness(unittest.TestCase):
    """Token IDs are valid and unique."""

    def test_all_ids_in_range(self):
        tokenizer = _make_tokenizer(["hello world test " * 50], vocab_size=300)
        text = "hello world test \u4f60\u597d \U0001f600"
        ids = tokenizer.encode(text)
        for tid in ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, tokenizer.vocab_size)

    def test_no_duplicate_ids(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=300)
        vocab = tokenizer.get_vocab()
        ids = list(vocab.values())
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_duplicate_tokens(self):
        tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=300)
        vocab = tokenizer.get_vocab()
        tokens = list(vocab.keys())
        self.assertEqual(len(tokens), len(set(tokens)))


class TestAnalysisUtilities(unittest.TestCase):
    """Compression ratio and token stats work correctly."""

    def setUp(self):
        self.tokenizer = _make_tokenizer(["hello world " * 50], vocab_size=280)

    def test_compression_ratio_positive(self):
        ratio = self.tokenizer.compression_ratio("hello world")
        self.assertGreater(ratio, 0)

    def test_tokens_per_character_positive(self):
        tpc = self.tokenizer.tokens_per_character("hello world")
        self.assertGreater(tpc, 0)
        self.assertLessEqual(tpc, 1.5)

    def test_empty_text_stats(self):
        self.assertEqual(self.tokenizer.compression_ratio(""), 0.0)
        self.assertEqual(self.tokenizer.tokens_per_character(""), 0.0)
        self.assertEqual(self.tokenizer.characters_per_token(""), 0.0)

    def test_token_to_bytes(self):
        b = self.tokenizer.token_to_bytes(ord("h"))
        self.assertEqual(b, b"h")

    def test_token_to_bytes_invalid_id(self):
        with self.assertRaises(ValueError):
            self.tokenizer.token_to_bytes(999999)


class TestMinFrequency(unittest.TestCase):
    """Minimum frequency threshold prevents rare merges."""

    def test_high_min_frequency_fewer_merges(self):
        corpus = ["aabb " * 50, "ccdd " * 2]
        trainer = BPETrainer()
        tok_low = trainer.train(corpus, vocab_size=270, min_frequency=1)
        tok_high = trainer.train(corpus, vocab_size=270, min_frequency=100)
        self.assertGreaterEqual(tok_low.vocab_size, tok_high.vocab_size)


if __name__ == "__main__":
    unittest.main()
