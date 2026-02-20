# Tokenization — Byte Pair Encoding (BPE)

**Phase 4 · Topic 18** — The input pipeline that determines everything downstream.

## What it is

Byte Pair Encoding is a subword tokenization algorithm used by GPT, LLaMA, Mistral, and virtually every modern LLM. It bridges the gap between character-level tokenization (small vocabulary, long sequences) and word-level tokenization (massive vocabulary, OOV problems).

BPE works by starting with a base vocabulary of individual characters (or bytes), then iteratively merging the most frequent adjacent pairs into new tokens. After training on a corpus, you end up with a vocabulary of subword units that efficiently compress common patterns while still being able to represent any input text.

The key insight: common words become single tokens ("the" -> [the]), while rare words get split into subwords ("tokenization" -> [token][ization]). This balances vocabulary size against sequence length, and critically, it means the model has seen components of rare words during training even if it never saw the exact word.

## The algorithm

### Training Phase: Learning Merge Rules

Given a text corpus and target vocabulary size $V$:

```
1. Initialize vocabulary with all unique bytes/characters in corpus
2. Split all words in corpus into character sequences
3. While vocabulary_size < V:
   a. Count frequency of all adjacent symbol pairs across corpus
   b. Find the most frequent pair (a, b)
   c. Create new symbol: ab (concatenation of a and b)
   d. Add ab to vocabulary
   e. Replace all occurrences of (a, b) with ab in corpus
   f. Store merge rule: (a, b) -> ab with priority = merge_order
4. Return vocabulary and ordered list of merge rules
```

**Worked example:**

Corpus (with word frequencies): `{"low": 5, "lower": 2, "newest": 6, "widest": 3}`

Initial representation:
```
l o w     (freq: 5)
l o w e r (freq: 2)
n e w e s t (freq: 6)
w i d e s t (freq: 3)
```

Iteration 1 - Count pairs:
```
(l, o): 7  (w, e): 8  (e, s): 9  (s, t): 9  (e, w): 6  ...
```

Most frequent: (e, s) with count 9. Merge e + s -> es.

After merge:
```
l o w     (freq: 5)
l o w e r (freq: 2)
n e w es t (freq: 6)
w i d es t (freq: 3)
```

Continue until vocabulary reaches target size.

### Encoding Phase: Text to Token IDs

Given text and learned merge rules (in priority order):

```
1. Pre-tokenize: split text into words (whitespace, punctuation rules)
2. For each word:
   a. Convert to character sequence
   b. For each merge rule (a, b) -> ab in priority order:
      - Find all adjacent (a, b) pairs in sequence
      - Replace with ab
   c. Map final symbols to token IDs
3. Return sequence of token IDs
```

**Efficient encoding with pair-priority:**

The naive approach applies merges sequentially. A more efficient approach:

```
1. Build a priority queue of all possible merges in the word
2. Pop highest-priority merge, apply it
3. Update affected neighbors in priority queue
4. Repeat until no valid merges remain
```

This is $O(n \log n)$ vs $O(n \cdot |M|)$ for naive, where $n$ is sequence length and $|M|$ is the number of merge rules.

### Decoding Phase: Token IDs to Text

```
1. Map each token ID to its string representation
2. Concatenate all strings
3. Handle special markers (space prefixes, byte fallbacks)
```

Decoding is simple because tokens directly store their string representation.

### Vocabulary Size Tradeoff

Let $|V|$ denote vocabulary size, $n$ denote sequence length, and $d$ denote model dimension.

**Smaller vocabulary** (e.g., $|V| = 8{,}000$):
- Longer sequences for same text
- More attention computation ($O(n^2)$)
- Larger KV cache
- Smaller embedding matrix

**Larger vocabulary** (e.g., $|V| = 128{,}000$):
- Shorter sequences
- Less attention computation
- Smaller KV cache per request
- Larger embedding matrix: $|V| \times d$
- May see rare tokens less during training

Typical choices: 32K (GPT-2), 50K (GPT-3/4), 32K (LLaMA), 32K (Mistral).

## Why it matters for inference

Tokenization is the first operation in every LLM inference call. Understanding it deeply reveals critical performance implications:

**1. Token count determines compute cost**

Attention is $O(n^2)$ in sequence length $n$. A poorly-tokenized prompt that produces $2\times$ the tokens costs $4\times$ the attention compute. This is why:
- GPT-4 pricing is per-token
- Context window limits are in tokens, not characters
- Prompt engineering often involves token optimization

**2. Token count determines KV cache memory**

$$\text{KV cache size} = 2 \cdot L \cdot h \cdot n \cdot d_h \cdot b$$

Where:
- $L$ is the number of layers
- $h$ is the number of attention heads
- $n$ is the sequence length
- $d_h$ is the head dimension
- $b$ is the bytes per element

Every additional token costs memory linearly. For a 7B model at FP16:
- 1000 tokens: ~500 MB KV cache
- 4000 tokens: ~2 GB KV cache

This directly limits how many concurrent requests fit in GPU memory.

**3. Vocabulary size determines embedding table size**

$$\text{Embedding memory} = |V| \cdot d \cdot b$$

Where:
- $|V|$ is the vocabulary size
- $d$ is the model dimension
- $b$ is the bytes per element

For LLaMA-7B ($|V| = 32{,}000$, $d = 4{,}096$, FP16):

$$32{,}000 \times 4{,}096 \times 2 = 250 \text{ MiB}$$

For a hypothetical 128K vocab: ~1 GB just for embeddings.

**4. Tokenization explains LLM behaviors**

- **Counting failures**: "How many r's in strawberry?" fails because the model sees tokens like ["str", "aw", "berry"], not characters
- **Arithmetic errors**: Numbers tokenize inconsistently ("380" might be one token, "381" might be two)
- **Code generation**: Different tokenizers handle whitespace/indentation differently
- **Multilingual quality**: Most tokenizers are trained on English-heavy corpora, producing $3$-$5\times$ more tokens for non-English text

**5. Byte fallback enables any input**

Modern BPE uses byte-level fallback: unknown characters are encoded as byte sequences (256 possible byte tokens). This means:
- Any UTF-8 text can be encoded
- No "unknown token" (`<unk>`) needed
- Rare Unicode still works, just less efficiently

## What to implement

### Core BPE Trainer

- [ ] `BPETrainer` class with `train(corpus: List[str], vocab_size: int)` method
- [ ] Pair frequency counting across entire corpus (use dict with tuple keys)
- [ ] Merge application: replace all $(a, b)$ pairs with $ab$ efficiently
- [ ] Store merge rules as ordered list of tuples: `[(a, b), (c, d), ...]`
- [ ] Build token-to-id and id-to-token mappings
- [ ] Handle minimum frequency threshold for merges

### BPE Encoder

- [ ] `encode(text: str) -> List[int]` method
- [ ] Pre-tokenization: split on whitespace/punctuation (configurable regex)
- [ ] Apply merges in correct priority order
- [ ] Option: greedy encoding vs. optimal encoding
- [ ] Return list of integer token IDs

### BPE Decoder

- [ ] `decode(token_ids: List[int]) -> str` method
- [ ] Handle space-prefix conventions (GPT-style: "Ghello" vs " hello")
- [ ] Byte fallback decoding for rare characters

### Special Tokens

- [ ] Support for special tokens: `<|endoftext|>`, `<|pad|>`, `<|unk|>`
- [ ] Special tokens should never be split by BPE
- [ ] Add special tokens to vocabulary with reserved IDs

### Serialization

- [ ] `save(path: str)` - save vocabulary and merges to files
- [ ] `load(path: str)` - load trained tokenizer from files
- [ ] Format: JSON or simple text format for merges

### Analysis Utilities

- [ ] `vocab_size` property
- [ ] `get_vocab() -> Dict[str, int]` - full vocabulary mapping
- [ ] `token_to_bytes(token_id: int) -> bytes` - for byte-level inspection
- [ ] Tokenization statistics: tokens per character ratio, compression ratio

## Test cases to cover

### Roundtrip Correctness

- [ ] `decode(encode(text)) == text` for ASCII text
- [ ] `decode(encode(text)) == text` for Unicode text (emojis, CJK, Arabic)
- [ ] `decode(encode(text)) == text` for empty string
- [ ] `decode(encode(text)) == text` for whitespace-only text
- [ ] `decode(encode(text)) == text` for text with special tokens

### Vocabulary Size Control

- [ ] Training with $|V| = 256$ produces only byte-level tokens
- [ ] Training with $|V| = 1{,}000$ produces expected merge count
- [ ] Increasing $|V|$ produces strictly shorter sequences
- [ ] Verify `vocab_size == len(get_vocab())`

### Merge Order Correctness

- [ ] Earlier merges have higher priority than later merges
- [ ] Verify specific known merge produces expected tokenization
- [ ] Training on "aaaa" corpus merges $(a, a) \to aa$ first, then $(aa, aa) \to aaaa$

### Edge Cases

- [ ] Empty corpus raises appropriate error
- [ ] Single-character corpus works correctly
- [ ] Corpus with only one unique word
- [ ] Very long words (>1000 characters)
- [ ] Text containing null bytes
- [ ] Text with mixed encodings (if supporting byte fallback)

### Unknown Character Handling

- [ ] Characters not in training corpus still encode (via byte fallback)
- [ ] Verify byte fallback produces valid UTF-8 on decode
- [ ] Rare Unicode characters roundtrip correctly

### Special Token Handling

- [ ] Special tokens encode to single token ID
- [ ] Special tokens are not split by merges
- [ ] Multiple special tokens in sequence encode correctly
- [ ] Special token at start/middle/end of text

### Determinism

- [ ] Same corpus + $|V|$ produces identical tokenizer
- [ ] Same text produces identical encoding on repeated calls
- [ ] Loaded tokenizer produces identical encodings as original

### Compression Efficiency

- [ ] English text: expect ~4-5 characters per token average
- [ ] Code: expect ~3-4 characters per token average
- [ ] Random characters: expect ~1-2 characters per token average
- [ ] Verify larger $|V|$ improves compression ratio

### Numerical Correctness

- [ ] All token IDs are in range $[0, |V|)$
- [ ] No duplicate token IDs for different tokens
- [ ] No duplicate tokens for same token ID

## Reference implementations to compare against

After implementing from scratch, verify behavior matches:
- `tiktoken` (OpenAI's fast BPE library)
- `tokenizers` (HuggingFace library)

Your implementation will be slower (pure Python/NumPy), but the tokenization output should be identical given the same vocabulary and merge rules.
