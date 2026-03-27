# Lagoon

Lagoon is a scoring library, based on the work (and exports) from [Windowsill](https://github.com/morimar32/windowsill). The Windowsill database export can also be found directly on [Huggingface](https://huggingface.co/datasets/morimar/windowsill).

Lagoon is meant to be the "glue" that supports a number of downstream projects meant to consume this information for different use cases. Once the project hits a certain degree of maturity, the plan is to create a Rust crate of this library (likely `lagoon-rs`).

**New here?** See the [Quickstart Guide](QUICKSTART.md) to get up and running in minutes.

## 1. What is Lagoon

Lagoon is a standalone text-scoring library that maps arbitrary input text to a four-level semantic hierarchy: 6 archipelagos, 40 islands, 298 towns, and 3,885 reefs. Towns are the primary scoring unit. It loads a set of pre-built binary data files at startup and scores text using weight accumulation with background subtraction — simple bag-of-words for short inputs, and a 4-pass contextual scorer for longer text (4+ matched words) with noise filtering and island-coherence boosting.

**What lagoon does:**
- Loads binary data files produced by the export tool from [Windowsill](https://github.com/morimar32/windowsill)
- Scores text against 298 towns using per-town min-max normalized weights (u8)
- Returns ranked towns (as "reefs" in the API), islands, and archipelagos with z-scores, confidence, and coverage
- Supports runtime vocabulary extension with custom words, optional consumer tags, and compound injection
- Provides compound-aware tokenization via Aho-Corasick automaton

**What lagoon does NOT do:**
- Build the reef hierarchy (that's the upstream pipeline)
- Produce the binary data files (that's the export tool)
- Make routing decisions or store documents (that's a downstream consumer)

**Design philosophy:** Python first, Rust later. All data structures and serialization formats are chosen to port cleanly to Rust without architectural changes. No pickle, no Python-specific formats, no dynamic dispatch.

---

## 2. Concepts

### 2.1 The Semantic Hierarchy

The data files encode a four-level hierarchy of semantic clusters, derived from embedding space decomposition:

**Archipelagos (6 total)** — The broadest semantic regions. Too coarse for meaningful topic discrimination — most sentences spread across all six.

**Islands (40 total)** — Mid-level semantic communities. Each island contains multiple towns. Islands play an active role in scoring: the contextual scorer (4+ matched words) tracks island activation levels and boosts town scores when multiple words converge on the same island ("island-coherence boosting"). Islands also appear in `TopicResult.top_islands` as a rollup of their child town scores.

**Towns (298 total)** — The primary scoring unit. Each town groups words into coherent semantic neighborhoods. Words map to towns with per-town min-max normalized weights. All weight accumulation, background subtraction, and result extraction operate at the town level. Towns are what the scorer refers to as "reefs" in its public API (for historical compatibility) — `ScoredReef.reef_id` is a town index, `top_reefs` returns scored towns, etc.

**Reefs (3,885 total)** — The finest-grained semantic clusters, nested within towns. Reef metadata is loaded via `V3ReefMeta` and per-word detail is available in `word_reef_detail` (5-element tuples: `island_id, town_id, reef_id, weight, level`). However, sub-town reef resolution is **not yet wired up** in the V3 scorer — `ScoredReef.resolved_sub_reef_id` is currently always `None` because the resolution guard (`_resolve_sub_reefs`) exits early when `sub_reef_meta` is empty (as it is in V3, where towns replaced the old sub-reef concept). The data to support reef-level resolution exists in `word_reef_detail` and `v3_reef_meta`, but the scorer does not yet use it. Reef-level scoring (accumulation, background subtraction) is also not implemented — reefs are metadata only.

**Note on scoring scope:** The three levels that actively participate in scoring are archipelagos (rollup), islands (contextual coherence + rollup), and towns (primary scoring unit). Reefs are loaded as metadata but do not currently participate in scoring or resolution.

**Implementation gaps:**
- `v3_reef_meta`, `town_meta`, `bucket_words`, and `bucket_only_word_ids` are loaded by `load_data()` but not passed to `ReefScorer` — they are available for downstream consumers but unused by the scorer
- `word_islands.bin` is checksum-validated but never parsed
- 770 "bucket-only" words (in bucket islands but no town) match in `word_lookup` but produce no scoring signal since they have no `word_reefs` entries

### 2.2 What a Town Score Means

A town score quantifies how strongly the input text's vocabulary converges on a particular semantic neighborhood. Individual word lookups are noisy — most words spread across multiple towns. **The signal emerges from convergence across words, not from any single word.** When multiple words in the input independently activate the same town, and that activation exceeds what random text would produce (background subtraction), the score is meaningful.

High-specificity words (specificity = +2) are an exception — they concentrate strongly in their top town and touch only a few towns total. A single specificity=+2 word is a strong signal on its own.

### 2.3 Scoring Approach

Lagoon uses two scoring paths depending on input length:

**Simple path (< 4 matched words):** Direct weight accumulation — each word's per-town weights (u8, min-max normalized at export time) are summed, then background-subtracted to produce z-scores. Islands are used only for result rollup (no coherence boosting). Fast and effective for short queries.

**Contextual path (4+ matched words):** A 4-pass scorer that builds a `TextResult` structure preserving word order, then:
1. **Build** — maps words to town hits with linked-list threading, groups hits by island (each town knows its parent island via `reef_meta[town_id].island_id`)
2. **Filter noise** — prunes low-weight hits (below score floor) and dampens hits from single-word islands (islands where only one input word contributes get their scores halved)
3. **Evaluate contextually** — walks words in text order, tracking per-island activation levels. When a word's town belongs to an already-active island, its contribution is boosted (island-coherence boosting, up to 30% via `_CONTEXT_GAMMA`). Also applies town-level corroboration penalty: towns where fewer than ~33% of matched words contribute get their scores dampened
4. **Background subtract** — converts raw scores to z-scores with alpha-ramped mean subtraction

Background subtraction converts raw scores to z-scores, suppressing towns that activate regardless of input topic (noise magnets) and amplifying towns where activation is genuinely surprising.

---

## 3. Data Format Specification

### 3.1 Overview

Lagoon loads a directory of binary data files at startup. All files use MessagePack serialization. The directory also contains a `manifest.json` with version, checksums, and build metadata.

```
lagoon_data/
  manifest.json          # version, file checksums, build timestamp
  word_lookup.bin        # HashMap<u64, WordInfo>
  word_islands.bin       # word_id -> list[island entries]
  word_towns.bin         # word_id -> list[town_id, weight]
  word_detail.bin        # word_id -> list[island_id, town_id, reef_id, weight, level]
  island_meta.bin        # [IslandMeta; 40]
  town_meta.bin          # [TownMeta; 298]
  reef_meta.bin          # [V3ReefMeta; 3885]
  background.bin         # town_bg_mean [f32; 298] + town_bg_std [f32; 298]
  compounds.bin          # compound strings + word_id mapping
  constants.bin          # runtime constants + town-level side arrays
  bucket_words.bin       # word_id -> list[bucket entries]
```

The manifest includes a `version` field. Lagoon validates this on load and fails fast with a clear error on mismatch.

### 3.2 word_lookup.bin

A `HashMap<u64, WordInfo>` of approximately 150K entries mapping FNV-1a u64 hashes to word metadata.

**Contents:**
- ~150K words from the vocabulary (each word's FNV-1a hash maps to its WordInfo), including base words, morphy variants, and Snowball stemmer mappings.

**Key format:** FNV-1a u64 hash of the lowercase, whitespace-normalized string. Multi-word compounds use spaces as separators. Zero collisions confirmed across the full vocabulary.

**MessagePack structure:**
```
{
  <u64_hash>: [word_hash, word_id, specificity, idf_q],
  ...
}
```

| Field | Type | Description |
|-------|------|-------------|
| word_hash | u64 | FNV-1a hash of the base word (also used as the lookup key for base entries) |
| word_id | u32 | Unique identifier, index into word_reefs |
| specificity | i8 | Sigma band: +2, +1, 0, -1, -2 (positive = specific, negative = universal) |
| idf_q | u8 | Quantized IDF: `round(idf * 51)`, decode: `idf_q / 51.0` |

### 3.3 word_towns.bin

An array indexed by word_id, where each entry is a list of `[town_id, weight]` pairs.

**MessagePack structure:**
```
[
  [],                              # index 0 (unused)
  [[town_id, weight], ...],       # word_id 1
  [[town_id, weight], ...],       # word_id 2
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| town_id | u16 | Which town (0..297) |
| weight | u8 | Min-max normalized weight (0-255), decode: `weight / WEIGHT_SCALE` (WEIGHT_SCALE=100) |

Weights are per-town min-max normalized at export time. At runtime, scoring is pure accumulation — no IDF lookup, no tf computation, no length normalization needed.

### 3.4 town_meta.bin

An array of 298 TownMeta records, indexed directly by town_id. Towns are the primary scoring unit.

**MessagePack structure:**
```
[
  {"island_id": <u8>, "name": "<string>", "n_words": <u32>, "tqf": <u8>, "avg_specificity": <f32>},
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| island_id | u8 | Parent island (0..39) |
| name | string | Human-readable name |
| n_words | u32 | Total words in this town |
| tqf | u8 | Town quality factor (0-255, placeholder 128) |
| avg_specificity | f32 | Average word specificity in this town |

### 3.5 reef_meta.bin

An array of 3,885 V3ReefMeta records — the finest-grained clusters, nested within towns.

**MessagePack structure:**
```
[
  {"town_id": <u16>, "name": "<string>", "n_words": <u32>, "avg_specificity": <f32>},
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| town_id | u16 | Parent town (0..297) |
| name | string | Human-readable name |
| n_words | u32 | Total words in this reef |
| avg_specificity | f32 | Average word specificity |

### 3.6 island_meta.bin

An array of 40 IslandMeta records, indexed by island_id.

**MessagePack structure:**
```
[
  {"arch_id": <u8>, "name": "<string>"},
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| arch_id | u8 | Parent archipelago (0..5) |
| name | string | Human-readable name |

### 3.7 background.bin

Two fixed-size arrays pre-computed at the town level by sampling random word subsets, scoring against all towns, and recording per-town statistics.

**MessagePack structure:**
```
{
  "town_bg_mean": [f32; 298],
  "town_bg_std": [f32; 298]
}
```

| Array | Type | Description |
|-------|------|-------------|
| town_bg_mean | [f32; 298] | Mean score per town across random samples |
| town_bg_std | [f32; 298] | Standard deviation per town across random samples |

At runtime: `z = (raw - town_bg_mean[town]) / town_bg_std[town]` converts noisy scores into "how surprising is this town activation given random input?"

Towns with high bg_mean are noise magnets that absorb vocabulary indiscriminately. Towns with low bg_mean are highly discriminating — when they activate, it means something.

### 3.8 compounds.bin

A list of compound (multi-word) entries for building an Aho-Corasick automaton at load time.

**MessagePack structure:**
```
[
  ["<compound string>", <word_id>],
  ...
]
```

**Current export:** ~64K compounds. Since a large fraction of the vocabulary is multi-word expressions, matching compounds as single units before falling back to individual word lookups significantly improves precision. "Heart attack" as a unit produces focused medical activation; "heart" + "attack" separately scatters across cardiac + violence towns.

The automaton is built at load time from this list. In Python, use `ahocorasick` or `pyahocorasick`. In Rust, use the `aho-corasick` crate.

### 3.9 word_detail.bin

Per-word detail entries providing the full hierarchy resolution (island → town → reef) for each word.

**MessagePack structure:**
```
[
  [],                                                      # index 0 (unused)
  [[island_id, town_id, reef_id, weight, level], ...],    # word_id 1
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| island_id | u8 | Island (0..39) |
| town_id | u16 | Town (0..297) |
| reef_id | u16 | Reef (0..3884) |
| weight | u8 | Weight at this level |
| level | u8 | Hierarchy level |

Used for sub-town resolution when resolving which reef within a town best matches the input.

### 3.10 bucket_words.bin

Sparse list indexed by word_id, containing bucket-level entries for words that fall into bucket islands (4 bucket islands in v3).

**MessagePack structure:**
```
[
  [[bucket_id, weight], ...],    # word_id 0
  ...
]
```

### 3.11 constants.bin

All runtime constants and town-level side arrays.

**MessagePack structure:**
```
{
  "N_ARCHS": 6,
  "WEIGHT_SCALE": 100,
  "IDF_SCALE": 51,
  "FNV1A_OFFSET": 14695981039346656037,
  "FNV1A_PRIME": 1099511628211,
  "town_n_words": [<u32>; 298],
  "town_domainless_word_ids": [<u32>, ...],
  "bucket_only_word_ids": [<u32>, ...]
}
```

Town-level side arrays:
- `town_n_words` — total words per town (298 entries)
- `town_domainless_word_ids` — word_ids of words recognized but not domain-specific (~22K)
- `bucket_only_word_ids` — word_ids only in bucket islands, not in any town (~770)

### 3.12 manifest.json

```json
{
  "version": "3.1",
  "format": "msgpack",
  "build_timestamp": "2026-03-04T07:53:47Z",
  "files": {
    "word_lookup.bin": "<sha256>",
    "word_islands.bin": "<sha256>",
    "word_towns.bin": "<sha256>",
    "island_meta.bin": "<sha256>",
    "background.bin": "<sha256>",
    "constants.bin": "<sha256>",
    "compounds.bin": "<sha256>",
    "word_detail.bin": "<sha256>",
    "town_meta.bin": "<sha256>",
    "reef_meta.bin": "<sha256>",
    "bucket_words.bin": "<sha256>"
  },
  "stats": {
    "n_islands": 40,
    "n_towns": 298,
    "n_reefs": 3885,
    "n_archs": 6,
    "n_bucket_islands": 4,
    "n_words": 149691,
    "n_lookup_entries": 149691,
    "n_topical_words": 126196,
    "n_compounds": 64182,
    "n_domainless_words": 22725,
    "n_bucket_only_words": 770
  }
}
```

**Note:** When deserializing `word_lookup.bin`, the MessagePack unpacker must be configured to accept integer map keys (e.g., `strict_map_key=False` in Python's `msgpack` library).

Lagoon validates the version on load (expects `"3.1"`) and verifies file checksums.

### 3.13 Quantization Scheme

All quantization is applied at export time only. The source database stores full DOUBLE precision values; the binary export files use fixed-point integer representations for compactness and cache efficiency.

**IDF quantization (u8, scale factor 51):**
- Encode: `idf_q = round(idf * 51)`
- Decode: `idf = idf_q / 51.0`
- Max quantization error: 1/102 = 0.0098

**Weight quantization (u8, scale factor 100):**
- Encode: per-town min-max normalized weights stored as u8 (0-255)
- Decode: `weight / WEIGHT_SCALE` (WEIGHT_SCALE=100)

---

## 4. Data Structures

### ReefScorer (top-level)

```python
@dataclass
class ReefScorer:
    word_lookup: dict[int, WordInfo]        # HashMap<u64, WordInfo>, ~150K entries
    word_reefs: list[list[tuple]]           # indexed by word_id, entries: (town_id, weight_q, sentinel)
    reef_meta: list[ReefMeta]               # indexed by town_id (towns as scoring reefs), len=298
    island_meta: list[IslandMeta]           # indexed by island_id, len=40
    bg_mean: list[float]                    # len=298 (town-level)
    bg_std: list[float]                     # len=298 (town-level)
    compound_ac: AhoCorasick                # built at load time from compounds.bin
    compound_word_ids: list[int]            # compound match index -> word_id
    reef_n_words: list[int]                 # [u32; 298]
    avg_reef_words: float                   # average words per town
```

**Rust equivalent:** `word_lookup` becomes `HashMap<u64, WordInfo>`, `word_reefs` becomes `Vec<Vec<(u16, u8, u16)>>`, and the f32 arrays become `[f32; 298]`. All fields are flat, owned types — no reference counting, no dynamic dispatch.

### WordInfo

```python
@dataclass
class WordInfo:
    word_hash: int        # u64 — FNV-1a hash (also the lookup key)
    word_id: int          # u32 — index into word_reefs
    specificity: int      # i8 — sigma band: +2 to -2
    idf_q: int            # u8 — quantized IDF
    tag: int = 0          # opaque consumer metadata, not interpreted by lagoon
```

The `tag` field is opaque consumer metadata — lagoon never inspects or acts on it. It defaults to 0 for all base vocabulary words and words loaded from data files. Non-zero tags can be assigned via `add_custom_word(tag=...)` to let downstream consumers (e.g., Shoal) distinguish custom-injected words from base vocabulary at query time.

**Rust:** All fields are fixed-size integers. `#[repr(C)]` layout matches serialized format. `tag` is `u32` (or whatever width the consumer needs).

### ReefMeta (town as scoring reef)

```python
@dataclass
class ReefMeta:
    reef_id: int          # positional index (= town_id)
    hierarchy_addr: int   # u32 bit-packed: arch(16)|reef_id
    n_words: int          # u32
    name: str             # UTF-8, pre-normalized
    island_id: int        # parent island
    arch_id: int          # parent archipelago
    avg_specificity: float
```

### TownMeta

```python
@dataclass
class TownMeta:
    town_id: int          # positional index
    island_id: int        # parent island (= lagoon reef_id)
    name: str
    n_words: int
    tqf: int              # town quality factor (0-255, placeholder 128)
    avg_specificity: float
```

### V3ReefMeta

```python
@dataclass
class V3ReefMeta:
    reef_id: int          # positional index
    town_id: int          # parent town
    name: str
    n_words: int
    avg_specificity: float
```

### IslandMeta

```python
@dataclass
class IslandMeta:
    island_id: int
    arch_id: int          # u8 — parent archipelago (0..5)
    name: str             # UTF-8
```

### TopicResult (scoring output)

```python
@dataclass
class TopicResult:
    top_reefs: list[ScoredReef]           # Top-K towns by z-score (named "reefs" for API compat)
    top_islands: list[ScoredIsland]       # Island-level rollup
    arch_scores: list[float]              # len=6, archipelago distribution
    confidence: float                      # top z-score (clamped to 0)
    coverage: float                        # matched_words / total_input_words
    matched_words: int                     # words that hit the dictionary
    unknown_words: list[str]              # words that failed lookup + stem (stop words excluded)
    matched_word_ids: frozenset[int]      # set of word_ids that matched (for cross-referencing)
    n_domainless: int                     # words recognized but not domain-specific
    valence_signal: float                 # z-score-weighted mean of town valences

@dataclass
class ScoredReef:
    reef_id: int                          # town index (0..297)
    z_score: float                        # f32 — background-subtracted
    raw_score: float                      # f32 — pre-subtraction
    n_contributing_words: int             # u16
    name: str
    quality_score: float                  # z-score (IQF modulation placeholder)
    valence: float
    avg_specificity: float
    resolved_sub_reef_id: int | None      # sub-town reef resolution
    resolved_sub_reef_name: str | None

@dataclass
class ScoredIsland:
    island_id: int                        # u8 (0..39)
    aggregate_z: float                    # f32 — sum of child town z-scores
    n_contributing_reefs: int             # u16
    name: str
```

**Rust:** `unknown_words` becomes `Vec<String>`, `matched_word_ids` becomes `HashSet<u32>`. The rest are flat numeric types. `top_reefs` and `top_islands` are small `Vec`s (typically K=10), allocated per call.

### TopicSegment (analysis output)

```python
@dataclass
class TopicSegment:
    sentences: list[str]                  # sentences in this segment
    start_idx: int                        # first sentence index (0-based)
    end_idx: int                          # last sentence index (inclusive)
    topic: TopicResult                    # segment-level scoring
    sentence_results: list[TopicResult]   # per-sentence scoring data
```

Each segment from `analyze()` includes `sentence_results` — one `TopicResult` per sentence. These provide per-sentence `matched_word_ids`, `unknown_words`, `top_reefs`, and `coverage` without requiring additional scoring calls.

### DocumentAnalysis (analysis output)

```python
@dataclass
class DocumentAnalysis:
    segments: list[TopicSegment]          # topic-coherent segments
    n_sentences: int                      # total sentences
    n_segments: int                       # total segments
    boundaries: list[int]                 # sentence indices where topic shifts occur
```

---

## 5. Scoring Algorithm

### 5.1 Pipeline Overview

```
Input text
    |
    v
Phase 1: Compound scan (Aho-Corasick)
    |
    v
Phase 2: Tokenize + normalize (lowercase, HashMap lookup, stem fallback)
    |
    v
Phase 3: Weight accumulation (precomputed weights -> [f32; 298])
    |
    v
Phase 4: Background subtraction (z = (raw - bg_mean) / bg_std)
    |
    v
Phase 5: Result extraction (top-K, confidence, coverage, island/arch rollup)
    |
    v
TopicResult
```

### 5.2 Phase 1: Compound Scan

Run the Aho-Corasick automaton over the lowercased input text. Use greedy leftmost-longest matching.

For each match:
1. Record the word_id via `compound_word_ids[match_pattern_index]`
2. Mark the matched character span as consumed

Consumed spans are excluded from individual word tokenization in Phase 2. This prevents double-counting — "heart attack" is scored as one compound, not "heart" + "attack".

```python
def scan_compounds(self, text_lower):
    matched_word_ids = set()
    consumed = []  # list of (start, end) spans
    for match in self.compound_ac.iter(text_lower):
        end = match[0] + 1
        start = end - len(self.compound_strings[match[1]])
        word_id = self.compound_word_ids[match[1]]
        matched_word_ids.add(word_id)
        consumed.append((start, end))
    return matched_word_ids, consumed
```

**Performance:** O(n) where n = text length. Typically ~1-3μs for a sentence, ~17μs for a 200-word paragraph (pure Python; Rust target: ~100-500ns).

### 5.3 Phase 2: Tokenize + Normalize

Split the unconsumed text segments on whitespace and punctuation. For each token:

1. Lowercase the token
2. Skip if it falls within a consumed compound span
3. Compute `fnv1a_u64(token)` and look up in `word_lookup`:
   - **Hit:** record the word_id
   - **Miss:** run Snowball stemmer on the token, compute `fnv1a_u64(stem)`, look up again
     - **Hit:** record the word_id
     - **Miss:** add to `unknown_words` list

**Deduplication:** Track unique word_ids. If the same word appears multiple times, it contributes only once (binary occurrence, not frequency). The signal comes from _how many different words_ converge on a town, not from repetition.

```python
def tokenize(self, text, consumed_spans):
    tokens = re.findall(r'[a-z]+', text.lower())
    matched = set()
    unknown = []
    for token in tokens:
        if is_in_consumed_span(token, consumed_spans):
            continue
        h = fnv1a_u64(token)
        info = self.word_lookup.get(h)
        if info is not None:
            matched.add(info[1])  # word_id
        else:
            stem = self.stemmer.stem(token)
            sh = fnv1a_u64(stem)
            info = self.word_lookup.get(sh)
            if info is not None:
                matched.add(info[1])
            else:
                unknown.append(token)
    return matched, unknown
```

**Performance per word (pure Python):**
- HashMap lookup: ~50ns
- Snowball stemmer fallback: ~120ns (fires ~30% of the time)
- Total per-word overhead including Python iteration: ~1μs/word

### 5.4 Phase 3: Weight Accumulation

Initialize a score array and accumulate precomputed weights:

```python
def accumulate_weights(self, word_ids):
    scores_q = [0] * 298
    word_counts = [0] * 298  # contributing words per town
    for word_id in word_ids:
        for town_id, weight_q, _sentinel in self.word_reefs[word_id]:
            scores_q[town_id] += weight_q
            word_counts[town_id] += 1
    return scores_q, word_counts
```

Weights are precomputed and min-max normalized at export time. At runtime, this is pure integer accumulation — no IDF lookup, no tf computation. Dequantize once after accumulation: `raw = scores_q / WEIGHT_SCALE`.

### 5.5 Phase 4: Background Subtraction

Convert raw scores to z-scores with alpha-ramping based on matched word count:

```python
def subtract_background(self, raw_scores, n_matched):
    z_scores = [0.0] * 298
    # Alpha ramps from 0 (1 word) to 1.0 (6+ words)
    alpha = min(1.0, max(0.0, (n_matched - 1) / 5))
    for town in range(298):
        std = max(bg_std[town], 0.01)  # floor to prevent division by near-zero
        z_scores[town] = (raw_scores[town] - alpha * bg_mean[town]) / std
    return z_scores
```

Alpha scaling: with 1 matched word, no background mean is subtracted (alpha=0), only std normalization is applied. At 6+ matched words, full background subtraction kicks in. This prevents over-penalizing single-word queries where accumulated common-reef noise isn't yet a factor.

### 5.6 Phase 5: Result Extraction

```python
def extract_results(self, z_scores, raw_scores, word_counts, matched, unknown):
    # Top-K towns by z-score
    indexed = sorted(range(298), key=lambda i: z_scores[i], reverse=True)
    top_reefs = [
        ScoredReef(reef_id=i, z_score=z_scores[i], raw_score=raw_scores[i],
                   n_contributing_words=word_counts[i], name=reef_meta[i].name, ...)
        for i in indexed[:10]
    ]

    # Confidence: top z-score (clamped to 0)
    confidence = max(0.0, z_scores[indexed[0]]) if indexed else 0.0

    # Coverage
    total_words = len(matched) + len(unknown)
    coverage = len(matched) / total_words if total_words > 0 else 0.0

    # Island rollup from selected towns
    island_agg = defaultdict(lambda: [0.0, 0])
    for i in indexed[:10]:
        iid = reef_meta[i].island_id
        island_agg[iid][0] += z_scores[i]
        island_agg[iid][1] += 1
    top_islands = sorted([
        ScoredIsland(island_id=iid, aggregate_z=agg, n_contributing_reefs=n, name=...)
        for iid, (agg, n) in island_agg.items()
    ], key=lambda x: x.aggregate_z, reverse=True)

    # Archipelago rollup
    arch_scores = [0.0] * 6
    for island in top_islands:
        arch_id = island_meta[island.island_id].arch_id
        arch_scores[arch_id] += island.aggregate_z

    return TopicResult(...)
```

### 5.7 Unknown Words

Words that fail both the HashMap lookup and the Snowball stem fallback are collected in `unknown_words`. These are:
- Proper nouns not in the vocabulary
- Neologisms, slang, brand names
- Domain-specific jargon (e.g., "Kubernetes", "GraphQL")
- Typos

**Stop word filtering:** Before inclusion in `unknown_words`, tokens are checked against a built-in set of ~130 English stop words (`STOP_WORDS`). Common function words like "the", "and", "is", etc. are excluded even when they aren't in the dictionary — they carry no topical signal and would create noise for downstream vocabulary extension workflows. Stop words do NOT affect coverage calculation — coverage uses the raw count of matched + unmatched tokens.

The unknown words list is a first-class output signal. Downstream consumers can use it to detect vocabulary gaps, trigger fallback strategies, or build corpus-specific extensions (see the vocabulary extension API in [Section 7](#7-api-surface)).

---

## 6. Scoring Mathematics

### IDF Formula

IDF is used for custom word injection via `calc_custom_idf()`:

```
IDF(word) = ln((N - n + 0.5) / (n + 0.5) + 1)
```

Where:
- N = 298 (total towns)
- n = number of towns containing the word

### Weight-Based Scoring

V3 uses pre-computed per-town min-max normalized weights (u8, 0-255) rather than BM25 term scores. Weights are divided by `WEIGHT_SCALE` (100) at scoring time. This simplifies the runtime to pure integer accumulation followed by a single dequantization step.

### Background Subtraction

Some towns consistently appear in top results regardless of input topic. Background subtraction handles noise continuously:

```
z[town] = (raw[town] - alpha * bg_mean[town]) / bg_std[town]
```

Where alpha ramps linearly from 0 (1 matched word) to 1.0 (6+ matched words). A `bg_std` floor of 0.01 prevents division by near-zero values.

A noisy town with high bg_mean needs a much higher raw score to achieve a high z-score than a clean town with low bg_mean. The penalty is proportional to noisiness.

---

## 7. API Surface

### load(data_dir) -> ReefScorer

Load all binary data files from the given directory. Validate the manifest version. Build the Aho-Corasick automaton from compounds.bin. Return a ready-to-use scorer.

**Errors:**
- Missing or corrupt data files
- Version mismatch in manifest
- Checksum mismatch
- Invalid data (e.g., reef_id >= N_REEFS)

### score(text) -> TopicResult

Score a single text string. Runs the full 5-phase pipeline. Returns a TopicResult.

**Edge cases:**
- Empty string: TopicResult with zero confidence, empty top_reefs, coverage=0
- All unknown words: TopicResult with zero confidence, all words in unknown_words
- Single matched word: TopicResult with low confidence, matched_words=1
- Very long input (1000+ words): works correctly, linear scaling

### score_batch(texts) -> list[TopicResult]

Score multiple texts. Semantically equivalent to `[score(t) for t in texts]` but may optimize memory allocation (reuse the `[f32; 298]` scratch array across calls).

### analyze(text, *, sensitivity=1.0, smooth_window=2, min_chunk_sentences=2, max_chunk_sentences=30) -> DocumentAnalysis

Segment a document by topic shifts. Accepts raw text (string) or a pre-split list of sentences.

Internally: scores each sentence to get z-score vectors and per-sentence `TopicResult` objects in a single pass, smooths vectors, computes cosine similarity between adjacent sentences, detects boundaries at similarity valleys, and enforces chunk size constraints.

**Parameters:**
- `text`: Raw text string or `list[str]` of pre-split sentences
- `sensitivity`: Boundary detection threshold (default 1.0). Lower = more boundaries.
- `smooth_window`: Sliding window size for z-score vector smoothing (default 2)
- `min_chunk_sentences`: Minimum sentences per segment (default 2). Undersized segments are merged with predecessors.
- `max_chunk_sentences`: Maximum sentences per segment (default 30). Oversized segments are split at weakest internal similarity boundaries. Set to 0 to disable.

Returns a `DocumentAnalysis` with `segments` (list of `TopicSegment`), `boundaries`, `n_sentences`, and `n_segments`. Each `TopicSegment` includes `sentence_results` — per-sentence `TopicResult` objects with `matched_word_ids`, `unknown_words`, `top_reefs`, and `coverage`.

### lookup_word(word) -> Optional[WordInfo]

Look up a single word in the dictionary. Applies the same normalization as Phase 2 (lowercase, then HashMap lookup, then Snowball stem fallback). Returns WordInfo if found, None if unknown.

Useful for debugging and for downstream consumers that want to inspect individual word properties (specificity, IDF) before scoring.

### filter_unknown(words) -> list[str]

Batch-filter a list of words, returning only those unknown to the dictionary. Applies the same lookup pipeline as scoring (HashMap + Snowball stemmer fallback). Stop words are excluded from the result — common function words are not considered "unknown" even if absent from the dictionary.

Useful for efficiently identifying vocabulary gaps across a document without full scoring.

### Vocabulary Extension API

These methods support extending the scorer's vocabulary at runtime with custom words learned from document context.

#### next_word_id (property) -> int

Returns the next available word_id (current length of the internal word_reefs list). Custom words are assigned sequential IDs starting from this value.

#### calc_custom_idf(n_associated_reefs) -> int

Compute quantized IDF (u8) for a custom word given the number of towns it appears in.

**Parameters:**
- `n_associated_reefs`: Number of towns containing the word (1 to 298)

**Returns:** Quantized IDF (u8, scale factor 51). Higher values = more specific (fewer towns).

**Validation:** Raises `ValueError` if n_associated_reefs < 1 or exceeds total towns.

#### calc_custom_weight(reef_id, strength) -> int

Compute a per-town calibrated weight_q for a custom word.

**Parameters:**
- `reef_id`: Town index (0..297)
- `strength`: Association strength (0.0 to 1.0+)

**Returns:** `round(p75[reef_id] * strength)`, clamped to u8 (0-255). The weight is calibrated to each town's 75th-percentile base vocabulary weight, so `strength=1.0` produces a weight equal to a typical word in that town.

#### add_custom_word(word, reef_weights, *, idf_q, specificity=2, tag=0) -> WordInfo

Full injection pipeline for a custom word with pre-computed weights:
1. Normalize to lowercase, whitespace-normalize
2. Compute FNV-1a hash
3. Validate: word not empty, not a duplicate, valid specificity (-2 to +2), non-empty reef_weights, valid reef_ids and weight_q values (0-255), valid idf_q (0-255)
4. Allocate next word_id
5. Inject into scorer's lookup structures
6. If `tag != 0`, record in internal tag index for `get_word_tags()` lookup

Use `calc_custom_idf()` and `calc_custom_weight()` to compute `idf_q` and weight values before calling.

**Parameters:**
- `word`: The word string (will be lowercased and whitespace-normalized)
- `reef_weights`: List of `(reef_id, weight_q)` pairs with pre-computed weights (u8, 0-255)
- `idf_q`: Pre-computed quantized IDF (u8, 0-255)
- `specificity`: Sigma band (-2 to +2), default 2 (highly specific)
- `tag`: Opaque consumer metadata (default 0). Stored on the `WordInfo` and indexed for O(1) lookup via `get_word_tags()`. Lagoon never inspects this value.

**Returns:** `WordInfo` with the assigned word_hash, word_id, specificity, idf_q, and tag.

**Errors:** Raises `ValueError` for empty word, duplicate word, invalid specificity, empty reef_weights, invalid reef_id (>= 298), or invalid weight_q/idf_q (outside 0-255).

#### get_word_tags(word_ids) -> dict[int, int]

Return non-zero tags for the given word_ids. Only entries whose `tag != 0` are included — base vocabulary words (tag 0) are omitted. This lets callers distinguish custom-injected words from base words by membership alone.

**Parameters:**
- `word_ids`: A `frozenset[int]` or `set[int]` of word_ids to query (typically `result.matched_word_ids`)

**Returns:** `{word_id: tag}` dict containing only non-zero-tagged entries. Empty dict if no tagged words are in the input set.

#### rebuild_compounds(additional_compounds)

Additively merge custom compound words into the Aho-Corasick automaton alongside the base ~64K compounds.

**Parameters:**
- `additional_compounds`: List of `(compound_string, word_id)` tuples

After calling this method, `score()` and `analyze()` will match the new compounds during Phase 1 (Aho-Corasick scan).

### STOP_WORDS

`lagoon.STOP_WORDS` — a `frozenset[str]` of ~130 minimal English stop words. Includes single letters, determiners, conjunctions, prepositions, pronouns, common verb forms, modals, and contraction fragments.

Used internally by `score()` and `filter_unknown()` to exclude noise from `unknown_words` lists. Also available for direct use by downstream consumers.

### split_sentences(text) -> list[str]

Split raw text into sentences using a regex-based sentence boundary detector. Used internally by `analyze()` when given a raw string.

---

## 8. Output Format

### TopicResult

| Field | Type | Description |
|-------|------|-------------|
| top_reefs | list[ScoredReef] | Top-K towns ranked by z-score (K=10 default) |
| top_islands | list[ScoredIsland] | Islands with contributing towns, ranked by aggregate z |
| arch_scores | [f32; 6] | Archipelago-level score distribution |
| confidence | f32 | Top z-score (clamped to 0) |
| coverage | f32 | Fraction of input words that matched the dictionary |
| matched_words | int | Count of matched words |
| unknown_words | list[str] | Words that failed lookup + stem (stop words excluded) |
| matched_word_ids | frozenset[int] | Set of word_ids that matched (for cross-referencing with custom vocabulary) |
| n_domainless | int | Words recognized but not domain-specific |
| valence_signal | f32 | z-score-weighted mean of town valences |

### ScoredReef

| Field | Type | Description |
|-------|------|-------------|
| reef_id | u16 | Town index (0..297) |
| z_score | f32 | Background-subtracted score |
| raw_score | f32 | Pre-subtraction score |
| n_contributing_words | u16 | Words that contributed to this town |
| name | str | Town name |
| quality_score | f32 | z-score (IQF modulation placeholder) |
| valence | f32 | Town valence |
| avg_specificity | f32 | Average word specificity in this town |
| resolved_sub_reef_id | int \| None | Resolved sub-town reef id (currently always None in V3 — not yet wired up) |
| resolved_sub_reef_name | str \| None | Resolved sub-town reef name (currently always None in V3) |

### ScoredIsland

| Field | Type | Description |
|-------|------|-------------|
| island_id | u8 | Island index (0..39) |
| aggregate_z | f32 | Sum of child town z-scores |
| n_contributing_reefs | u16 | Towns contributing to this island |
| name | str | Island name |

**Interpretation guidance:**
- `confidence > 3.0`: Strong topic signal, #1 town is clearly dominant
- `confidence < 1.0`: Ambiguous — text may be generic or cross-domain
- `coverage > 0.8`: Good dictionary coverage
- `coverage < 0.5`: Many unknown words — results may be unreliable
- `z_score > 4.0` for a town: Very strong match (rare for generic text)

---

## 9. Performance Characteristics

### Memory Footprint

| Structure | Size | Notes |
|-----------|------|-------|
| word_lookup | ~4 MB | u64 keys, ~150K entries |
| word_towns | varies | entries/word x ~150K words |
| town_meta | ~20 KB | 298 records |
| reef_meta | ~260 KB | 3,885 records |
| island_meta | ~2 KB | 40 records |
| bg_mean + bg_std | ~2.4 KB | 298 x 2 x f32 |
| compound automaton | ~1.3 MB | Aho-Corasick over ~64K strings |
| word_detail | varies | per-word hierarchy detail |

### Latency (Pure Python, Measured)

All timings measured with `pytest-benchmark` on CPython 3.11. A Rust port is expected to be significantly faster — Rust targets are noted in Section 5 per-phase breakdowns.

| Input size | Measured | Breakdown |
|------------|----------|-----------|
| 8-word phrase | ~56 μs | ~1μs compounds + ~7μs tokenize + ~5μs accumulate + ~9μs bg-sub + ~25μs extract |
| 30-word sentence | ~96 μs | ~3μs compounds + ~31μs tokenize + ~20μs accumulate + ~9μs bg-sub + ~25μs extract |
| 140-word passage | ~207 μs | Dominated by tokenization and weight accumulation |
| 200-word paragraph | ~337 μs | Linear scaling, tokenization dominant |
| 1000+ words | ~2.2 ms | Linear scaling |

### Startup

~1.5s — deserialize binary files + build Aho-Corasick automaton (pure Python MessagePack deserialization + pyahocorasick automaton construction over ~64K compounds). V3 loads additional files (word_detail.bin, bucket_words.bin, town_meta.bin) but startup time remains similar.

### Cache Behavior

The `[f32; 298]` town score array is ~1.2 KB — fits in L1 cache. No memory pressure during scoring.

### Bottleneck Analysis (Pure Python)

- **Python loop overhead:** The dominant cost in pure Python. Individual operations (hash lookup ~50ns, stemmer ~120ns) are fast, but Python's per-iteration overhead (~1μs/word for tokenization, ~650ns/word for accumulation) adds up.
- **Result extraction:** ~25μs due to dataclass construction and Python sort over 298 elements — disproportionately expensive for short inputs.
- **Background subtraction:** ~9μs for 298-element loop — trivial in Rust (~200ns) but meaningful in Python.
- **Aho-Corasick scan:** O(n) in text length, implemented in C (pyahocorasick), fast even in Python (~1-3μs for sentences).
- **Snowball stemmer fallback:** ~120ns per word (C extension), fires ~30% of the time.
- **HashMap lookups:** ~50ns per lookup. Dominant cost for long inputs in Rust. In Python, the loop overhead around lookups is the real bottleneck.
- **Startup:** ~1.5s is dominated by MessagePack deserialization of ~150K word_lookup entries and Aho-Corasick automaton construction over ~64K compounds.

---

## 10. Constants Reference

| Constant | Value | Description |
|----------|-------|-------------|
| N_TOWNS | 298 | Total number of towns (scoring units) |
| N_REEFS | 3,885 | Total number of reefs (finest-grained clusters) |
| N_ISLANDS | 40 | Total number of islands |
| N_ARCHS | 6 | Total number of archipelagos |
| WEIGHT_SCALE | 100 | Dequantization factor for u8 weights |
| IDF_SCALE | 51 | u8 quantization factor for IDF |
| FNV1A_OFFSET | 14695981039346656037 | FNV-1a 64-bit offset basis |
| FNV1A_PRIME | 1099511628211 | FNV-1a 64-bit prime |

---

## 11. FNV-1a Hash Implementation

FNV-1a (Fowler-Noll-Vo) is the hash function used for all word lookups. It produces a 64-bit hash with zero collisions across the full ~150K vocabulary.

### Algorithm

1. Start with the offset basis (14695981039346656037)
2. For each byte of the UTF-8 encoded input string:
   a. XOR the hash with the byte
   b. Multiply by the FNV prime (1099511628211)
   c. Mask to 64 bits

### Python Implementation

```python
FNV1A_OFFSET = 14695981039346656037
FNV1A_PRIME = 1099511628211

def fnv1a_u64(s: str) -> int:
    h = FNV1A_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV1A_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
```

### Rust Implementation

```rust
const FNV1A_OFFSET: u64 = 14695981039346656037;
const FNV1A_PRIME: u64 = 1099511628211;

fn fnv1a_u64(s: &str) -> u64 {
    let mut h = FNV1A_OFFSET;
    for byte in s.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(FNV1A_PRIME);
    }
    h
}
```

### Properties

- **Deterministic:** Same input always produces the same hash
- **UTF-8 aware:** Operates on the byte-level encoding, handling all Unicode correctly
- **Zero collisions:** Verified across the full ~150K vocabulary — no two distinct strings hash to the same u64 value
- **Fast:** Linear in string length, no allocations, branch-free inner loop
- **Cross-language:** Byte-level operation means Python and Rust produce identical hashes for the same input

**Input normalization:** All lookups use lowercased, whitespace-normalized strings. Multi-word compounds use single spaces as separators. This normalization must be applied before hashing.

---

## 12. Canonical Test Cases

These test sentences establish ground-truth expectations for any lagoon implementation. Available town names include Neurology (61), Neuropsychology (177), Literary Criticism (208), Novel (210), Drama & Playwriting (224), Theater (227), Acoustics (124), and many more — see the V3.1 town list for the full set.

### Test 1: Generic text

**Input:** `"now is the time for all good men to come to the aid of the country"`

**Expected:**
- Low confidence (< 1.5) — flat z-score distribution
- No dominant topic — the text IS generic
- Positive coverage (most words are known)

### Test 2: Huck Finn passage

**Input:** `"It was after sun-up now, but we went right on and didn't tie up. The king and the duke turned out by-and-by looking pretty rusty; but after they'd jumped overboard and took a swim it chippered them up a good deal..."`

**Expected:**
- A literary/dramatic/theatrical town in top results — the passage is from a novel and includes a theatrical rehearsal of Romeo & Juliet. Towns like "Literary Criticism", "Novel", "Drama & Playwriting", or "Theater" should appear in the top 5
- Confidence > 2.0 — the passage has strong domain vocabulary
- **V3.1 data quality issue:** Currently produces "Theology" #1, "Board Games" #2. Root cause: "jackass" maps to Board Games (w=153), "king" maps to Organic Chemistry (w=255), "juliet" is not in vocabulary at all, and most passage words are treated as domainless. No literary/dramatic town appears anywhere in the top results despite towns 208, 210, 224, 227 existing

### Test 3: Waveform passage

**Input:** `"The sine wave has a pattern that repeats. The frequency determines the pitch. Amplitude controls the volume. The waveform oscillates between positive and negative values."`

**Expected:**
- Physics/acoustics/mathematics towns should dominate the top 3 — "Acoustics", "Electromagnetism", "Algebra" are all valid
- No music/performance towns in top 3 — this is physics, not music
- **V3.1 data quality issue:** "Opera" currently appears as #2 (z=4.58) for a pure physics passage. "Algebra" at #1 is correct

### Test 4: Neuroscience words

**Input:** `"neuron synapse axon dendrite cortex brain neural hippocampus"`

**Expected:**
- A neuroscience-related town as #1 — "Neurology" (town 61) or "Neuropsychology" (town 177) should dominate when given 8 pure neuroscience words
- Full coverage (1.0), all 8 words matched
- **V3.1 data quality issue:** Currently produces "Obstetrics" as #1 (z=6.37). Root cause: 5 of 8 words ("neuron", "cortex", "brain", "neural", "synapse") have specificity 0 or -1 and spread uniformly across dozens of unrelated towns (Algorithms, AI, Cloud Computing, etc.). Only "axon", "dendrite", and "hippocampus" have specific associations — and those map primarily to Obstetrics and Nanotechnology instead of Neurology. Town 61 (Neurology) is not hit by any of the 8 neuroscience words

### Test 5: Topic shift (split Huck Finn)

Split the Huck Finn passage at the midpoint:
- Two halves should have different top-5 town rankings — the first half describes a physical scene while the second half focuses on theatrical rehearsal
- Two halves' z-score distributions should be measurably divergent

---

## 13. Known Limitations

1. **Mostly bag-of-words.** For short inputs (< 4 matched words), word order is fully ignored — "dog bites man" and "man bites dog" produce identical results. For longer inputs (4+ matched words), the contextual scorer processes words in text order and applies island-coherence boosting based on sequential activation patterns. This means word order can influence scores slightly through the coherence bonus, but the effect is modest (up to 30% boost via `_CONTEXT_GAMMA`). There is no compositional semantics — the system does not understand phrases or syntax.

2. **Fixed base vocabulary.** Neologisms, slang, brand names, and domain-specific jargon not in the base vocabulary will be missed. The stemmer fallback helps with inflections but not with truly unknown words. The vocabulary extension API (`add_custom_word()`) allows runtime injection of custom words with learned reef associations — see [Section 7](#7-api-surface).

3. **Abstract domains are weak.** Software engineering, philosophy, and other highly abstract domains produce low-confidence results because their vocabulary spreads thinly across many reefs. The system's strength is concrete, domain-specific vocabulary (science, literature, physical descriptions).

4. **No compositional semantics.** Phrase meaning isn't captured beyond compound matching. "Not good" scores the same as "good" for reef convergence.

5. **English only.** The source vocabulary and embedding model are English-specific.

6. **Static base vocabulary.** The base dictionary is frozen at build time. New base words require a rebuild of the data files. However, the vocabulary extension API allows injecting custom words at runtime with learned reef associations — useful for domain-specific corpora where the base vocabulary has gaps.

7. **298-town ceiling for scoring.** Topic detection granularity at the scoring level is bounded by 298 towns. The 3,885 reefs provide finer resolution via `V3ReefMeta` and sub-town reef resolution, but the primary scoring operates at the town level.

---

## 14. Portability Notes

Lagoon is designed to port from Python to Rust without architectural changes:

- **No pickle.** All serialized data uses MessagePack — language-agnostic, supported in both Python and Rust (`rmp-serde`).

- **Flat-array / mmap-friendly.** The `[f32; 298]` score arrays and metadata arrays are contiguous and fixed-stride. In Rust, the fixed-size arrays can potentially be mmapped directly with zero deserialization.

- **UTF-8 strings, pre-normalized.** All strings in the data files are lowercase, whitespace-normalized UTF-8. No runtime Unicode normalization needed beyond lowercasing the input.

- **No dynamic dispatch.** All types are concrete, all methods are monomorphic. No trait objects needed in the Rust port.

- **No reference counting / GC dependency.** All data structures are owned. In Python, plain dataclasses with no circular references. In Rust, owned `Vec`, `HashMap`, and `String`.

- **Fixed-size numeric types.** town_id is u16, word_id is u32, weight is u8. These are documented in the data structure definitions and must match between the export tool and lagoon.

- **FNV-1a is byte-level.** The hash function operates on UTF-8 bytes, producing identical results in Python and Rust for the same input string. No platform-dependent behavior.

- **Aho-Corasick is standard.** The `aho-corasick` crate in Rust is mature and well-optimized. Python has `pyahocorasick` or `ahocorasick-rs`.

- **No external ML dependencies.** Lagoon has zero dependency on embedding models, NLTK, or any ML framework. The only non-trivial dependency is the Aho-Corasick implementation and a Snowball stemmer for runtime fallback lookups.
