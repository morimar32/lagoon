# Lagoon

Lagoon is a scoring library, based on the work (and exports) from [Windowsill](https://github.com/morimar32/windowsill). The Windowsill database export can also be found directly on [Huggingface](https://huggingface.co/datasets/morimar/windowsill).

Lagoon is meant to be the "glue" that supports a number of downstream projects meant to consume this information for different use cases. Once the project hits a certain degree of maturity, the plan is to create a Rust crate of this library (likely `lagoon-rs`).

**New here?** See the [Quickstart Guide](QUICKSTART.md) to get up and running in minutes.

## 1. What is Lagoon

Lagoon is a standalone text-scoring library that maps arbitrary input text to a hierarchy of 207 semantic reefs. It loads a set of pre-built binary data files at startup (~16 MB on disk) and scores text in ~10-15 microseconds per sentence using depth-weighted BM25 with background subtraction.

**What lagoon does:**
- Loads binary data files produced by the export tool from [Windowsill](https://github.com/morimar32/windowsill)
- Scores text against 207 reefs using precomputed BM25 term scores
- Returns ranked reefs, islands, and archipelagos with z-scores, confidence, and coverage
- Provides compound-aware tokenization via Aho-Corasick automaton

**What lagoon does NOT do:**
- Build the reef hierarchy (that's the upstream pipeline)
- Produce the binary data files (that's the export tool)
- Make routing decisions or store documents (that's a downstream consumer)

**Design philosophy:** Python first, Rust later. All data structures and serialization formats are chosen to port cleanly to Rust without architectural changes. No pickle, no Python-specific formats, no dynamic dispatch.

---

## 2. Concepts

### 2.1 The Reef Hierarchy

The data files encode a three-generation hierarchy of semantic clusters, derived from embedding space decomposition:

**Archipelagos (4 total)** — The broadest semantic regions:
- Natural sciences and taxonomy
- Physical world and materiality
- Abstract processes and systems
- Social order and assessment

Too coarse for meaningful topic discrimination — most sentences spread across all four.

**Islands (52 total)** — Mid-level semantic communities. Each island contains 2-8 reefs. Useful for moderate-confidence topic summarization (e.g., "perception and attributes", "behavior and emotional states").

**Reefs (207 total)** — The finest-grained semantic clusters. Each reef groups 2-17 embedding dimensions that share statistically significant word overlap. Reef names describe coherent semantic neighborhoods: "archaic literary terms", "coastal landscapes and frontiers", "neural and structural". Reefs are the primary scoring target.

The hierarchy is encoded as a packed u16 address per reef: `arch(2)|island(6)|reef(8)`. Extract fields:
- `reef_id = addr & 0xFF`
- `island_id = (addr >> 8) & 0x3F`
- `arch_id = (addr >> 14) & 0x03`

### 2.2 What a Reef Score Means

A reef score quantifies how strongly the input text's vocabulary converges on a particular semantic neighborhood. Individual word lookups are noisy — most words spread across 10-20 reefs with only 10-15% concentration in their top reef. **The signal emerges from convergence across words, not from any single word.** When multiple words in the input independently activate the same reef, and that activation exceeds what random text would produce (background subtraction), the reef score is meaningful.

High-specificity words (specificity = +2, only ~1,615 in the vocabulary) are an exception — they concentrate 30%+ in their top reef and touch only 1-4 reefs total. A single specificity=+2 word is a strong signal on its own.

### 2.3 Why BM25 Over Reefs

The scoring engine treats reefs as "documents" and input words as "query terms", applying BM25 — a well-studied information retrieval formula. This works because the problem is structurally identical to document retrieval: find which reef (document) best matches the input words (query), accounting for term frequency, inverse document frequency, and document length.

BM25 over reefs has three advantages over raw enrichment counting:
1. **IDF** naturally up-weights domain-specific words and down-weights generic ones
2. **Length normalization** via the `b` parameter penalizes large reefs that catch words by chance
3. **Depth-weighted tf** (`n_dims / reef_total_dims` instead of binary 0/1) captures how "at home" a word is in a reef, distinguishing residents from visitors

Background subtraction then converts raw BM25 scores to z-scores, suppressing reefs that activate regardless of input topic (noise magnets) and amplifying reefs where activation is genuinely surprising.

---

## 3. Data Format Specification

### 3.1 Overview

Lagoon loads a directory of binary data files at startup. All files use MessagePack serialization. The directory also contains a `manifest.json` with version, checksums, and build metadata.

```
lagoon_data/
  manifest.json          # version, file checksums, build timestamp
  word_lookup.bin        # HashMap<u64, WordInfo>
  word_reefs.bin         # word_id -> list[ReefEntry]
  reef_meta.bin          # [ReefMeta; 207]
  island_meta.bin        # [IslandMeta; 52]
  background.bin         # bg_mean [f32; 207] + bg_std [f32; 207]
  compounds.bin          # compound strings + word_id mapping
  constants.bin          # runtime constants + reef side arrays
```

The manifest includes a `version` field. Lagoon validates this on load and fails fast with a clear error on mismatch.

### 3.2 word_lookup.bin

A `HashMap<u64, WordInfo>` of approximately 173K entries mapping FNV-1a u64 hashes to word metadata.

**Contents:**
- ~147K base words from the vocabulary (each word's FNV-1a hash maps to its WordInfo). Morphy variants resolve to existing base word hashes, so they don't add new entries.
- ~27K Snowball stemmer mappings for additional coverage (stems of base words and morphy variants that hash to values not already in the dictionary)

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

### 3.3 word_reefs.bin

An array indexed by word_id, where each entry is a list of `[reef_id, bm25_q]` pairs. Average ~13 reef entries per word. Index 0 is unused (word_ids are 1-indexed).

**MessagePack structure:**
```
[
  [],                           # index 0 (unused)
  [[reef_id, bm25_q], ...],    # word_id 1
  [[reef_id, bm25_q], ...],    # word_id 2
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| reef_id | u8 | Which reef (0..206) |
| bm25_q | u16 | Precomputed BM25 term score, quantized: `round(score * 8192)`, decode: `bm25_q / 8192.0` |

Entries are sorted by reef_id within each word's list for cache-friendly iteration during BM25 accumulation.

BM25 term scores are **fully precomputed** at export time. At runtime, scoring is pure accumulation — no IDF lookup, no tf computation, no length normalization needed.

### 3.4 reef_meta.bin

An array of 207 ReefMeta records, indexed directly by reef_id.

**MessagePack structure:**
```
[
  {"hierarchy_addr": <u16>, "n_words": <u32>, "name": "<string>"},
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| hierarchy_addr | u16 | Bit-packed hierarchy: `arch(2)\|island(6)\|reef(8)` |
| n_words | u32 | Total words in this reef |
| name | string | Human-readable name (e.g., "archaic literary terms") |

### 3.5 island_meta.bin

An array of 52 IslandMeta records, indexed by island_id.

**MessagePack structure:**
```
[
  {"arch_id": <u8>, "name": "<string>"},
  ...
]
```

| Field | Type | Description |
|-------|------|-------------|
| arch_id | u8 | Parent archipelago (0..3) |
| name | string | Human-readable name (e.g., "perception and attributes") |

### 3.6 background.bin

Two fixed-size arrays pre-computed by sampling 1000 random 15-word subsets from the single-word vocabulary, scoring each against all reefs, and recording per-reef statistics.

**MessagePack structure:**
```
{
  "bg_mean": [f32; 207],
  "bg_std": [f32; 207]
}
```

| Array | Type | Description |
|-------|------|-------------|
| bg_mean | [f32; 207] | Mean BM25 score per reef across random samples |
| bg_std | [f32; 207] | Standard deviation per reef across random samples |

**Size:** 207 x 2 x 4 bytes = 1,656 bytes.

At runtime: `z = (raw_bm25 - bg_mean[reef]) / bg_std[reef]` converts noisy BM25 scores into "how surprising is this reef activation given random input?"

Reefs with high bg_mean (~1.74) are noise magnets that absorb vocabulary indiscriminately. Reefs with low bg_mean (~0.69) are highly discriminating — when they activate, it means something.

### 3.7 compounds.bin

A list of compound (multi-word) entries for building an Aho-Corasick automaton at load time.

**MessagePack structure:**
```
[
  ["<compound string>", <word_id>],
  ...
]
```

Since ~43% of the vocabulary is multi-word expressions, matching compounds as single units before falling back to individual word lookups significantly improves precision. "Heart attack" as a unit produces focused medical activation; "heart" + "attack" separately scatters across cardiac + violence reefs.

The automaton is built at load time from this list. In Python, use `ahocorasick` or `pyahocorasick`. In Rust, use the `aho-corasick` crate.

### 3.8 constants.bin

All runtime constants and reef-level side arrays.

**MessagePack structure:**
```
{
  "N_REEFS": 207,
  "N_ISLANDS": 52,
  "N_ARCHS": 4,
  "avg_reef_words": <float>,
  "k1": 1.2,
  "b": 0.75,
  "IDF_SCALE": 51,
  "BM25_SCALE": 8192,
  "FNV1A_OFFSET": 14695981039346656037,
  "FNV1A_PRIME": 1099511628211,
  "reef_total_dims": [<u8>; 207],
  "reef_n_words": [<u32>; 207]
}
```

The reef-level side arrays are stored here rather than duplicated per word-reef entry:
- `reef_total_dims` — total embedding dimensions per reef
- `reef_n_words` — total words per reef

### 3.9 manifest.json

```json
{
  "version": "1.0",
  "format": "msgpack",
  "build_timestamp": "2026-02-13T06:09:55Z",
  "files": {
    "word_lookup.bin": "<sha256>",
    "word_reefs.bin": "<sha256>",
    ...
  },
  "stats": {
    "n_reefs": 207,
    "n_islands": 52,
    "n_archs": 4,
    "n_words": 146698,
    "n_lookup_entries": 173286,
    "n_words_with_reefs": 146695,
    "n_compounds": 63912
  }
}
```

**Note:** When deserializing `word_lookup.bin`, the MessagePack unpacker must be configured to accept integer map keys (e.g., `strict_map_key=False` in Python's `msgpack` library).

Lagoon should validate the version on load and verify file checksums.

### 3.10 Quantization Scheme

All quantization is applied at export time only. The source database stores full DOUBLE precision values; the binary export files use fixed-point integer representations for compactness and cache efficiency.

**IDF quantization (u8, scale factor 51):**
- Encode: `idf_q = round(idf * 51)`
- Decode: `idf = idf_q / 51.0`
- IDF range: [1.80, 4.93] -> u8 range: [92, 251] — well within u8 capacity (0-255)
- Max quantization error: 1/102 = 0.0098

**BM25 quantization (u16, scale factor 8192):**
- Encode: `bm25_q = clamp(round(score * 8192), 0, 65535)`
- Decode: `score = bm25_q / 8192.0`
- Max quantization error: 1/16384 = 0.000061
- Simulation results: **0 ranking errors** at u16 precision across the full vocabulary

At u8 BM25 precision (scale=255), 1.3% ranking errors were observed, confirming u16 as the correct choice for BM25 scores.

---

## 4. Data Structures

### ReefScorer (top-level)

```python
@dataclass
class ReefScorer:
    word_lookup: dict[int, WordInfo]        # HashMap<u64, WordInfo>, ~173K entries
    word_reefs: list[list[ReefEntry]]       # indexed by word_id
    reef_meta: list[ReefMeta]               # indexed by reef_id, len=207
    island_meta: list[IslandMeta]           # indexed by island_id, len=52
    bg_mean: list[float]                    # len=207
    bg_std: list[float]                     # len=207
    compound_ac: AhoCorasick                # built at load time from compounds.bin
    compound_word_ids: list[int]            # compound match index -> word_id
    reef_total_dims: list[int]              # [u8; 207]
    reef_n_words: list[int]                 # [u32; 207]
    avg_reef_words: float                   # ~5,282
```

**Rust equivalent:** `word_lookup` becomes `HashMap<u64, WordInfo>`, `word_reefs` becomes `Vec<Vec<ReefEntry>>`, and the f32 arrays become `[f32; 207]`. All fields are flat, owned types — no reference counting, no dynamic dispatch. BM25 parameters are baked into precomputed scores at export time and are not needed at runtime for scoring; `avg_reef_words` is retained only for reference.

### WordInfo

```python
@dataclass
class WordInfo:
    word_hash: int        # u64 — FNV-1a hash (also the lookup key)
    word_id: int          # u32 — index into word_reefs
    specificity: int      # i8 — sigma band: +2 to -2
    idf_q: int            # u8 — quantized IDF
```

**Rust:** All fields are fixed-size integers. `#[repr(C)]` layout matches serialized format.

### ReefEntry

```python
@dataclass
class ReefEntry:
    reef_id: int          # u8 — which reef (0..206)
    bm25_q: int           # u16 — precomputed BM25 term score
```

**Rust:** 3 bytes packed. `reef_id: u8, bm25_q: u16`.

### ReefMeta

```python
@dataclass
class ReefMeta:
    hierarchy_addr: int   # u16 — bit-packed arch|island|reef
    n_words: int          # u32
    name: str             # UTF-8, pre-normalized
```

### IslandMeta

```python
@dataclass
class IslandMeta:
    arch_id: int          # u8 — parent archipelago (0..3)
    name: str             # UTF-8
```

### TopicResult (scoring output)

```python
@dataclass
class TopicResult:
    top_reefs: list[ScoredReef]           # Top-K reefs by z-score
    top_islands: list[ScoredIsland]       # Island-level rollup
    arch_scores: list[float]              # len=4, archipelago distribution
    confidence: float                      # z-score gap: #1 - #2 reef
    coverage: float                        # matched_words / total_input_words
    matched_words: int                     # words that hit the dictionary
    unknown_words: list[str]              # words that failed lookup + stem

@dataclass
class ScoredReef:
    reef_id: int                          # u8
    z_score: float                        # f32 — background-subtracted
    raw_bm25: float                       # f32 — pre-subtraction
    n_contributing_words: int             # u16

@dataclass
class ScoredIsland:
    island_id: int                        # u8
    aggregate_z: float                    # f32 — sum of child reef z-scores
    n_contributing_reefs: int             # u16
```

**Rust:** `unknown_words` becomes `Vec<String>`. The rest are flat numeric types. `top_reefs` and `top_islands` are small `Vec`s (typically K=10), allocated per call.

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
Phase 3: BM25 accumulation (precomputed scores -> [f32; 207])
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

**Performance:** O(n) where n = text length. Typically ~100-500ns for a sentence.

### 5.3 Phase 2: Tokenize + Normalize

Split the unconsumed text segments on whitespace and punctuation. For each token:

1. Lowercase the token
2. Skip if it falls within a consumed compound span
3. Compute `fnv1a_u64(token)` and look up in `word_lookup`:
   - **Hit:** record the word_id
   - **Miss:** run Snowball stemmer on the token, compute `fnv1a_u64(stem)`, look up again
     - **Hit:** record the word_id
     - **Miss:** add to `unknown_words` list

**Deduplication:** Track unique word_ids. If the same word appears multiple times, it contributes only once to BM25 (binary occurrence, not frequency). The signal comes from _how many different words_ converge on a reef, not from repetition.

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

**Performance per word:**
- HashMap lookup: ~50ns
- Snowball stemmer fallback: ~200ns (fires ~30% of the time)

### 5.4 Phase 3: BM25 Accumulation

Initialize a score array and accumulate precomputed BM25 scores:

```python
def accumulate_bm25(self, word_ids):
    scores = [0.0] * 207
    word_counts = [0] * 207  # contributing words per reef
    for word_id in word_ids:
        for reef_id, bm25_q in self.word_reefs[word_id]:
            scores[reef_id] += bm25_q / 8192.0
            word_counts[reef_id] += 1
    return scores, word_counts
```

BM25 term scores are fully precomputed. At runtime, this is pure accumulation — no IDF lookup, no tf computation, no length normalization. One multiply-add per reef entry.

**Performance per word:** ~80ns (iterate ~13 reef entries, one multiply-add each).

### 5.5 Phase 4: Background Subtraction

Convert raw BM25 scores to z-scores:

```python
def subtract_background(self, scores):
    z_scores = [0.0] * 207
    for reef in range(207):
        z_scores[reef] = (scores[reef] - self.bg_mean[reef]) / self.bg_std[reef]
    return z_scores
```

Guard: if `bg_std[reef]` is zero (should not happen — epsilon floor applied at export), set z to 0.0.

**Performance:** ~200ns total (two float ops per reef x 207).

### 5.6 Phase 5: Result Extraction

```python
def extract_results(self, z_scores, raw_scores, word_counts, matched, unknown):
    # Top-K reefs by z-score
    indexed = [(z, i) for i, z in enumerate(z_scores)]
    indexed.sort(reverse=True)
    top_reefs = [
        ScoredReef(reef_id=i, z_score=z, raw_bm25=raw_scores[i],
                   n_contributing_words=word_counts[i])
        for z, i in indexed[:10]
    ]

    # Confidence: gap between #1 and #2
    confidence = indexed[0][0] - indexed[1][0] if len(indexed) >= 2 else 0.0

    # Coverage
    total_words = len(matched) + len(unknown)
    coverage = len(matched) / total_words if total_words > 0 else 0.0

    # Island rollup
    island_z = defaultdict(lambda: [0.0, 0])
    for reef in top_reefs:
        island_id = (self.reef_meta[reef.reef_id].hierarchy_addr >> 8) & 0x3F
        island_z[island_id][0] += reef.z_score
        island_z[island_id][1] += 1
    top_islands = sorted([
        ScoredIsland(island_id=iid, aggregate_z=agg, n_contributing_reefs=n)
        for iid, (agg, n) in island_z.items()
    ], key=lambda x: x.aggregate_z, reverse=True)

    # Archipelago rollup
    arch_scores = [0.0] * 4
    for island in top_islands:
        arch_id = self.island_meta[island.island_id].arch_id
        arch_scores[arch_id] += island.aggregate_z

    return TopicResult(
        top_reefs=top_reefs, top_islands=top_islands,
        arch_scores=arch_scores, confidence=confidence,
        coverage=coverage, matched_words=len(matched),
        unknown_words=unknown,
    )
```

**Performance:** ~500ns for partial sort + rollup.

### 5.7 Unknown Words

Words that fail both the HashMap lookup and the Snowball stem fallback are collected in `unknown_words`. These are:
- Proper nouns not in the vocabulary
- Neologisms, slang, brand names
- Domain-specific jargon (e.g., "Kubernetes", "GraphQL")
- Typos

The unknown words list is a first-class output signal. Downstream consumers can use it to detect vocabulary gaps, trigger fallback strategies, or build corpus-specific extensions.

---

## 6. BM25 Mathematics

### IDF Formula

```
IDF(word) = ln((N - n + 0.5) / (n + 0.5) + 1)
```

Where:
- N = 207 (total reefs)
- n = number of reefs containing the word

**IDF distribution across the vocabulary:**

| Reefs per word | Example count | IDF | Interpretation |
|---------------|--------------|-----|----------------|
| 1 | 3 words | 4.93 | Extremely specific |
| 4 | 249 words | 3.83 | Highly specific |
| 10 | 6,600 words | 2.99 | Moderate |
| 13 | 8,828 words | 2.74 | Most common (peak) |
| 20 | 1,732 words | 2.32 | Somewhat universal |
| 30 | 3 words | 1.92 | Maximum spread |

The distribution peaks at 13 reefs per word. IDF naturally up-weights domain-specific words and down-weights generic ones.

### BM25 Term Score

```
bm25(word, reef) = IDF(word) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |reef| / avgdl))
```

Where:
- `tf = n_dims / reef_total_dims` — depth-weighted term frequency
- `k1 = 1.2` — term frequency saturation
- `b = 0.75` — length normalization strength
- `|reef|` = reef_n_words — words in this reef
- `avgdl` = avg_reef_words (~5,282)

### Depth-Weighted tf vs Binary tf

Binary tf (word is in reef: yes/no) works for broad categorization but fails at fine-grained reef discrimination. Depth-weighted tf captures how "at home" a word is in a reef.

**The neuroscience test case illustrates this:**

| Approach | #1 Reef | Score | #2 Reef | Score |
|----------|---------|-------|---------|-------|
| Binary tf | dinosaurs and fossils | 10.87 | neural and structural | 9.19 |
| Depth-weighted tf | **neural and structural** | **7.45** | scalable aspects | 5.62 |

With binary tf, "dinosaurs and fossils" wins because more neuroscience words appear in that reef. But with depth-weighted tf, "neural and structural" wins because "cortex" activates 100% of its dimensions (2/2), while the neuroscience words only activate 40-60% of the dinosaur reef's dimensions.

A word activating all of a reef's dimensions is a resident; a word activating one of five is a visitor.

### Background Subtraction

Some reefs consistently appear in top results regardless of input topic. Testing showed 8 reefs in 4+ of 14 diverse test sentences' top-15. These noise magnets correlate with large word sets, but it's a gradient, not a clean cutoff.

Background subtraction handles noise continuously:

```
z[reef] = (raw_bm25[reef] - bg_mean[reef]) / bg_std[reef]
```

A noisy reef with bg_mean=1.74 needs a much higher raw BM25 to achieve a high z-score than a clean reef with bg_mean=0.69. The penalty is proportional to noisiness.

**Example — Huck Finn passage:** "archaic literary terms" achieves a very high z-score because its raw BM25 score far exceeds the reef's background mean. The signal is genuinely surprising.

**Implementation cost: essentially zero.** 1.6 KB of pre-computed data. Two float ops per reef (~200ns for all 207).

### Parameter Rationale: k1 and b

- **k1 = 1.2** (term frequency saturation): Controls how quickly tf saturates. Lower k1 approaches binary tf; higher k1 allows deeper words to contribute proportionally more. 1.2 is well-studied in IR literature and works well here.

- **b = 0.75** (length normalization): Controls how much reef size penalizes scores. b=0 means no normalization; b=1 means full normalization. 0.75 appropriately penalizes the largest reefs (8,000-14,000 words) without over-penalizing moderate ones.

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
- Single matched word: TopicResult with low confidence (no #2 to gap against), matched_words=1
- Very long input (1000+ words): works correctly, linear scaling

### score_batch(texts) -> list[TopicResult]

Score multiple texts. Semantically equivalent to `[score(t) for t in texts]` but may optimize memory allocation (reuse the `[f32; 207]` scratch array across calls).

### lookup_word(word) -> Optional[WordInfo]

Look up a single word in the dictionary. Applies the same normalization as Phase 2 (lowercase, then HashMap lookup, then Snowball stem fallback). Returns WordInfo if found, None if unknown.

Useful for debugging and for downstream consumers that want to inspect individual word properties (specificity, IDF) before scoring.

---

## 8. Output Format

### TopicResult

| Field | Type | Description |
|-------|------|-------------|
| top_reefs | list[ScoredReef] | Top-K reefs ranked by z-score (K=10 default) |
| top_islands | list[ScoredIsland] | Islands with contributing reefs, ranked by aggregate z |
| arch_scores | [f32; 4] | Archipelago-level score distribution |
| confidence | f32 | z-score gap between #1 and #2 reef |
| coverage | f32 | Fraction of input words that matched the dictionary |
| matched_words | int | Count of matched words |
| unknown_words | list[str] | Words that failed lookup + stem |

### ScoredReef

| Field | Type | Description |
|-------|------|-------------|
| reef_id | u8 | Reef index (0..206) |
| z_score | f32 | Background-subtracted score |
| raw_bm25 | f32 | Pre-subtraction BM25 score |
| n_contributing_words | u16 | Words that contributed to this reef |

### ScoredIsland

| Field | Type | Description |
|-------|------|-------------|
| island_id | u8 | Island index (0..51) |
| aggregate_z | f32 | Sum of child reef z-scores |
| n_contributing_reefs | u16 | Reefs contributing to this island |

**Interpretation guidance:**
- `confidence > 3.0`: Strong topic signal, #1 reef is clearly dominant
- `confidence < 1.0`: Ambiguous — text may be generic or cross-domain
- `coverage > 0.8`: Good dictionary coverage
- `coverage < 0.5`: Many unknown words — results may be unreliable
- `z_score > 4.0` for a reef: Very strong match (rare for generic text)

---

## 9. Performance Characteristics

### Memory Footprint

| Structure | Size | Notes |
|-----------|------|-------|
| word_lookup | ~4.3 MB | u64 keys, ~173K entries (on disk; in-memory HashMap ~2.8 MB) |
| word_reefs | ~11 MB | ~13 entries/word x ~147K words (on disk; in-memory with decoded entries) |
| reef_meta | ~13 KB | 207 records |
| island_meta | ~2.3 KB | 52 records |
| reef side arrays | ~1.2 KB | reef_total_dims [u8; 207] + reef_n_words [u32; 207] |
| bg_mean + bg_std | 3.7 KB | 207 x 2 x f32 (MessagePack overhead) |
| compound automaton | ~1.3 MB | Aho-Corasick over ~64K strings |
| **Total on disk** | **~16 MB** | |

### Latency

| Input size | Target | Breakdown |
|------------|--------|-----------|
| 30-word sentence | ~10-15 us | ~500ns compounds + 30 x ~310ns/word + ~200ns bg-sub + ~500ns extract |
| 200-word paragraph | ~100 us | Dominated by HashMap lookups |
| 1000+ words | < 1 ms | Linear scaling, hash lookups dominant |

### Startup

< 100 ms — deserialize binary files + build Aho-Corasick automaton.

### Cache Behavior

The `[f32; 207]` reef score array is 828 bytes — fits in L1 cache. No memory pressure during scoring. The word_reefs entries (~13 per word x 4 bytes = ~52 bytes) also fit in cache lines during iteration.

### Bottleneck Analysis

- **Aho-Corasick scan:** O(n) in text length, good cache behavior. Negligible for sentences.
- **Snowball stemmer fallback:** ~200ns per word, fires ~30% of the time. Pre-expansion of morphy variants and snowball stems at export time reduces this to uncommon cases (proper nouns, neologisms, typos).
- **HashMap lookups:** Dominant cost for long inputs. A perfect-hash function (e.g., `phf` crate in Rust) could reduce to ~20ns/word vs ~50ns for standard HashMap.
- **Top-K extraction:** If only top-1 or top-3 is needed, a min-heap avoids sorting all 207 scores.

---

## 10. Constants Reference

| Constant | Value | Description |
|----------|-------|-------------|
| N_REEFS | 207 | Total number of reefs |
| N_ISLANDS | 52 | Total number of islands |
| N_ARCHS | 4 | Total number of archipelagos |
| avg_reef_words | ~5,282 | Average word count across reefs |
| k1 | 1.2 | BM25 term frequency saturation |
| b | 0.75 | BM25 length normalization strength |
| IDF_SCALE | 51 | u8 quantization factor for IDF |
| BM25_SCALE | 8192 | u16 quantization factor for BM25 |
| FNV1A_OFFSET | 14695981039346656037 | FNV-1a 64-bit offset basis |
| FNV1A_PRIME | 1099511628211 | FNV-1a 64-bit prime |

All constants except `avg_reef_words` are fixed across builds. `avg_reef_words` is computed from the data and stored in `constants.bin`.

---

## 11. FNV-1a Hash Implementation

FNV-1a (Fowler-Noll-Vo) is the hash function used for all word lookups. It produces a 64-bit hash with zero collisions across the full ~147K base vocabulary.

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
- **Zero collisions:** Verified across the full ~147K base vocabulary — no two distinct strings hash to the same u64 value
- **Fast:** Linear in string length, no allocations, branch-free inner loop
- **Cross-language:** Byte-level operation means Python and Rust produce identical hashes for the same input

**Input normalization:** All lookups use lowercased, whitespace-normalized strings. Multi-word compounds use single spaces as separators. This normalization must be applied before hashing.

---

## 12. Canonical Test Cases

These test sentences establish ground-truth expectations for any lagoon implementation. An implementation that does not match these results has a bug.

### Test 1: Generic text

**Input:** `"now is the time for all good men to come to the aid of the country"`

**Expected:**
- Low confidence (< 1.5) — flat z-score distribution
- Top reefs are broad/abstract: "perception and attributes", "behavior and emotional states"
- This is correct — the text IS generic

### Test 2: Huck Finn passage

**Input:** `"It was after sun-up now, but we went right on and didn't tie up. The king and the duke turned out by-and-by looking pretty rusty; but after they'd jumped overboard and took a swim it chippered them up a good deal..."`

**Expected:**
- **"archaic literary terms" as clear #1** — far ahead of #2
- Supporting reefs: "movement and transit", "coastal landscapes and frontiers", "shabby appearance and manner"
- High confidence (> 3.0)

### Test 3: Waveform passage

**Input:** `"The sine wave has a pattern that repeats..."`

**Expected:**
- "prediction and certainty" near top
- Mathematical/technical signal present
- Moderate confidence

### Test 4: Neuroscience words

**Input:** `"neuron synapse axon dendrite cortex brain neural hippocampus"`

**Expected:**
- **"neural and structural" as #1** (after background subtraction)
- "dinosaurs and fossils" drops from #1 (raw) to lower rank (z-score)
- Demonstrates the critical importance of background subtraction

### Test 5: Topic shift (split Huck Finn)

Split the Huck Finn passage at the midpoint:
- **First half** (physical scene): "archaic literary terms", "branching and tree structure", "shabby appearance and manner"
- **Second half** (theater): "movement and transit", "poetic religious texts", "skill and expertise"
- Two halves' z-score distributions should be measurably divergent

---

## 13. Known Limitations

1. **Bag-of-words.** Word order is ignored. "Dog bites man" and "man bites dog" produce identical results.

2. **Fixed vocabulary.** Neologisms, slang, brand names, and domain-specific jargon not in the vocabulary will be missed. The stemmer fallback helps with inflections but not with truly unknown words.

3. **Abstract domains are weak.** Software engineering, philosophy, and other highly abstract domains produce low-confidence results because their vocabulary spreads thinly across many reefs. The system's strength is concrete, domain-specific vocabulary (science, literature, physical descriptions).

4. **No compositional semantics.** Phrase meaning isn't captured beyond compound matching. "Not good" scores the same as "good" for reef convergence.

5. **English only.** The source vocabulary and embedding model are English-specific.

6. **Static vocabulary.** The dictionary is frozen at build time. New words require a rebuild of the data files.

7. **207-reef ceiling.** Topic detection granularity is bounded by the 207 reefs. Sub-reef distinctions (e.g., "organic chemistry" vs "inorganic chemistry" within a chemistry reef) are not possible.

---

## 14. Portability Notes

Lagoon is designed to port from Python to Rust without architectural changes:

- **No pickle.** All serialized data uses MessagePack — language-agnostic, supported in both Python and Rust (`rmp-serde`).

- **Flat-array / mmap-friendly.** The `[f32; 207]` score arrays, ReefEntry lists, and metadata arrays are contiguous and fixed-stride. In Rust, the fixed-size arrays can potentially be mmapped directly with zero deserialization.

- **UTF-8 strings, pre-normalized.** All strings in the data files are lowercase, whitespace-normalized UTF-8. No runtime Unicode normalization needed beyond lowercasing the input.

- **No dynamic dispatch.** All types are concrete, all methods are monomorphic. No trait objects needed in the Rust port.

- **No reference counting / GC dependency.** All data structures are owned. In Python, plain dataclasses with no circular references. In Rust, owned `Vec`, `HashMap`, and `String`.

- **Fixed-size numeric types.** reef_id is u8, word_id is u32, hierarchy_addr is u16. These are documented in the data structure definitions and must match between the export tool and lagoon.

- **FNV-1a is byte-level.** The hash function operates on UTF-8 bytes, producing identical results in Python and Rust for the same input string. No platform-dependent behavior.

- **Aho-Corasick is standard.** The `aho-corasick` crate in Rust is mature and well-optimized. Python has `pyahocorasick` or `ahocorasick-rs`.

- **No external ML dependencies.** Lagoon has zero dependency on embedding models, NLTK, or any ML framework. The only non-trivial dependency is the Aho-Corasick implementation and a Snowball stemmer for runtime fallback lookups.
