# Lagoon Quickstart

## Install

```bash
pip install -e .
```

Lagoon bundles its data files (~16 MB) inside the package. No additional downloads needed.

## Load the scorer

```python
import lagoon

scorer = lagoon.load()
```

This loads all data files, validates checksums, and builds the Aho-Corasick automaton for compound matching. Takes ~100ms on first call.

To load from a custom data directory instead of the bundled data:

```python
scorer = lagoon.load("/path/to/data")
```

## Score text

Pass any text to `scorer.score()` and get back a `TopicResult`:

```python
result = scorer.score("photosynthesis chlorophyll leaf plant sunlight carbon dioxide")

print(result.top_reefs[0].name)       # "botanical classification systems"
print(result.top_reefs[0].z_score)    # 5.87
print(result.confidence)               # 2.63
print(result.coverage)                 # 1.0 (all words matched)
print(result.matched_words)            # 6
print(result.unknown_words)            # [] (stop words excluded automatically)
print(len(result.matched_word_ids))    # 6 (set of word_ids that matched)
```

### What's in a TopicResult

| Field | Type | What it tells you |
|-------|------|-------------------|
| `top_reefs` | `list[ScoredReef]` | Top 10 reefs ranked by z-score |
| `top_islands` | `list[ScoredIsland]` | Island-level rollup (mid-level clusters) |
| `arch_scores` | `list[float]` | 5 archipelago-level scores |
| `confidence` | `float` | Gap between #1 and #2 reef z-scores |
| `coverage` | `float` | Fraction of input words found in the dictionary |
| `matched_words` | `int` | Number of words that hit the dictionary |
| `unknown_words` | `list[str]` | Words that failed lookup + stemmer fallback (stop words excluded) |
| `matched_word_ids` | `frozenset[int]` | Set of word_ids that matched (for cross-referencing) |

Each `ScoredReef` has:

```python
reef = result.top_reefs[0]
reef.reef_id               # 0-282
reef.z_score               # background-subtracted score
reef.raw_bm25              # raw BM25 before background subtraction
reef.n_contributing_words   # how many input words activated this reef
reef.name                   # human-readable label
```

### Interpreting results

- **`confidence > 2.0`** — clear dominant topic
- **`confidence < 1.0`** — ambiguous, generic, or cross-domain text
- **`coverage > 0.8`** — good dictionary coverage, results are reliable
- **`coverage < 0.5`** — many unknown words, treat results with caution
- **`z_score > 4.0`** on a reef — very strong match

## Batch scoring

```python
results = scorer.score_batch([
    "neuron synapse axon dendrite cortex brain",
    "photosynthesis chlorophyll leaf plant sunlight",
    "plaintiff defendant jury verdict trial testimony",
])

for r in results:
    print(f"{r.top_reefs[0].name}: z={r.top_reefs[0].z_score:.2f}")
```

## Look up individual words

```python
info = scorer.lookup_word("cortex")
info.specificity   # +1 (domain-specific)
info.idf_q / 51.0  # 3.63 (high IDF = appears in few reefs)
info.word_id       # index into internal data

info = scorer.lookup_word("brain")
info.specificity   # -1 (universal, appears across many reefs)

scorer.lookup_word("xyzzy")  # None (not in vocabulary)
```

Specificity ranges from -2 (very universal) to +2 (very specific). High-specificity words are strong topical signals on their own.

## Document topic segmentation

`scorer.analyze()` splits a document into topic-coherent segments. It scores each sentence to get z-score vectors and per-sentence `TopicResult` objects in a single pass, measures cosine similarity between adjacent smoothed vectors, detects valleys (topic shifts), and enforces chunk size constraints.

### From a string

```python
doc = """The neural pathways in the brain connect through axons and dendrites.
Synaptic transmission occurs at the junction between neurons.
The hippocampus plays a key role in memory.

The stock market experienced significant volatility yesterday.
Trading volumes surged as investors reacted to the Fed announcement.
Bond yields dropped sharply across the board."""

analysis = scorer.analyze(doc)

print(analysis.n_sentences)   # 6
print(analysis.n_segments)    # 2
print(analysis.boundaries)    # [3] (topic shift before sentence 3)

for seg in analysis.segments:
    print(f"Sentences {seg.start_idx}-{seg.end_idx}: {seg.topic.top_reefs[0].name}")
    # Segment 0: neuroscience-related reef
    # Segment 1: finance-related reef
```

### Per-sentence results

Each segment includes `sentence_results` — per-sentence `TopicResult` objects with `matched_word_ids`, `unknown_words`, `top_reefs`, and `coverage`:

```python
for seg in analysis.segments:
    for i, sr in enumerate(seg.sentence_results):
        print(f"  Sentence {seg.start_idx + i}: "
              f"{sr.matched_words} matched, "
              f"{len(sr.unknown_words)} unknown, "
              f"top reef: {sr.top_reefs[0].name if sr.top_reefs else 'none'}")
        print(f"    matched_word_ids: {sr.matched_word_ids}")
```

### From pre-segmented sentences

If you have your own sentence splitter, pass a list of strings:

```python
analysis = scorer.analyze([
    "Neurons fire electrical signals through axons.",
    "The cortex processes sensory information.",
    "Stock prices surged on Wall Street today.",
    "Investors bought bonds and equities.",
])
```

### Tuning segmentation

```python
# Higher sensitivity = fewer boundaries (less sensitive to shifts)
analysis = scorer.analyze(doc, sensitivity=2.0)

# Lower sensitivity = more boundaries
analysis = scorer.analyze(doc, sensitivity=0.5)

# Smoothing window (default 2): larger = smoother, fewer false splits
analysis = scorer.analyze(doc, smooth_window=3)

# Minimum sentences per segment (default 2): merge small segments
analysis = scorer.analyze(doc, min_chunk_sentences=2)

# Maximum sentences per segment (default 30): split large segments
analysis = scorer.analyze(doc, max_chunk_sentences=15)

# Disable maximum size enforcement
analysis = scorer.analyze(doc, max_chunk_sentences=0)
```

## Stop words and filtering unknown words

Lagoon automatically excludes ~130 English stop words (determiners, conjunctions, prepositions, etc.) from `unknown_words` in scoring results — they carry no topical signal.

```python
from lagoon import STOP_WORDS

print("the" in STOP_WORDS)   # True
print("neuron" in STOP_WORDS) # False
```

Use `filter_unknown()` to batch-identify vocabulary gaps (stop words excluded):

```python
unknowns = scorer.filter_unknown(["kubernetes", "brain", "terraform", "the", "xyzzy"])
print(unknowns)  # ["kubernetes", "terraform", "xyzzy"]
# "brain" is known, "the" is a stop word — both excluded
```

## Vocabulary extension

Lagoon's base vocabulary (~147K words) can be extended at runtime with custom words. This is the primary API for downstream vocabulary learning systems.

### Add a custom word

```python
from lagoon._types import WordInfo

# Define reef associations: (reef_id, strength) where strength is 0.0-1.0
info = scorer.add_custom_word(
    "kubernetes",
    reef_associations=[(42, 0.9), (17, 0.5)],
    specificity=2,  # default: highly specific
    tag=1,          # optional: opaque consumer metadata (default 0)
)

print(info.word_id)       # next available word_id
print(info.idf_q)         # quantized IDF (u8)
print(info.specificity)   # 2
print(info.tag)           # 1

# Word is now immediately scorable
result = scorer.score("kubernetes")
print(result.matched_words)   # 1
print(result.coverage)        # 1.0
```

### Tag custom words for downstream consumers

The `tag` parameter is opaque metadata that lagoon stores but never interprets. Downstream consumers (e.g., Shoal) can use tags to distinguish base vocabulary words from custom-injected words:

```python
info1 = scorer.add_custom_word("kubernetes", reef_associations=[(42, 0.9)], tag=1)
info2 = scorer.add_custom_word("terraform", reef_associations=[(17, 0.8)], tag=2)

result = scorer.score("kubernetes and terraform")

# get_word_tags() returns only non-zero tags
tags = scorer.get_word_tags(result.matched_word_ids)
print(tags)  # {<word_id>: 1, <word_id>: 2}

# Base vocabulary words have tag 0 and are omitted from the result
base_info = scorer.lookup_word("brain")
print(base_info.tag)  # 0
print(scorer.get_word_tags({base_info.word_id}))  # {}
```

### Compute BM25 scores without injecting

```python
idf_q, reef_scores = scorer.compute_custom_word_scores(
    n_associated_reefs=3,
    associations=[(42, 1.0), (17, 0.6), (103, 0.3)],
)
print(idf_q)         # quantized IDF (u8, high value = specific)
print(reef_scores)   # [(42, bm25_q), (17, bm25_q), (103, bm25_q)]
```

### Add custom compound words

```python
# First add individual words
info1 = scorer.add_custom_word("zorblax", reef_associations=[(42, 0.8)])
info2 = scorer.add_custom_word("frimble", reef_associations=[(17, 0.7)])

# Add the compound
compound_info = scorer.add_custom_word(
    "zorblax frimble",
    reef_associations=[(42, 0.95), (17, 0.85)],
)

# Rebuild automaton to match the new compound
scorer.rebuild_compounds([("zorblax frimble", compound_info.word_id)])
```

### Validation

All inputs are validated with clear `ValueError` messages:

```python
scorer.add_custom_word("brain", ...)           # ValueError: already exists
scorer.add_custom_word("", ...)                # ValueError: must not be empty
scorer.add_custom_word("x", ..., specificity=5) # ValueError: specificity
scorer.add_custom_word("x", reef_associations=[]) # ValueError: must not be empty
scorer.add_custom_word("x", reef_associations=[(999, 0.5)]) # ValueError: reef_id out of range
scorer.add_custom_word("x", reef_associations=[(42, 2.0)])  # ValueError: strength must be in
```

## Edge cases

```python
# Empty input
r = scorer.score("")
# confidence=0.0, matched_words=0, top_reefs=[]

# All unknown words
r = scorer.score("xyzzy plugh qwerty")
# confidence=0.0, coverage=0.0, unknown_words=["xyzzy", "plugh", "qwerty"]

# Repeated words don't increase scores (binary occurrence model)
scorer.score("cortex").top_reefs[0].z_score == scorer.score("cortex cortex cortex").top_reefs[0].z_score
```

## Hierarchy

Lagoon's 283 reefs are organized into a three-level hierarchy:

- **5 Archipelagos** — broadest level (medical/physical sciences, physical world/human artifacts, abstract concepts/states, formal systems/qualities, activities/performance/relations)
- **67 Islands** — mid-level communities (2-8 reefs each)
- **283 Reefs** — finest-grained semantic clusters (the primary scoring target)

Access island and archipelago rollups through the result:

```python
result = scorer.score("neuron synapse axon dendrite cortex brain")

# Islands
for island in result.top_islands[:3]:
    print(f"{island.name}: z={island.aggregate_z:.2f} ({island.n_contributing_reefs} reefs)")

# Archipelagos (indexed 0-4)
print(result.arch_scores)
```
