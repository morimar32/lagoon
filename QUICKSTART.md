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
print(result.top_reefs[0].z_score)    # 4.46
print(result.confidence)               # 2.11
print(result.coverage)                 # 1.0 (all words matched)
print(result.matched_words)            # 7
print(result.unknown_words)            # []
```

### What's in a TopicResult

| Field | Type | What it tells you |
|-------|------|-------------------|
| `top_reefs` | `list[ScoredReef]` | Top 10 reefs ranked by z-score |
| `top_islands` | `list[ScoredIsland]` | Island-level rollup (mid-level clusters) |
| `arch_scores` | `list[float]` | 4 archipelago-level scores |
| `confidence` | `float` | Gap between #1 and #2 reef z-scores |
| `coverage` | `float` | Fraction of input words found in the dictionary |
| `matched_words` | `int` | Number of words that hit the dictionary |
| `unknown_words` | `list[str]` | Words that failed lookup + stemmer fallback |

Each `ScoredReef` has:

```python
reef = result.top_reefs[0]
reef.reef_id               # 0-206
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

`scorer.analyze()` splits a document into topic-coherent segments. It works by computing z-score vectors per sentence, measuring cosine similarity between adjacent sentences, and detecting valleys (topic shifts).

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

# Minimum sentences per segment
analysis = scorer.analyze(doc, min_segment_sentences=2)
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

Lagoon's 207 reefs are organized into a three-level hierarchy:

- **4 Archipelagos** — broadest level (natural sciences, physical world, abstract processes, social order)
- **52 Islands** — mid-level communities (2-8 reefs each)
- **207 Reefs** — finest-grained semantic clusters (the primary scoring target)

Access island and archipelago rollups through the result:

```python
result = scorer.score("neuron synapse axon dendrite cortex brain")

# Islands
for island in result.top_islands[:3]:
    print(f"{island.name}: z={island.aggregate_z:.2f} ({island.n_contributing_reefs} reefs)")

# Archipelagos (indexed 0-3)
print(result.arch_scores)
```
