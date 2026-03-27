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
| `top_reefs` | `list[ScoredReef]` | Top 10 towns ranked by z-score (named "reefs" for API compat) |
| `top_islands` | `list[ScoredIsland]` | Island-level rollup (mid-level clusters) |
| `arch_scores` | `list[float]` | 6 archipelago-level scores |
| `confidence` | `float` | Top z-score (clamped to 0) |
| `coverage` | `float` | Fraction of input words found in the dictionary |
| `matched_words` | `int` | Number of words that hit the dictionary |
| `unknown_words` | `list[str]` | Words that failed lookup + stemmer fallback (stop words excluded) |
| `matched_word_ids` | `frozenset[int]` | Set of word_ids that matched (for cross-referencing) |
| `n_domainless` | `int` | Words recognized but not domain-specific |
| `valence_signal` | `float` | z-score-weighted mean of town valences |

Each `ScoredReef` has:

```python
reef = result.top_reefs[0]
reef.reef_id               # 0-297 (town index)
reef.z_score               # background-subtracted score
reef.raw_score             # raw score before background subtraction
reef.n_contributing_words   # how many input words activated this town
reef.name                   # human-readable label
reef.quality_score          # z-score (IQF modulation placeholder)
reef.valence                # town valence
reef.avg_specificity        # average word specificity in this town
reef.resolved_sub_reef_id   # sub-town reef resolution (always None in V3 — not yet wired up)
reef.resolved_sub_reef_name # sub-town reef name (always None in V3)
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
info.idf_q / 51.0  # high IDF = appears in few towns
info.word_id       # index into internal data

info = scorer.lookup_word("brain")
info.specificity   # -1 (universal, appears across many towns)

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

Lagoon's base vocabulary (~150K words) can be extended at runtime with custom words. This is the primary API for downstream vocabulary learning systems.

### Add a custom word

```python
# Compute calibrated weights for each town association
idf_q = scorer.calc_custom_idf(n_associated_reefs=2)
reef_weights = [
    (42, scorer.calc_custom_weight(42, strength=0.9)),
    (17, scorer.calc_custom_weight(17, strength=0.5)),
]

info = scorer.add_custom_word(
    "kubernetes",
    reef_weights=reef_weights,
    idf_q=idf_q,
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
idf_q = scorer.calc_custom_idf(1)
info1 = scorer.add_custom_word("kubernetes", reef_weights=[(42, scorer.calc_custom_weight(42, 0.9))], idf_q=idf_q, tag=1)
info2 = scorer.add_custom_word("terraform", reef_weights=[(17, scorer.calc_custom_weight(17, 0.8))], idf_q=idf_q, tag=2)

result = scorer.score("kubernetes and terraform")

# get_word_tags() returns only non-zero tags
tags = scorer.get_word_tags(result.matched_word_ids)
print(tags)  # {<word_id>: 1, <word_id>: 2}

# Base vocabulary words have tag 0 and are omitted from the result
base_info = scorer.lookup_word("brain")
print(base_info.tag)  # 0
print(scorer.get_word_tags({base_info.word_id}))  # {}
```

### Compute calibrated weights without injecting

```python
# Compute IDF for a word appearing in 3 towns
idf_q = scorer.calc_custom_idf(n_associated_reefs=3)
print(idf_q)  # quantized IDF (u8, high value = specific)

# Compute calibrated weight for a specific town
weight_q = scorer.calc_custom_weight(reef_id=42, strength=0.9)
print(weight_q)  # u8 weight calibrated to the town's 75th-percentile
```

### Add custom compound words

```python
# First add individual words
idf_q = scorer.calc_custom_idf(1)
info1 = scorer.add_custom_word("zorblax", reef_weights=[(42, scorer.calc_custom_weight(42, 0.8))], idf_q=idf_q)
info2 = scorer.add_custom_word("frimble", reef_weights=[(17, scorer.calc_custom_weight(17, 0.7))], idf_q=idf_q)

# Add the compound
compound_info = scorer.add_custom_word(
    "zorblax frimble",
    reef_weights=[(42, scorer.calc_custom_weight(42, 0.95)), (17, scorer.calc_custom_weight(17, 0.85))],
    idf_q=scorer.calc_custom_idf(2),
)

# Rebuild automaton to match the new compound
scorer.rebuild_compounds([("zorblax frimble", compound_info.word_id)])
```

### Validation

All inputs are validated with clear `ValueError` messages:

```python
scorer.add_custom_word("brain", ...)             # ValueError: already exists
scorer.add_custom_word("", ...)                  # ValueError: must not be empty
scorer.add_custom_word("x", ..., specificity=5)  # ValueError: specificity
scorer.add_custom_word("x", reef_weights=[])     # ValueError: must not be empty
scorer.add_custom_word("x", reef_weights=[(999, 100)], idf_q=100) # ValueError: reef_id out of range
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

Lagoon uses a four-level semantic hierarchy. Three levels actively participate in scoring:

- **6 Archipelagos** — broadest level, score rollup only
- **40 Islands** — mid-level communities. The contextual scorer (4+ matched words) tracks island activation and boosts scores when multiple words converge on the same island
- **298 Towns** — the primary scoring unit. All weight accumulation, background subtraction, and result extraction happen here. Referred to as "reefs" in the API (`top_reefs`, `ScoredReef`, etc.)
- **3,885 Reefs** — finest-grained clusters nested within towns. Available as metadata via `V3ReefMeta` and through `ScoredReef.resolved_sub_reef_id` for sub-town resolution, but do not have their own scoring pass

Access island and archipelago rollups through the result:

```python
result = scorer.score("neuron synapse axon dendrite cortex brain")

# Islands (active in contextual scoring — coherence boosting)
for island in result.top_islands[:3]:
    print(f"{island.name}: z={island.aggregate_z:.2f} ({island.n_contributing_reefs} towns)")

# Archipelagos (indexed 0-5, rollup only)
print(result.arch_scores)

# Sub-town reef resolution (not yet wired up in V3 — always None currently)
# When implemented, will resolve which of the 3,885 reefs within a town best matches
for reef in result.top_reefs[:3]:
    print(f"  {reef.name}: sub-reef resolved = {reef.resolved_sub_reef_id}")
```
