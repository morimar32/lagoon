# Export Enrichment Opportunities

Status: planning document for expanding the data windowsill exports to lagoon.

## Current Export Summary

The export pipeline distills a rich database into a compact binary format optimized
for fast scoring. Here's what crosses the boundary today vs. what's available:

### Per word-reef pair (`word_reefs.bin`)

Currently exported as a single `(reef_id, bm25_q)` tuple per pair.

| Column | In DB | Exported | How it's used |
|---|---|---|---|
| `n_dims` | yes | **indirectly** | Fed into BM25 as TF; the raw count is discarded |
| `max_z` | yes | no | Strongest single-dimension z-score for this word in this reef |
| `sum_z` | yes | no | Sum of z-scores across all dimensions |
| `max_weighted_z` | yes | no | Like max_z but weighted by dim_weight (abstractness penalty) |
| `sum_weighted_z` | yes | no | Like sum_z but dim_weight-weighted |
| `centroid_similarity` | yes | no | Cosine similarity between the word's 768-dim embedding and the reef's depth-2+ member centroid |

### Per reef (`reef_meta.bin`)

| Column | In DB | Exported | Notes |
|---|---|---|---|
| `hierarchy_addr` | derived | yes | Bit-packed arch/island/reef address |
| `n_words` | yes | yes | Total member count |
| `name` | yes | yes | LLM-generated label |
| `n_dims` | yes | **in constants.bin** | As `reef_total_dims` array |
| `valence` | yes | no | Mean negation-vector projection; indicates positive/negative pole |
| `noun_frac` | yes | no | Sense-aware POS composition |
| `verb_frac` | yes | no | " |
| `adj_frac` | yes | no | " |
| `adv_frac` | yes | no | " |
| `avg_specificity` | yes | no | Mean specificity of member words; how niche vs. broad the reef is |
| `avg_internal_jaccard` | yes | no | Cohesion of the reef's dimensions |
| `n_core_words` | yes | no | Words with depth >= 2 |
| `median_word_depth` | yes | no | Median n_dims across members |

### Per island (`island_meta.bin`)

| Column | In DB | Exported | Notes |
|---|---|---|---|
| `arch_id` | yes | yes | Parent archipelago |
| `name` | yes | yes | LLM-generated label |
| `valence` | yes | no | Same as reef but aggregated to island level |
| `noun_frac` etc. | yes | no | POS composition at island level |
| `avg_specificity` | yes | no | Specificity at island level |

### Per reef edge (`reef_edges.bin`)

| Column | In DB | Exported | Notes |
|---|---|---|---|
| `weight` (composite) | yes | yes | Used for score propagation |
| `containment` | yes | no | Fraction of source words also in target |
| `lift` | yes | no | P(target\|source) / P(target) |
| `pos_similarity` | yes | no | Cosine of POS fraction vectors |
| `valence_gap` | yes | no | Signed valence difference |
| `specificity_gap` | yes | no | Signed specificity difference |

---

## Planned Change: Export `centroid_similarity`

### What it is

For each (word, reef) pair, the cosine similarity between the word's 768-dimensional
embedding and the reef's centroid (mean embedding of depth-2+ members). This was
identified in noise analysis as the single strongest per-entry quality signal for
word-reef associations.

**Range:** [0.54, 1.00], mean 0.78, std 0.03.

Genuine associations (e.g., "military" in the warfare reef) score ~0.86+, while
noise associations (word appears in reef due to a single spurious dimension) score
near the mean or below.

### How lagoon should use it

**Primary use: per-reef quality signal during result extraction.**

During `_accumulate_bm25`, track the weighted-average centroid_similarity across
contributing words per reef. Expose this as a `centroid_quality` field on `ScoredReef`.
Downstream consumers (shoal, etc.) can use this to:

- Filter out reefs where the score is high but quality is low (noise amplification)
- Weight reef scores by quality when ranking results
- Set minimum quality thresholds for high-precision applications

**Secondary use: BM25 modulation.**

Optionally multiply `bm25_q` by a function of centroid_similarity during accumulation.
A word with 0.90 similarity to a reef's centroid should contribute more than one with
0.65. Something like `bm25_q * (centroid_sim ** alpha)` where alpha is tunable.

### Export format

Add a quantized `centroid_sim_q` (u8, scale 0-255 mapping [0.0, 1.0]) alongside
each `bm25_q` in `word_reefs.bin`. The per-entry cost goes from 4 bytes to 6 bytes
(+50%), but this is the smallest file — the total export size increase is minor.

```
Current:  (reef_id: u16, bm25_q: u16)           = 4 bytes/entry
Proposed: (reef_id: u16, bm25_q: u16, csim: u8, pad: u8) = 6 bytes/entry
```

Padding to 6 bytes keeps alignment clean. Alternatively, pack into 5 bytes if size
matters more than alignment.

---

## Unused Signals: Analysis and Recommendations

### Tier 1 — High value, should export

#### `centroid_similarity` (word-reef)
Already discussed above. Strongest quality signal we have.

#### `reef.valence` and `island.valence`
The negation-vector projection tells you whether a reef/island leans positive (joy,
approval, growth) or negative (disapproval, deprivation, transgression). Range is
roughly [-0.42, +0.72].

**Value to lagoon:** Sentiment-aware topic scoring. When scoring a document, lagoon
could report not just "this chunk is about reef X" but "this chunk is about reef X
which has negative valence." Shoal could use this for sentiment analysis without
needing a separate model — the topic structure already encodes it.

**Export cost:** One f32 per reef in `reef_meta.bin`. Trivial.

#### `reef.avg_specificity`
How niche vs. broad the reef is. Ranges from -0.41 (very broad, universal words)
to +0.18 (highly specific, domain terminology).

**Value to lagoon:** Confidence calibration. A high z-score on a highly specific reef
is much more meaningful than a high z-score on a broad reef. Lagoon could use this
to weight or normalize reef scores. Also useful for shoal: "this document is about
very specific topics" vs. "this document covers broad themes."

**Export cost:** One f32 per reef in `reef_meta.bin`. Trivial.

### Tier 2 — Moderate value, export if the use case materializes

#### `max_weighted_z` / `sum_weighted_z` (word-reef)
These incorporate dim_weight (an abstractness penalty: -log2 of universal_pct). They
downweight dimensions that are dominated by universal words — a word's activation in
a "concrete" dimension is worth more than activation in an "abstract" one.

**Value to lagoon:** An alternative to raw n_dims as the TF signal. Currently BM25
uses `n_dims / reef_total_dims` as TF, which treats all dimensions equally. Using
`sum_weighted_z` would give credit for *quality* of dimensional membership, not just
quantity.

**Trade-off:** This is partially redundant with centroid_similarity (both are quality
signals). If centroid_similarity proves sufficient, these may not be needed. But if
lagoon wants to build a richer scoring model (e.g., linear combination of signals),
having both available gives more degrees of freedom.

**Export cost:** One or two additional u16 values per word-reef entry. Non-trivial
at ~2.4M entries.

#### `reef.noun_frac / verb_frac / adj_frac / adv_frac`
POS composition of reef members, computed with sense-aware disambiguation.

**Value to lagoon:** POS-aware scoring. If a document chunk is mostly verbs and a reef
is 90% nouns, the match is probably lower quality than the z-score suggests. Could also
power queries like "find topics that are action-oriented" (high verb_frac).

**Export cost:** Four f32 values per reef. Small.

#### `reef.n_core_words` and `reef.median_word_depth`
How many words are "real" members (depth >= 2) and how deep the typical member is.

**Value to lagoon:** Reef reliability signal. A reef with 500 n_core_words and median
depth 4 is much more trustworthy than one with 50 core words and median depth 2.
Could feed into confidence scoring.

**Export cost:** Two values per reef. Trivial.

### Tier 3 — Low value for lagoon, keep in windowsill

#### `max_z` / `sum_z` (word-reef, unweighted)
Raw z-scores without dim_weight correction. Superseded by the weighted variants.
No reason to export both weighted and unweighted.

#### Reef edge components (`containment`, `lift`, `pos_similarity`, `valence_gap`, `specificity_gap`)
These are already baked into the composite `weight` that lagoon uses for propagation.
Exporting the components would only matter if lagoon wanted to recompute the composite
weight with different coefficients — a tuning scenario that hasn't arisen.

#### `avg_internal_jaccard` (reef)
Measures how tightly the reef's dimensions correlate. Useful for pipeline diagnostics
but not directly useful for document scoring.

---

## Suggested Implementation Order

1. **centroid_similarity** — Immediate. Already computed, strongest signal.
2. **reef.valence + reef.avg_specificity** — Next. Cheap to export, high
   information value for downstream consumers.
3. **reef POS composition** — When lagoon needs POS-aware features.
4. **sum_weighted_z** — When/if lagoon moves beyond simple BM25 to a
   multi-signal scoring model.
