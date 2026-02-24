# New Export Fields (v2.1)

Reference for integrating the new fields from windowsill's export pipeline into lagoon's scoring engine. Everything below describes data that is now present in the exported `.bin` files but was not available in v2.0.

---

## Overview of Changes

The word-reef tuple expanded from 2 elements to 4:

```
v2.0:  (reef_id, bm25_q)
v2.1:  (reef_id, bm25_q, csim_q, swz_q)
```

Reef metadata gained 6 float fields: `valence`, `avg_specificity`, `noun_frac`, `verb_frac`, `adj_frac`, `adv_frac`.

The IDF computation was also reworked (two-tier), which changes the distribution of existing `idf_q` values.

---

## 1. csim_q — Centroid Similarity

### What it is

Cosine similarity between a word's raw embedding vector and the unit-normalized centroid of the reef it belongs to. Measures how geometrically close a word sits to the "center of mass" of its reef in the original 768-dimensional embedding space.

### How it is computed

1. For each reef, collect all member words with depth >= 2 (activate in >= 2 of the reef's dimensions). If a reef has no depth-2+ members, fall back to all members.
2. Compute the reef centroid as the mean of those member embeddings, then L2-normalize it to a unit vector.
3. For each `(word, reef)` pair in `word_reef_affinity`, compute:
   ```
   cosine_sim = dot(word_embedding, centroid) / ||word_embedding||
   ```
   (The centroid is already unit-length, so dividing by its norm is unnecessary.)

### Quantization

```
csim_q = clamp(round(centroid_similarity * 255), 0, 255)
```

- **Type**: u8
- **Scale constant**: `CSIM_SCALE = 255` (exported in `constants.bin`)
- **Recovery**: `centroid_similarity ~ csim_q / 255.0`
- **Resolution**: ~0.004 per step

### Value semantics

| csim_q | Meaning |
|--------|---------|
| 230-255 | Core member — the word's embedding is tightly aligned with the reef's centroid |
| 180-230 | Solid member — clearly belongs, somewhat off-center |
| 100-180 | Peripheral — included by dimension overlap but embedding is drifting from the cluster |
| 0-100 | Weak/incidental — the word activates in reef dimensions but its embedding points elsewhere |

### Why it matters for scoring

BM25 and z-score signals are derived from discrete dimension membership (a word either crosses a dimension's threshold or it doesn't). Centroid similarity is a **continuous geometric signal** from the original embedding space. It captures information that the discretization process loses.

Practical use cases:

- **Re-ranking**: Two words with identical BM25 scores for a reef can be disambiguated by centroid similarity — the one with higher csim_q is a more natural member.
- **Confidence weighting**: When aggregating reef scores across a document, weight each word's contribution by csim_q. Words that are geometrically aligned with their reef contribute more confident signal.
- **Noise filtering**: A word with high BM25 but low csim_q may be a dimension-overlap artifact rather than a genuine semantic match. A threshold (e.g., csim_q >= 100) could filter these out.
- **Blended scoring**: A linear combination like `score = alpha * bm25_q + beta * csim_q` would fuse the discrete (dimension-counting) and continuous (embedding-space) perspectives.

---

## 2. swz_q — Sum of Weighted Z-scores

### What it is

The sum of `z_score * dim_weight` across all dimensions where the word activates within the reef. This is an information-weighted measure of how strongly a word activates in the reef's dimensions, with higher credit given to rare/specific dimensions.

### How it is computed

For each `(word, reef)` pair:

```sql
SUM(z_score * dim_weight) AS sum_weighted_z
```

Where:
- **z_score** = how many standard deviations the word's activation exceeds the dimension's mean (from the original z-score thresholding in phase 4)
- **dim_weight** = `-log2(max(universal_pct, 0.01))`, an IDF-like weight at the dimension level

**dim_weight explained**: `universal_pct` is the fraction of a dimension's members that are universal words (appear in 23+ dimensions). Dimensions dominated by common/abstract words get low weight; dimensions populated by rare/specific words get high weight.

| universal_pct | dim_weight | Interpretation |
|--------------|------------|----------------|
| 1.0 (all universal) | 0.0 | Dimension is maximally generic — zero information |
| 0.50 | 1.0 | Half universal, moderate weight |
| 0.10 | 3.32 | Mostly specific words — good discriminator |
| 0.01 (floor) | 6.64 | Highly specific — maximum weight |

### Quantization

```
swz_q = clamp(round(sum_weighted_z * 100), 0, 65535)
```

- **Type**: u16
- **Scale constant**: `SWZ_SCALE = 100` (exported in `constants.bin`)
- **Recovery**: `sum_weighted_z ~ swz_q / 100.0`
- **Resolution**: 0.01 per step
- Values exceeding 655.35 are clamped (rare, warns at export time)

### Value semantics

Higher swz_q means the word activates strongly in dimensions that are informationally rich for this reef. Two words might have the same `n_dims` in a reef, but the one whose activated dimensions are rarer/more specific will have a higher swz_q.

### Why it matters for scoring

BM25 treats all reef dimensions equally (via `tf = n_dims / reef_total_dims`). swz_q adds a second axis: **which** dimensions the word fires in, not just how many.

- **Specificity-aware ranking**: A word that fires in 3 highly specific dimensions of a reef (swz_q = 1500) is a more diagnostic hit than one that fires in 3 generic dimensions (swz_q = 300), even though their BM25 contribution is identical.
- **Complementary to BM25**: BM25 is the right primary signal (it handles document-length normalization and IDF). swz_q is a secondary signal that can break ties or adjust confidence. Consider it as a quality multiplier on the BM25 signal.
- **Bridge detection**: Universal words that act as bridges between archipelagos tend to have moderate BM25 everywhere but low swz_q (because they activate in generic dimensions). High swz_q distinguishes words that are genuinely diagnostic for a reef from universal words that merely happen to appear.

---

## 3. Reef Metadata Fields

These are per-reef floats now included in `reef_meta.bin`. They describe static properties of each reef and do not vary per word.

### 3a. valence

**What it is**: The mean evaluative polarity of the reef's dimensions, derived from a negation vector.

**How it is computed**: A negation vector is built from ~1,600 directed antonym pairs (e.g., happy -> unhappy). For each pair, compute `embedding(negated) - embedding(positive)` and average. Each dimension's valence is its component of this negation vector. Reef valence = mean of its dimensions' valence values.

**Value range**: Roughly [-0.44, 0.64] at reef level. Negative = positive-pole (negation decreases activation), positive = negative-pole (negation increases activation).

**Use in scoring**: Reefs with strong valence encode evaluative content. A reef with valence = -0.3 captures positive sentiment concepts; one with valence = +0.4 captures negative/critical concepts. This allows lagoon to:
- Weight sentiment-bearing reefs differently in sentiment-aware applications
- Provide an evaluative axis without needing a separate sentiment model
- Detect when a document's reef profile is skewed toward positive or negative evaluation

### 3b. avg_specificity

**What it is**: How concrete vs. abstract the reef's vocabulary is, on average.

**How it is computed**: Each word has a specificity band (integer from -2 to +2) based on how many dimensions it belongs to relative to the population. Each dimension's avg_specificity = mean of its member words' bands. Reef avg_specificity = mean of its dimensions' avg_specificity values.

**Value range**: Practically concentrated around [-1.0, +1.0] since it is a double average.

| Value | Interpretation |
|-------|---------------|
| > 0.5 | Concrete/taxonomic reef — populated by narrow, specific vocabulary |
| -0.2 to 0.5 | Mixed reef |
| < -0.2 | Abstract/social reef — populated by broad, universal vocabulary |

**Use in scoring**: Concrete reefs (e.g., chemical compounds, species names) produce high-confidence narrow matches — a hit is very informative. Abstract reefs (e.g., social evaluation, process words) produce broader, lower-confidence matches. Lagoon could:
- Use avg_specificity as a confidence modifier: hits in concrete reefs count more toward topical classification
- Distinguish between documents that score highly in specific reefs (domain expertise) vs. abstract reefs (general language)

### 3c-f. noun_frac, verb_frac, adj_frac, adv_frac

**What they are**: The grammatical composition of the reef, expressed as fractions summing to ~1.0.

**How they are computed**: A sense-aware, three-tier system:
1. Unambiguous words (92% of vocab): full weight to their known POS
2. Ambiguous words with sense activations in the dimension: POS weight proportional to the number of matching senses
3. Ambiguous words with no sense activation: equal weight across all their POS

These per-dimension POS fractions are averaged across the reef's dimensions.

**Value range**: [0.0, 1.0] per fraction. The four sum to approximately 1.0.

**Use in scoring**: Reefs have distinct grammatical profiles. A reef with verb_frac = 0.6 captures actions/processes; one with noun_frac = 0.9 captures entities/objects. Lagoon could:
- Weight reef contributions differently based on the query's grammatical intent (entity search vs. action search)
- Use POS composition to interpret what a reef activation means (a noun-heavy reef hit suggests the document discusses the reef's entities; a verb-heavy hit suggests the document describes its processes)
- Build grammatically-aware scoring profiles for documents

---

## 4. Two-Tier Reef IDF (affects existing idf_q)

This changes the distribution of the existing `idf_q` field, not its format.

### What changed

Previously, IDF was computed from all `word_reef_affinity` rows regardless of depth. Now:

- **Tier 1**: Words with depth >= 2 in at least one reef get IDF computed from depth-filtered counts only (only reefs where `n_dims >= 2` count toward document frequency).
- **Tier 2**: Words that only ever appear at depth 1 (no strong reef membership) get IDF from all-membership counts.

### Effect on values

Tier-1 words get **higher IDF** than before because their effective document frequency drops (depth-1 noise memberships are excluded). Tier-2 words get the same IDF they would have gotten under the old formula.

### Impact on lagoon

If lagoon caches or hardcodes any IDF-related expectations, this shift matters. The overall effect is that strong reef members become more discriminative (higher IDF -> higher BM25 contribution), while depth-1-only words remain weak. This should improve scoring quality without requiring code changes — the same BM25 formula produces better results with cleaner IDF inputs.

---

## Binary Layout Reference

### word_reefs.bin (v2 flat binary)

```
Header: magic(4) = "WSWR", count(u32) = max_word_id + 1
Index:  [offset(u32), count(u32)] × count     // 8 bytes per word
Data:   [reef_id(u16), bm25_q(u16), csim_q(u8), pad(u8), swz_q(u16)] × total_entries  // 8 bytes per entry
```

Previous v2.0 data entries were 4 bytes (`reef_id(u16) + bm25_q(u16)`). Now 8 bytes.

### reef_meta.bin (v2 flat binary)

```
Header: magic(4) = "WSRM", count(u32) = n_reefs
Record: hierarchy_addr(u32) + n_words(u32) + valence(f32) + avg_specificity(f32)
        + noun_frac(f32) + verb_frac(f32) + adj_frac(f32) + adv_frac(f32)
        + name(64 bytes, null-padded UTF-8)
        = 96 bytes per record
```

Previous v2.0 records were 72 bytes (no float fields between n_words and name).

### constants.bin

Two new scalar fields appended to the v2 struct:

```
..., FNV1A_PRIME(u64), SWZ_SCALE(u32), CSIM_SCALE(u32)
```

### msgpack (v1) format

- `word_reefs.bin`: Inner lists now have 4 elements `[reef_id, bm25_q, csim_q, swz_q]` instead of 2.
- `reef_meta.bin`: Each dict gains keys `"valence"`, `"avg_specificity"`, `"noun_frac"`, `"verb_frac"`, `"adj_frac"`, `"adv_frac"`.
- `constants.bin`: Gains keys `"SWZ_SCALE"` and `"CSIM_SCALE"`.

---

## Integration Suggestions

### Minimal integration (use new signals as tie-breakers)

Keep BM25 as the primary scoring signal. Use csim_q and swz_q only when BM25 produces ties or near-ties:

```
if abs(score_a - score_b) < epsilon:
    prefer the candidate with higher csim_q (or swz_q)
```

### Medium integration (weighted blend)

Combine all three signals into a composite word-reef score:

```
composite = w1 * (bm25_q / BM25_SCALE) + w2 * (csim_q / CSIM_SCALE) + w3 * (swz_q / SWZ_SCALE)
```

Suggested starting weights: `w1 = 0.6, w2 = 0.2, w3 = 0.2`. Tune from there.

### Full integration (reef-aware scoring)

Use the reef metadata to modulate how reef scores are aggregated:

1. Score each word against its reefs using the composite signal above.
2. When aggregating reef scores across a document, weight by reef properties:
   - High `|valence|` reefs contribute to a sentiment sub-score
   - High `avg_specificity` reefs contribute more to topical classification
   - POS composition informs whether a reef hit represents entity mention vs. action description
3. The background model (`bg_mean`, `bg_std`) already handles z-score normalization for BM25. The new signals don't need separate background normalization — they are per-word-reef properties, not aggregated document scores.

### What not to do

- Don't use csim_q as a hard filter (e.g., dropping entries below a threshold) without testing. Some legitimate word-reef associations have moderate centroid similarity because the reef is broad.
- Don't multiply BM25 by swz_q — they are additive signals, not multiplicative. Both are already on different scales and the product would be dominated by outliers.
- Don't treat the POS fractions as a query-time filter. They are descriptive metadata for understanding what reefs encode, not for restricting which reefs can match.
