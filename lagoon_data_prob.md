# Windowsill Data Quality: Root Cause Analysis

**In response to:** shoal retrieval stress test report (2026-02-17)
**Method:** Direct queries against `vector_distillery.duckdb` + reconstruction of
the full export pipeline (BM25 scoring, reef ID remapping, background model)

---

## Executive Summary

The shoal stress test report identifies real retrieval quality problems, but
**misattributes their origin**. After reconstructing the export pipeline
end-to-end and comparing database values against exported data, the issues fall
into three categories:

1. **Two real Windowsill export bugs** — the export includes 2.28M noise entries
   that should be filtered, and IDF computation is inflated by the same noise
2. **At least one lagoon/shoal bug** — several z-scores reported are
   mathematically impossible from the exported data, and one word-reef
   association cited doesn't exist in any format of the export
3. **A genuine embedding limitation** — some common words produce weak,
   scattered activations that no amount of filtering will fix

The fixed-point quantization (u8 IDF, u16 BM25) is **not** a contributing
factor. The v1 and v2 binary formats are consistent and the quantization
precision is adequate.

---

## Issue A: Depth-1 Noise in the Export (Windowsill bug)

This is the single largest data quality issue and the root cause of most
retrieval problems described in the report.

### The problem

`word_reef_affinity` contains entries at every depth (n_dims >= 1). The export
pipeline (`export.py:load_word_reef_affinity`, line 326-344) loads **all**
entries without a depth filter:

```python
rows = con.execute("""
    SELECT word_id, reef_id, n_dims
    FROM word_reef_affinity
    ORDER BY word_id, reef_id
""").fetchall()
```

The depth distribution is:

| Depth | Entries | % of total |
|-------|---------|------------|
| 1 | 2,278,228 | **93.3%** |
| 2 | 151,059 | 6.2% |
| 3 | 11,875 | 0.5% |
| 4+ | 1,327 | 0.1% |
| **Total** | **2,442,489** | |

**93.3% of all exported word-reef associations are depth-1 noise** — a word
activating a single dimension out of a reef's 2-8 dimensions. These are
statistically expected at the z >= 2.0 threshold and carry no semantic signal.

The pipeline already defines `REEF_MIN_DEPTH = 2` in `config.py` and uses it
during reef analytics (phase 9) for core-word counts, but the export ignores it.

### Impact on the reported words

| Word | word_id | Total reefs | Depth >= 2 | Depth-1 noise |
|------|---------|-------------|------------|---------------|
| attack | 9085 | 31 | 4 | 27 (87%) |
| cannon | 19702 | 38 | 4 | 34 (89%) |
| chess | 23347 | 14 | **0** | 14 (100%) |
| navy | 88401 | 17 | 1 | 16 (94%) |
| military | 84287 | 14 | **0** | 14 (100%) |
| earthquake | 39747 | 10 | **0** | 10 (100%) |
| yamamoto | 145842 | 17 | 1 | 16 (94%) |
| admiral | 1692 | 16 | **0** | 16 (100%) |

### What the data actually looks like at depth >= 2

**"attack"** — depth >= 2 associations (vs report's claim of "taxonomic scientific terms" #1):

| Reef (DB ID) | Name | n_dims | sum_weighted_z |
|--------------|------|--------|----------------|
| 132 | military assault operations | 2 | 12.16 |
| 198 | emphasis and intensification | 2 | 10.97 |
| 39 | chemical and scientific terminology | 2 | 9.88 |
| 176 | harm and inadequacy | 2 | 9.88 |

At depth >= 2, attack's **#1 reef IS "military assault operations"** with the
highest sum_weighted_z. The report's claim that it ranks 7th is an artifact of
depth-1 noise entries dominating the score.

**"cannon"** — depth >= 2 associations (vs report's "weather and frontal" #1):

| Reef (DB ID) | Name | n_dims | sum_weighted_z |
|--------------|------|--------|----------------|
| 136 | archaic objects and titles | 2 | 12.36 |
| 31 | culinary descriptive terms | 2 | 12.29 |
| 148 | unrefined and unconventional | 2 | 9.76 |
| 73 | historical periods events | 2 | 9.12 |

Cannon's mapping to "weather and frontal" was a single depth-1 entry
(dim 787, z=2.000 — the absolute minimum threshold) in a reef with only 18
words total. BM25 length normalization then amplified this noise 12.6x relative
to large reefs.

### Impact on reef frequency distribution

The report's Issue 3 (degenerate reef frequency) is primarily caused by depth-1
noise. Filtering to depth >= 2 reduces reef word counts by 85-90%:

| Reef | Name | Words (all) | Words (d>=2) | Retained |
|------|------|-------------|--------------|----------|
| 164 | careless negative manner | 24,073 | 3,479 | 14.5% |
| 123 | people and vehicles | 21,955 | 2,798 | 12.7% |
| 150 | skepticism and discernment | 21,132 | 2,730 | 12.9% |
| 189 | written symbols and notation | 18,942 | 2,411 | 12.7% |

At depth >= 2, the largest reef drops from 24K to 3.5K words. This alone would
dramatically reduce the reef frequency saturation the report describes.

---

## Issue B: IDF Inflated by Depth-1 Noise (Windowsill bug)

`compute_reef_idf()` in `post_process.py:395-409` counts ALL reef entries
(including depth-1) when computing each word's IDF:

```sql
SELECT word_id,
       LN(({n_reefs} - COUNT(*) + 0.5) / (COUNT(*) + 0.5) + 1) AS idf
FROM word_reef_affinity
GROUP BY word_id
```

This means every word's reef count is inflated by noise, deflating IDF:

| Word | Reefs (all) | IDF (actual) | Reefs (d>=2) | IDF (corrected) | IDF penalty |
|------|-------------|--------------|--------------|-----------------|-------------|
| attack | 31 | 2.199 | 4 | 3.944 | -44% |
| cannon | 38 | 1.998 | 4 | 3.944 | -49% |
| chess | 14 | 2.975 | 0 | N/A | total loss |

Words like "attack" lose 44% of their IDF weight because depth-1 noise makes
them appear in 31 reefs instead of 4.

---

## Issue C: Report Z-Scores Don't Match Exported Data (Lagoon/Shoal issue)

Several specific claims in the report are **not reproducible** from the exported
data. The v1 msgpack and v2 flat binary formats were both verified to be
consistent and contain identical data.

### C1: "chess" → "aggression and weaponry" (z=77.46)

**This association does not exist.** Chess (word_id=23347) has zero affinity to
reef 129 ("aggression and weaponry") in the database. The `word_reef_affinity`
table has no row for (word_id=23347, reef_id=129). The export contains no entry
for this pair in either v1 or v2 format.

Chess has 14 reef entries, all depth-1, none of which is "aggression and
weaponry":

| DB Reef | Name | n_dims | z_score |
|---------|------|--------|---------|
| 71 | medical conditions terminology | 1 | 2.97 |
| 144 | spatial arrangement and deformation | 1 | 2.71 |
| 161 | deception and mythology | 1 | 2.68 |
| ... | (11 more, all depth-1, all unrelated) | 1 | 2.0-2.37 |

If the shoal engine reports chess → "aggression and weaponry", it is reading
data from the wrong word_id or has a lookup bug.

### C2: "attack" → "taxonomic scientific terms" (z=95.57)

The report claims this is attack's #1 reef with z=95.57. Reconstructing the
full scoring pipeline (BM25 precomputation + background model normalization)
from the exported data:

| Export Reef | Name | bg-normalized z | bm25_q |
|-------------|------|-----------------|--------|
| 274 | military compound terms | 3.53 | 15,720 |
| 120 | informal name suffixes | 1.09 | 19,143 |
| 153 | speech and human attributes | 0.53 | 22,530 |
| 131 | military assault operations | 0.42 | 14,567 |

"Taxonomic scientific terms" (export reef 27) has a background-normalized
z-score of approximately **-0.4** for attack — not 95.57. A z-score of 95.57
would require either a radically different background model or a data
corruption.

### C3: Cannon z-scores DO match

For comparison, cannon's reported z-scores **are** reproducible:

| Reef | Report z | Computed z | Match? |
|------|----------|------------|--------|
| weather and frontal | 22.34 | 22.34 | Yes |
| muscle nerve function | 17.01 | 17.01 | Yes |

This confirms the background model and BM25 computation are correct in
principle. The discrepancy for attack and chess points to a word-level lookup
bug in lagoon, not a systemic scoring error.

---

## Issue D: Fixed-Point Quantization (NOT the issue)

The user asked whether fixed-decimal optimization could be the cause. After
analysis: **no**.

### IDF quantization (u8)

- Formula: `idf_q = clamp(round(reef_idf * 51), 0, 255)`
- reef_idf range: [1.09, 5.24]
- idf_q range: [56, 267] → 7 words clip at 255 (0.005% of vocabulary)
- Precision: 1/51 = 0.0196 (~1% resolution)
- **Not used in BM25 precomputation** — the export computes BM25 scores from
  the original float `reef_idf`, not from `idf_q`. The quantized `idf_q` is
  stored in `word_lookup.bin` as metadata only.

### BM25 quantization (u16)

- Formula: `bm25_q = round(score * 8192)`
- Score range: ~0.1 to ~2.8
- bm25_q range: ~800 to ~23,000 (well within u16 max of 65,535)
- Precision: 1/8192 = 0.00012 (~0.01% resolution)
- Only 0 words were clamped at the u16 maximum during the most recent export

### V1/V2 consistency

The v1 (msgpack) and v2 (flat binary) exports were verified byte-for-byte for
attack (word_id=9085): all 31 reef entries match exactly between formats. The
v2 binary reader correctly decodes the index (offset=144814, count=31) and all
data entries.

---

## Issue E: Genuine Embedding Limitation (not a bug)

Some words reported as problematic have **zero depth >= 2 reef associations**.
This is not a pipeline bug — these words genuinely produce weak, scattered
activations across the embedding dimensions:

| Word | total_dims | Depth >= 2 reefs | Assessment |
|------|-----------|------------------|------------|
| chess | 14 | 0 | All 14 dims land in different reefs |
| military | 14 | 0 | Same: no reef coherence |
| admiral | 16 | 0 | Same |
| earthquake | 10 | 0 | Same |
| yamamoto | 18 | 1 (compound trade occupations) | Noise |

These words activate enough dimensions to pass the z >= 2.0 threshold, but
their dimensions are scattered uniformly across unrelated reefs rather than
concentrated in a coherent cluster. This means the nomic-embed-text-v1.5 model
does not form sharp dimension clusters for these concepts.

For "chess" specifically: it activates 14 dimensions at z-scores of 2.0-2.97
(barely above threshold), spread across 14 different reefs. No two of its
dimensions share a reef. The domain-anchored sense enrichment doesn't help
because "chess" is unambiguous in WordNet (single POS, no domain compounds
generated for the bare word).

### "manga" missing from vocabulary

Confirmed: "manga" is not in the ~146K WordNet-derived vocabulary. This is a
WordNet coverage limitation — WordNet's lexicon skews toward traditional English
and underrepresents loanwords that have entered mainstream usage.

---

## BM25 Length Normalization Amplification

While not a bug per se, the BM25 parameter choice (`b=0.75`) creates a 12.6x
score amplification between the smallest reefs (7-57 words) and the largest
(16,325 words):

| Reef size | BM25 for 1 dim | bm25_q |
|-----------|----------------|--------|
| 57 words (smallest) | 2.158 | 17,674 |
| 4,731 words (average) | 0.535 | 4,382 |
| 16,325 words (largest) | 0.171 | 1,404 |

Combined with depth-1 noise, this means a spurious single-dimension activation
in a tiny reef outscores a genuine multi-dimension activation in a large reef.
This directly causes the mismatched associations in the report (e.g., cannon →
"weather and frontal" via a 18-word reef at z=2.000).

---

## Recommendations

### Fix in Windowsill (export.py)

1. **Filter depth-1 entries from the export.** In `load_word_reef_affinity()`,
   add `WHERE n_dims >= 2` (or reference `config.REEF_MIN_DEPTH`). This removes
   93.3% of noise entries. Words with only depth-1 associations (35.1% of
   vocabulary) would have empty reef profiles, which is more honest than
   exporting noise.

2. **Recompute reef_idf with the same depth filter.** In
   `compute_reef_idf()` (`post_process.py:395-409`), add
   `WHERE n_dims >= 2` to the affinity count. This corrects IDF inflation
   (e.g., attack's IDF recovers from 2.20 → 3.94).

3. **Consider reducing BM25 `b` parameter** from 0.75 to 0.3-0.5 to reduce
   the amplification between small and large reefs. Alternatively, set a
   minimum reef size floor in BM25 normalization.

### Investigate in Lagoon/Shoal

4. **Audit the word_id → reef lookup path.** The chess → "aggression and
   weaponry" association does not exist in the exported data. If shoal reports
   it, there is a lookup or indexing bug. Verify that word_reefs[23347] returns
   chess's entries and not another word's.

5. **Verify the z-score computation for attack.** The reported z=95.57 for
   "taxonomic scientific terms" is not reproducible from the exported background
   model. Either the background model is being loaded incorrectly, or the
   scoring accumulation has a bug that affects specific words.

6. **Cross-validate cannon's scores as a known-good baseline.** Cannon's
   z-scores (22.34, 17.01) match the exported data exactly. Use this as a
   reference point when debugging the attack/chess discrepancies.

---

## Appendix: Source Values from Database

All values below queried directly from `vector_distillery.duckdb`.

### Reef ID Mapping

The export remaps database reef IDs to contiguous export IDs (sorted by
archipelago → island → reef). The report uses export IDs. Key mappings for
reefs mentioned in the report:

| Export ID | DB ID | Name | n_dims | n_words |
|-----------|-------|------|--------|---------|
| 27 | 28 | taxonomic scientific terms | 4 | 7,170 |
| 45 | 46 | medical and bodily | 4 | 5,976 |
| 128 | 129 | aggression and weaponry | 3 | 6,198 |
| 163 | 164 | careless negative manner | 3 | 5,436 |
| 188 | 189 | written symbols and notation | 6 | 11,091 |
| 275 | 276 | weather and frontal | 3 | 18 |
| 279 | 280 | muscle nerve function | 3 | 25 |

### Background Model Statistics

From the exported `background.bin` (1000 samples, 15 words each, seed=42):

- bg_mean range: [0.0000, 1.6423], median 1.2052
- bg_std range: [0.000001, 2.2247], median 1.2151
- 3 reefs with bg_std < 0.01 (export reefs 107, 159, 263)

### Word Properties

| Word | word_id | total_dims | specificity | pos | category | reef_idf |
|------|---------|-----------|-------------|-----|----------|----------|
| attack | 9085 | 22 | 0 | NULL | single | 2.1990 |
| cannon | 19702 | 15 | 0 | NULL | single | 1.9983 |
| chess | 23347 | 14 | 0 | noun | single | 2.9748 |
| navy | 88401 | 18 | 0 | noun | single | 2.6119 |
| military | 84287 | 14 | 0 | NULL | single | 2.9748 |
| admiral | 1692 | 16 | 0 | noun | single | 2.8199 |
| yamamoto | 145842 | 18 | 0 | noun | single | 2.6119 |
| earthquake | 39747 | 10 | 1 | noun | single | 3.3068 |
| crater | 30765 | 12 | 1 | noun | single | 3.2282 |
| lunar | 79269 | 14 | 0 | adj | single | 2.9748 |
| yakuza | 145828 | 15 | 0 | noun | single | 2.9340 |
| manga | — | — | — | — | — | NOT IN VOCABULARY |
