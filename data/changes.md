# Export v1.1 Changes — Handoff for Lagoon

## What's new

1. **reef_edges.bin** added to the export (both v1 and v2 formats)
2. **v2 flat binary format** alongside existing v1 msgpack
3. FORMAT_VERSION bumped from `1.0` to `1.1`

---

## reef_edges.bin

### What it contains

Directed edges between reefs with a pre-computed composite weight.
Each edge `(src, tgt, weight)` represents the strength of the
semantic relationship from reef `src` to reef `tgt`. Only edges
with `weight > 0.01` (EXPORT_WEIGHT_THRESHOLD) are exported.

Edges are sorted by `(src, tgt)` for binary search.

### Weight formula

```
weight = (containment * lift)
       * pos_similarity ^ ALPHA_POS
       * exp(-ALPHA_VAL * |valence_gap|)
       * exp(-ALPHA_SPEC * |specificity_gap|)
```

The five component scores:

| Component | Range | Meaning |
|-----------|-------|---------|
| **containment** | [0, 1] | Fraction of source reef's words that also appear in target reef |
| **lift** | [0, +inf) | `P(target \| source) / P(target)` — how much more likely target words appear given source, vs. baseline |
| **pos_similarity** | [0, 1] | Cosine similarity of the POS-fraction vectors (noun/verb/adj/adv) between source and target |
| **valence_gap** | (-inf, +inf) | `target.valence - source.valence` — signed difference in sentiment polarity |
| **specificity_gap** | (-inf, +inf) | `target.avg_specificity - source.avg_specificity` — signed difference in domain specificity |

The three tuning knobs (from `config.py`):

| Knob | Value | Effect |
|------|-------|--------|
| `COMPOSITE_ALPHA_POS` | 2.0 | POS similarity exponent — higher = stronger gating on POS mismatch |
| `COMPOSITE_ALPHA_VAL` | 2.92 | Valence gap decay rate — higher = more suppression for valence mismatch |
| `COMPOSITE_ALPHA_SPEC` | 0.5 | Specificity gap decay rate — mild effect |

### How to use it for single-hop propagation

When scoring a query, after computing direct BM25 scores per reef,
propagate scores through reef edges:

```
for (src, tgt, w) in reef_edges:
    propagated_score[tgt] += direct_score[src] * w
```

The weight already encodes all the gating logic (POS compatibility,
valence alignment, specificity match), so no additional filtering
is needed at query time.

---

## v2 Format Specification (flat binary)

All v2 files live in `output_dir/v2/`. Every file shares the same
header layout:

```
[0..4)   magic    4 bytes ASCII
[4..8)   count    u32 LE — number of records (or index entries)
[8..)    records  fixed-stride (see below)
```

All multi-byte values are **little-endian**.

### File layouts

#### reef_edges.bin (magic: `WSRE`)

| Offset | Type | Field |
|--------|------|-------|
| 0 | u8 | src reef ID |
| 1 | u8 | tgt reef ID |
| 2 | f32 | weight |

**6 bytes/record.** Sorted by (src, tgt).

#### word_lookup.bin (magic: `WSWL`)

| Offset | Type | Field |
|--------|------|-------|
| 0 | u64 | lookup_hash (FNV-1a of the lookup string) |
| 8 | u64 | word_hash (FNV-1a of the canonical word) |
| 16 | u32 | word_id |
| 20 | i8 | specificity (-2 to 2) |
| 21 | u8 | idf_q (quantized IDF, 0-255) |
| 22 | 2 bytes | padding |

**24 bytes/record.** Sorted by lookup_hash for binary search.

#### word_reefs.bin (magic: `WSWR`)

Two sections after the header:

**Index** (count entries, 8 bytes each):

| Offset | Type | Field |
|--------|------|-------|
| 0 | u32 | offset (into data section) |
| 4 | u32 | count (number of reef entries) |

**Data** (variable length, 4 bytes each):

| Offset | Type | Field |
|--------|------|-------|
| 0 | u8 | reef_id |
| 1 | 1 byte | padding |
| 2 | u16 | bm25_q (quantized BM25 score) |

The index is sized `max_word_id + 1`, so you can look up
`word_id` directly by index position.

#### reef_meta.bin (magic: `WSRM`)

| Offset | Type | Field |
|--------|------|-------|
| 0 | u16 | hierarchy_addr (arch\|island\|reef packed) |
| 2 | u16 | n_words |
| 4 | 64 bytes | name (UTF-8, null-padded) |

**68 bytes/record.**

#### island_meta.bin (magic: `WSIM`)

| Offset | Type | Field |
|--------|------|-------|
| 0 | u8 | arch_id |
| 1 | 1 byte | padding |
| 2 | 64 bytes | name (UTF-8, null-padded) |

**66 bytes/record.**

#### background.bin (magic: `WSBG`)

After the header (`count` = N_REEFS):

- `bg_mean`: N_REEFS x f32
- `bg_std`: N_REEFS x f32

**4 bytes per value**, 2 * N_REEFS values total.

#### compounds.bin (magic: `WSCP`)

Two sections after the header:

**Index** (count entries, 8 bytes each):

| Offset | Type | Field |
|--------|------|-------|
| 0 | u32 | str_offset (byte offset into string pool) |
| 4 | u32 | word_id |

**String pool**: null-terminated UTF-8 strings, concatenated.

#### constants.bin (magic: `WSCN`)

After the header (`count` = N_REEFS), a fixed scalar struct:

| Offset | Type | Field |
|--------|------|-------|
| 0 | u32 | N_REEFS |
| 4 | u32 | N_ISLANDS |
| 8 | u32 | N_ARCHS |
| 12 | f32 | avg_reef_words |
| 16 | f32 | k1 (BM25) |
| 20 | f32 | b (BM25) |
| 24 | u32 | IDF_SCALE |
| 28 | u32 | BM25_SCALE |
| 32 | u64 | FNV1A_OFFSET |
| 40 | u64 | FNV1A_PRIME |

Then two arrays:
- `reef_total_dims`: N_REEFS x f32
- `reef_n_words`: N_REEFS x f32

---

## Consuming v2 in Rust

The v2 format is designed for zero-copy access via mmap.
With `bytemuck`, each file can be consumed as:

```rust
use bytemuck::{Pod, Zeroable};
use std::fs::File;
use memmap2::Mmap;

#[repr(C, packed)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReefEdge {
    src: u8,
    tgt: u8,
    weight: f32,
}

fn load_reef_edges(path: &str) -> &[ReefEdge] {
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let magic = &mmap[0..4];
    assert_eq!(magic, b"WSRE");
    let count = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
    let data = &mmap[8..8 + count * 6];
    bytemuck::cast_slice(data)
}
```

For word_lookup.bin, use binary search on the sorted lookup_hash
field to find entries in O(log n).

---

## FORMAT_VERSION bump

v1.0 -> v1.1:
- Added `reef_edges.bin` to both v1 and v2 exports
- Added v2 flat binary format alongside v1 msgpack
- manifest.json now includes `v2_files`, `v2_format`, `n_edges`, and `edge_weight_threshold`

Lagoon should check `manifest.stats.n_edges` to confirm reef edges
are present before attempting single-hop propagation.
