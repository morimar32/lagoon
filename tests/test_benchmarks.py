"""Benchmark suite for lagoon scoring engine.

Measures actual performance of the scoring pipeline to validate
(or correct) the README's performance claims.

Run:  pytest tests/test_benchmarks.py --benchmark-enable
Skip: pytest tests/ -m "not benchmark"
"""

from __future__ import annotations

import pytest

import lagoon
from lagoon._hash import fnv1a_u64

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Sample texts (canonical texts from README Section 12 + synthetic targets)
# ---------------------------------------------------------------------------

NEURO_8W = "neuron synapse axon dendrite cortex brain neural hippocampus"

GENERIC_16W = (
    "now is the time for all good men to come to the aid of the country"
)

SENTENCE_30W = (
    "The human brain contains approximately eighty six billion neurons "
    "connected through trillions of synapses forming complex neural "
    "networks that process sensory information and generate behavioral "
    "responses across cortical and subcortical regions"
)

HUCK_FINN = (
    "It was after sun-up now, but we went right on and didn't tie up. "
    "The king and the duke turned out by and by looking pretty rusty "
    "but after they had jumped overboard and took a swim it chippered "
    "them up a good deal. After breakfast the king he took a seat on "
    "the corner of the raft and pulled off his boots and rolled up his "
    "britches and let his legs dangle in the water so as to be "
    "comfortable and lit his pipe and went to getting his Romeo and "
    "Juliet by heart. When he had got it pretty good him and the duke "
    "begun to practice it together. The duke had to learn him over and "
    "over again how to say every speech and he made him sigh and put "
    "his hand on his heart and after a while he said he done it pretty "
    "well."
)

PARAGRAPH_200W = " ".join([SENTENCE_30W] * 7)

DOCUMENT_1000W = " ".join([PARAGRAPH_200W] * 5)

SAMPLE_TEXTS = {
    "neuro_8w": NEURO_8W,
    "generic_16w": GENERIC_16W,
    "sentence_30w": SENTENCE_30W,
    "huck_finn": HUCK_FINN,
    "paragraph_200w": PARAGRAPH_200W,
    "document_1000w": DOCUMENT_1000W,
}

# ---------------------------------------------------------------------------
# 1. Startup (1 run)
# ---------------------------------------------------------------------------


def test_bench_startup(benchmark):
    """Measure lagoon.load() time. README claims <100ms."""
    result = benchmark.pedantic(lagoon.load, rounds=5, iterations=1, warmup_rounds=0)
    benchmark.extra_info["readme_claim"] = "<100ms"


# ---------------------------------------------------------------------------
# 2. End-to-End Scoring (6 runs)
# ---------------------------------------------------------------------------

_E2E_CLAIMS = {
    "sentence_30w": "~10-15us",
    "paragraph_200w": "~100us",
    "document_1000w": "<1ms",
}


@pytest.mark.parametrize("text_key", list(SAMPLE_TEXTS.keys()))
def test_bench_score_e2e(benchmark, scorer, text_key):
    """End-to-end scorer.score() across all text sizes."""
    text = SAMPLE_TEXTS[text_key]
    benchmark.extra_info["text_key"] = text_key
    benchmark.extra_info["n_words"] = len(text.split())
    claim = _E2E_CLAIMS.get(text_key)
    if claim:
        benchmark.extra_info["readme_claim"] = claim
    benchmark(scorer.score, text)


# ---------------------------------------------------------------------------
# 3. Per-Phase Isolation (11 runs)
# ---------------------------------------------------------------------------

# Phase 1: Compound scan (Aho-Corasick) — 3 text sizes


@pytest.mark.parametrize("text_key", ["neuro_8w", "sentence_30w", "paragraph_200w"])
def test_bench_phase1_compound_scan(benchmark, scorer, text_key):
    """Phase 1: Aho-Corasick compound scan. Claims ~100-500ns/sentence."""
    text_lower = SAMPLE_TEXTS[text_key].lower()
    benchmark.extra_info["readme_claim"] = "~100-500ns/sentence"
    benchmark.extra_info["text_key"] = text_key
    benchmark(scorer._tokenizer.scan_compounds, text_lower)


# Phase 2: Tokenize + normalize — 3 text sizes


@pytest.mark.parametrize("text_key", ["neuro_8w", "sentence_30w", "paragraph_200w"])
def test_bench_phase2_tokenize(benchmark, scorer, text_key):
    """Phase 2: tokenize + hash lookup + stem fallback. Claims ~50-200ns/word."""
    text = SAMPLE_TEXTS[text_key]
    text_lower = text.lower()
    _, consumed = scorer._tokenizer.scan_compounds(text_lower)
    benchmark.extra_info["readme_claim"] = "~50-200ns/word"
    benchmark.extra_info["text_key"] = text_key
    benchmark(scorer._tokenizer.tokenize, text, consumed)


# Phase 3: BM25 accumulation — 3 word counts


@pytest.mark.parametrize("n_words", [1, 8, 30])
def test_bench_phase3_bm25_accumulate(benchmark, scorer, n_words):
    """Phase 3: BM25 accumulation over precomputed scores. Claims ~80ns/word."""
    # Get real word_ids from neuroscience text and slice to desired count
    all_ids, _ = scorer._tokenizer.process(NEURO_8W + " " + SENTENCE_30W)
    word_ids = set(list(all_ids)[:n_words])
    benchmark.extra_info["readme_claim"] = "~80ns/word"
    benchmark.extra_info["n_word_ids"] = len(word_ids)
    benchmark(scorer._accumulate_bm25, word_ids)


# Phase 4: Background subtraction — 1 run


def test_bench_phase4_bg_subtract(benchmark, scorer):
    """Phase 4: z = (raw - bg_mean) / bg_std over 207 reefs. Claims ~200ns."""
    word_ids, _ = scorer._tokenizer.process(SENTENCE_30W)
    scores_q, _ = scorer._accumulate_bm25(word_ids)
    raw = [sq / 8192.0 for sq in scores_q]
    benchmark.extra_info["readme_claim"] = "~200ns total"
    benchmark(scorer._subtract_background, raw)


# Phase 5: Result extraction — 1 run


def test_bench_phase5_extract(benchmark, scorer):
    """Phase 5: top-K sort, island/arch rollup, coverage. Claims ~500ns."""
    word_ids, unknown = scorer._tokenizer.process(SENTENCE_30W)
    scores_q, word_counts = scorer._accumulate_bm25(word_ids)
    raw = [sq / 8192.0 for sq in scores_q]
    z = scorer._subtract_background(raw)
    benchmark.extra_info["readme_claim"] = "~500ns"
    benchmark(scorer._extract_results, z, raw, word_counts, word_ids, unknown, 10)


# ---------------------------------------------------------------------------
# 4. Micro-benchmarks (3 runs) — pedantic mode for sub-μs operations
# ---------------------------------------------------------------------------


def test_bench_fnv1a_hash(benchmark):
    """FNV-1a u64 hash of a single word."""
    benchmark.pedantic(
        fnv1a_u64, args=("neuron",), rounds=1000, iterations=1000,
    )


def test_bench_stemmer(benchmark):
    """Snowball stemmer on a single word. README claims ~200ns fallback."""
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    benchmark.extra_info["readme_claim"] = "~200ns/word (stemmer fallback)"
    benchmark.pedantic(
        stemmer.stemWord, args=("neurons",), rounds=1000, iterations=1000,
    )


def test_bench_dict_lookup(benchmark, scorer):
    """Dictionary hash-map lookup for a known word. README claims ~50ns."""
    key = fnv1a_u64("neuron")
    benchmark.extra_info["readme_claim"] = "~50ns (HashMap lookup)"
    benchmark.pedantic(
        scorer._word_lookup.get, args=(key,), rounds=1000, iterations=1000,
    )


# ---------------------------------------------------------------------------
# 5. Batch / Analyze (3 runs)
# ---------------------------------------------------------------------------


def test_bench_score_batch(benchmark, scorer):
    """score_batch() over 10 mixed texts."""
    texts = [
        NEURO_8W, GENERIC_16W, SENTENCE_30W, HUCK_FINN,
        NEURO_8W, GENERIC_16W, SENTENCE_30W, HUCK_FINN,
        NEURO_8W, GENERIC_16W,
    ]
    benchmark(scorer.score_batch, texts)


def test_bench_score_raw(benchmark, scorer):
    """score_raw() returning full 207-element z-score vector."""
    benchmark(scorer.score_raw, SENTENCE_30W)


def test_bench_analyze(benchmark, scorer):
    """analyze() on a 6-sentence multi-topic document."""
    doc = (
        "The brain processes information through neural circuits and "
        "synaptic connections. "
        "Neurons communicate via electrical and chemical signals across "
        "complex networks. "
        "Meanwhile, the stock market experienced unprecedented volatility "
        "last quarter. "
        "Investors scrambled to rebalance portfolios amid rising interest "
        "rates. "
        "In other news, the deep ocean floor contains diverse ecosystems "
        "near hydrothermal vents. "
        "Marine biologists discovered several new species adapted to "
        "extreme pressure and temperature."
    )
    benchmark(scorer.analyze, doc)


# ---------------------------------------------------------------------------
# 6. Vocabulary Extension (2 runs)
# ---------------------------------------------------------------------------


def test_bench_filter_unknown(benchmark, scorer):
    """filter_unknown() with mixed known/unknown/stop words."""
    words = [
        "neuron", "brain", "the", "and", "kubernetes",
        "synapse", "graphql", "cortex", "is", "hippocampus",
        "xylophone", "hello", "of", "a", "dendrite",
    ]
    benchmark(scorer.filter_unknown, words)


def test_bench_add_custom_word(benchmark):
    """add_custom_word() — fresh scorer, unique word name per round."""
    scorer = lagoon.load()
    counter = {"n": 0}

    def add_word():
        counter["n"] += 1
        word = f"benchword{counter['n']}"
        scorer.add_custom_word(
            word,
            [(0, 0.8), (1, 0.5), (2, 0.3)],
            specificity=2,
        )

    benchmark.pedantic(add_word, rounds=100, iterations=1, warmup_rounds=0)
