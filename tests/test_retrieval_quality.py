"""Retrieval quality tests for lagoon's base vocabulary scoring.

22 queries across 6 categories stress-tested in shoal identified patterns where
lagoon's base vocabulary works well and where it fails.  109 of 120 tested words
exist in lagoon's base vocabulary.  By testing these patterns directly we can
validate data export changes without re-running the full shoal pipeline.

Currently-failing tests are marked ``xfail(strict=True)`` to capture known
vocabulary/scoring limitations.  When a lagoon data rebuild fixes a word-reef
association the corresponding xfail will start passing (xpass), flagging the
improvement.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_reef_keyword(result, keywords, top_n=10):
    """Return first ScoredReef in *top_n* whose name contains any keyword, or None."""
    for reef in result.top_reefs[:top_n]:
        name_lower = reef.name.lower()
        if any(kw in name_lower for kw in keywords):
            return reef
    return None


def _has_island_keyword(result, keywords):
    """Return first ScoredIsland whose name contains any keyword, or None."""
    for island in result.top_islands:
        name_lower = island.name.lower()
        if any(kw in name_lower for kw in keywords):
            return island
    return None


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Keyword sets matched to gen-1 island names (includes coral-promoted islands)
# After noise cleanup rebuild, island names changed significantly.
MILITARY_KW = {"military", "warfare", "conflict", "combat", "war", "weapon"}
BIOLOGY_KW = {"biolog", "organism", "taxonomy", "anatom", "zoolog", "microscop", "life science"}
PLANT_KW = {"plant", "flora", "botan", "greenery"}
CHEMISTRY_KW = {"chemical", "compound", "substance", "decay", "contamination"}
HISTORY_KW = {"histor", "european geograph", "classical", "mediterranean"}
MUSIC_KW = {"music", "artistic", "performance", "sound", "acoustic"}
EARTH_KW = {"archaeolog", "earth science", "geography", "geological"}
ASTRO_KW = {"archaeolog", "earth science", "geography", "geological"}
GEOLOGY_KW = {"archaeolog", "earth science", "geography", "geological"}
FAUNA_KW = {"fauna", "animal", "livestock", "wildlife", "zoolog", "creature"}
FOOD_KW = {"botan", "plant", "flora", "agricult", "greenery"}
GAME_KW = {"game", "play", "recreation", "sport", "entertainment"}
MACHINE_KW = {"mechanical", "device", "physical movement", "object", "tool"}
TECH_KW = {"technical", "process", "method", "logic"}
MONARCHY_KW = {"aristocra", "hierarchi", "power structure", "authorit"}
VIOLENCE_KW = {"violence", "violent", "conflict", "disorder", "threat", "misconduct", "wrongdoing", "brutal"}


# ===================================================================
# Group 1: Domain Clustering
# ===================================================================
# Multi-word queries where words share a semantic domain.
# Tests that the correct domain reef appears in top 10.


class TestDomainClustering:
    """Multi-word queries where words share a semantic domain."""

    def test_pearl_harbor_military_reef(self, scorer):
        """'Pearl Harbor attack Japan' → military reef in top 10."""
        result = scorer.score("Pearl Harbor attack Japan")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MILITARY_KW)
        assert reef is not None, (
            f"No military reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_dna_genetics_biology_reef(self, scorer):
        """'DNA genetic mutation heredity' → biology reef in top 10."""
        result = scorer.score("DNA genetic mutation heredity")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, BIOLOGY_KW)
        assert reef is not None, (
            f"No biology reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'lunar/crater/impact/basin' map to visual/descriptive reefs, no geological signal",
    )
    def test_lunar_crater_earth_reef(self, scorer):
        """'lunar crater impact basin' → earth/space reef in top 10."""
        result = scorer.score("lunar crater impact basin")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, EARTH_KW)
        assert reef is not None, (
            f"No earth/space reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="3-word query 'python snake reptile' scatters across fluid/scheduling reefs",
    )
    def test_python_snake_fauna_reef(self, scorer):
        """'python snake reptile' → fauna reef in top 10."""
        result = scorer.score("python snake reptile")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, FAUNA_KW)
        assert reef is not None, (
            f"No fauna reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'apple/fruit/nutrition' map to warmth/cheer reef, no food signal",
    )
    def test_apple_fruit_food_reef(self, scorer):
        """'apple fruit nutrition' → food reef in top 10."""
        result = scorer.score("apple fruit nutrition")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, FOOD_KW)
        assert reef is not None, (
            f"No food reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'music/opera/symphony/orchestra' scatter across labels/measurement reefs",
    )
    def test_music_domain_clustering(self, scorer):
        """'music opera symphony orchestra' → music reef in top 10."""
        result = scorer.score("music opera symphony orchestra")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MUSIC_KW)
        assert reef is not None, (
            f"No music reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'French/Revolution/guillotine' map to human roles/virtue reefs, no history signal",
    )
    def test_french_revolution_history_reef(self, scorer):
        """'French Revolution guillotine' → history reef in top 10."""
        result = scorer.score("French Revolution guillotine")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, HISTORY_KW)
        assert reef is not None, (
            f"No history reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )


# ===================================================================
# Group 2: Disambiguation
# ===================================================================
# Ambiguous words steered by context.


class TestDisambiguation:
    """Ambiguous words steered by context."""

    def test_python_context_shifts_profile(self, scorer):
        """'python programming language' vs 'python snake reptile' → different top reefs."""
        prog = scorer.score("python programming language")
        animal = scorer.score("python snake reptile")
        prog_names = {r.name for r in prog.top_reefs[:5]}
        animal_names = {r.name for r in animal.top_reefs[:5]}
        assert prog_names != animal_names, (
            f"Top-5 reefs are identical for both queries: {prog_names}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'python/programming/language' scatter across measurement/contamination reefs",
    )
    def test_python_programming_tech_reef(self, scorer):
        """'python programming language' → tech reef in top 10."""
        result = scorer.score("python programming language")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, TECH_KW)
        assert reef is not None, (
            f"No tech reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_chess_opening_strategy(self, scorer):
        """'chess opening strategy' → game reef in top 10 with conf > 0.1."""
        result = scorer.score("chess opening strategy")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, GAME_KW)
        assert reef is not None, (
            f"No game reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )
        assert result.confidence > 0.1, (
            f"Confidence too low: {result.confidence:.3f}"
        )


# ===================================================================
# Group 3: Single-Word Signal
# ===================================================================
# Individual words that should produce meaningful signal.


class TestSingleWordSignal:

    @pytest.mark.xfail(
        strict=True,
        reason="'Yamamoto' is effectively unknown — confidence 0.001",
    )
    def test_yamamoto_single_word(self, scorer):
        """'Yamamoto' → conf > 0.1."""
        result = scorer.score("Yamamoto")
        assert result.matched_words >= 1
        assert result.confidence > 0.1, (
            f"Confidence too low: {result.confidence:.3f}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'computer' produces no signal above z=0, no tech reef in top 10",
    )
    def test_computer_technology_reef(self, scorer):
        """'computer' → tech reef in top 10."""
        result = scorer.score("computer")
        assert result.matched_words >= 1
        reef = _has_reef_keyword(result, TECH_KW)
        assert reef is not None, (
            f"No tech reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'queen' maps to 'offensive language slang' island — no monarchy signal",
    )
    def test_queen_monarchy_reef(self, scorer):
        """'queen' → monarchy reef in top 10."""
        result = scorer.score("queen")
        assert result.matched_words >= 1
        reef = _has_reef_keyword(result, MONARCHY_KW)
        assert reef is not None, (
            f"No monarchy reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_pacific_war_fleet_commander(self, scorer):
        """'Pacific War fleet commander' → military reef + conf > 0.05."""
        result = scorer.score("Pacific War fleet commander")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MILITARY_KW)
        assert reef is not None, (
            f"No military reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )
        assert result.confidence > 0.05, (
            f"Confidence too low: {result.confidence:.3f}"
        )


# ===================================================================
# Group 4: Base Vocabulary Analogues
# ===================================================================
# Base-vocab-only equivalents of custom-word stress test queries.


class TestBaseVocabAnalogues:
    """Base-vocab equivalents of shoal custom-word stress test queries."""

    def test_cannon_weapon_military(self, scorer):
        """'cannon weapon military' → military reef in top 10 (analogue of Q1)."""
        result = scorer.score("cannon weapon military")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MILITARY_KW)
        assert reef is not None, (
            f"No military reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_yamamoto_pearl_harbor_attack(self, scorer):
        """'Yamamoto Pearl Harbor attack' → military or history reef (analogue of Q4)."""
        result = scorer.score("Yamamoto Pearl Harbor attack")
        assert result.matched_words >= 2
        reef_mil = _has_reef_keyword(result, MILITARY_KW)
        reef_hist = _has_reef_keyword(result, HISTORY_KW)
        assert reef_mil is not None or reef_hist is not None, (
            f"No military or history reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'commander/fleet/navy' map to triggering/latin reefs, no military signal",
    )
    def test_commander_fleet_navy(self, scorer):
        """'commander fleet navy' → military reef in top 10 (analogue of Q6)."""
        result = scorer.score("commander fleet navy")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MILITARY_KW)
        assert reef is not None, (
            f"No military reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_yakuza_violence_killer(self, scorer):
        """'yakuza violence killer' → violence reef in top 10 (analogue of Q10)."""
        result = scorer.score("yakuza violence killer")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, VIOLENCE_KW)
        assert reef is not None, (
            f"No violence reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_battleship_naval_military(self, scorer):
        """'battleship naval unit' → military reef in top 10 (analogue of Q13)."""
        result = scorer.score("battleship naval unit")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, MILITARY_KW)
        assert reef is not None, (
            f"No military reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'computer' produces no signal above z=0",
    )
    def test_apple_computer_tech(self, scorer):
        """'apple computer' → tech reef in top 10 (analogue of Q21)."""
        result = scorer.score("apple computer")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, TECH_KW)
        assert reef is not None, (
            f"No tech reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )


# ===================================================================
# Group 5: Shoal Retrieval Failures — Word-Level Root Causes
# ===================================================================
# Individual words whose reef associations are wrong or missing,
# causing downstream retrieval failures in shoal.  Each test targets
# a specific word that maps to the wrong reef domain.


class TestWordReefAssociations:
    """Individual base-vocab words that map to wrong or irrelevant reefs."""

    @pytest.mark.xfail(
        strict=True,
        reason="'mercury' maps to 'physical elements hazards' z=0.08 — no science signal",
    )
    def test_mercury_planet_or_chemistry(self, scorer):
        """'mercury' → should have earth/space or chemistry reef in top 5."""
        result = scorer.score("mercury")
        assert result.matched_words >= 1
        reef_space = _has_reef_keyword(result, ASTRO_KW, top_n=5)
        reef_chem = _has_reef_keyword(result, CHEMISTRY_KW, top_n=5)
        assert reef_space is not None or reef_chem is not None, (
            f"No planet/chemistry reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'jupiter' maps to 'decay and affliction' island — no science signal",
    )
    def test_jupiter_planet(self, scorer):
        """'jupiter' → should have earth/space reef in top 5."""
        result = scorer.score("jupiter")
        assert result.matched_words >= 1
        reef = _has_reef_keyword(result, ASTRO_KW, top_n=5)
        assert reef is not None, (
            f"No planet/space reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    def test_chloroplast_biology(self, scorer):
        """'chloroplast' → should have biology or plant reef in top 5."""
        result = scorer.score("chloroplast")
        assert result.matched_words >= 1
        reef_bio = _has_reef_keyword(result, BIOLOGY_KW, top_n=5)
        reef_plant = _has_reef_keyword(result, PLANT_KW, top_n=5)
        assert reef_bio is not None or reef_plant is not None, (
            f"No biology/plant reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    def test_photosynthesis_biology(self, scorer):
        """'photosynthesis' → should have biology or plant reef in top 3."""
        result = scorer.score("photosynthesis")
        assert result.matched_words >= 1
        reef_bio = _has_reef_keyword(result, BIOLOGY_KW, top_n=3)
        reef_plant = _has_reef_keyword(result, PLANT_KW, top_n=3)
        assert reef_bio is not None or reef_plant is not None, (
            f"No biology/plant reef in top 3: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'earthquake' produces no signal above z=0 — no geology signal",
    )
    def test_earthquake_geology(self, scorer):
        """'earthquake' → should have geology/earth reef in top 5 with z > 1.0."""
        result = scorer.score("earthquake")
        assert result.matched_words >= 1
        reef = _has_reef_keyword(result, GEOLOGY_KW, top_n=5)
        assert reef is not None and reef.z_score > 1.0, (
            f"No geology reef with z > 1.0 in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    def test_tectonic_geology(self, scorer):
        """'tectonic' → should have geology/earth reef in top 5."""
        result = scorer.score("tectonic")
        assert result.matched_words >= 1
        reef = _has_reef_keyword(result, GEOLOGY_KW, top_n=5)
        assert reef is not None, (
            f"No geology reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )


# ===================================================================
# Group 6: Multi-Word Query Retrieval Gaps
# ===================================================================
# Queries where multiple domain-specific words SHOULD reinforce each
# other but the combined reef profile lacks the expected domain reef.
# These are the exact queries that fail in shoal retrieval.


class TestMultiWordRetrievalGaps:
    """Multi-word queries that produce off-domain reef profiles."""

    @pytest.mark.xfail(
        strict=True,
        reason="'Mercury/closest/sun' scatter across wind/material reefs, no earth science signal",
    )
    def test_mercury_planet_query(self, scorer):
        """'Mercury closest to the sun' → earth/space reef in top 10."""
        result = scorer.score("Mercury closest to the sun")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, ASTRO_KW)
        assert reef is not None, (
            f"No planet/space reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_crane_machine_query(self, scorer):
        """'crane origami arcade claw' → machine or game reef in top 3."""
        result = scorer.score("crane origami arcade claw")
        assert result.matched_words >= 2
        reef_mach = _has_reef_keyword(result, MACHINE_KW, top_n=3)
        reef_game = _has_reef_keyword(result, GAME_KW, top_n=3)
        assert reef_mach is not None or reef_game is not None, (
            f"No machine/game reef in top 3: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    def test_photosynthesis_chloroplast_query(self, scorer):
        """'photosynthesis chloroplast' → biology or plant reef in top 10."""
        result = scorer.score("photosynthesis chloroplast")
        assert result.matched_words >= 2
        reef_bio = _has_reef_keyword(result, BIOLOGY_KW)
        reef_plant = _has_reef_keyword(result, PLANT_KW)
        assert reef_bio is not None or reef_plant is not None, (
            f"No biology/plant reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'jupiter/giant/spot' map to decay/capability islands, no earth science signal",
    )
    def test_jupiter_planet_query(self, scorer):
        """'Jupiter Great Red Spot gas giant' → earth/space reef in top 10."""
        result = scorer.score("Jupiter Great Red Spot gas giant")
        assert result.matched_words >= 3
        reef = _has_reef_keyword(result, ASTRO_KW)
        assert reef is not None, (
            f"No planet/space reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'crater/moon' map to titles/deception islands, no earth science signal",
    )
    def test_crater_moon_query(self, scorer):
        """'Yamamoto crater moon' → earth/space reef in top 10."""
        result = scorer.score("Yamamoto crater moon")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, ASTRO_KW)
        assert reef is not None, (
            f"No space reef in top 10: "
            f"{[r.name for r in result.top_reefs[:5]]}"
        )


# ===================================================================
# Group 7: Scoring Produces Domain Signal (Regression Guards)
# ===================================================================
# Queries where lagoon scoring IS correct — the right domain reef
# appears.  Shoal retrieval still fails for these, but the problem
# is downstream (chunk ranking), not in lagoon's reef profiles.
# These are regression guards: if these start failing, the data
# rebuild broke something that was working.


class TestScoringProducesDomainSignal:
    """Queries where lagoon correctly produces domain-relevant reefs."""

    def test_mercury_toxic_has_chemistry_signal(self, scorer):
        """'Mercury toxic heavy metal' → metallic/chemical reef in top 3."""
        result = scorer.score("Mercury toxic heavy metal")
        assert result.matched_words >= 2
        chem_kw = {"metal", "chemical", "physical sciences", "sciences", "decay", "contamination", "material"}
        reef = _has_reef_keyword(result, chem_kw, top_n=3)
        assert reef is not None, (
            f"No metallic/chemical reef in top 3: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'crane/migratory/bird/wingspan' map to warfare/geometry reefs, no fauna in top 3",
    )
    def test_crane_bird_has_fauna_signal(self, scorer):
        """'crane migratory bird wingspan' → fauna reef in top 3."""
        result = scorer.score("crane migratory bird wingspan")
        assert result.matched_words >= 3
        reef = _has_reef_keyword(result, FAUNA_KW, top_n=3)
        assert reef is not None, (
            f"No fauna reef in top 3: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="'Python/constrictor/snake' map to fluid/scheduling reefs, no fauna in top 5",
    )
    def test_python_snake_has_fauna_signal(self, scorer):
        """'Python constrictor snake' → fauna reef in top 5."""
        result = scorer.score("Python constrictor snake")
        assert result.matched_words >= 2
        reef = _has_reef_keyword(result, FAUNA_KW, top_n=5)
        assert reef is not None, (
            f"No fauna reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:5]]}"
        )

    def test_earthquake_query_has_earth_signal(self, scorer):
        """'earthquake tectonic plates seismic' → earth/space reef in top 5."""
        result = scorer.score("earthquake tectonic plates seismic")
        assert result.matched_words >= 3
        reef = _has_reef_keyword(result, EARTH_KW, top_n=5)
        assert reef is not None, (
            f"No earth/space reef in top 5: "
            f"{[(r.name, round(r.z_score, 2)) for r in result.top_reefs[:7]]}"
        )
