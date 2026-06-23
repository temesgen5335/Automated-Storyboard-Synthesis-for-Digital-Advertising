"""Smoke + unit tests. Uses the offline mock image provider (no network/keys)."""
from __future__ import annotations

import os

os.environ.setdefault("ADSYNTH_IMAGE_PROVIDER", "mock")
os.environ.setdefault("ADSYNTH_LLM_PROVIDER", "none")

from adsynth import StoryboardPipeline, get_settings, load_raw_concepts, parse_concept  # noqa: E402
from adsynth.schemas import BrandKit  # noqa: E402
from adsynth.taxonomy import normalize_category  # noqa: E402


def test_taxonomy_normalization():
    assert normalize_category("CTA Button") == "Call-To-Action (CTA) Button"
    assert normalize_category("Background Animation") == "Background Image"
    assert normalize_category("Brand Logo") == "Logo"
    assert normalize_category("Tagline") == "Text Elements"


def test_concepts_load_and_parse():
    raws = load_raw_concepts()
    assert len(raws) > 100
    c = parse_concept(raws[0])
    assert c.name
    assert c.frames and all(f.id for f in c.frames)
    # at least one frame carries asset briefs
    assert any(f.assets for f in c.frames)


def test_text_brief_extracts_copy():
    raws = load_raw_concepts()
    c = parse_concept(raws[0])
    text_assets = [a for f in c.frames for a in f.assets if a.is_text]
    assert text_assets, "expected at least one text asset"
    assert all(a.text for a in text_assets)


def test_pipeline_produces_storyboard():
    raws = load_raw_concepts()
    concept = parse_concept(raws[0])
    pipe = StoryboardPipeline(get_settings(), brand=BrandKit(), fmt_key="fs")
    board = pipe.run(concept)
    assert board.image is not None
    assert board.image.size[0] > 0 and board.image.size[1] > 0
    assert len(board.frames) == len(concept.frames)
    # every composed frame has an image and a critique
    for cf in board.frames:
        assert cf.image is not None
        assert cf.critique and 0.0 <= cf.critique["overall"] <= 1.0


def test_dco_ranks_variants():
    raws = load_raw_concepts()
    pipe = StoryboardPipeline(get_settings(), brand=BrandKit(), fmt_key="mpu")
    result = pipe.run_dco(raws[1], n_variants=2)
    assert len(result.storyboards) == 2
    scores = [(b.meta.get("grade") or {}).get("overall", 0) for b in result.storyboards]
    assert scores == sorted(scores, reverse=True)  # ranked best-first
