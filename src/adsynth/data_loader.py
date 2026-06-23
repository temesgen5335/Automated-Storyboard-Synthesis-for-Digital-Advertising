"""Load and parse concepts.json into typed ``Concept`` objects."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .config import CONCEPTS_PATH
from .schemas import AssetBrief, Concept, Frame
from .taxonomy import normalize_category

_QUOTED = re.compile(r"['\"‘’“”]([^'\"‘’“”]{2,40})['\"‘’“”]")


def _extract_copy(description: str) -> Optional[str]:
    """Best-effort pull of literal ad copy from a prose asset description.

    Asset descriptions often embed the literal text in quotes, e.g.
    "...the tagline 'YOUR CITY, NO LIMITS'...". Used by the keyless planner.
    """
    m = _QUOTED.search(description or "")
    return m.group(1).strip() if m else None


def _build_briefs(suggestion_set: dict, frame_id: str) -> list[AssetBrief]:
    """Turn one frame's ``{asset_name: description}`` map into AssetBriefs."""
    frame_assets = suggestion_set.get(frame_id)
    if not isinstance(frame_assets, dict):
        return []
    briefs: list[AssetBrief] = []
    for raw_name, desc in frame_assets.items():
        if not isinstance(desc, str):
            continue
        category = normalize_category(raw_name)
        brief = AssetBrief(category=category, description=desc.strip())
        if brief.is_text:
            brief.text = _extract_copy(desc) or raw_name
        briefs.append(brief)
    return briefs


def parse_concept(raw: dict, variant: int = 0) -> Concept:
    """Parse one raw concepts.json record into a ``Concept``.

    ``variant`` selects which of the (often multiple) asset-suggestion sets to
    use — this is the seed of the DCO variant axis.
    """
    impl = raw.get("implementation", {}) or {}
    suggestions = raw.get("asset_suggestions", []) or []
    if isinstance(suggestions, dict):
        suggestions = [suggestions]
    chosen = suggestions[variant % len(suggestions)] if suggestions else {}

    frames: list[Frame] = []
    for fid, fdata in impl.items():
        if not isinstance(fdata, dict):
            continue
        frames.append(
            Frame(
                id=fid,
                description=str(fdata.get("description", "")).strip(),
                interaction_type=str(fdata.get("interaction_type") or "Tap").strip() or "Tap",
                next_frame=fdata.get("next_frame"),
                duration=fdata.get("duration"),
                assets=_build_briefs(chosen, fid),
            )
        )

    return Concept(
        name=str(raw.get("concept", "Untitled Concept")).strip(),
        explanation=str(raw.get("explanation", "")).strip(),
        frames=frames,
    )


def variant_count(raw: dict) -> int:
    s = raw.get("asset_suggestions", []) or []
    if isinstance(s, dict):
        return 1
    return max(1, len(s))


def load_raw_concepts(path: Path | str = CONCEPTS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def load_concepts(path: Path | str = CONCEPTS_PATH, variant: int = 0) -> list[Concept]:
    return [parse_concept(r, variant=variant) for r in load_raw_concepts(path)]


def load_concept_by_name(name: str, path: Path | str = CONCEPTS_PATH, variant: int = 0) -> Optional[Concept]:
    for raw in load_raw_concepts(path):
        if str(raw.get("concept", "")).strip().lower() == name.strip().lower():
            return parse_concept(raw, variant=variant)
    return None
