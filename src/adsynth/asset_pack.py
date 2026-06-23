"""Real asset packs — the modern equivalent of the original challenge's
``Challenge_Data/Assets/<creative>/`` folders.

``build_asset_pack`` downloads real, CC-licensed assets (via Openverse) for a
concept and writes them to disk with challenge-style filenames plus a
``manifest.json`` recording attribution/licenses. ``render_pack`` recomposes a
storyboard from an on-disk pack — the "real artifacts → frames → storyboard"
path, mirroring the notebooks' ``create_ad_frame``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from PIL import Image

from .agents.critic import Critic
from .agents.planner import Planner
from .assembly.storyboard import assemble_storyboard
from .composition.compositor import compose_frame
from .config import PROJECT_ROOT, Settings, get_settings
from .data_loader import parse_concept
from .formats import AdFormat, get_format
from .generation.text_render import render_text_asset
from .providers import Providers
from .providers.image import OpenverseImageProvider
from .schemas import AssetBrief, BrandKit, Concept, Frame, GeneratedAsset, Storyboard

log = logging.getLogger(__name__)

ASSET_PACKS_DIR = PROJECT_ROOT / "data" / "asset_packs"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "concept"


# ---------------------------------------------------------------------------
# Build a pack from a concept by downloading real assets
# ---------------------------------------------------------------------------
def build_asset_pack(
    raw_concept: dict,
    *,
    settings: Optional[Settings] = None,
    brand: Optional[BrandKit] = None,
    fmt_key: str = "fs",
    variant: int = 0,
    out_root: Path = ASSET_PACKS_DIR,
    provider=None,
) -> Path:
    settings = settings or get_settings()
    brand = brand or BrandKit()
    fmt = get_format(fmt_key)
    concept = parse_concept(raw_concept, variant=variant)

    # default to real-asset retrieval; injectable for tests / alternative sources
    provider = provider or OpenverseImageProvider(settings)
    planner = Planner(Providers(settings).llm)
    # per-concept seed so distinct concepts don't collide on the shared image cache
    import hashlib as _hl
    base_seed = int(_hl.sha1(concept.name.encode()).hexdigest(), 16) % 100000

    pack_dir = out_root / _slug(concept.name)
    pack_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "concept": concept.name,
        "explanation": concept.explanation,
        "format": fmt.key,
        "frames": {},
    }

    for frame in concept.frames:
        entries = []
        for i, brief in enumerate(frame.assets):
            stem = f"{frame.id}__{_slug(brief.category)}__{i}"
            if brief.is_text:
                img = render_text_asset(brief, brand, max_width=int(fmt.width * 0.9))
                fname = f"{stem}.png"
                img.save(pack_dir / fname)
                entries.append({"category": brief.category, "kind": "text",
                                 "text": brief.text or brief.description, "file": fname})
            else:
                query = brief.description
                if "logo" in brief.category.lower():
                    query = f"{brand.name} {brief.description}"
                w, h = (fmt.width, fmt.height) if "background" in brief.category.lower() else (512, 512)
                img = provider.generate(
                    planner.image_prompt(brief, concept, brand, fmt), w, h,
                    seed=base_seed + i, query=query, category=brief.category,
                )
                ext = "jpg"
                fname = f"{stem}.{ext}"
                img.convert("RGB").save(pack_dir / fname, quality=90)
                entries.append({
                    "category": brief.category, "kind": "image", "file": fname,
                    "description": brief.description,
                    "attribution": getattr(provider, "last_attribution", None),
                })
        manifest["frames"][frame.id] = {
            "interaction_type": frame.interaction_type,
            "next_frame": frame.next_frame,
            "duration": frame.duration,
            "description": frame.description,
            "assets": entries,
        }

    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Built asset pack at %s", pack_dir)
    return pack_dir


# ---------------------------------------------------------------------------
# Load + render a pack from disk
# ---------------------------------------------------------------------------
def load_pack(pack_dir: Path | str) -> tuple[Concept, dict[str, list[GeneratedAsset]]]:
    pack_dir = Path(pack_dir)
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))

    frames: list[Frame] = []
    assets_by_frame: dict[str, list[GeneratedAsset]] = {}
    for fid, fdata in manifest["frames"].items():
        briefs, gen = [], []
        for entry in fdata["assets"]:
            brief = AssetBrief(
                category=entry["category"],
                description=entry.get("description", entry.get("text", "")),
                text=entry.get("text"),
            )
            briefs.append(brief)
            img = Image.open(pack_dir / entry["file"]).convert("RGBA")
            gen.append(GeneratedAsset(brief=brief, image=img, source="asset-pack"))
        frames.append(Frame(
            id=fid,
            description=fdata.get("description", ""),
            interaction_type=fdata.get("interaction_type", "Tap"),
            next_frame=fdata.get("next_frame"),
            duration=fdata.get("duration"),
            assets=briefs,
        ))
        assets_by_frame[fid] = gen

    concept = Concept(name=manifest["concept"], explanation=manifest.get("explanation", ""), frames=frames)
    return concept, assets_by_frame


def render_pack(
    pack_dir: Path | str,
    *,
    brand: Optional[BrandKit] = None,
    fmt_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Storyboard:
    settings = settings or get_settings()
    brand = brand or BrandKit()
    concept, assets_by_frame = load_pack(pack_dir)
    manifest = json.loads((Path(pack_dir) / "manifest.json").read_text(encoding="utf-8"))
    fmt: AdFormat = get_format(fmt_key or manifest.get("format", "fs"))

    composed = [compose_frame(f, assets_by_frame[f.id], brand, fmt) for f in concept.frames]
    critic = Critic(Providers(settings).llm, brand)
    grade = critic.grade_concept(concept, composed)
    image = assemble_storyboard(concept, composed, brand, show_grades=True)

    return Storyboard(
        concept_name=concept.name, format_key=fmt.key, frames=composed, image=image,
        meta={"source": "asset-pack", "pack_dir": str(pack_dir), "grade": grade},
    )
