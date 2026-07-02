#!/usr/bin/env python
"""Render the COOL-Company demo campaign: storyboards + animatic + DCO variants.

Usage (from repo root):
    PYTHONPATH=src python scripts/render_cool_demo.py                # real providers
    ADSYNTH_IMAGE_PROVIDER=mock PYTHONPATH=src python scripts/render_cool_demo.py

Outputs land in outputs/cool_demo/. The main concepts.json dataset is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adsynth.animatic import export_animatic  # noqa: E402
from adsynth.data_loader import parse_concept  # noqa: E402
from adsynth.pipeline import StoryboardPipeline  # noqa: E402
from adsynth.schemas import BrandKit  # noqa: E402

OUT = ROOT / "outputs" / "cool_demo"
OUT.mkdir(parents=True, exist_ok=True)

# COOL-inspired brand kit. Hex values are approximations of their public
# branding — eyedropper the real logo (cool.co) and adjust before the demo.
COOL_BRAND = BrandKit(
    name="The COOL Company",
    primary="#0EA5E9",      # electric cyan-blue
    secondary="#0B1220",    # deep ink navy
    accent="#38BDF8",       # lighter cyan accent
    text_color="#F8FAFC",
    style_keywords=["confident", "modern", "sleek", "tech-forward", "high-contrast"],
)

FORMATS = ["fs", "story", "mpu"]


def _progress(msg: str, frac: float) -> None:
    bar = "#" * int(30 * frac)
    print(f"\r[{bar:<30}] {int(frac * 100):3d}%  {msg:<50}", end="", flush=True)


def main() -> int:
    raw = json.loads((ROOT / "data" / "demo" / "cool_concept.json").read_text())

    # 1) hero render per format (variant 0) + animatic on the FS one
    for fkey in FORMATS:
        concept = parse_concept(raw, variant=0)
        pipe = StoryboardPipeline(brand=COOL_BRAND, fmt_key=fkey)
        board = pipe.run(concept, progress=_progress)
        print()
        png = OUT / f"cool_{fkey}.png"
        board.image.convert("RGB").save(png)
        grade = (board.meta.get("grade") or {}).get("overall")
        print(f"  {fkey:>6}: {png.name}  (score {grade:.3f})" if grade else f"  {fkey}: {png.name}")
        if fkey == "fs":
            gif = export_animatic(board, str(OUT / "cool_fs_animatic.gif"))
            print(f"  {'':>6}  animatic -> {Path(gif).name}")

    # 2) DCO: ranked variants on FS (both suggestion sets)
    pipe = StoryboardPipeline(brand=COOL_BRAND, fmt_key="fs")
    result = pipe.run_dco(raw, n_variants=2, progress=_progress)
    print()
    for rank, board in enumerate(result.storyboards, 1):
        p = OUT / f"cool_dco_rank{rank}.png"
        board.image.convert("RGB").save(p)
        grade = (board.meta.get("grade") or {}).get("overall", 0.0)
        print(f"  DCO #{rank}: {p.name}  (score {grade:.3f})")

    print(f"\nAll outputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
