#!/usr/bin/env python
"""Render the playable-interstitial demo (fictional brand: Roam).

    PYTHONPATH=src python scripts/render_travel_demo.py
    ADSYNTH_IMAGE_PROVIDER=mock PYTHONPATH=src python scripts/render_travel_demo.py

Outputs -> outputs/travel_demo/. Only the in-device app screens touch the
image provider; background, chips, copy and CTA are deterministic brand-kit
rendering (the 'designed UI' composition mode).
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

OUT = ROOT / "outputs" / "travel_demo"
OUT.mkdir(parents=True, exist_ok=True)

ROAM_BRAND = BrandKit(
    name="Roam",
    primary="#1D4ED8",     # deep travel blue (the solid field)
    secondary="#FFFFFF",   # copy color
    accent="#D97706",      # warm amber (circle, accent word, CTA plate)
    text_color="#0B1B3A",
    style_keywords=["clean", "modern", "trustworthy", "warm"],
)


def _progress(msg: str, frac: float) -> None:
    bar = "#" * int(30 * frac)
    print(f"\r[{bar:<30}] {int(frac * 100):3d}%  {msg:<50}", end="", flush=True)


def main() -> int:
    raw = json.loads((ROOT / "data" / "demo" / "travel_interstitial.json").read_text())
    for fkey in ("fs", "story"):
        concept = parse_concept(raw, variant=0)
        pipe = StoryboardPipeline(brand=ROAM_BRAND, fmt_key=fkey)
        board = pipe.run(concept, progress=_progress)
        print()
        png = OUT / f"roam_{fkey}.png"
        board.image.convert("RGB").save(png)
        grade = (board.meta.get("grade") or {}).get("overall")
        print(f"  {fkey:>6}: {png.name}" + (f"  (score {grade:.3f})" if grade else ""))
        if fkey == "fs":
            gif = export_animatic(board, str(OUT / "roam_fs_animatic.gif"))
            print(f"  {'':>6}  animatic (chip tap-through) -> {Path(gif).name}")
    print(f"\nAll outputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
