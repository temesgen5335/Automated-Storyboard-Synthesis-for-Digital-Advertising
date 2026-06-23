"""Download a REAL asset pack for a concept and render a storyboard from it.

Sources real, CC-licensed assets from Openverse (keyless) and writes them to
``data/asset_packs/<concept>/`` with a manifest of attributions — the modern
stand-in for the original challenge's ``Challenge_Data/Assets/<creative>/``.

Usage
-----
    python scripts/build_asset_pack.py --list
    python scripts/build_asset_pack.py --index 0 --format fs
    python scripts/build_asset_pack.py --name "Escape Challenge Teaser"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adsynth.asset_pack import build_asset_pack, render_pack  # noqa: E402
from adsynth.config import OUTPUT_DIR  # noqa: E402
from adsynth.data_loader import load_raw_concepts  # noqa: E402
from adsynth.schemas import BrandKit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build a real asset pack + storyboard")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--name", type=str)
    ap.add_argument("--format", default="fs")
    ap.add_argument("--variant", type=int, default=0)
    args = ap.parse_args(argv)

    raws = load_raw_concepts()
    if args.list:
        for i, r in enumerate(raws):
            print(f"{i:3d}  {r.get('concept', '?')}")
        return 0

    if args.index is not None:
        raw = raws[args.index]
    elif args.name:
        raw = next((r for r in raws if r.get("concept", "").lower() == args.name.lower()), None)
        if raw is None:
            print(f"Concept not found: {args.name}", file=sys.stderr)
            return 2
    else:
        raw = raws[0]

    brand = BrandKit(name=raw.get("concept", "Brand").split()[0])
    print(f"Downloading real assets for: {raw.get('concept')!r} ...")
    pack_dir = build_asset_pack(raw, brand=brand, fmt_key=args.format, variant=args.variant)
    print(f"Asset pack written → {pack_dir}")
    print(f"  files: {len(list(pack_dir.glob('*')))}  (see manifest.json for attributions)")

    board = render_pack(pack_dir, brand=brand, fmt_key=args.format)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{pack_dir.name}_real_{args.format}.png"
    board.image.convert("RGB").save(out)
    print(f"Storyboard (from real assets) → {out}")
    print(f"Critic score: {(board.meta.get('grade') or {}).get('overall', 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
