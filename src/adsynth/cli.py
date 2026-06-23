"""Command-line entry point: render a storyboard for a concept.

Examples
--------
    python -m adsynth.cli --list
    python -m adsynth.cli --index 0 --format fs --out outputs/board.png
    python -m adsynth.cli --name "Escape Challenge Teaser" --dco 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import OUTPUT_DIR
from .data_loader import load_raw_concepts, parse_concept
from .pipeline import StoryboardPipeline
from .schemas import BrandKit


def _progress(msg: str, frac: float) -> None:
    bar = "#" * int(frac * 30)
    sys.stdout.write(f"\r[{bar:<30}] {int(frac*100):3d}%  {msg[:40]:<40}")
    sys.stdout.flush()
    if frac >= 1.0:
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="AdSynth — text concept → storyboard")
    ap.add_argument("--list", action="store_true", help="list available concepts")
    ap.add_argument("--index", type=int, help="concept index in concepts.json")
    ap.add_argument("--name", type=str, help="concept name")
    ap.add_argument("--format", default="fs", help="ad format key (fs, mpu, story, ...)")
    ap.add_argument("--variant", type=int, default=0, help="asset-suggestion variant")
    ap.add_argument("--dco", type=int, default=0, help="generate N DCO variants (ranked)")
    ap.add_argument("--out", type=str, default=None, help="output PNG path")
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

    brand = BrandKit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dco and args.dco > 1:
        pipe = StoryboardPipeline(brand=brand, fmt_key=args.format)
        result = pipe.run_dco(raw, n_variants=args.dco, progress=_progress)
        for i, b in enumerate(result.storyboards):
            score = (b.meta.get("grade") or {}).get("overall", 0)
            path = Path(args.out or OUTPUT_DIR / "dco") .with_suffix("")
            path.parent.mkdir(parents=True, exist_ok=True)
            out = path.parent / f"{path.name}_v{i+1}_{score:.2f}.png"
            b.image.convert("RGB").save(out)
            print(f"variant {i+1}: score={score:.3f} -> {out}")
        return 0

    concept = parse_concept(raw, variant=args.variant)
    pipe = StoryboardPipeline(brand=brand, fmt_key=args.format, variant=args.variant)
    board = pipe.run(concept, progress=_progress)
    out = Path(args.out) if args.out else OUTPUT_DIR / f"{concept.name[:40].replace(' ', '_')}_{args.format}.png"
    board.image.convert("RGB").save(out)
    print(f"Saved storyboard -> {out}")
    if board.meta.get("grade"):
        print(f"Overall creative score: {board.meta['grade']['overall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
