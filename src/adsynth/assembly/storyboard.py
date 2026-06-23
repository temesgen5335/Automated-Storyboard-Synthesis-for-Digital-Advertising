"""Assemble composed AdFrames into one storyboard image showing user flow.

Frames are laid out left-to-right; arrows between them are labelled with the
interaction (Tap/Swipe/…) that advances the story. Branching concepts get a
note band (full multi-path rendering is a documented follow-up).
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ..schemas import BrandKit, ComposedFrame, Concept
from ..composition.compositor import annotate_frame


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _font(size: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def assemble_storyboard(
    concept: Concept,
    frames: list[ComposedFrame],
    brand: BrandKit,
    *,
    show_grades: bool = False,
) -> Image.Image:
    if not frames:
        raise ValueError("No frames to assemble.")

    tiles = [annotate_frame(cf, brand, show_grade=show_grades) for cf in frames]
    fw, fh = tiles[0].size

    title_h = 64
    gap = max(70, fw // 3)          # room for arrows
    margin = 30
    n = len(tiles)
    total_w = margin * 2 + n * fw + (n - 1) * gap
    total_h = title_h + margin * 2 + fh

    canvas = Image.new("RGBA", (total_w, total_h), (245, 246, 248, 255))
    draw = ImageDraw.Draw(canvas)

    # title
    draw.rectangle([0, 0, total_w, title_h], fill=_hex(brand.primary) + (255,))
    draw.text((margin, 18), f"Storyboard — {concept.name}", font=_font(26), fill=(255, 255, 255, 255))
    fmt_note = f"{n} frames"
    draw.text((total_w - 160, 22), fmt_note, font=_font(18), fill=(255, 255, 255, 220))

    # frames + arrows
    y = title_h + margin
    arrow_font = _font(15)
    for i, tile in enumerate(tiles):
        x = margin + i * (fw + gap)
        canvas.alpha_composite(tile, (x, y))
        draw.rectangle([x, y, x + fw - 1, y + fh - 1], outline=(180, 184, 190, 255), width=2)
        if i < n - 1:
            ax0 = x + fw + 12
            ax1 = x + fw + gap - 12
            ay = y + fh // 2
            draw.line([(ax0, ay), (ax1, ay)], fill=_hex(brand.accent) + (255,), width=4)
            draw.polygon([(ax1, ay), (ax1 - 12, ay - 8), (ax1 - 12, ay + 8)], fill=_hex(brand.accent) + (255,))
            label = frames[i].frame.interaction_type or "Tap"
            tw = draw.textlength(label, font=arrow_font)
            draw.text(((ax0 + ax1) / 2 - tw / 2, ay - 24), label, font=arrow_font, fill=_hex(brand.text_color) + (255,))

    if concept.is_branching:
        draw.text(
            (margin, total_h - 22),
            "⤳ Concept contains branching paths — shown linearized.",
            font=_font(14),
            fill=(120, 120, 120, 255),
        )
    return canvas
