"""Render text assets (headlines, CTAs, taglines) to RGBA images with PIL.

Task 1 of the brief asks to "convert provided text into images while considering
size, font and visual properties". CTA-type text gets a filled button plate;
other text gets a translucent plate for legibility over any background.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..schemas import AssetBrief, BrandKit


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _load_font(brand: BrandKit, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if brand.font:
        candidates.append(brand.font)
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_text_asset(
    brief: AssetBrief,
    brand: BrandKit,
    max_width: int,
    *,
    font_size: Optional[int] = None,
) -> Image.Image:
    text = (brief.text or brief.description or "").strip()
    is_cta = "cta" in brief.category.lower() or "button" in brief.category.lower()
    if is_cta and len(text.split()) > 4:
        text = "Learn More"  # CTAs must stay short

    font_size = font_size or max(14, int(max_width * (0.11 if is_cta else 0.09)))
    font = _load_font(brand, font_size)

    probe = Image.new("RGBA", (max_width, 10))
    d = ImageDraw.Draw(probe)
    pad = max(8, font_size // 2)
    lines = _wrap(d, text, font, max_width - 2 * pad)

    line_h = font_size + 6
    text_w = max((d.textlength(ln, font=font) for ln in lines), default=max_width // 2)
    box_w = int(min(max_width, text_w + 2 * pad))
    box_h = int(len(lines) * line_h + 2 * pad)

    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if is_cta:
        plate, fg = _hex(brand.accent) + (255,), _hex(brand.secondary) + (255,)
        radius = box_h // 2
    else:
        plate, fg = (0, 0, 0, 130), _hex(brand.secondary) + (255,)
        radius = 10
    draw.rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=radius, fill=plate)

    y = pad
    for ln in lines:
        w = draw.textlength(ln, font=font)
        draw.text(((box_w - w) / 2, y), ln, font=font, fill=fg)
        y += line_h
    return img
