"""Design-composited elements — deterministic PIL renderers for 'designed UI'
ads (playable interstitials) as opposed to photo-composited ones.

The Booking-style interstitial is solid brand field + one accent shape + one
raster asset in a device frame + rendered text/buttons/chips. Everything here
is deterministic brand-kit rendering; only the *content inside the device
frame* ever touches the image provider.

Opt-in via description prefixes, so existing concepts are untouched:
  * ``design: <spec>``  on a Background asset  -> solid field + accent shape
  * ``device: <prompt>`` on any visual asset   -> provider image inside a phone bezel
  * ``chips: A|B|*C*``   on any asset          -> chip row, starred = selected
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from ..schemas import BrandKit
from ..generation.text_render import _hex, _load_font  # shared brand font/hex helpers

# ---------------------------------------------------------------------------
# 1) design background: solid brand field + one accent shape
# ---------------------------------------------------------------------------


def render_design_background(spec: str, brand: BrandKit, w: int, h: int) -> Image.Image:
    """``spec`` keywords: 'accent circle' | 'accent arc' | 'accent band',
    optionally 'top'/'bottom' (default: lower third, behind the hero)."""
    img = Image.new("RGBA", (w, h), _hex(brand.primary) + (255,))
    draw = ImageDraw.Draw(img)
    s = spec.lower()
    accent = _hex(brand.accent) + (255,)

    cy = int(h * (0.28 if "top" in s else 0.68))
    if "band" in s:
        band_h = int(h * 0.22)
        draw.rectangle([0, cy - band_h // 2, w, cy + band_h // 2], fill=accent)
    elif "arc" in s:
        r = int(w * 0.9)
        draw.ellipse([w // 2 - r, cy, w // 2 + r, cy + 2 * r], fill=accent)
    else:  # circle (default)
        r = int(min(w, h) * 0.34)
        draw.ellipse([w // 2 - r, cy - r, w // 2 + r, cy + r], fill=accent)
    return img


# ---------------------------------------------------------------------------
# 2) device frame: wrap any raster in a phone bezel
# ---------------------------------------------------------------------------


def render_device_frame(screen: Image.Image, scale_w: int = 512) -> Image.Image:
    """Draw a minimal modern phone around ``screen`` (9:16-ish content)."""
    sw = scale_w
    sh = int(sw * 16 / 9)
    screen = screen.convert("RGBA").resize((sw, sh), Image.LANCZOS)

    bezel = max(8, sw // 34)
    W, H = sw + 2 * bezel, sh + 2 * bezel
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = sw // 7

    # body + subtle edge highlight
    draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=(12, 14, 20, 255))
    draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius,
                           outline=(90, 96, 110, 255), width=2)

    # screen with rounded corners
    mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw - 1, sh - 1],
                                           radius=radius - bezel // 2, fill=255)
    img.paste(screen, (bezel, bezel), mask)

    # punch-hole camera
    r = max(3, sw // 60)
    draw.ellipse([W // 2 - r, bezel + r, W // 2 + r, bezel + 3 * r], fill=(5, 5, 8, 255))
    return img


# ---------------------------------------------------------------------------
# 3) chip row: A|B|*C* -> pills, starred selected
# ---------------------------------------------------------------------------


def render_chip_row(spec: str, brand: BrandKit, max_width: int) -> Image.Image:
    labels = [c.strip() for c in spec.split("|") if c.strip()]
    if not labels:
        labels = ["Option"]
    selected = [lb.startswith("*") and lb.endswith("*") for lb in labels]
    labels = [lb.strip("*").strip() for lb in labels]

    font_size = max(13, max_width // 22)
    font = _load_font(brand, font_size)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    pad_x, pad_y, gap = font_size, int(font_size * 0.65), max(6, font_size // 2)

    widths = [int(probe.textlength(lb, font=font)) + 2 * pad_x for lb in labels]
    chip_h = font_size + 2 * pad_y
    tray_pad = gap
    total_w = sum(widths) + gap * (len(labels) - 1) + 2 * tray_pad
    total_h = chip_h + 2 * tray_pad

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # white tray so the row reads as one interactive control
    draw.rounded_rectangle([0, 0, total_w - 1, total_h - 1],
                           radius=total_h // 3, fill=(255, 255, 255, 255))
    x = tray_pad
    primary = _hex(brand.primary)
    for lb, w, sel in zip(labels, widths, selected):
        box = [x, tray_pad, x + w, tray_pad + chip_h]
        if sel:
            draw.rounded_rectangle(box, radius=chip_h // 2, fill=primary + (255,))
            fg = (255, 255, 255, 255)
        else:
            draw.rounded_rectangle(box, radius=chip_h // 2,
                                   outline=(200, 204, 212, 255), width=2)
            fg = (120, 126, 138, 255)
        tw = probe.textlength(lb, font=font)
        draw.text((x + (w - tw) / 2, tray_pad + pad_y - 1), lb, font=font, fill=fg)
        x += w + gap
    return img


# ---------------------------------------------------------------------------
# prefix dispatch helpers (used by the asset generator)
# ---------------------------------------------------------------------------

PREFIXES = ("design:", "device:", "chips:")


def split_prefix(description: str) -> tuple[str | None, str]:
    d = (description or "").strip()
    low = d.lower()
    for p in PREFIXES:
        if low.startswith(p):
            return p[:-1], d[len(p):].strip()
    return None, d
