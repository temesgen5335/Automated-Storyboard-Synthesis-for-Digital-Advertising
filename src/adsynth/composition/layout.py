"""Layout engine — decides pixel placement & size for each asset in a frame.

A lightweight, deterministic design system: assets are routed to vertical bands
by their category role (top / hero / bottom), sized by category, then stacked
and centered within the band. This is the modern, rule-based replacement for the
original notebook's hardcoded VERTICAL_POSITIONING dict.
"""
from __future__ import annotations

from ..formats import AdFormat
from ..schemas import GeneratedAsset, Placement

# band assignment by category (top=0, hero=1, bottom=2)
_BAND = {
    "Logo": 0, "Banner": 0, "Social Proof": 0, "Seal or Badge": 0, "Testimonial Quotes": 0,
    "Text Elements": 0,
    "Product Image": 1, "Mascot": 1, "Illustration": 1, "Photograph": 1,
    "Interactive Elements": 1, "Infographic": 1, "Animation Frames": 1,
    "Graphs and Charts": 1, "Map or Location Image": 1,
    "Call-To-Action (CTA) Button": 2, "Legal Disclaimers or Terms": 2,
    "Contact Information": 2, "Coupon or Offer Code": 2, "QR Code": 2, "Icon": 2,
    "Decorative Elements": 1,
}

# target width as fraction of frame width, by band
_WIDTH_FRAC = {
    "Logo": 0.34, "Banner": 0.9, "Text Elements": 0.88, "Social Proof": 0.5,
    "Seal or Badge": 0.25, "Testimonial Quotes": 0.8,
    "Call-To-Action (CTA) Button": 0.6, "Legal Disclaimers or Terms": 0.9,
    "Contact Information": 0.7, "Coupon or Offer Code": 0.6, "QR Code": 0.22, "Icon": 0.18,
}
_DEFAULT_HERO_FRAC = 0.82
_BANDS = [(0.0, 0.30), (0.26, 0.74), (0.72, 1.0)]  # (top, bottom) as fraction of H


def _band_index(category: str, used_text_top: bool) -> int:
    if category == "Text Elements" and used_text_top:
        return 2  # secondary text drops to bottom
    return _BAND.get(category, 1)


def plan_layout(assets: list[GeneratedAsset], fmt: AdFormat) -> list[Placement]:
    W, H = fmt.width, fmt.height
    placements: list[Placement | None] = [None] * len(assets)

    # 1) backgrounds fill the frame
    bands: dict[int, list[int]] = {0: [], 1: [], 2: []}
    used_text_top = False
    for i, ga in enumerate(assets):
        cat = ga.brief.category
        if "background" in cat.lower():
            placements[i] = Placement(category=cat, x=0, y=0, width=W, height=H, z=0)
            continue
        bi = _band_index(cat, used_text_top)
        if cat == "Text Elements" and bi == 0:
            used_text_top = True
        bands[bi].append(i)

    # 2) lay out each band: size by category, stack centered vertically
    for bi, idxs in bands.items():
        if not idxs:
            continue
        top = int(_BANDS[bi][0] * H)
        bottom = int(_BANDS[bi][1] * H)
        band_h = bottom - top
        gap = max(6, band_h // (len(idxs) + 4))

        sized: list[tuple[int, int, int]] = []  # (asset_index, w, h)
        for i in idxs:
            ga = assets[i]
            cat = ga.brief.category
            frac = _WIDTH_FRAC.get(cat, _DEFAULT_HERO_FRAC)
            tw = int(W * frac)
            img_w, img_h = ga.image.size
            aspect = img_h / img_w if img_w else 1.0
            th = int(tw * aspect)
            # clamp to band height share
            max_h = (band_h - gap * (len(idxs) + 1)) // len(idxs)
            if th > max_h and max_h > 0:
                th = max_h
                tw = int(th / aspect) if aspect else tw
            sized.append((i, max(8, tw), max(8, th)))

        total_h = sum(h for _, _, h in sized) + gap * (len(sized) - 1)
        y = top + max(0, (band_h - total_h) // 2)
        for (i, w, h) in sized:
            x = (W - w) // 2
            placements[i] = Placement(category=assets[i].brief.category, x=x, y=y, width=w, height=h, z=bi + 1)
            y += h + gap

    return [p for p in placements if p is not None]
