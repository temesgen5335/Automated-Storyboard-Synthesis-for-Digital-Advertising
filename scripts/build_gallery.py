#!/usr/bin/env python
"""Build a self-contained demo gallery (gallery.html) from rendered outputs.

Scans outputs/cool_demo/ for storyboards + animatic, reads critic scores from
filenames' sibling metadata where present, and writes a single static HTML
page styled as a creative review board. No server, no JS dependencies —
open the file in a browser and screenshare it.

Usage:
    python scripts/build_gallery.py            # -> outputs/cool_demo/gallery.html
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "outputs" / "cool_demo"

FORMAT_META = {
    "fs": ("Full Screen", "320\u00d7480"),
    "story": ("Story / Vertical", "1080\u00d71920"),
    "mpu": ("Mid-Page Unit", "300\u00d7250"),
}

CSS = """
:root{
  --ink:#0B1220; --panel:#101a2e; --line:#1e2a44;
  --text:#e7edf7; --muted:#8fa3c0; --cyan:#38bdf8; --amber:#fbbf24;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--text);
  font:15px/1.6 "Inter",system-ui,-apple-system,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1160px;margin:0 auto;padding:56px 32px 96px}
.eyebrow{font:600 11px/1 "IBM Plex Mono",monospace;letter-spacing:.22em;
  text-transform:uppercase;color:var(--cyan)}
h1{font-family:"Archivo",system-ui,sans-serif;font-weight:800;
  font-size:clamp(34px,5vw,56px);line-height:1.04;letter-spacing:-.02em;
  margin:14px 0 10px}
.sub{color:var(--muted);max-width:56ch}
.pipeline{margin-top:26px;font:500 12px/1 "IBM Plex Mono",monospace;
  color:var(--muted);letter-spacing:.06em}
.pipeline b{color:var(--text);font-weight:600}
.hero{display:grid;grid-template-columns:minmax(240px,320px) 1fr;
  gap:56px;align-items:center;margin-bottom:84px}
.phone{background:#05080f;border:1px solid var(--line);border-radius:34px;
  padding:14px;box-shadow:0 30px 80px rgba(0,0,0,.55)}
.phone img{display:block;width:100%;border-radius:22px}
.phone .slate{margin-top:12px}
section{margin-top:72px}
h2{font-family:"Archivo",sans-serif;font-weight:700;font-size:22px;
  letter-spacing:-.01em;margin-bottom:6px}
.note{color:var(--muted);font-size:14px;margin-bottom:26px;max-width:64ch}
.row{display:flex;flex-wrap:wrap;gap:28px;align-items:flex-end}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:14px;flex:0 1 auto}
.card img{display:block;border-radius:6px;max-height:420px;width:auto;max-width:100%}
.slate{display:flex;gap:14px;align-items:baseline;margin-top:12px;
  font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.08em;
  color:var(--muted);text-transform:uppercase}
.slate .score{color:var(--amber);font-weight:600}
.slate .rank{color:var(--cyan);font-weight:600}
footer{margin-top:88px;padding-top:26px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px;max-width:70ch}
footer code{font-family:"IBM Plex Mono",monospace;color:var(--text);font-size:12px}
@media(max-width:760px){.hero{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""


def slate(*parts: str) -> str:
    return '<div class="slate">' + "".join(f"<span>{p}</span>" for p in parts if p) + "</div>"


def card(img: Path, slate_html: str) -> str:
    src = img.as_posix()
    return (f'<div class="card"><img src="{src}" alt="{html.escape(img.stem)}" '
            f'loading="lazy">{slate_html}</div>')


def main() -> int:
    if not DEMO.exists():
        print("No outputs/cool_demo — run scripts/render_cool_demo.py first", file=sys.stderr)
        return 1

    travel = ROOT / "outputs" / "travel_demo"

    animatic = DEMO / "cool_fs_animatic.gif"
    formats = [(k, DEMO / f"cool_{k}.png") for k in ("fs", "story", "mpu")]
    dco = sorted(DEMO.glob("cool_dco_rank*.png"))

    fmt_cards = "".join(
        card(p, slate(f"{FORMAT_META[k][0]}", FORMAT_META[k][1], "SAME CONCEPT"))
        for k, p in formats if p.exists()
    )
    dco_cards = "".join(
        card(p, slate(f'<span class="rank">RANK {i}</span>', "CRITIC-SCORED", "FS 320\u00d7480"))
        for i, p in enumerate(dco, 1)
    )
    hero = (
        f'<div class="phone"><img src="{animatic.name}" alt="animatic preview">'
        + slate("ANIMATIC", "FS 320\u00d7480", '<span class="score">MOVING-AD PREVIEW</span>')
        + "</div>"
        if animatic.exists() else "<div></div>"
    )

    travel_section = ""
    if travel.exists():
        t_cards = ""
        t_anim = travel / "roam_fs_animatic.gif"
        if t_anim.exists():
            t_cards += card(Path("../travel_demo") / t_anim.name,
                            slate("ANIMATIC", "CHIP TAP-THROUGH",
                                  '<span class="score">PLAYABLE</span>'))
        for name, lab in (("roam_fs.png", "Full Screen 320\u00d7480"),
                          ("roam_story.png", "Story 1080\u00d71920")):
            p = travel / name
            if p.exists():
                t_cards += card(Path("../travel_demo") / name, slate(lab, "DESIGNED UI"))
        if t_cards:
            travel_section = f"""
<section>
  <h2>Playable interstitial &mdash; designed-UI mode</h2>
  <p class="note">A second composition mode: solid brand field, accent shape,
  app screen in a device frame, rendered copy, button and an interactive chip
  row &mdash; each frame is one chip state, so the interaction itself is the
  story. Only the in-device screen touches the image model; everything else is
  deterministic brand-kit rendering.</p>
  <div class="row">{t_cards}</div>
</section>"""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdSynth — Creative Review Board</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=IBM+Plex+Mono:wght@500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">

<div class="hero">
{hero}
<div>
  <div class="eyebrow">AdSynth &mdash; Creative Automation + DCO</div>
  <h1>Concept in.<br>Campaign out.</h1>
  <p class="sub">A text ad concept, planned, generated, composed, critiqued and
  rendered into a moving-ad preview &mdash; every asset constrained by a brand kit,
  every variant scored before it earns an impression.</p>
  <p class="pipeline"><b>PLANNER</b> &rarr; <b>ASSETS</b> &rarr; <b>COMPOSE</b>
  &rarr; <b>CRITIC</b> &rarr; <b>ANIMATIC</b></p>
</div>
</div>

<section>
  <h2>One concept, every placement</h2>
  <p class="note">The DCO layer adapts a single concept to IAB formats —
  layout, asset sizing and copy placement re-derive per aspect ratio.</p>
  <div class="row">{fmt_cards}</div>
</section>

<section>
  <h2>Ranked variants</h2>
  <p class="note">Each variant draws a different asset-suggestion set; the critic
  grades composition, CTA presence, palette alignment and legibility, and the
  set is returned best-first — selection is gated before spend.</p>
  <div class="row">{dco_cards}</div>
</section>
{travel_section}

<footer>
  Rendered by the AdSynth pipeline (<code>scripts/render_cool_demo.py</code>).
  Storyboard = the inspectable plan; the animatic is its deterministic render.
  A video-generation stage replaces the renderer shot-by-shot — the plan,
  brand constraints and critic gate stay exactly where they are.
</footer>
</div></body></html>"""

    out = DEMO / "gallery.html"
    out.write_text(page)
    print(f"Gallery -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
