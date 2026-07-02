"""Animatic export — turn a composed storyboard into a *moving ad* preview.

A storyboard is the intermediate representation between a text campaign and a
video ad. This module walks the composed frames per their scripted ``duration``,
applies interaction-aware transitions (swipe -> slide, tap -> cut, default ->
crossfade) and a subtle Ken Burns drift on holds, and writes an animated GIF
(PIL-only, zero extra deps) or MP4 (if ``imageio-ffmpeg`` is installed).

This is deliberately *not* generative video: it is the deterministic animatic
layer that a video-generation stage can later replace shot-by-shot — each
frame is a keyframe, each transition a cut instruction. The plan stays
inspectable; only the renderer gets smarter.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from PIL import Image

from .schemas import Storyboard

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_DUR_RE = re.compile(r"([\d.]+)\s*(s|sec|second|seconds|ms)?", re.I)


def parse_duration(raw: Optional[str], default: float = 2.0) -> float:
    """'5 seconds' -> 5.0, '750ms' -> 0.75, None -> default (clamped 0.5–8s)."""
    if not raw:
        return default
    m = _DUR_RE.search(str(raw))
    if not m:
        return default
    val = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    if unit == "ms":
        val /= 1000.0
    return max(0.5, min(val, 8.0))


def _ken_burns(img: Image.Image, t: float, max_zoom: float = 1.06) -> Image.Image:
    """Subtle push-in: t in [0,1] -> zoom from 1.0 to max_zoom, center-anchored."""
    w, h = img.size
    zoom = 1.0 + (max_zoom - 1.0) * t
    cw, ch = int(w / zoom), int(h / zoom)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def _crossfade(a: Image.Image, b: Image.Image, steps: int) -> List[Image.Image]:
    return [Image.blend(a, b, (i + 1) / (steps + 1)) for i in range(steps)]


def _slide(a: Image.Image, b: Image.Image, steps: int) -> List[Image.Image]:
    """b pushes a off to the left — reads as a 'swipe'."""
    w, h = a.size
    frames = []
    for i in range(steps):
        off = int(w * (i + 1) / (steps + 1))
        canvas = Image.new("RGB", (w, h))
        canvas.paste(a, (-off, 0))
        canvas.paste(b, (w - off, 0))
        frames.append(canvas)
    return frames


_TRANSITIONS = {
    "swipe": _slide,
    "drag": _slide,
    "scrub": _slide,
    "tap": None,        # hard cut — a tap advances instantly
    "hotspot": None,
}


def _pick_transition(interaction: str):
    key = (interaction or "").lower()
    for name, fn in _TRANSITIONS.items():
        if name in key:
            return fn
    return _crossfade


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def render_animatic_frames(
    board: Storyboard,
    fps: int = 12,
    transition_s: float = 0.4,
) -> tuple[List[Image.Image], int]:
    """Expand the storyboard into a flat frame list. Returns (frames, ms/frame)."""
    if not board.frames:
        raise ValueError("Storyboard has no composed frames")

    size = board.frames[0].image.size
    stills = [cf.image.convert("RGB").resize(size, Image.LANCZOS) for cf in board.frames]
    frame_ms = int(1000 / fps)
    out: List[Image.Image] = []

    for i, cf in enumerate(board.frames):
        hold_s = parse_duration(cf.frame.duration)
        n_hold = max(2, int(hold_s * fps))
        # Ken Burns drift across the hold so nothing is ever static
        out.extend(_ken_burns(stills[i], t / max(1, n_hold - 1)) for t in range(n_hold))

        if i + 1 < len(stills):
            trans = _pick_transition(cf.frame.interaction_type)
            if trans is not None:
                out.extend(trans(out[-1], stills[i + 1], max(1, int(transition_s * fps))))

    return out, frame_ms


def export_animatic(
    board: Storyboard,
    path: str,
    fps: int = 12,
    transition_s: float = 0.4,
) -> str:
    """Write the animatic. ``.gif`` needs only PIL; ``.mp4`` needs imageio-ffmpeg."""
    frames, frame_ms = render_animatic_frames(board, fps=fps, transition_s=transition_s)

    if path.lower().endswith(".mp4"):
        try:
            import imageio.v3 as iio  # optional dependency
            import numpy as np

            iio.imwrite(path, [np.asarray(f) for f in frames], fps=fps, codec="libx264")
            log.info("Animatic (mp4) -> %s (%d frames)", path, len(frames))
            return path
        except ImportError:
            log.warning("imageio-ffmpeg not installed; falling back to GIF")
            path = path[:-4] + ".gif"

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    log.info("Animatic (gif) -> %s (%d frames)", path, len(frames))
    return path
