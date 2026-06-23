"""AdSynth — Streamlit demo.

    streamlit run app/streamlit_app.py

Pick (or paste) an ad concept and watch it become a storyboard: generated assets
→ composed AdFrames → a full user-flow storyboard. Runs with zero API keys.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

# make src/ importable when run via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adsynth import BrandKit, StoryboardPipeline, get_settings  # noqa: E402
from adsynth.data_loader import load_raw_concepts, parse_concept, variant_count  # noqa: E402
from adsynth.formats import AD_FORMATS  # noqa: E402
from adsynth.providers import Providers  # noqa: E402

st.set_page_config(page_title="AdSynth — Storyboard Synthesis", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner=False)
def _concepts() -> list[dict]:
    return load_raw_concepts()


def _png_bytes(img) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎬 AdSynth")
st.sidebar.caption("Text concept → storyboard · Creative Automation + DCO")

settings = get_settings()
providers = Providers(settings)
desc = providers.describe()
st.sidebar.markdown("**Providers**")
st.sidebar.write(
    f"- image: `{desc['image_provider']}`\n"
    f"- llm: `{desc['llm_provider']}` ({'on' if desc['llm_available'] else 'heuristic'})"
)

fmt_key = st.sidebar.selectbox(
    "Ad format",
    options=list(AD_FORMATS.keys()),
    format_func=lambda k: f"{AD_FORMATS[k].label} ({AD_FORMATS[k].width}×{AD_FORMATS[k].height})",
)

st.sidebar.markdown("**Brand kit**")
primary = st.sidebar.color_picker("Primary", "#1A73E8")
accent = st.sidebar.color_picker("Accent / CTA", "#FF6D00")
secondary = st.sidebar.color_picker("Surface / text-on-dark", "#FFFFFF")
style = st.sidebar.text_input("Style keywords", "clean, modern, high-contrast")
brand = BrandKit(
    primary=primary,
    accent=accent,
    secondary=secondary,
    style_keywords=[s.strip() for s in style.split(",") if s.strip()],
)

st.sidebar.markdown("**Generation**")
dco_on = st.sidebar.toggle("DCO mode (ranked variants)", value=False)
n_variants = st.sidebar.slider("Variants", 2, 5, 3) if dco_on else 1
do_grade = st.sidebar.toggle("Critic grading", value=True)

# ---------------------------------------------------------------- main
st.title("Automated Storyboard Synthesis")

concepts = _concepts()
names = [c.get("concept", f"Concept {i}") for i, c in enumerate(concepts)]

tab_pick, tab_custom = st.tabs(["📚 Pick a concept", "✍️ Paste your own"])
raw_concept = None
with tab_pick:
    idx = st.selectbox("Concept", range(len(names)), format_func=lambda i: f"{i:>3}  {names[i]}")
    vc = variant_count(concepts[idx])
    variant = st.number_input("Asset-suggestion variant", 0, max(0, vc - 1), 0) if vc > 1 else 0
    raw_concept = concepts[idx]
    with st.expander("Concept details"):
        st.json(raw_concept)

with tab_custom:
    st.caption("Provide a concept JSON with keys: concept, implementation, explanation, asset_suggestions.")
    txt = st.text_area("Concept JSON", height=200, placeholder='{"concept": "...", "implementation": {...}}')
    if txt.strip():
        try:
            raw_concept = json.loads(txt)
            variant = 0
            st.success("Parsed custom concept.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

go = st.button("✨ Generate storyboard", type="primary", use_container_width=True)

if go and raw_concept:
    bar = st.progress(0.0, text="Starting…")

    def _progress(msg: str, frac: float) -> None:
        bar.progress(min(1.0, frac), text=msg)

    if dco_on:
        pipe = StoryboardPipeline(settings, brand=brand, fmt_key=fmt_key)
        result = pipe.run_dco(raw_concept, n_variants=n_variants, progress=_progress)
        bar.empty()
        st.subheader("DCO variants (ranked by critic score)")
        for rank, board in enumerate(result.storyboards):
            score = (board.meta.get("grade") or {}).get("overall", 0)
            with st.container(border=True):
                st.markdown(f"**#{rank+1} · score {score:.2f}** · variant {board.meta.get('variant')}")
                st.image(_png_bytes(board.image), use_container_width=True)
                st.download_button(
                    "⬇ PNG", _png_bytes(board.image),
                    file_name=f"{board.concept_name[:30]}_v{rank+1}.png",
                    key=f"dl_{rank}",
                )
    else:
        concept = parse_concept(raw_concept, variant=int(variant))
        pipe = StoryboardPipeline(settings, brand=brand, fmt_key=fmt_key, variant=int(variant))
        board = pipe.run(concept, progress=_progress, grade=do_grade)
        bar.empty()

        st.subheader("Storyboard")
        st.image(_png_bytes(board.image), use_container_width=True)
        st.download_button(
            "⬇ Download storyboard PNG", _png_bytes(board.image),
            file_name=f"{board.concept_name[:30]}_{fmt_key}.png",
        )

        grade = board.meta.get("grade")
        if grade:
            st.subheader("Creative critique")
            cols = st.columns(len(board.frames) + 1)
            cols[0].metric("Overall", f"{grade['overall']:.2f}")
            for c, cf in zip(cols[1:], board.frames):
                cq = cf.critique or {}
                c.metric(cf.frame.id, f"{cq.get('overall', 0):.2f}")
            if grade.get("llm_note"):
                st.info(grade["llm_note"])
            notes = [n for cf in board.frames for n in (cf.critique or {}).get("notes", [])]
            if notes:
                st.warning("  \n".join(f"• {n}" for n in notes))

        st.subheader("Frames & assets")
        for cf in board.frames:
            with st.expander(f"{cf.frame.id} · {cf.frame.interaction_type} · {len(cf.placements)} assets"):
                fc, ac = st.columns([1, 2])
                fc.image(_png_bytes(cf.image), width=AD_FORMATS[fmt_key].width)
                ac.write(cf.frame.description)
                ac.json({"placements": [p.model_dump() for p in cf.placements]})
elif go:
    st.error("No concept selected or parsed.")
