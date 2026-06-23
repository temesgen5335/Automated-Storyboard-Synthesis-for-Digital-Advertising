# 🎬 AdSynth — Automated Storyboard Synthesis for Digital Advertising

Turn a **textual ad concept** into a **visual storyboard** — generated assets composed
into AdFrames and assembled into a single user-flow image. Built as a **Creative
Automation + Dynamic Creative Optimization (DCO)** system.

> Origin: 10 Academy "Semantic Image & Text Alignment" challenge (Adludio).
> See [`docs/CONTEXT.md`](docs/CONTEXT.md) for full background and the original briefs.

---

## What it does

```
Concept (text)  →  Planner  →  Asset generation  →  Composition  →  Critic  →  Storyboard
                 (LLM/heuristic)  (image + text)     (layout engine)  (grading)  (user-flow image)
```

- **Two asset sources, one compositor** — either *generate* assets from the concept text
  (text-to-image + PIL text) **or** *retrieve real, CC-licensed assets* (Openverse) and
  compose those. The original challenge's raw asset archive is unavailable, so this
  reconstructs it from real, freely-licensed sources.
- **Runs with zero API keys** — keyless [Pollinations](https://pollinations.ai) generation,
  keyless [Openverse](https://openverse.org) real-image retrieval, and a deterministic
  offline mock fallback. No RAM-heavy local models.
- **DCO layer** — generate N ranked creative variants and adapt one concept to many
  ad formats (FS, MPU, story, leaderboard, …).
- **Critic agent** — heuristic creative-quality grading (composition, CTA presence,
  brand-palette alignment, text legibility), with an optional LLM note.
- **Provider abstraction** — swap image/LLM backends via env vars; nothing is vendor-locked.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Web demo (recommended)
streamlit run app/streamlit_app.py

# Or the CLI
python -m adsynth.cli --list
PYTHONPATH=src python -m adsynth.cli --index 0 --format fs --out outputs/board.png
PYTHONPATH=src python -m adsynth.cli --index 0 --dco 3        # ranked DCO variants
```

Everything works out of the box. To upgrade quality, copy `.env.example` → `.env` and add
a free key (Gemini / Groq / OpenRouter for the agents, Hugging Face for image gen).

## Real asset packs (test data)

The original challenge shipped an `Assets/<creative>/` archive of real image files
(`landing.jpg`, `endframe.jpg`, `cta.png`, `logo*`, backgrounds, …). That archive is no
longer available, so this builds an equivalent from **real, CC-licensed assets** via
Openverse:

```bash
PYTHONPATH=src python scripts/build_asset_pack.py --index 0 --format fs
```

This downloads real assets to `data/asset_packs/<concept>/`, writes a `manifest.json`
recording each asset's **title / creator / license / source URL** (for attribution), and
renders a storyboard from those real files (`render_pack`). In the web app, choose
**Image source → Real assets (Openverse)** to compose any concept from real imagery live.

> Retrieval quality depends on the concept: concrete brand/product terms (e.g. "LEGO CITY")
> resolve to highly relevant images; abstract frame descriptions are weaker. Downloaded
> binaries are git-ignored (mixed licenses, reproducible); manifests are kept as provenance.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ADSYNTH_IMAGE_PROVIDER` | `pollinations` | `pollinations` (keyless) · `huggingface` · `mock` |
| `ADSYNTH_LLM_PROVIDER` | `none` | `none` (heuristic) · `gemini` · `groq` · `openrouter` |
| `HF_TOKEN`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` | — | optional free-tier keys |
| `ADSYNTH_SEED` | `42` | deterministic generation |

## Project layout

```
src/adsynth/
  config.py            settings from env (zero-config defaults)
  formats.py           AdFormat definitions (FS, MPU, …)
  schemas.py           Concept / Frame / AssetBrief / BrandKit / Placement / Storyboard
  taxonomy.py          23 ad-asset categories + fuzzy normalization
  data_loader.py       parse concepts.json → typed Concept objects
  providers/           image.py (pollinations/openverse/hf/mock), llm.py (gemini/groq/openrouter)
  agents/              planner.py (briefs→prompts/copy), critic.py (grading)
  generation/          assets.py (image gen), text_render.py (PIL text)
  composition/         layout.py (band layout engine), compositor.py (paste)
  assembly/            storyboard.py (frames + flow arrows → storyboard image)
  asset_pack.py        build/load/render real asset packs (Openverse + manifest)
  pipeline.py          orchestration + DCO
  cli.py               command-line entry point
scripts/build_asset_pack.py   download real assets → pack → storyboard
app/streamlit_app.py   web demo
notebooks/             CV tutorial + (optional) Colab GPU generation
data/                  concepts.json (115 concepts), categories.txt
docs/                  CONTEXT.md, ARCHITECTURE.md, original challenge briefs
```

## Optional: object detection / asset QA

`pip install -r requirements-cv.txt` adds YOLOv8 (ultralytics) for asset analysis.
Heavy — intended for Colab/GPU, not the core path. See `notebooks/`.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) — performance-prediction model on KPI data,
real multi-path (branching) storyboards, brand-kit ingestion from a logo, and a
FastAPI service for platform integration.
