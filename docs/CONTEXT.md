# Project Context — AdSynth

_Automated Storyboard Synthesis for Digital Advertising → reframed as a **Creative Automation + Dynamic Creative Optimization (DCO)** system._

## 1. Origin

This started as **10 Academy Cohort B, Week 12** (and its refined **July 2024** rewrite — see
[`challenge/cv_challenge_july2024.md`](challenge/cv_challenge_july2024.md) and
[`challenge/week12_brief.md`](challenge/week12_brief.md)). The original client framing is **Adludio**, a
mobile interactive-advertising company. The brief: build an ML system that turns **textual ad concepts**
into **visual storyboards** that depict the narrative flow and user interactions of an ad campaign.

It is being revived as a portfolio "spotlight" project targeting a role at an **end-to-end adtech platform
(buy-side + sell-side under one roof)**. The bar is therefore higher than the original coursework: it must be
**functional, deployable, and recognizable to industry as DCO + Creative Automation**.

## 2. Domain vocabulary (from the brief)

| Term | Meaning |
|---|---|
| **Creative** | The ad a user encounters/interacts with. |
| **Concept** | The envisioned appearance/structure/idea of the ad. |
| **AdFrame** | One scene/segment of a creative; frames together tell the story. |
| **AdFormat** | Display dimensions. FS (Full Screen) = 320×480, MPU (Mid-Page Unit) = 300×250. |
| **Storyboard** | All frames laid out to show the user-flow through the ad. |

## 3. The three tasks (July 2024 framing — what we implement)

1. **Image & Text Generation** — generate realistic ad assets (images + rendered text) from descriptions.
2. **Image Composition** — dynamically position & size assets into AdFrames (visual balance, brand consistency).
3. **Storyboard Building** — synthesize frames into one storyboard image conveying user flow (taps/swipes, branching).

The original Week 12 brief also emphasized **AutoGen agents** (image-analysis skill, text-analysis skill,
critic/grading agent) and CV models (YOLO, segmentation). We keep the agent + critic idea but modernize it.

## 4. Data inventory (what we actually have)

- ✅ `data/concepts.json` — **115 ad concepts**. Each: `concept` (name), `implementation`
  (frame-by-frame dict: `description`, `interaction_type`, `next_frame`, `duration`),
  `explanation` (narrative), `asset_suggestions` (per-frame `category → description`).
  **This is the primary input and it already contains per-frame asset briefs.**
- ✅ `data/categories.txt` — 23 asset categories (Logo, CTA Button, Product Image, Background, …).
- ✅ `yolov8m.pt` — generic COCO YOLOv8 (optional; used for asset QA, not core path).
- ✅ `docs/challenge/storyboard_examples/` — 20 example storyboard thumbnails (visual target reference).
- ❌ **The raw `Assets/` image archive is NOT available** (it was a `[LINK]`; the DVC remote is unconfigured).

**Consequence / design pivot:** because the raw asset images are unavailable, the system is **generative** —
it *creates* assets from the concept text rather than retrieving a fixed archive. This matches the July 2024
Task 1 ("Image & Text Generation") and is the modern Creative-Automation approach anyway.

## 5. Key decisions (locked with the user)

- **Stack:** Python pipeline + **Streamlit** web demo (free to deploy on Streamlit Cloud / HF Spaces).
- **AI backends:** open-source / free APIs only; **no RAM-heavy local models**. Colab for any GPU-heavy work.
  - **Image generation:** [Pollinations](https://pollinations.ai) (free, **keyless**, FLUX-based) as default;
    Hugging Face Inference API (FLUX.1-schnell, free tier) as an alternative; deterministic **PIL mock** fallback.
  - **LLM (planner/critic agents):** free tiers (Google Gemini / Groq Llama / OpenRouter) behind an abstraction;
    a **keyless template planner** uses the asset briefs already in `concepts.json` so the pipeline runs with **zero keys**.
- **Provider abstraction** (`src/adsynth/providers/`) means no code is tied to one vendor — swap via env/config.

## 6. What "functional v1" means

A user opens the Streamlit app, picks (or pastes) a concept, and gets: generated assets → composed AdFrames →
a full storyboard image with interaction arrows — running end-to-end with **no API keys** (keyless providers + mock).

## 7. Modernization axes (DCO + Creative Automation maturity)

- **Brand kit** constraints (palette, logo, fonts) for on-brand consistency.
- **Multi-format adaptation** — one concept → FS + MPU + custom formats.
- **DCO variants** — N creative variants per concept (headline/CTA/palette permutations).
- **Critic/scoring agent** — automated creative-quality grading + regeneration loop.
- **(Later) performance prediction** — learn from KPI data to rank variants.
