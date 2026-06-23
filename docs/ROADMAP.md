# Roadmap

The functional v1 covers concept → assets → frames → critique → storyboard, with a DCO
variant/format layer. Below is the path to a production-grade adtech creative system.

## Near term (creative quality)
- [ ] **Transparent-background asset generation** (logos/products) via background removal
      (`rembg`) so composited assets don't carry boxy backplates.
- [ ] **Prompt enrichment with the LLM** on the image path (not just copy), with a
      negative-prompt template per category.
- [ ] **Brand-kit ingestion**: upload a logo → extract palette + place real logo asset.
- [ ] **Font controls** in the brand kit surfaced in the UI.

## DCO / optimization
- [ ] **Performance-prediction model** — train a CTR/engagement-rate regressor on the KPI
      data referenced in the brief; use it (not just heuristics) to rank variants.
- [ ] **Multivariate variant matrix** — headline × CTA × palette × hero permutations with
      automatic de-duplication and diversity sampling.
- [ ] **A/B export** — emit variant metadata (JSON) alongside images for ad-server ingestion.

## Storyboard fidelity
- [ ] **True multi-path rendering** for branching concepts (tree/graph layout, not linearized).
- [ ] **Interaction iconography** — render the actual gesture (tap ripple, swipe hand).
- [ ] **Animation preview** — export an MP4/GIF walking the frames per their `duration`.

## Platform / MLOps
- [ ] **FastAPI service** — `POST /storyboard` returning images + metadata for integration
      with a buy-side/sell-side platform.
- [ ] **Async batch generation** with a job queue + caching layer.
- [ ] **DVC + MLflow** — version concept datasets and track variant scores/experiments
      (the original brief's MLOps learning outcome).
- [ ] **Eval harness** — golden-set concepts + automated layout/score regression tests.

## CV asset analysis (optional track)
- [ ] Fine-tune YOLOv8 on the 23 ad-asset categories for **asset QA** (detect that a
      generated frame actually contains a logo/CTA where intended).
- [ ] Segmentation (UNet++/SAM) for precise asset cut-outs.
