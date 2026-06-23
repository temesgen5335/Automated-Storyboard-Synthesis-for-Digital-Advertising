# Architecture

AdSynth is a staged pipeline with a strict **provider-abstraction** boundary so the
creative logic never depends on a specific AI vendor.

```
                 ┌─────────────────────────────────────────────────────────┐
   concepts.json │  data_loader → Concept{ frames[], asset_briefs[] }        │
        │        └─────────────────────────────────────────────────────────┘
        ▼
  ┌───────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐   ┌──────────────┐
  │  Planner  │──▶│  Generation  │──▶│  Composition  │──▶│  Critic  │──▶│   Assembly   │
  │  (agent)  │   │ image + text │   │ layout engine │   │ (agent)  │   │  storyboard  │
  └───────────┘   └──────────────┘   └───────────────┘   └──────────┘   └──────────────┘
        │                │                                     │
        ▼                ▼                                     ▼
  LLMProvider      ImageProvider                         LLMProvider
  (gemini/groq/    (pollinations/hf/mock)                (optional note)
   openrouter/none)
```

## Stage responsibilities

| Stage | Module | In → Out | Keyless behaviour |
|---|---|---|---|
| Load | `data_loader` | concepts.json → `Concept` | n/a |
| Plan | `agents/planner` | `AssetBrief` → image prompt / final copy | heuristic prompt + quoted-copy extraction |
| Generate | `generation/{assets,text_render}` | prompt → PIL image | Pollinations (keyless) or mock; PIL text |
| Compose | `composition/{layout,compositor}` | assets → `ComposedFrame` | pure PIL, deterministic band layout |
| Critique | `agents/critic` | frame → scores | heuristic (composition/CTA/brand/legibility) |
| Assemble | `assembly/storyboard` | frames → storyboard PNG | pure PIL |
| DCO | `pipeline.run_dco` | concept → ranked variants × formats | reuses all of the above |

## Why these choices

- **Generative over retrieval.** The raw asset archive from the original challenge was
  never available, and modern creative automation generates on-brand assets on demand.
  This also removes a data dependency that would block reproduction.
- **Provider abstraction + keyless defaults.** A reviewer can `git clone && streamlit run`
  with no signup. Quality scales up by adding a free key — no code changes.
- **Deterministic layout engine** instead of an LLM "designer": fast, free, explainable,
  and reproducible (seeded). The LLM is reserved for language tasks (copy, critique notes).
- **Critic as a first-class stage.** Scoring is what turns "generate a creative" into
  "optimize creatives" — the foundation of DCO and (later) performance prediction.

## Extension points

- **New image/LLM backend:** implement `ImageProvider` / `LLMProvider`, register in
  `providers/{image,llm}.py`. Nothing else changes.
- **New ad format:** add an `AdFormat` to `formats.py`; layout adapts automatically.
- **Brand kit:** `BrandKit` already threads palette/style through generation, composition,
  and critique. Future: derive a kit from an uploaded logo.
- **Performance model:** the critic returns structured scores; swap/augment with a learned
  CTR/engagement predictor trained on the KPI data referenced in the brief.
