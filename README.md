# Slide Quality Rater (MVP)

This project rates lecture slide decks (PDF/PPTX) with five explainable metrics:

- Origination
- Visual
- Engagement
- Clarity of formulas
- Easy to follow

It outputs:

- overall score (0-100)
- five metric scores (0-10)
- 3-8 findings per metric with slide references
- top 5 prioritized improvements

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

System requirements:

- `pdftoppm` (Poppler) for slide image rendering
- `libreoffice` only if using PPTX input

## CLI Use with Qwen Flash

Set your API key through environment variable (recommended):

```bash
export QWEN_API_KEY="<your_key>"
```

Run:

```bash
slide-rater ./deck.pdf -o ./report.json --model qwen-flash --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --mode student
```

Or pass key directly:

```bash
slide-rater ./deck.pdf -o ./report.json --api-key "<your_key>" --model qwen-flash
```

## Web UI for Testing

Run local UI:

```bash
source .venv/bin/activate
export QWEN_API_KEY="<your_key>"
python web/app.py
```

Open:

- [http://127.0.0.1:5050](http://127.0.0.1:5050)

The UI supports:

- upload `.pdf` or `.pptx`
- choose `student` or `expert`
- set `model` and `base_url`
- optional API key and custom weights JSON
- view overall/metric scores and top improvements

## Optional custom weights

```bash
slide-rater ./deck.pdf --weights '{"visual":0.2,"easy_to_follow":0.3,"engagement":0.2,"formula_clarity":0.2,"origination":0.1}'
```

## Output schema

Top-level fields in `report.json`:

- `overall_score`
- `metric_scores.{origination|visual|engagement|formula_clarity|easy_to_follow}.score`
- `metric_scores.*.findings[]` with `slide`, `evidence`, `fix`
- `top_5_improvements[]` with `priority`, `action`, `impact`, `reason`, `slides`
- `deck_stats`
- `slides` (per-slide extracted feature records)

## Notes

- If API key is missing or LLM call fails, the tool falls back to deterministic heuristic scoring.
- `mode=student` prioritizes novice comprehension.
- `mode=expert` allows denser technical delivery.
