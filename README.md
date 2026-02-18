# 🏫 SLIDE SCOLDER

> **"Is this a slide or a wiki dump? No excuses!"**

Welcome to **Slide Scolder**, the AI-powered critique engine that doesn't hold back. If your lecture slides are a wall of text or a visual nightmare, the Scolder will find it, score it, and tell you exactly how to fix it with a bit of "tough love."

![Slide Scolder UI Mockup](https://raw.githubusercontent.com/YifanXu1999/rate-prof-slide/main/web/static/screenshot.png) *(Placeholder if you have an actual screenshot)*

---

## 🎨 The Vibe: Cartoon UI & Neubrutalism

This project isn't just a tool; it's a **whole vibe**. We've ditched the boring corporate SaaS look for a **stylish cartoon UI** inspired by Neubrutalism—bold borders, vibrant colors, and high-contrast shadows.

### Powered by:
- **🍌 Nano Banana**: Used for generating and editing the consistent "Scolding Teacher" character. Built on Gemini 3 Pro, Nano Banana allowed us to maintain scene consistency and character personality across the UI.
- **🧵 Google Stitch**: The UI architecture was crafted using Google Stitch. By feeding it natural language "vibe" prompts and rough sketches, Stitch generated the raw HTML/CSS components that give this app its unique, punchy look.

---

## ⚡ How it was built: Vibe Coding

This codebase was developed using **Vibe Coding**. Instead of meticulously writing every line of CSS by hand, we defined the "soul" of the application:
1. **The Aesthetic**: "Neubrutalism meets Sunday Comics."
2. **The Voice**: "A strict but fair professor who has seen too many 50-bullet-point slides."
3. **The Workflow**: Describing the desired interaction to our AI agents and letting them "stitch" together the logic and layout while we focused on the high-level vision.

---

## 🧠 The Brain: LLM Calibration & Tuning

To ensure the "strict professor" persona actually scores accurately, we implemented a data-driven tuning loop:

### 1. The Gold Standard Dataset
We curated two specific directories to act as our "Ground Truth":
- **`good_slides/`**: Exemplary decks (e.g., from Stanford, MIT) that demonstrate high attribution hygiene, clear visual hierarchy, and strong engagement.
- **`bad_slides/`**: Decks with tiny fonts, wall-of-text layouts, and uncredited diagrams.

### 2. Prompt Engineering Loop
We didn't just write a prompt and hope for the best. We used an LLM-in-the-loop to:
- **Analyze Failures**: If a `bad_slide` received a high score, we fed the extraction features back to the LLM to identify why the prompt failed to penalize it.
- **Hard Caps & Anchors**: Based on these insights, we implemented "HARD CAPS" in our system prompts (e.g., *if font < 16pt AND words > 60, score ≤ 5*). This forces the LLM to respect strict pedagogical boundaries.
- **Tone Alignment**: We tuned the system instructions to ensure the feedback was "scathing yet constructive," maintaining the teacher persona across all 5 metrics.

### 3. Parameter Calibration
By running the rater over both datasets, we generated a **baseline distribution**. We then adjusted the heuristic weights and LLM temperature until the "good" slides consistently ranked in the 80s-90s and "bad" slides stayed in the "failure zone" (<50).

---

## 🛠️ Features

- **🎯 Five Core Metrics**:
    - **Origination**: Where did this come from?
    - **Visual**: Is it easy on the eyes?
    - **Engagement**: Will your students stay awake?
    - **Clarity of Formulas**: Are the math bits readable?
    - **Easy to Follow**: Does the logic flow or jump off a cliff?
- **📜 Detailed Findings**: 3-8 specific findings per metric with slide references.
- **🔝 Top 5 Prioritized Fixes**: Don't know where to start? We prioritize the biggest wins for you.
- **🎭 Dual Personas**: Switch between `Student` (novice comprehension) and `Expert` (technical delivery) modes.

---

## 🚀 Getting Started

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**System Requirements:**
- `pdftoppm` (Poppler) for rendering slides.
- `libreoffice` (optional, for PPTX support).

### 2. Configure

Set your Qwen API key (we recommend Qwen-Flash for that speedy scolding):

```bash
export QWEN_API_KEY="your_api_key_here"
```

### 3. Run the Web UI

```bash
python web/app.py
```
Open [http://127.0.0.1:5050](http://127.0.0.1:5050) and prepare to be scolded.

---

## 🖥️ CLI Usage

Prefer the terminal? Scold your slides from the command line:

```bash
slide-rater ./deck.pdf -o ./report.json --model qwen-flash --mode expert
```

---

## 📝 Output Schema

The `report.json` includes:
- `overall_score` (0-100)
- `metric_scores` with `findings` (slide, evidence, fix)
- `top_5_improvements` (priority, action, impact, reason)
- `deck_stats` & `slides` metadata.

---

*Built with ♥️ and a lot of attitude using **Nano Banana**, **Google Stitch**, and the power of **Vibe Coding**.*

