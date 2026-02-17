"""Per-metric specialised prompts for slide-deck evaluation."""

from __future__ import annotations

import json
from typing import Any

from .models import SlideFeatures


def _slim_features(features: list[SlideFeatures], max_text_len: int = 400) -> list[dict[str, Any]]:
    out = []
    for f in features:
        text = f.text.replace("\n", " ").strip()
        if len(text) > max_text_len:
            text = text[:max_text_len] + " ..."
        out.append({
            "slide": f.slide_number,
            "words": f.words,
            "bullets": f.bullet_lines,
            "formula_tokens": f.formula_like_tokens,
            "prompts": f.prompt_markers,
            "citations": f.citation_markers,
            "signposting": f.signpost_markers,
            "avg_font_pt": f.avg_font_size_pt,
            "contrast": f.contrast_proxy,
            "clutter": f.visual_clutter_proxy,
            "images": f.image_count,
            "text_excerpt": text,
        })
    return out


# Shared framing: judge lecturer-produced slides, identify good vs bad
JUDGE_FRAMING = (
    "You are judging lecture slides produced by a lecturer. "
    "Your aim is to identify good slides and bad slides: reward clear, original, well-attributed teaching; "
    "penalize poor design, unattributed borrowing, and especially copying of diagrams/figures from other sources (including other top universities) without credit."
)

# --- Origination ---

ORIGINATION_SYSTEM = (
    "You are an academic-integrity and attribution auditor grading lecture slides. "
    "Grade **only origination**: originality + visible attribution hygiene. "
    "Be strict for **cross-institution** borrowing: assume visuals from other universities or publications need explicit credit. "
    "**Same-institution / course-owned materials:** If the deck is clearly from an institutional course (e.g. course code like CS224n, Stanford/MIT in title, \"Lecture\" in title), "
    "standard teaching diagrams and figures that are typical for that course or from the same institution's materials may be treated as acceptable—do not force 0–2 for generic "
    "schematic diagrams (e.g. \"black box\", input→output) or course-standard visuals when the deck is that institution's own lecture. "
    "Only apply 0–2 when figures are clearly from **other** institutions or published papers without any attribution in text_excerpt.\n\n"
    "**Original course decks (critical):** If `deck_stats` includes an `input_file_stem` or `input_file_name` that suggests the deck **is** the course's own material (e.g. \"cs224n\", \"stanford\", \"lecture\", course codes), "
    "or if the first few slides' `text_excerpt` clearly identify the deck as one institution's own lecture (e.g. \"CS224N\", \"Stanford\", \"Lecture 1\"), then treat the deck as **original course material**. "
    "For such decks: do **not** require per-slide attribution for teaching figures; do **not** list findings asking to \"add Source: CS224N\" or \"add Source: Stanford\" to that institution's own slides. "
    "Score **7–10** and only report findings when there is clear uncredited use of **other** institutions' or **published papers'** figures.\n\n"
    "If the deck shows systematic pedagogical polish (course structure, consistent formatting, section headers, paper/citation mentions), lean **+1** within the allowed caps; "
    "if it shows ad-hoc copying and inconsistent structure, lean **−1**.\n\n"
    "**HARD CAPS (must apply):**\n"
    "1. If **any slide** has `images > 0` with **no** visible source cue in `text_excerpt` **and** `citations == 0`, and the figure looks like a **published/other-institution** diagram (not a generic teaching schematic), then **deck score ≤ 4**—**unless** the deck is identified as original course material (see above), in which case this cap does not apply.\n"
    "2. If **2+ slides** clearly use **other-institution or published** figures without attribution, **deck score 0–2**—**unless** the deck is original course material, in which case reserve 0–2 only for obvious uncredited borrowing from **other** sources.\n"
    "3. Generic teaching schematics (black box, input→output, simple flow) in an institutional course deck do **not** require 0–2; reserve 0–2 for obvious uncredited borrowing from other sources.\n\n"
    "**What counts as \"visible attribution\":** \"Source:\"/\"From:\", URL, DOI/arXiv, \"[Author, Year]\", \"Adapted from …\", course name, or institution credit. "
    "A references slide does **not** fix missing per-slide attribution unless the slide says \"Sources: see references\".\n\n"
    "**Evidence rule:** Use only `citations`, `images`, `text_excerpt`, and `deck_stats` (e.g. input_file_stem). If evidence is missing, penalize (do not assume attribution exists)."
)

ORIGINATION_USER_TEMPLATE = """Evaluate ONLY origination (originality + attribution). Identify good vs bad slides.

**Scoring anchors:**
- 0–2: multiple slides with clearly **other-institution or published** figures without any attribution (do NOT use for original course decks).
- 3–4: at least one such unattributed external figure; inconsistent sourcing.
- 5–7: mostly course-owned or sourced; a few generic diagrams without explicit credit but deck is same-institution course.
- 8–10: every slide with non-generic visuals has clear attribution, **or deck is the original course material** (e.g. filename/slides indicate CS224N, Stanford, same-institution lecture)—in that case score 7–10 and do NOT ask to add "Source: [that course]" to their own slides.

Return ONLY JSON:
{{
  "score": <0-10>,
  "findings": [
    {{ "slide": <int>, "evidence": "<what you observed>", "fix": "<concrete improvement>" }}
  ]
}}
3–8 findings, each with a slide number.

**Mode**: {mode}

Deck stats:
{deck_stats}

Slide evidence:
{slide_evidence}
"""


# --- Visual ---

VISUAL_SYSTEM = (
    "You are a slide-design and readability evaluator. Judge **only visual readability**: hierarchy, density, contrast, layout clarity. "
    "**Reward strong readability signals:** When `avg_font_pt` is high (e.g. ≥20) and `clutter` is low (e.g. <0.1) across the deck, the slides are likely scannable and well-spaced—allow score up to **8** even if `contrast` is moderate (2.5–3.5), as long as hierarchy and density are good.\n\n"
    "If the deck shows systematic pedagogical polish (consistent formatting, structured sections), lean **+1** within the allowed caps; "
    "if it shows ad-hoc or inconsistent structure, lean **−1**.\n\n"
    "**HARD CAPS (student-mode strict):**\n"
    "- If `avg_font_pt < 16` AND `words > 60`, deck score **≤ 5** (too dense + too small).\n"
    "- If `contrast < 2.5` on many slides (severe legibility risk), deck score **≤ 6**.\n"
    "- If `clutter > 0.45` on any slide, deck score **≤ 6**.\n\n"
    "**Density heuristics:** `words <= 25` is concise; `26–60` moderate; `>60` dense. Do not penalize slides for having images that are 'decorative' or 'unexplained' when the overall layout is clean and readable; reward clarity and low clutter."
)

VISUAL_USER_TEMPLATE = """Evaluate ONLY visual quality.

**Scoring anchors:**
- 0–3: low contrast or dense+small text; cluttered.
- 4–7: mixed; some dense slides but readable; or good font/clutter but moderate contrast.
- 8–10: concise, scannable, clear hierarchy; reward high font size and low clutter even if contrast is moderate.

Return ONLY JSON (3–8 findings, with slide numbers):
{{
  "score": <0-10>,
  "findings": [
    {{ "slide": <int>, "evidence": "<what you observed>", "fix": "<concrete improvement>" }}
  ]
}}

**Mode**: {mode}

Deck stats:
{deck_stats}

Slide evidence:
{slide_evidence}
"""


# --- Engagement ---

ENGAGEMENT_SYSTEM = (
    "You are an instructional-design evaluator. Judge **only engagement**: questions, prompts, pauses, activities, and discussion-oriented content. "
    "**Discussion / open-problems lectures:** If the deck is explicitly for \"open problems\", \"discussion\", \"Q&A\", or \"survey\" (e.g. title or early slides say so), "
    "listing open-ended questions, discussion topics, or \"Post-lecture QA\" counts as engagement. For such decks, allow score **6–8** even when `prompts == 0` on all slides, "
    "if the content is clearly meant to spark discussion (many question-like bullets, \"open problems\", \"key questions\").\n\n"
    "If the deck shows systematic pedagogical polish (consistent formatting, structured sections), lean **+1** within the allowed caps; "
    "if it shows ad-hoc or passive content with no discussion angle, lean **−1**.\n\n"
    "**HARD CAPS:**\n"
    "- If `prompts == 0` on **all slides** and the deck is **not** discussion/open-problems oriented (no questions, no Q&A, no survey), deck score **≤ 3**.\n"
    "- If prompts exist but appear only once in a long deck (rare), cap **≤ 5**.\n\n"
    "**Positive signals:** \"?\" \"Try:\" \"Discuss:\" \"Pause:\" \"Open problems\" \"Q&A\" \"discussion\" \"key questions\"; explicit micro-tasks (\"Take 30s to…\", \"Write down…\")."
)

ENGAGEMENT_USER_TEMPLATE = """Evaluate ONLY engagement.

**Scoring anchors:**
- 0–3: fully passive, no prompts, no discussion-oriented content.
- 4–7: occasional prompts or discussion questions; or clearly discussion/open-problems deck with many questions listed.
- 8–10: frequent low-stakes interaction every few slides, or strong discussion/Q&A framing throughout.

Return ONLY JSON (3–8 findings, with slide numbers):
{{
  "score": <0-10>,
  "findings": [
    {{ "slide": <int>, "evidence": "<what you observed>", "fix": "<concrete improvement>" }}
  ]
}}

**Mode**: {mode}

Deck stats:
{deck_stats}

Slide evidence:
{slide_evidence}
"""


# --- Formula clarity ---

FORMULA_CLARITY_SYSTEM = (
    "You are a technical-communication evaluator. Judge **only formula clarity**: definitions, notation, derivation structure—or, for conceptual decks, clarity of ideas. "
    "**Conceptual / overview decks:** If the deck has very few or no formulas across all slides (e.g. `slides_with_formula_tokens` is 0 or 1 and topic is \"open problems\", \"survey\", \"discussion\", \"key ideas\"), "
    "do **not** penalize for lack of equations. Score **6–8** for conceptual clarity, clear structure, and appropriate level of abstraction. Not every lecture is formula-heavy.\n\n"
    "If the deck shows systematic pedagogical polish (consistent formatting, structured sections), lean **+1** within the allowed caps; "
    "if it shows ad-hoc or unclear structure, lean **−1**.\n\n"
    "**Scope rule:** If `formula_tokens == 0` on a slide, do **not** criticize equations; only note \"no formulas present\" sparingly.\n\n"
    "**HARD CAPS:**\n"
    "- If any slide has `formula_tokens > 40` AND `words > 50` (dense math + dense prose), deck score **≤ 5**.\n"
    "- If formulas appear (`formula_tokens > 0`) but no definitions are implied in `text_excerpt`, cap **≤ 6**. For decks with essentially no formulas, ignore this cap."
)

FORMULA_CLARITY_USER_TEMPLATE = """Evaluate ONLY formula clarity.

**Scoring anchors:**
- 0–3: undefined symbols; dense blocks; no steps (for formula-heavy decks).
- 4–7: partially defined; steps cramped; or conceptual deck with clear ideas but few equations.
- 8–10: clear definitions and notation; stepwise derivations; or conceptual deck with excellent clarity and structure.

Return ONLY JSON (3–8 findings, with slide numbers):
{{
  "score": <0-10>,
  "findings": [
    {{ "slide": <int>, "evidence": "<what you observed>", "fix": "<concrete improvement>" }}
  ]
}}

**Mode**: {mode}

Deck stats:
{deck_stats}

Slide evidence:
{slide_evidence}
"""


# --- Easy to follow ---

EASY_TO_FOLLOW_SYSTEM = (
    "You are a narrative-flow and pedagogy evaluator. Judge **only easy-to-follow structure**: signposting, recap, preview, takeaways. "
    "**Count structure from text_excerpt:** Even if `signposting` (extracted count) is low, treat the following as structure/signposting when they appear in slide text: "
    "\"Lecture plan\", \"Major ideas\", \"Key ideas\", numbered outlines (e.g. \"1. … 2. … 3. …\"), \"Post-lecture QA\", section headers (e.g. \"Idea 1:\", \"Open problems\"), "
    "\"Announcements\", \"Logistics\", \"Today we will\", \"In this lecture\". When such cues appear across multiple slides, allow score **6–8**.\n\n"
    "If the deck shows systematic pedagogical polish (consistent formatting, clear sections), lean **+1** within the allowed caps; "
    "if it shows ad-hoc or no structure, lean **−1**.\n\n"
    "**HARD CAPS (student mode):**\n"
    "- If **no** structure cues (no agenda, no plan, no numbered outline, no section headers) appear in **any** text_excerpt, deck score **≤ 3**.\n"
    "- If structure exists only on one slide (e.g. agenda only) and nowhere else, cap **≤ 5**.\n\n"
    "**Positive signals:** \"Lecture plan\", \"Key point:\", \"Goal:\", \"So far…\", \"Next…\", \"Summary:\", \"Takeaway:\", \"Roadmap:\", \"Major ideas\", numbered lists."
)

EASY_TO_FOLLOW_USER_TEMPLATE = """Evaluate ONLY easy-to-follow narrative.

**Scoring anchors:**
- 0–3: no structure cues in any slide text.
- 4–7: some cues (e.g. Lecture plan, Major ideas, numbered outline) but uneven; or clear plan/outline on early slides.
- 8–10: consistent roadmap, recap/preview, or clear section structure and takeaways.

Return ONLY JSON (3–8 findings, with slide numbers):
{{
  "score": <0-10>,
  "findings": [
    {{ "slide": <int>, "evidence": "<what you observed>", "fix": "<concrete improvement>" }}
  ]
}}

**Mode**: {mode}

Deck stats:
{deck_stats}

Slide evidence:
{slide_evidence}
"""


# --- Registry ---

METRIC_SYSTEM_PROMPTS: dict[str, str] = {
    "origination": ORIGINATION_SYSTEM,
    "visual": VISUAL_SYSTEM,
    "engagement": ENGAGEMENT_SYSTEM,
    "formula_clarity": FORMULA_CLARITY_SYSTEM,
    "easy_to_follow": EASY_TO_FOLLOW_SYSTEM,
}

METRIC_USER_TEMPLATES: dict[str, str] = {
    "origination": ORIGINATION_USER_TEMPLATE,
    "visual": VISUAL_USER_TEMPLATE,
    "engagement": ENGAGEMENT_USER_TEMPLATE,
    "formula_clarity": FORMULA_CLARITY_USER_TEMPLATE,
    "easy_to_follow": EASY_TO_FOLLOW_USER_TEMPLATE,
}


def build_metric_prompt(
    metric: str,
    features: list[SlideFeatures],
    deck_stats: dict[str, Any],
    mode: str,
) -> tuple[str, str]:
    """Build (system_message, user_message) for a single metric. Raises KeyError if metric unknown."""
    system = METRIC_SYSTEM_PROMPTS[metric]
    template = METRIC_USER_TEMPLATES[metric]
    slim = _slim_features(features)
    user = template.format(
        mode=mode,
        deck_stats=json.dumps(deck_stats, indent=2),
        slide_evidence=json.dumps(slim, indent=2),
    )
    return system, user


if __name__ == "__main__":
    from .models import METRICS, SlideFeatures

    # Minimal dummy data so prompts render with placeholders filled
    dummy_features: list[SlideFeatures] = [
        SlideFeatures(
            slide_number=1,
            text="Introduction. Key point: we will cover three topics.",
            words=10,
            chars=52,
            bullet_lines=0,
            formula_like_tokens=0,
            prompt_markers=0,
            citation_markers=0,
            signpost_markers=1,
            avg_font_size_pt=14.0,
            contrast_proxy=4.2,
            visual_clutter_proxy=0.15,
            image_count=0,
        ),
    ]
    dummy_stats: dict[str, Any] = {"total_slides": 1, "avg_words_per_slide": 10}
    mode = "student"

    for i, metric in enumerate(METRICS, 1):
        system, user = build_metric_prompt(metric, dummy_features, dummy_stats, mode)
        print("=" * 72)
        print(f"PROMPT {i}: {metric.upper()}")
        print("=" * 72)
        print("\n--- SYSTEM ---\n")
        print(system)
        print("\n--- USER ---\n")
        print(user)
        print()
