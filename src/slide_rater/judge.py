from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from .metric_prompts import build_metric_prompt
from .models import DEFAULT_WEIGHTS, METRICS, MetricFinding, MetricScore, SlideFeatures


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rate(features: list[SlideFeatures], predicate: Callable[[SlideFeatures], bool]) -> float:
    n = max(1, len(features))
    return sum(1 for f in features if predicate(f)) / n


def _window_score(value: float, center: float, half_span: float) -> float:
    return _clamp(1.0 - abs(value - center) / max(half_span, 1e-6), 0.0, 1.0)


def _collect_deck_profile(features: list[SlideFeatures]) -> dict[str, float]:
    n = max(1, len(features))
    avg_contrast = sum((f.contrast_proxy or 3.0) for f in features) / n
    avg_clutter = sum((f.visual_clutter_proxy or 0.02) for f in features) / n

    contrast_low_rate = _rate(features, lambda f: (f.contrast_proxy or 3.0) < 1.5)
    contrast_high_rate = _rate(features, lambda f: (f.contrast_proxy or 3.0) > 6.0)
    contrast_penalty = 1.3 * contrast_low_rate + 1.0 * contrast_high_rate
    if avg_contrast < 1.6:
        contrast_penalty += min(0.6, (1.6 - avg_contrast) / 0.5)
    if avg_contrast > 6.0:
        contrast_penalty += min(0.6, (avg_contrast - 6.0) / 6.0)

    return {
        "slide_count": float(len(features)),
        "avg_words": sum(f.words for f in features) / n,
        "words20_rate": _rate(features, lambda f: f.words >= 20),
        "words5_rate": _rate(features, lambda f: f.words >= 5),
        "dense_rate": _rate(features, lambda f: f.words > 80),
        "formula_rate": _rate(features, lambda f: f.formula_like_tokens > 0),
        "prompt_rate": _rate(features, lambda f: f.prompt_markers > 0),
        "signpost_rate": _rate(features, lambda f: f.signpost_markers > 0),
        "citation_rate": _rate(features, lambda f: f.citation_markers > 0),
        "image_rate": _rate(features, lambda f: f.image_count > 0),
        "cited_image_rate": (
            _rate(features, lambda f: f.image_count > 0 and f.citation_markers > 0)
            / max(1e-6, _rate(features, lambda f: f.image_count > 0))
        ),
        "tiny_font_rate": _rate(features, lambda f: (f.avg_font_size_pt or 0.0) > 0 and (f.avg_font_size_pt or 0.0) < 16.0),
        "sparse_image_rate": _rate(features, lambda f: f.image_count > 0 and f.words < 10),
        "avg_contrast": avg_contrast,
        "avg_clutter": avg_clutter,
        "contrast_penalty": _clamp(contrast_penalty, 0.0, 1.5),
    }


def _build_fallback_findings(
    metric: str,
    features: list[SlideFeatures],
) -> list[MetricFinding]:
    candidates: list[tuple[float, int, str, str]] = []

    def add(severity: float, slide: int, evidence: str, fix: str) -> None:
        candidates.append((severity, slide, evidence, fix))

    for f in features:
        if metric == "visual":
            if f.words > 85:
                add(3.0 + (f.words - 85) / 30.0, f.slide_number, "Slide has very high text density.", "Split content into multiple slides and keep one key message per slide.")
            if (f.avg_font_size_pt or 99.0) < 16.0:
                add(3.0, f.slide_number, "Average font size appears too small for projection.", "Increase body font size and reduce line count.")
            if (f.contrast_proxy or 3.0) < 1.5 or (f.contrast_proxy or 3.0) > 6.0:
                add(2.6, f.slide_number, "Contrast level is outside the stable readability range.", "Use darker text on light background and avoid extreme contrast artifacts.")
            if f.image_count > 0 and f.words < 8:
                add(2.0, f.slide_number, "Image-heavy slide has little explanatory text.", "Add a caption or one-sentence takeaway to anchor the visual.")
        elif metric == "engagement":
            if f.words >= 20 and f.prompt_markers == 0:
                add(2.8, f.slide_number, "No explicit audience interaction cue on a content slide.", "Add a quick question, pause point, or 30-second mini task.")
            if f.signpost_markers == 0 and f.words >= 30:
                add(2.0, f.slide_number, "Dense content appears without guiding transition language.", "Add 'why this matters' and 'what comes next' prompts.")
        elif metric == "formula_clarity":
            if f.formula_like_tokens > 25 and f.words > 45:
                add(3.3, f.slide_number, "Formula-heavy slide is packed with text.", "Break derivation into numbered steps and define symbols near each equation.")
            if f.formula_like_tokens > 0 and f.words < 10:
                add(2.7, f.slide_number, "Formula appears with minimal surrounding explanation.", "Add one-line intuition and variable definitions.")
            if f.formula_like_tokens > 0 and f.signpost_markers == 0:
                add(2.0, f.slide_number, "Technical step lacks explicit transition cue.", "Add a short signpost such as 'Step 1', 'Assumption', or 'Result'.")
        elif metric == "origination":
            if f.image_count > 0 and f.citation_markers == 0:
                add(3.2 + min(2.0, f.image_count * 0.2), f.slide_number, "Slide includes visual content without visible attribution marker.", "Add Source/URL/DOI or 'adapted from' credit on the same slide.")
            if f.image_count >= 3 and f.citation_markers > 0:
                add(1.5, f.slide_number, "Multiple visuals are used; attribution format may be inconsistent.", "Standardize attribution style across all visual slides.")
        elif metric == "easy_to_follow":
            if f.signpost_markers == 0 and f.words >= 25:
                add(2.9, f.slide_number, "Slide lacks explicit structure cue (agenda/recap/next).", "Add a short transition or section marker.")
            if f.words < 5 and f.image_count > 0:
                add(2.4, f.slide_number, "Very sparse slide may weaken narrative continuity.", "Add one sentence that links this slide to the previous and next ideas.")
            if f.words > 90:
                add(2.0, f.slide_number, "Long slide text can obscure key storyline.", "Promote key takeaway to title and move details to speaker notes.")

    defaults = {
        "origination": "Apply one consistent attribution template to all non-original visuals.",
        "visual": "Keep each slide scannable: one key message, fewer lines, larger text.",
        "engagement": "Insert interaction checkpoints every 4-6 slides.",
        "formula_clarity": "Pair each formula with symbol definitions and one-sentence intuition.",
        "easy_to_follow": "Use roadmap and recap language at section boundaries.",
    }
    default_fix = defaults.get(metric, "Improve slide clarity and teaching flow.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    findings: list[MetricFinding] = []
    seen: set[tuple[int, str]] = set()
    for _, slide, evidence, fix in candidates:
        key = (slide, evidence)
        if key in seen:
            continue
        seen.add(key)
        findings.append(MetricFinding(slide=slide, evidence=evidence, fix=fix))
        if len(findings) >= 8:
            break

    if not findings:
        findings.append(
            MetricFinding(
                slide=1,
                evidence="Limited direct evidence in deterministic fallback mode.",
                fix=default_fix,
            )
        )

    while len(findings) < 3:
        findings.append(
            MetricFinding(
                slide=1,
                evidence="Deck-level pattern suggests room for improvement.",
                fix=default_fix,
            )
        )
    return findings[:8]


def fallback_scoring(features: list[SlideFeatures], mode: str = "student") -> dict[str, MetricScore]:
    profile = _collect_deck_profile(features)

    readability = (
        0.40 * profile["words20_rate"]
        + 0.30 * profile["words5_rate"]
        + 0.15 * _window_score(profile["avg_clutter"], center=0.024, half_span=0.014)
        + 0.10 * _window_score(profile["avg_contrast"], center=2.6, half_span=2.0)
        + 0.05 * (1.0 - _clamp(profile["dense_rate"] / 0.35, 0.0, 1.0))
    )

    visual = _clamp(
        2.0
        + 7.0 * readability
        - 1.2 * profile["tiny_font_rate"]
        - 1.3 * profile["contrast_penalty"]
        - 1.0 * profile["sparse_image_rate"],
        0,
        10,
    )

    engagement = _clamp(
        1.0
        + 4.5 * _clamp(profile["prompt_rate"] / 0.06, 0.0, 1.0)
        + 2.0 * profile["signpost_rate"]
        + 2.2 * profile["words20_rate"]
        - 1.2 * (1.0 - profile["words5_rate"])
        - 1.0 * profile["sparse_image_rate"],
        0,
        10,
    )

    conceptual_bonus = 1.0 if profile["formula_rate"] < 0.08 and profile["words20_rate"] > 0.6 else 0.0
    formula = _clamp(
        2.0
        + 4.0 * _clamp(profile["formula_rate"] / 0.35, 0.0, 1.0)
        + 1.8 * profile["words20_rate"]
        + 1.2 * profile["words5_rate"]
        - 1.0 * profile["dense_rate"]
        + 0.8 * conceptual_bonus,
        0,
        10,
    )

    easy = _clamp(
        1.5
        + 2.2 * profile["words5_rate"]
        + 2.6 * profile["words20_rate"]
        + 2.2 * _clamp(profile["signpost_rate"] / 0.05, 0.0, 1.0)
        - 1.1 * profile["contrast_penalty"]
        - 0.8 * profile["sparse_image_rate"],
        0,
        10,
    )

    uncited_image_penalty = profile["image_rate"] * (1.0 - profile["cited_image_rate"])
    citation_balance_bonus = _window_score(profile["citation_rate"], center=0.06, half_span=0.06)
    origination = _clamp(
        2.0
        + 2.4 * citation_balance_bonus
        + 1.8 * profile["words5_rate"]
        + 1.2 * _clamp(profile["formula_rate"] / 0.35, 0.0, 1.0)
        - 2.2 * uncited_image_penalty
        - 1.0 * profile["sparse_image_rate"],
        0,
        10,
    )

    if mode == "expert":
        formula = _clamp(formula + 0.8, 0, 10)
        engagement = _clamp(engagement - 0.3, 0, 10)
        easy = _clamp(easy - 0.2, 0, 10)

    return {
        "origination": MetricScore(origination, _build_fallback_findings("origination", features)),
        "visual": MetricScore(visual, _build_fallback_findings("visual", features)),
        "engagement": MetricScore(engagement, _build_fallback_findings("engagement", features)),
        "formula_clarity": MetricScore(formula, _build_fallback_findings("formula_clarity", features)),
        "easy_to_follow": MetricScore(easy, _build_fallback_findings("easy_to_follow", features)),
    }


def _llm_prompt(features: list[SlideFeatures], deck_stats: dict[str, Any], mode: str) -> str:
    slim = []
    for f in features:
        text = f.text.replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:400] + " ..."
        slim.append(
            {
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
            }
        )

    rubric = {
        "score_bands": {"poor": "0-3", "ok": "4-7", "excellent": "8-10"},
        "metrics": {
            "origination": "originality + attribution. Same-institution course decks may use standard teaching figures; reserve 0-2 for clear uncredited borrowing from other institutions or publications.",
            "visual": "design readability, hierarchy; reward high font and low clutter even if contrast is moderate.",
            "engagement": "interaction and discussion; open-problems/discussion decks get credit for listed questions and Q&A even without explicit prompt markers.",
            "formula_clarity": "notation and definitions, or conceptual clarity for decks with few/no formulas (e.g. open problems, survey).",
            "easy_to_follow": "narrative flow; count Lecture plan, Major ideas, numbered outline, section headers as structure even if signpost count is low.",
        },
    }

    return (
        "You are judging lecture slides produced by a lecturer. Your aim is to identify good slides and bad slides: "
        "reward clear, original, well-attributed teaching; penalize poor design and unattributed borrowing from other sources. "
        "For origination: same-institution course materials and generic teaching diagrams are acceptable; only score 0-2 when figures are clearly from other institutions or publications without attribution. "
        "Evaluate using the rubric below. Use only provided evidence. If evidence is insufficient for a metric, cap score at 7. "
        "Return ONLY valid JSON with schema:\n"
        "{\n"
        '  "metric_scores": {\n'
        '    "origination": {"score": number, "findings": [{"slide": int, "evidence": str, "fix": str}]},\n'
        '    "visual": {"score": number, "findings": [...]},\n'
        '    "engagement": {"score": number, "findings": [...]},\n'
        '    "formula_clarity": {"score": number, "findings": [...]},\n'
        '    "easy_to_follow": {"score": number, "findings": [...]}\n'
        "  },\n"
        '  "top_improvements": [{"priority": 1-5 int, "action": str, "impact": "high|medium|low", "reason": str, "slides": [int]}]\n'
        "}\n"
        "Each metric must contain 3-8 findings with slide references. "
        f"Mode: {mode}. In student mode, prioritize novice comprehension and scaffolding. "
        "In expert mode, allow denser content if technically coherent.\n\n"
        f"Deck stats:\n{json.dumps(deck_stats, indent=2)}\n\n"
        f"Rubric:\n{json.dumps(rubric, indent=2)}\n\n"
        f"Slide evidence:\n{json.dumps(slim, indent=2)}"
    )


def llm_scoring(
    *,
    api_key: str,
    base_url: str,
    model: str,
    features: list[SlideFeatures],
    deck_stats: dict[str, Any],
    mode: str,
) -> tuple[dict[str, MetricScore], list[dict[str, Any]]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = _llm_prompt(features, deck_stats, mode)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator of lecture slides produced by a lecturer. Your aim is to identify good and bad slides. Score consistently using the rubric; cite evidence by slide number. Penalize heavily any uncredited use of diagrams or figures from other top universities or published materials.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    metric_data = data.get("metric_scores", {})

    scores: dict[str, MetricScore] = {}
    for name in METRICS:
        entry = metric_data.get(name, {})
        score = float(entry.get("score", 0.0))
        findings_json = entry.get("findings", [])
        findings: list[MetricFinding] = []
        for item in findings_json:
            try:
                findings.append(
                    MetricFinding(
                        slide=int(item.get("slide", 1)),
                        evidence=str(item.get("evidence", ""))[:240],
                        fix=str(item.get("fix", ""))[:240],
                    )
                )
            except Exception:
                continue

        if len(findings) < 3:
            fallback = fallback_scoring(features, mode)[name].findings
            findings.extend(fallback[: 3 - len(findings)])
        findings = findings[:8]

        scores[name] = MetricScore(score=_clamp(score, 0, 10), findings=findings)

    top_improvements = data.get("top_improvements", [])
    if not isinstance(top_improvements, list):
        top_improvements = []
    return scores, top_improvements[:5]


def _parse_single_metric_response(raw: str, metric: str, features: list[SlideFeatures], mode: str) -> MetricScore:
    """Parse JSON from a single-metric prompt response into MetricScore."""
    data = json.loads(raw or "{}")
    score = float(data.get("score", 0.0))
    findings_json = data.get("findings", [])
    findings: list[MetricFinding] = []
    for item in findings_json:
        try:
            findings.append(
                MetricFinding(
                    slide=int(item.get("slide", 1)),
                    evidence=str(item.get("evidence", ""))[:240],
                    fix=str(item.get("fix", ""))[:240],
                )
            )
        except Exception:
            continue
    if len(findings) < 3:
        fallback = fallback_scoring(features, mode)[metric].findings
        findings.extend(fallback[: 3 - len(findings)])
    return MetricScore(score=_clamp(score, 0, 10), findings=findings[:8])


def _score_one_metric(
    *,
    metric: str,
    api_key: str,
    base_url: str,
    model: str,
    features: list[SlideFeatures],
    deck_stats: dict[str, Any],
    mode: str,
    step_start_callback: Callable[[str], None] | None,
) -> tuple[str, MetricScore]:
    """Score a single metric (used by parallel workers). Returns (metric_name, MetricScore)."""
    if step_start_callback:
        step_start_callback(metric)
    client = OpenAI(api_key=api_key, base_url=base_url)
    system_msg, user_msg = build_metric_prompt(metric, features, deck_stats, mode)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    score = _parse_single_metric_response(raw, metric, features, mode)
    return (metric, score)


def llm_scoring_per_metric(
    *,
    api_key: str,
    base_url: str,
    model: str,
    features: list[SlideFeatures],
    deck_stats: dict[str, Any],
    mode: str,
    progress_callback: Callable[[str, MetricScore], None] | None = None,
    step_start_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, MetricScore], list[dict[str, Any]]]:
    """
    Score each metric with a dedicated prompt in parallel and merge results.
    Returns (metric_scores, top_improvements). top_improvements is empty; caller may use defaults.
    If progress_callback is set, it is called after each metric completes with (metric_name, MetricScore).
    If step_start_callback is set, it is called at the start of each metric with (metric_name).
    """
    scores: dict[str, MetricScore] = {}
    with ThreadPoolExecutor(max_workers=len(METRICS)) as executor:
        futures = {
            executor.submit(
                _score_one_metric,
                metric=metric,
                api_key=api_key,
                base_url=base_url,
                model=model,
                features=features,
                deck_stats=deck_stats,
                mode=mode,
                step_start_callback=step_start_callback,
            ): metric
            for metric in METRICS
        }
        for future in as_completed(futures):
            metric_name, score = future.result()
            scores[metric_name] = score
            if progress_callback:
                progress_callback(metric_name, score)
    # Return in METRICS order for deterministic output
    ordered_scores = {m: scores[m] for m in METRICS}
    return ordered_scores, []


def weighted_overall(scores: dict[str, MetricScore], weights: dict[str, float] | None = None) -> float:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total = sum(w.values()) or 1.0
    normalized = {k: v / total for k, v in w.items()}
    value = 0.0
    for metric, weight in normalized.items():
        value += scores[metric].score * weight
    return round(value * 10.0, 2)
