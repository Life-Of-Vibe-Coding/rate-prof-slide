from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


METRICS = [
    "origination",
    "visual",
    "engagement",
    "formula_clarity",
    "easy_to_follow",
]


DEFAULT_WEIGHTS = {
    "origination": 0.15,
    "visual": 0.20,
    "engagement": 0.20,
    "formula_clarity": 0.20,
    "easy_to_follow": 0.25,
}


@dataclass
class SlideFeatures:
    slide_number: int
    text: str
    words: int
    chars: int
    bullet_lines: int
    formula_like_tokens: int
    prompt_markers: int
    citation_markers: int
    signpost_markers: int
    avg_font_size_pt: float | None
    contrast_proxy: float | None
    visual_clutter_proxy: float | None
    image_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricFinding:
    slide: int
    evidence: str
    fix: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricScore:
    score: float
    findings: list[MetricFinding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 2),
            "findings": [f.to_dict() for f in self.findings],
        }
