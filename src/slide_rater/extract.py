from __future__ import annotations

import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image
from pypdf import PdfReader

from .models import SlideFeatures

PROMPT_PATTERN = re.compile(r"\b(think|try|question|exercise|pause|discuss|poll|checkpoint)\b", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"(source\s*:|reference\s*:|https?://|doi\s*:|©|copyright)", re.IGNORECASE)
SIGNPOST_PATTERN = re.compile(r"\b(agenda|outline|recap|summary|next|key takeaway|in this section|we will)\b", re.IGNORECASE)
FORMULA_PATTERN = re.compile(r"(=|\b(sum|int|lim|argmin|argmax|loss|gradient|lambda|theta|sigma|mu)\b|\^|_\{|\\frac)", re.IGNORECASE)
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def convert_to_pdf(input_path: Path, tmp_dir: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return input_path
    if suffix != ".pptx":
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF or PPTX.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(tmp_dir),
        str(input_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PPTX conversion requires libreoffice. Install it or provide a PDF export."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to convert PPTX to PDF: {exc.stderr.strip()}") from exc

    out_pdf = tmp_dir / f"{input_path.stem}.pdf"
    if not out_pdf.exists():
        raise RuntimeError("PPTX conversion completed but output PDF was not found.")
    return out_pdf


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prevent stale images from previous runs from polluting current deck features.
    for old_png in out_dir.glob("slide-*.png"):
        old_png.unlink(missing_ok=True)
    prefix = out_dir / "slide"
    cmd = ["pdftoppm", "-png", str(pdf_path), str(prefix)]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # suppress Poppler "Ignoring wrong pointing object" etc.
            text=True,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, None)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftoppm is required for PDF slide rendering. Install Poppler: "
            "macOS: brew install poppler; Ubuntu/Debian: sudo apt install poppler-utils; "
            "Fedora: sudo dnf install poppler-utils."
        ) from exc
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"PDF rendering failed: {err or 'pdftoppm returned non-zero'}") from exc

    pages = sorted(out_dir.glob("slide-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
    if not pages:
        raise RuntimeError("No slide images were generated from PDF.")
    return pages


def _image_proxies(image_path: Path) -> tuple[float, float]:
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0

    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    p5 = float(np.percentile(luminance, 5))
    p95 = float(np.percentile(luminance, 95))
    contrast = (p95 + 0.05) / (p5 + 0.05)

    gx = np.abs(np.diff(luminance, axis=1)).mean()
    gy = np.abs(np.diff(luminance, axis=0)).mean()
    clutter = float(gx + gy)
    return contrast, clutter


def _line_stats(text: str) -> tuple[int, int]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    bullets = sum(1 for ln in lines if BULLET_PATTERN.search(ln))
    return len(lines), bullets


def _avg_font_size(words: list[dict]) -> float | None:
    sizes = [w.get("size") for w in words if isinstance(w.get("size"), (int, float))]
    if not sizes:
        return None
    return float(np.mean(sizes))


@contextmanager
def _suppress_stderr():
    """Redirect process stderr to devnull to suppress PDF library warnings (e.g. Poppler 'wrong pointing object')."""
    stderr_fd = 2
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr = os.dup(stderr_fd)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


def extract_slide_features(pdf_path: Path, slide_images: list[Path]) -> tuple[list[SlideFeatures], dict]:
    features: list[SlideFeatures] = []
    total_chars = 0
    with _suppress_stderr():
        reader = PdfReader(str(pdf_path))
        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                words = re.findall(r"\b\w+\b", text)
                lines_count, bullet_count = _line_stats(text)

                plumber_page = pdf.pages[idx - 1]
                word_objs = plumber_page.extract_words(extra_attrs=["size"])
                avg_font = _avg_font_size(word_objs)

                contrast = None
                clutter = None
                if idx - 1 < len(slide_images):
                    contrast, clutter = _image_proxies(slide_images[idx - 1])

                feat = SlideFeatures(
                    slide_number=idx,
                    text=text.strip(),
                    words=len(words),
                    chars=len(text),
                    bullet_lines=bullet_count,
                    formula_like_tokens=len(FORMULA_PATTERN.findall(text)),
                    prompt_markers=len(PROMPT_PATTERN.findall(text)),
                    citation_markers=len(CITATION_PATTERN.findall(text)),
                    signpost_markers=len(SIGNPOST_PATTERN.findall(text)),
                    avg_font_size_pt=avg_font,
                    contrast_proxy=contrast,
                    visual_clutter_proxy=clutter,
                    image_count=len([x for x in (plumber_page.images or [])]),
                )
                total_chars += feat.chars
                features.append(feat)

    slide_count = len(features)
    avg_words = float(np.mean([f.words for f in features])) if features else 0.0
    avg_bullets = float(np.mean([f.bullet_lines for f in features])) if features else 0.0
    avg_contrast = float(np.nanmean([f.contrast_proxy for f in features if f.contrast_proxy is not None])) if features else 0.0
    avg_clutter = float(np.nanmean([f.visual_clutter_proxy for f in features if f.visual_clutter_proxy is not None])) if features else 0.0

    deck_stats = {
        "slide_count": slide_count,
        "total_chars": total_chars,
        "avg_words_per_slide": round(avg_words, 2),
        "avg_bullets_per_slide": round(avg_bullets, 2),
        "avg_contrast_proxy": round(avg_contrast, 3),
        "avg_clutter_proxy": round(avg_clutter, 4),
        "slides_with_citations": sum(1 for f in features if f.citation_markers > 0),
        "slides_with_prompts": sum(1 for f in features if f.prompt_markers > 0),
        "slides_with_formula_tokens": sum(1 for f in features if f.formula_like_tokens > 0),
        "slides_with_signposting": sum(1 for f in features if f.signpost_markers > 0),
    }
    return features, deck_stats


def adjacent_text_similarity(features: list[SlideFeatures]) -> list[float]:
    def tokenize(s: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", s.lower()))

    sims: list[float] = []
    for i in range(1, len(features)):
        a = tokenize(features[i - 1].text)
        b = tokenize(features[i].text)
        if not a and not b:
            sims.append(1.0)
            continue
        inter = len(a.intersection(b))
        union = len(a.union(b)) or 1
        sims.append(inter / union)
    return sims
