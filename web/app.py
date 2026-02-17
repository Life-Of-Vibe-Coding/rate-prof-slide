from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, render_template, request
from werkzeug.utils import secure_filename

from slide_rater.pipeline import run_rating_pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "runs" / "uploads"
REPORT_DIR = BASE_DIR / "runs" / "reports"
ALLOWED_EXT = {".pdf", ".pptx"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", result=None, error=None)


@app.post("/rate")
def rate():
    deck = request.files.get("deck")
    if not deck or not deck.filename:
        return render_template("index.html", result=None, error="Please upload a PDF or PPTX file.")

    filename = secure_filename(deck.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        return render_template("index.html", result=None, error="Unsupported file type. Use .pdf or .pptx.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_path = UPLOAD_DIR / f"{stamp}-{filename}"
    report_path = REPORT_DIR / f"{stamp}-{Path(filename).stem}.json"
    deck.save(input_path)

    api_key = request.form.get("api_key", "").strip() or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = request.form.get("model", "qwen-flash").strip() or "qwen-flash"
    base_url = request.form.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").strip()
    mode = request.form.get("mode", "student").strip()
    per_metric = request.form.get("per_metric") == "on"

    weights_raw = request.form.get("weights", "").strip()
    weights = None
    if weights_raw:
        try:
            weights = json.loads(weights_raw)
        except json.JSONDecodeError:
            return render_template("index.html", result=None, error="Weights must be valid JSON.")

    try:
        result = run_rating_pipeline(
            input_file=str(input_path),
            output_file=str(report_path),
            mode=mode,
            model=model,
            api_key=api_key,
            base_url=base_url,
            weights=weights,
            per_metric_prompts=per_metric,
        )
    except Exception as exc:
        return render_template("index.html", result=None, error=f"Rating failed: {type(exc).__name__}: {exc}")

    return render_template(
        "index.html",
        result={
            "overall_score": result["overall_score"],
            "judge_mode": result["judge_mode"],
            "metric_scores": result["metric_scores"],
            "top_5_improvements": result["top_5_improvements"],
            "report_path": str(report_path),
        },
        error=None,
    )


def _stream_rating_events(
    input_path: Path,
    report_path: Path,
    api_key: str,
    model: str,
    base_url: str,
    mode: str,
    weights: dict | None,
    per_metric: bool,
):
    """Generator that yields SSE events: steps and metrics, then a final 'done' event."""
    event_queue = queue.Queue()

    def on_metric(name: str, data: dict):
        event_queue.put(("metric", name, data))

    def on_step(step_id: str, status: str, detail: str | None):
        event_queue.put(("step", step_id, status, detail or ""))

    def run():
        try:
            result = run_rating_pipeline(
                input_file=str(input_path),
                output_file=str(report_path),
                mode=mode,
                model=model,
                api_key=api_key or None,
                base_url=base_url,
                weights=weights,
                per_metric_prompts=per_metric,
                progress_callback=on_metric,
                step_callback=on_step,
            )
            event_queue.put(("done", result))
        except Exception as exc:
            event_queue.put(("error", str(exc)))

    thread = threading.Thread(target=run)
    thread.start()

    while True:
        try:
            msg = event_queue.get(timeout=300)
        except queue.Empty:
            yield "data: " + json.dumps({"error": "timeout"}) + "\n\n"
            break
        if msg[0] == "step":
            _, step_id, status, detail = msg
            yield "data: " + json.dumps({"type": "step", "step_id": step_id, "status": status, "detail": detail}) + "\n\n"
        elif msg[0] == "metric":
            _, name, data = msg
            yield "data: " + json.dumps({"type": "metric", "name": name, "data": data}) + "\n\n"
        elif msg[0] == "done":
            result = msg[1]
            payload = {
                "type": "done",
                "overall_score": result["overall_score"],
                "judge_mode": result["judge_mode"],
                "metric_scores": result["metric_scores"],
                "top_5_improvements": result["top_5_improvements"],
                "report_path": str(report_path),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            break
        else:
            yield "data: " + json.dumps({"type": "error", "message": msg[1]}) + "\n\n"
            break
    thread.join(timeout=1)


@app.post("/rate_stream")
def rate_stream():
    """Stream metric evaluation results as Server-Sent Events (one event per metric)."""
    deck = request.files.get("deck")
    if not deck or not deck.filename:
        return Response(
            json.dumps({"error": "Please upload a PDF or PPTX file."}),
            status=400,
            content_type="application/json",
        )

    filename = secure_filename(deck.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        return Response(
            json.dumps({"error": "Unsupported file type. Use .pdf or .pptx."}),
            status=400,
            content_type="application/json",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    input_path = UPLOAD_DIR / f"{stamp}-{filename}"
    report_path = REPORT_DIR / f"{stamp}-{Path(filename).stem}.json"
    deck.save(input_path)

    api_key = request.form.get("api_key", "").strip() or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = request.form.get("model", "qwen-flash").strip() or "qwen-flash"
    base_url = request.form.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").strip()
    mode = request.form.get("mode", "student").strip()
    per_metric = request.form.get("per_metric") == "on"

    weights = None
    weights_raw = request.form.get("weights", "").strip()
    if weights_raw:
        try:
            weights = json.loads(weights_raw)
        except json.JSONDecodeError:
            return Response(
                json.dumps({"error": "Weights must be valid JSON."}),
                status=400,
                content_type="application/json",
            )

    return Response(
        _stream_rating_events(
            input_path=input_path,
            report_path=report_path,
            api_key=api_key,
            model=model,
            base_url=base_url,
            mode=mode,
            weights=weights,
            per_metric=per_metric,
        ),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
