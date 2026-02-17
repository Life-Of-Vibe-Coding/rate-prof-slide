from __future__ import annotations

import argparse
import json

from .pipeline import run_rating_pipeline


def _weights_arg(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("weights must be a JSON object")
    return {str(k): float(v) for k, v in data.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rate lecture slide quality with deterministic + LLM rubric scoring.")
    p.add_argument("input_file", help="Path to input PDF or PPTX")
    p.add_argument("-o", "--output", default="report.json", help="Output JSON report path")
    p.add_argument("--mode", choices=["student", "expert"], default="student", help="Scoring viewpoint")
    p.add_argument("--model", default="qwen-flash", help="Model name")
    p.add_argument("--api-key", default=None, help="API key (or use QWEN_API_KEY env var)")
    p.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base URL",
    )
    p.add_argument(
        "--weights",
        default=None,
        help='Optional JSON object, e.g. "{\"visual\":0.2,\"easy_to_follow\":0.3}"',
    )
    p.add_argument(
        "--per-metric",
        action="store_true",
        help="Use a dedicated prompt per metric (5 API calls instead of 1).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    weights = _weights_arg(args.weights)

    result = run_rating_pipeline(
        input_file=args.input_file,
        output_file=args.output,
        mode=args.mode,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        weights=weights,
        per_metric_prompts=args.per_metric,
    )

    print(f"Overall score: {result['overall_score']}")
    print(f"Report written to: {args.output}")
    print(f"Judge mode: {result['judge_mode']}")


if __name__ == "__main__":
    main()
