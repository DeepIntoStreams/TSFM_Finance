"""Smoke test that simply loads Chronos and TimesFM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml
from chronos import ChronosPipeline
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams

DEFAULT_CHECKPOINTS = {
    "chronos": {
        "checkpoint": "amazon/chronos-t5-mini",
        "prediction_length": 14,
        "num_samples": 20,
    },
    "timesfm": {
        "checkpoint": "google/timesfm-1.0-200m-pytorch",
        "prediction_length": 14,
        "past_length": 128,
        "horizon_len": 128,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Chronos and TimesFM can be imported.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML path to override the built-in checkpoint configuration.",
    )
    return parser.parse_args()


def _load_config(path: Path | None) -> Dict[str, Dict]:
    if path is None:
        return DEFAULT_CHECKPOINTS
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _check_chronos(config: Dict[str, Dict]) -> Dict[str, str]:
    checkpoint = config["chronos"]["checkpoint"]
    ChronosPipeline.from_pretrained(checkpoint, device_map="auto")
    return {"checkpoint": checkpoint, "status": "loaded"}


def _check_timesfm(config: Dict[str, Dict]) -> Dict[str, str]:
    cfg = config["timesfm"]
    checkpoint_id = cfg["checkpoint"]
    context_len = cfg.get("context_len") or cfg.get("past_length", 512)
    horizon_len = cfg.get("horizon_len") or cfg.get("prediction_length", 32)
    input_patch = cfg.get("input_patch_len", 32)
    output_patch = cfg.get("output_patch_len", horizon_len)

    hparams = TimesFmHparams(
        backend="cpu",
        context_len=int(context_len),
        horizon_len=int(horizon_len),
        input_patch_len=int(input_patch),
        output_patch_len=int(output_patch),
    )
    checkpoint = TimesFmCheckpoint(version="torch", huggingface_repo_id=checkpoint_id)
    TimesFm(hparams=hparams, checkpoint=checkpoint)
    return {"checkpoint": checkpoint_id, "status": "loaded"}


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    summaries: Dict[str, Dict] = {}
    failures = []

    try:
        summaries["chronos"] = _check_chronos(config)
    except Exception as exc:
        failures.append(f"Chronos loader failed: {exc}")

    try:
        summaries["timesfm"] = _check_timesfm(config)
    except Exception as exc:
        failures.append(f"TimesFM loader failed: {exc}")

    print(json.dumps(summaries, indent=2))

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
