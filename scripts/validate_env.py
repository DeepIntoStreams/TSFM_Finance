"""Smoke test that simply loads Chronos and TimesFM checkpoints."""

from __future__ import annotations

import json
import sys
from typing import Dict

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
    config = DEFAULT_CHECKPOINTS
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
