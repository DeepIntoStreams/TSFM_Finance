import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder


MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "8M": {
        "num_layers": 7,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 66,
        "hidden_size": 264,
        "intermediate_size": 1024,
    },
    "20M": {
        "num_layers": 9,
        "num_heads": 6,
        "num_kv_heads": 6,
        "head_dim": 72,
        "hidden_size": 432,
        "intermediate_size": 1248,
    },
}


def _init_timesfm(model_size: str, prediction_length: int) -> TimesFm:
    """Instantiate the base TimesFM class with large defaults before patching."""
    size_key = model_size.upper()
    if size_key not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_size '{model_size}'. Choose from {list(MODEL_CONFIGS)}.")

    cfg_overrides = MODEL_CONFIGS[size_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "gpu" if device.type == "cuda" else "cpu"

    # Use the large TimesFM defaults (50-layer) so the internal loader can
    # initialize without state-dict mismatches. We'll override the decoder
    # configuration immediately after instantiation.
    hparams = TimesFmHparams(
        backend=backend,
        per_core_batch_size=2048,
        horizon_len=prediction_length,
        context_len=512,
        num_layers=50,
        num_heads=16,
    )

    model = TimesFm(
        hparams=hparams,
        checkpoint=TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )

    cfg = model._model_config
    decoder_keys = ("num_layers", "num_heads", "num_kv_heads", "head_dim", "hidden_size", "intermediate_size")
    for key in decoder_keys:
        setattr(cfg, key, cfg_overrides[key])

    return model, cfg, device


def _load_state_dict(weights_path: Path) -> OrderedDict:
    state_dict = load_safetensors(weights_path)
    clean_state_dict = OrderedDict(
        (k.replace("module.", "", 1).replace("model.", "", 1), v)
        for k, v in state_dict.items()
    )
    return clean_state_dict


def _finalize_model(model: TimesFm, cfg, state_dict: OrderedDict, device: torch.device) -> TimesFm:
    decoder = PatchedTimeSeriesDecoder(cfg).to(device)
    decoder.load_state_dict(state_dict, strict=True)

    model._model = decoder
    model._model.to(model._device)
    model._model.eval()
    return model


def _resolve_weights_path(source: str) -> Path:
    """
    Return a path to the `model.safetensors` file. If `source` exists locally,
    use it directly (folder or file). Otherwise assume it is a repo_id.
    """
    local_path = Path(source)
    if local_path.exists():
        if local_path.is_dir():
            candidate = local_path / "model.safetensors"
        else:
            candidate = local_path

        if not candidate.is_file():
            raise FileNotFoundError(f"Could not find 'model.safetensors' under {local_path}.")

        print(f"TimesFM weights loaded from local path: {candidate}")
        return candidate

    weights_path = Path(hf_hub_download(repo_id=source, filename="model.safetensors"))
    print(f"HF weights downloaded from: {weights_path}")
    return weights_path


def load_timesfm(source: str, model_size: str = "20M", prediction_length: int = 1) -> TimesFm:
    """
    Load a TimesFM model either from a Hugging Face repo or a local path.

    Args:
        source: Hugging Face repo_id or a local directory/file that contains `model.safetensors`.
        model_size: Determines which decoder config overrides to apply.
    """
    model, cfg, device = _init_timesfm(model_size, prediction_length)
    weights_path = _resolve_weights_path(source)
    state_dict = _load_state_dict(weights_path)
    return _finalize_model(model, cfg, state_dict, device)


def load_timesfm_from_hf(repo_id: str, model_size: str = "20M", prediction_length: int = 1) -> TimesFm:
    """Backwards-compatible helper that forces the Hugging Face code path."""
    return load_timesfm(source=repo_id, model_size=model_size, prediction_length=prediction_length)


def load_timesfm_from_local(path: str, model_size: str = "20M", prediction_length: int = 1) -> TimesFm:
    """Load a TimesFM model from a local directory or `model.safetensors` file."""
    return load_timesfm(source=path, model_size=model_size, prediction_length=prediction_length)


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    repo_id = "FinText/TimesFM_20M_2023_Augmented"
    print(f"Loading TimesFM model from Hugging Face repo: {repo_id}")
    load_timesfm_from_hf(repo_id=repo_id, model_size="20M")
    print("Model loaded successfully.")
