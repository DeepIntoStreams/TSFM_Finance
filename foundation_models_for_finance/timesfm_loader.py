import os
from collections import OrderedDict
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


def load_timesfm_from_hf(repo_id: str, model_size: str = "20M", prediction_length: int = 1) -> TimesFm:
    """
    Load a TimesFM model from a Hugging Face repo and return a ready-to-use model.

    Args:
        repo_id: Hugging Face repository that hosts the `model.safetensors` file.
        model_size: Either '8M' or '20M'. Determines the architecture that is
            applied before loading the checkpoint.
    """
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

    weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"HF weights downloaded from: {weights_path}")

    state_dict = load_safetensors(weights_path)
    clean_state_dict = OrderedDict(
        (k.replace("module.", "", 1).replace("model.", "", 1), v)
        for k, v in state_dict.items()
    )

    decoder = PatchedTimeSeriesDecoder(cfg).to(device)
    decoder.load_state_dict(clean_state_dict, strict=True)

    model._model = decoder
    model._model.to(model._device)
    model._model.eval()
    return model


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    repo_id = "FinText/TimesFM_20M_2023_Augmented"
    print(f"Loading TimesFM model from Hugging Face repo: {repo_id}")
    load_timesfm_from_hf(repo_id=repo_id, model_size="20M")
    print("Model loaded successfully.")
