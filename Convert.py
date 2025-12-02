# -*- coding: utf-8 -*-

import os
import torch
from collections import OrderedDict
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from transformers import PreTrainedModel, PretrainedConfig

base_dir = r"checkpoints"
out_base_dir = os.path.join(base_dir, "out_hf_batch")
os.makedirs(out_base_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = TimesFmHparams(
    backend=device,
    per_core_batch_size=2048,
    horizon_len=128,
    num_layers=50,
    # use_positional_embedding=False,
    context_len=512,
)

model_wrapper = TimesFm(
    hparams=hparams,
    checkpoint=TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
)

cfg = model_wrapper._model_config
cfg.num_layers = 9
cfg.num_heads = 6
cfg.num_kv_heads = 6
cfg.head_dim = 72
cfg.hidden_size = 432
cfg.intermediate_size = 1248


# cfg.num_layers        = 7
# cfg.num_heads         = 4
# cfg.num_kv_heads      = 4
# cfg.head_dim          = 66
# cfg.hidden_size       = 264
# cfg.intermediate_size = 1024


cfg.patch_len = 128

class TimesFMConfig(PretrainedConfig):
    model_type = "timesfm_custom"

    def __init__(
        self,
        num_layers=None,
        num_heads=None,
        num_kv_heads=None,
        head_dim=None,
        hidden_size=None,
        intermediate_size=None,
        horizon_len=None,
        context_len=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.horizon_len = horizon_len
        self.context_len = context_len


class TimesFMForHF(PreTrainedModel):
    config_class = TimesFMConfig

    def __init__(self, config: TimesFMConfig, model_module: torch.nn.Module):
        super().__init__(config)
        self.model = model_module

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


for year in range(2000, 2024):
    ckpt_name = f"timesfm_combined_pretrained_{year}.pt"
    checkpoint_path = os.path.join(base_dir, ckpt_name)

    if not os.path.isfile(checkpoint_path):
        print(f"Skipping {ckpt_name} (not found)")
        continue

    print(f"Converting {ckpt_name}...")

    # Load and clean checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = OrderedDict(
        (k.replace("module.", "", 1), v) for k, v in state_dict.items()
    )

    decoder = PatchedTimeSeriesDecoder(cfg).to(device)
    decoder.load_state_dict(clean_state_dict, strict=True)
    decoder.eval()

    hf_config = TimesFMConfig(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        horizon_len=hparams.horizon_len,
        context_len=hparams.context_len,
        patch_len=cfg.patch_len,
        input_patch_len=hparams.input_patch_len,
        output_patch_len=hparams.output_patch_len,
        model_dims=getattr(cfg, 'model_dims', 1280),
    )

    hf_model = TimesFMForHF(hf_config, decoder).to(device)
    hf_model.eval()

    save_dir = os.path.join(out_base_dir, f"{year}")
    os.makedirs(save_dir, exist_ok=True)

    # Save
    hf_model.save_pretrained(save_dir)
    hf_config.save_pretrained(save_dir)

    print("Saved Hugging Face model for {year} to: {save_dir}")

print("All available checkpoints processed successfully!")
