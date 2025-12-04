# Time Series Foundation Models (TSFMs) for Finance
[![SSRN](https://img.shields.io/badge/SSRN-5770562-1a5dab?logo=ssrn&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5770562)
[![arXiv](https://img.shields.io/badge/arXiv-2511.18578-b31b1b?logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2511.18578)
[![Website - FinText.ai](https://img.shields.io/badge/Website-FinText.ai-0A66C2?logo=google-chrome&logoColor=white)](https://fintext.ai)
[![Hugging Face - FinText](https://img.shields.io/badge/Hugging%20Face-FinText-f77f00?logo=huggingface&logoColor=white)](https://huggingface.co/FinText)

<p align="center">
  <img src="https://fintext.ai/FinText_Github.png" alt="Logo" width="200"/>
</p>

Sample workflows to load Amazon Chronos 1.x and Google TimesFM checkpoints that are published on Hugging Face. The repo favors short, explicit instructions so users can copy the environment, download weights, and run the demos immediately.

## Quickstart (uv)
With the following steps, user shall be able to build an envrionment that allows to load both Chronos 1.x and TimesFM 1.x models.
1. Create virtual environment with `uv venv --python 3.11 .venv`
2. Activate the environment (`.venv\\Scripts\\activate` on PowerShell, `source .venv/bin/activate` on bash).
3. `uv pip sync requirements.txt`
4. Run `python scripts/validate_env.py` to confirm the Chronos and TimesFM checkpoints load correctly, then open `notebooks/predict-with-chronos1.ipynb` or `predict-with-timesfm1` to load data and generate forecasts.


## Repository layout
```
├── data/
│   └── two_stocks_excess_returns.csv
├── notebooks/
│   └── predict-with-x.ipynb
├── pyproject.toml
├── requirements.txt
└── scripts/
    └── validate_env.py
├── tsfm_finance    # utility module to load TSFM
```

`two_stocks_excess_returns.csv` is a tiny daily-frequency example that contains daily excess returns for Microsoft and Apple stocks.

## Working surfaces
- `python scripts/validate_env.py`: light-weight smoke test that loads Chronos (`amazon/chronos-t5-mini`) and TimesFM (`google/timesfm-1.0-200m-pytorch`) using the baked-in defaults. Edit the script if you want to change checkpoints.
- `notebooks/predict-with-x.ipynb`: interactive workflow that reads `data/two_stocks_excess_returns.csv`, keeps the final `prediction_length` rows as ground truth, and produces Chronos/TimesFM forecasts for comparison or plotting. Update the notebook cells (for example, the TimesFM loader arguments) to experiment with different checkpoints or datasets, then rerun the cells.

## Hugging Face checkpoints
- Chronos defaults to `amazon/chronos-t5-mini` to keep downloads small. Edit the validation script or notebook to load another Chronos 1.x release (e.g., `amazon/chronos-t5-small`).
- TimesFM defaults to `google/timesfm-1.0-200m-pytorch`. Modify the validation script or tweak the notebook's loader call to try another repo slug.

All tooling respects `HF_HOME` for caching and reads `HF_TOKEN` (if set) for gated models.

## Import the models
`notebooks/predict-with-chronos1.ipynb` and `notebooks/predict-with-timesfm1.ipynb` contain end-to-end walkthroughs. The snippet below shows the core imports so you can script Chronos/TimesFM forecasts without opening a notebook.

```python
import torch
import pandas as pd
from chronos import ChronosPipeline
from tsfm_finance.timesfm_loader import load_timesfm_from_hf

df_excess_ret = pd.read_csv("data/two_stocks_excess_returns.csv", index_col=0, parse_dates=True)
window_size = 21
input_tensor = torch.tensor(df_excess_ret.iloc[:window_size].values.T)

chronos = ChronosPipeline.from_pretrained(
    "FinText/Chronos_Small_2022_Global",
    device_map="auto",
    dtype=torch.bfloat16,
)
chronos_forecast = chronos.predict(input_tensor, prediction_length=1, num_samples=20)

timesfm = load_timesfm_from_hf(
    repo_id="FinText/TimesFM_8M_2023_Augmented",
    model_size="8M",
    prediction_length=1,
)
point_forecast, quantile_forecast = timesfm.forecast(input_tensor)

# Want the “vanilla” checkpoints from Hugging Face instead of the FinText ones?
# Reuse the exact defaults from scripts/validate_env.py:
chronos_default = ChronosPipeline.from_pretrained("amazon/chronos-t5-mini", device_map="auto")

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams

timesfm_hparams = TimesFmHparams(
    backend="cpu",
    context_len=128,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
)
timesfm_default = TimesFm(
    hparams=timesfm_hparams,
    checkpoint=TimesFmCheckpoint(version="torch", huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
)
```

## Citation
If this repo or the accompanying study helps your research, please cite the SSRN preprint:

```bibtex
@article{rahimikia2025revisiting,
  title        = {Re(Visiting) Time Series Foundation Models in Finance},
  author       = {Rahimikia, Eghbal and Ni, Hao and Wang, Weiguan},
  journal      = {SSRN Electronic Journal},
  year         = {2025},
  month        = nov,
  doi          = {10.2139/ssrn.5770562},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5770562},
  note         = {SSRN working paper 5770562}
}
```

## Environment options
- `pyproject.toml`: source of truth for high-level dependencies. Edit this file when you need to add/remove packages.
- `requirements.txt`: uv-generated lockfile; refresh with `uv pip compile pyproject.toml -o requirements.txt` before committing dependency changes.
