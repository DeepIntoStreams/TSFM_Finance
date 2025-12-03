# Time Series Foundation Models for Finance

Sample workflows to load Amazon Chronos 1.x and Google TimesFM checkpoints that are published on Hugging Face. The repo favors short, explicit instructions so users can copy the environment, download weights, and run the demos immediately.

## Quickstart (uv)
1. `uv venv --python 3.11 .venv`
2. Activate the environment (`.venv\\Scripts\\activate` on PowerShell, `source .venv/bin/activate` on bash).
3. `uv pip sync requirements.txt`
4. Run `python scripts/validate_env.py` to confirm the Chronos and TimesFM checkpoints load correctly, then open `notebooks/model_walkthrough.ipynb` to load data and generate forecasts.

> Need to access private checkpoints? export `HF_TOKEN=<your_hf_token>` before running any script so both loaders can authenticate.

## Environment options
- `pyproject.toml`: source of truth for high-level dependencies. Edit this file when you need to add/remove packages.
- `requirements.txt`: uv-generated lockfile; refresh with `uv pip compile pyproject.toml -o requirements.txt` before committing dependency changes.

## Repository layout (rolling plan)
```
.
├── codex.md                 # working plan / status notes
├── data/
│   └── two_stocks_excess_returns.csv
├── notebooks/
│   └── model_walkthrough.ipynb
├── PROJECT_STYLE.md
├── pyproject.toml
├── requirements.txt
└── scripts/
    └── validate_env.py
```

`two_stocks_excess_returns.csv` is a tiny daily-frequency example that will be documented in the README once the loaders are wired up. All future datasets follow the `<dataset>_<freq>.csv` convention from `PROJECT_STYLE.md`.

## Working surfaces
- `python scripts/validate_env.py`: light-weight smoke test that loads Chronos (`amazon/chronos-t5-mini`) and TimesFM (`google/timesfm-1.0-200m-pytorch`) using built-in settings. Pass `--config path/to/custom.yaml` if you want to override those defaults.
- `notebooks/model_walkthrough.ipynb`: interactive workflow that reads `data/two_stocks_excess_returns.csv`, keeps the final `prediction_length` rows as ground truth, and produces Chronos/TimesFM forecasts for comparison or plotting. Update the notebook cells (for example, the TimesFM loader arguments) to experiment with different checkpoints or datasets, then rerun the cells.

## Hugging Face checkpoints
- Chronos defaults to `amazon/chronos-t5-mini` to keep downloads small. Provide a custom YAML file to `scripts/validate_env.py --config` or edit the notebook to load another Chronos 1.x release (e.g., `amazon/chronos-t5-small`).
- TimesFM defaults to `google/timesfm-1.0-200m-pytorch`. Override it via `--config` for the validation script or tweak the notebook's loader call to try another repo slug.

All tooling respects `HF_HOME` for caching and reads `HF_TOKEN` (if set) for gated models.

## Validation
Run `python scripts/validate_env.py` after syncing dependencies. Pass `HF_TOKEN` if the checkpoints are private. The command prints a JSON summary; exit code 0 means both models loaded successfully. Because it pulls from Hugging Face, ensure outbound network access or cache the weights under `HF_HOME`.
