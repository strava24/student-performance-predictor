# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early-stage end-to-end ML project (student performance dataset). Infrastructure (`src/exception.py`, `src/logger.py`) plus data ingestion (`src/components/data_ingestion.py`) and data transformation (`src/components/data_transformation.py`) are implemented, along with `save_object()` in `src/utils.py` (pickles an object to disk via `dill`). `src/components/model_trainer.py` and `src/pipeline/*` are still empty stubs, so there is no test suite, linter or build config yet. Exploratory work lives in `notebook/` (EDA and model training notebooks, data in `notebook/data/stud.csv`), which is untracked.

## Environment and commands

- Conda env is path-based in `./venv`, defined by `environment.yml` (Python 3.14, `llvm-openmp`, `ipykernel`, then pip installs `requirements.txt`): `conda env create -p ./venv -f environment.yml` (or `conda env update -p ./venv -f environment.yml`), then `conda activate ./venv` (activating by name does not work). Non-Python dependencies such as `llvm-openmp` (libomp, required by xgboost on macOS) go in `environment.yml`, not `requirements.txt`.
- Python packages: `pip install -r requirements.txt`. The final `-e .` line is commented out (`# -e .`), so neither this nor `environment.yml` installs the project. Install it separately as `ml-project` in editable mode with `pip install -e .` from the project root (`setup.py` reads `requirements.txt` by relative path), which makes `src` importable. `get_requirements()` only strips an exact `-e .` line, so the commented line reaches `install_requires`, where setuptools ignores it as a comment.
- Notebooks use a registered kernel named `generic-ml` ("Python 3.14 (generic-ml)") backed by `./venv`. `ipykernel` comes from `environment.yml`, not `requirements.txt`. Dependencies are unpinned.
- Run scripts from the project root, either as `python -m src.<module>` or directly (`python src/components/data_ingestion.py`) — both work identically because the project is installed in editable mode, which makes `src` importable regardless of invocation style or `sys.path[0]`. What does depend on the working directory is the relative paths each script uses (`logs/`, `artifacts/`, `notebook/data/stud.csv`), so run from the project root either way.

## Architecture

Planned layout follows the common components/pipeline pattern: `src/components/` (data ingestion, transformation, model trainer) is orchestrated by `src/pipeline/train_pipeline.py`, with `predict_pipeline.py` for inference and `utils.py` for shared helpers.

- `data_ingestion.py`: reads `notebook/data/stud.csv`, saves a raw copy plus an 80/20 train/test split (`random_state=42`) to `artifacts/`, returns the train/test paths. Its `__main__` block chains straight into `DataTransformation`.
- `data_transformation.py`: builds a `ColumnTransformer` (numeric: median-impute → scale; categorical: mode-impute → one-hot → scale) via `get_data_transformer_object()`, then `initiate_data_transformation()` fits it on the training features only, transforms both splits (no leakage into test), reattaches the `math_score` target, and pickles the **fitted** preprocessor to `artifacts/preprocessor.pkl` via `save_object()`.

Conventions that span files:

- Each pipeline stage follows a `<Stage>Config` (a `@dataclass` holding just output paths, e.g. `DataIngestionConfig`, `DataTransformationConfig`) plus a `<Stage>` class holding the logic, which builds its config in `__init__`.
- Generated files go under `artifacts/` (raw/train/test CSVs, `preprocessor.pkl`, later the trained model) — git-ignored, not committed.
- **Logging**: importing `src.logger` configures logging globally (a new timestamped file in `logs/` per run, plus console, INFO level). Always use `from src.logger import logging`. Plain `logging` stays unconfigured and drops INFO messages.
- **Errors**: `CustomException(error, sys)` in `src/exception.py` builds a message with file name and line number from `sys.exc_info()`, so it must be raised inside an `except` block.
