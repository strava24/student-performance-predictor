# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early-stage scaffold for an end-to-end ML project (student performance dataset). Only infrastructure is implemented: `src/exception.py` and `src/logger.py`. `src/components/*`, `src/pipeline/*` and `src/utils.py` are empty or stubs, so there is no test suite, linter or build config yet. Exploratory work lives in `notebook/` (EDA and model training notebooks, data in `notebook/data/stud.csv`), which is untracked.

## Environment and commands

- Conda env is path-based in `./venv`, defined by `environment.yml` (Python 3.14, `llvm-openmp`, `ipykernel`, then pip installs `requirements.txt`): `conda env create -p ./venv -f environment.yml` (or `conda env update -p ./venv -f environment.yml`), then `conda activate ./venv` (activating by name does not work). Non-Python dependencies such as `llvm-openmp` (libomp, required by xgboost on macOS) go in `environment.yml`, not `requirements.txt`.
- Python packages: `pip install -r requirements.txt`. The final `-e .` line is commented out (`# -e .`), so neither this nor `environment.yml` installs the project. Install it separately as `ml-project` in editable mode with `pip install -e .` from the project root (`setup.py` reads `requirements.txt` by relative path), which makes `src` importable. `get_requirements()` only strips an exact `-e .` line, so the commented line reaches `install_requires`, where setuptools ignores it as a comment.
- Notebooks use a registered kernel named `generic-ml` ("Python 3.14 (generic-ml)") backed by `./venv`. `ipykernel` comes from `environment.yml`, not `requirements.txt`. Dependencies are unpinned.
- Run modules from the project root as `python -m src.<module>`. `logs/` is created relative to the current working directory.

## Architecture

Planned layout follows the common components/pipeline pattern: `src/components/` (data ingestion, transformation, model trainer) is orchestrated by `src/pipeline/train_pipeline.py`, with `predict_pipeline.py` for inference and `utils.py` for shared helpers.

Conventions that span files:

- **Logging**: importing `src.logger` configures logging globally (a new timestamped file in `logs/` per run, plus console, INFO level). Always use `from src.logger import logging`. Plain `logging` stays unconfigured and drops INFO messages.
- **Errors**: `CustomException(error, sys)` in `src/exception.py` builds a message with file name and line number from `sys.exc_info()`, so it must be raised inside an `except` block.
