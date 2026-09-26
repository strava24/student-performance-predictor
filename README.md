## End to End Machine Learning Project

A reusable, end-to-end machine learning project in Python. This README tracks what has been set up so far.

## Project Structure

```
generic-ml/
├── .gitignore
├── README.md
├── requirements.txt          # project dependencies
├── setup.py                  # makes the project an installable package
├── src/
│   ├── __init__.py           # marks src as a Python package
│   ├── exception.py          # custom exception handling
│   ├── logger.py             # logging configuration
│   ├── utils.py              # shared helper functions
│   ├── components/           # data ingestion, transformation, model training
│   └── pipeline/             # training and prediction pipelines
├── logs/                     # generated at runtime (not tracked by git)
└── venv/                     # local conda environment (not tracked by git)
```

## Environment Setup

The environment lives in the project folder as `venv/` and was created with conda using a path prefix:

```bash
conda create -p ./venv python=3.11 -y
```

Use whichever Python version you prefer in place of `3.11`.

### Activate

Run this from the project root. Because the environment is path-based, activate it by path, not by name (`conda activate venv` will not work):

```bash
conda activate ./venv
```

Your prompt will change to show the active environment. Check with `which python`, which should point into `generic-ml/venv/bin/python`.

### Deactivate

```bash
conda deactivate
```

### Install dependencies

With the environment active, from the project root:

```bash
pip install -r requirements.txt
```

This installs `pandas`, `numpy` and `seaborn`. The last line of `requirements.txt`, `-e .`, also installs this project itself in editable mode by running `setup.py`, which makes `src` importable from anywhere. Run it from the project root, because `setup.py` reads `requirements.txt` by relative path. It also generates an `ml_project.egg-info/` folder, which is git-ignored.

## How It Works

### Packaging (`setup.py`)

- `get_requirements()` reads `requirements.txt` and returns the package list, dropping the `-e .` line because it is a pip instruction, not a package.
- `setup()` registers the project as `ml-project`, discovers packages with `find_packages()` (any folder with an `__init__.py`) and passes the dependency list to `install_requires`.

### Exception handling (`src/exception.py`)

`CustomException` wraps any error and builds a readable message containing the file name, line number and original error message
It must be raised inside an `except` block, because it reads the active exception from `sys.exc_info()`.

### Logging (`src/logger.py`)

Importing this module configures logging for the whole project:

- Creates a `logs/` folder and a new timestamped file for each run, e.g. `logs/09_26_2026_23_47_53.log`.
- Writes to both the log file and the terminal.
- Level is `INFO` and above.
- Format: `timestamp | level | logger name | file:line | message`

  ```
  2026-09-26 21:05:12 | INFO     | root | data_ingestion.py:42 | Loaded 1000 rows
  ```

Import it once wherever you log. Otherwise the plain `logging` module stays unconfigured and `INFO` messages are silently dropped:

```python
from src.logger import logging

logging.info("Training started")
```

`logs/` is created relative to where you run the command, so run scripts from the project root, for example `python -m src.exception`.

## Next Steps

- Implement the data ingestion, transformation and model training components under `src/components/`
- Implement the training and prediction pipelines under `src/pipeline/`
- Add model evaluation
