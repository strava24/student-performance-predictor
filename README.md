## End to End Machine Learning Project

A reusable, end-to-end machine learning project in Python. This README tracks what has been set up so far.

## Project Structure

```
generic-ml/
├── .gitignore
├── README.md
├── environment.yml           # conda environment (Python, system libraries, pip deps)
├── requirements.txt          # Python package dependencies
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

The environment lives in the project folder as `venv/` and is created with conda from `environment.yml`, using a path prefix. From the project root:

```bash
conda env create -p ./venv -f environment.yml
```

This installs Python 3.14, `llvm-openmp` and `ipykernel` from conda-forge, then runs `pip install -r requirements.txt`. `llvm-openmp` provides `libomp.dylib`, which xgboost needs on macOS. It is a system library, not a Python package, so pip cannot install it and it has to live in `environment.yml` instead of `requirements.txt`.

To apply changes to `environment.yml` or `requirements.txt` to an existing environment:

```bash
conda env update -p ./venv -f environment.yml
```

### Activate

Run this from the project root. Because the environment is path-based, activate it by path, not by name (`conda activate venv` will not work):

```bash
conda activate ./venv
```

Your prompt will change to show the active environment. Check with `which python`, which should point into `generic-ml/venv/bin/python`.

### Notebook kernel

Notebooks must run on this environment's interpreter, not the base Anaconda one, or the installed packages won't be visible. `ipykernel` comes from `environment.yml`. Register the kernel once, with the environment active:

```bash
python -m ipykernel install --user --name generic-ml --display-name "Python 3.14 (generic-ml)"
```

Then select "Python 3.14 (generic-ml)" as the kernel in the notebook.

### Deactivate

```bash
conda deactivate
```

### Install dependencies

`conda env create` already runs this step. To reinstall only the Python packages, with the environment active and from the project root:

```bash
pip install -r requirements.txt
```

This installs `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `catboost` and `xgboost` (unpinned, so the latest versions for your Python). The `-e .` line at the end of `requirements.txt` is currently commented out (`# -e .`), so this step does not install the project itself. To make `src` importable from anywhere, install the project in editable mode separately, from the project root:

```bash
pip install -e .
```

This runs `setup.py`, which reads `requirements.txt` by relative path, so it must be run from the project root. It also generates an `ml_project.egg-info/` folder, which is git-ignored.

## How It Works

### Packaging (`setup.py`)

- `get_requirements()` reads `requirements.txt` and returns the package list, dropping an exact `-e .` line because it is a pip instruction, not a package. The commented `# -e .` line is not dropped by this check, but setuptools ignores comment lines in `install_requires`.
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
