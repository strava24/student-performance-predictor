## End to End Machine Learning File

A reusable, end-to-end machine learning project in Python. This README tracks what has been set up so far.

## Progress So Far

### Project structure

```
generic-ml/
├── .gitignore         # ignores the /venv virtual environment
├── README.md
├── requirements.txt   # project dependencies
├── setup.py           # makes the project an installable package
├── src/
│   └── __init__.py    # marks src as a Python package
└── venv/              # local virtual environment (not tracked by git)
```

### What each file does

- **`.gitignore`**: keeps the `venv/` folder out of version control.
- **`requirements.txt`**: lists the third-party packages the project needs: `pandas`, `numpy` and `seaborn`. The last line, `-e .`, tells pip to also install this project itself in editable mode, which runs `setup.py`.
- **`setup.py`**: packages the project as `ml-project` (version `0.0.1`).
  - `get_requirements()` reads `requirements.txt` and returns the package list. It drops the `-e .` line, because that is a pip instruction and not a package name.
  - `setup()` uses `find_packages()` to discover packages (any folder with an `__init__.py`) and passes the list from `get_requirements()` to `install_requires`.
- **`src/__init__.py`**: turns `src` into a package so `find_packages()` picks it up.

### Bug fixed

`setup.py` originally had `with(filen_path) as file:`, which tries to use a string as a context manager and would make the install fail. It now uses `with open(file_path) as file:`.

## Setup

```bash
# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies and the project itself (editable mode)
pip install -r requirements.txt
```

Run the install from the project root, because `setup.py` reads `requirements.txt` by relative path.

## Next Steps

- Add data ingestion, transformation and model training components under `src/`
- Add logging and custom exception handling
- Add model evaluation and a prediction pipeline
