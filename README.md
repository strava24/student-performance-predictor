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
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # exploratory data analysis
│   ├── 2. MODEL TRAINING.ipynb              # model comparison and selection
│   └── data/stud.csv                        # dataset
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

## Notebook Findings

The notebooks in `notebook/` explore the data and prototype the model before the work moves into `src/`.

### Dataset

[Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) (Kaggle): 1,000 students, 8 columns.

| Column | Type | Values |
|---|---|---|
| `gender` | categorical | female, male |
| `race_ethnicity` | categorical | group A to group E |
| `parental_level_of_education` | categorical | some high school, high school, some college, associate's, bachelor's, master's |
| `lunch` | categorical | standard, free/reduced |
| `test_preparation_course` | categorical | none, completed |
| `math_score`, `reading_score`, `writing_score` | numeric | 0 to 100 |

### EDA (`1 . EDA STUDENT PERFORMANCE .ipynb`)

The goal is to understand how a student's test scores relate to their background.

- **Data quality:** no missing values or duplicates, and every column has the correct type.
- **Score statistics:** the three subjects have similar means (math 66.1, reading 69.2, writing 68.1) and standard deviations (about 15). Math has the lowest minimum score (0, against 17 for reading and 10 for writing).
- **Subject difficulty:** students do worst in math (7 full marks, 4 scores of 20 or less) and best in reading (17 full marks, 1 score of 20 or less).
- **Gender:** the classes are balanced (518 female, 482 male). Females have the higher overall average (69.6 vs 65.8), but males score higher in math (68.7 vs 63.6).
- **Race/ethnicity:** group C is the largest and group A the smallest. Group E scores highest in every subject and group A lowest.
- **Parental education:** "some college" and "associate's degree" are the most common levels. Students whose parents have a bachelor's or master's degree score higher.
- **Lunch:** most students get standard lunch. Students with standard lunch score higher than those on free/reduced lunch, for both genders.
- **Test preparation course:** most students did not take the course. Those who completed it score higher in all three subjects.
- **Score correlation:** math, reading and writing scores increase linearly with each other.

**Conclusion:** performance is related to lunch type, race/ethnicity, parental education and gender, and completing the test preparation course helps.

### Model Training (`2. MODEL TRAINING.ipynb`)

The goal is a regression model that predicts `math_score`.

- **Features:** the other 7 columns (the 5 categorical columns plus `reading_score` and `writing_score`).
- **Preprocessing:** a `ColumnTransformer` one-hot encodes the categorical columns and standard-scales the numeric ones, giving 19 features.
- **Split:** 80% train and 20% test (`random_state=42`).
- **Evaluation:** RMSE, MAE and R² on both sets.

Test-set R² for the 9 models compared:

| Model | R² |
|---|---|
| Ridge | 0.881 |
| Linear Regression | 0.880 |
| CatBoost Regressor | 0.852 |
| AdaBoost Regressor | 0.850 |
| Random Forest Regressor | 0.847 |
| Lasso | 0.825 |
| XGBoost Regressor | 0.822 |
| K-Neighbors Regressor | 0.784 |
| Decision Tree | 0.760 |

**Result:** the linear models perform best, because math scores are strongly linearly correlated with reading and writing scores. Linear Regression was chosen as the final model (R² of 0.880, test RMSE of about 5.4 and MAE of about 4.2 marks). Its actual vs predicted plot follows the diagonal closely.

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

`logs/` is created relative to where you run the command. The project is installed in editable mode, so `src` imports work whether you run a script directly (`python src/exception.py`) or as a module (`python -m src.exception`) — just run it from the project root so the relative `logs/`/`artifacts/` paths resolve correctly.

### Data Ingestion (`src/components/data_ingestion.py`)

- `DataIngestionConfig`: a `@dataclass` holding the output paths (`artifacts/data.csv`, `train.csv`, `test.csv`).
- `DataIngestion.initiate_data_ingestion()`: reads `notebook/data/stud.csv`, saves an untouched copy to `data.csv`, splits it 80/20 into train/test (`random_state=42`), saves both, and returns their paths.
- Running the file directly also chains into `DataTransformation`.

### Data Transformation (`src/components/data_transformation.py`)

- `DataTransformationConfig`: holds the path for the fitted preprocessor (`artifacts/preprocessor.pkl`).
- `get_data_transformer_object()`: builds an (unfitted) `ColumnTransformer` — numeric columns go through median-impute → scale, categorical columns through mode-impute → one-hot → scale (`with_mean=False`, since one-hot output is sparse).
- `initiate_data_transformation()`: reads the train/test CSVs, separates the `math_score` target from the features, fits the preprocessor on the training features only and transforms both train and test (so no test-set statistics leak into training), reattaches the target column, and saves the **fitted** preprocessor with `save_object()` (`src/utils.py`, uses `dill`) to `preprocessor.pkl`.
- **Fitted vs. unfitted:** a fresh preprocessor only knows *what* to do (impute, scale, encode); fitting is what makes it learn the actual numbers (medians, means, categories) from training data. It's pickled so `predict_pipeline.py` can later reload that exact fitted state and transform new input the same way, without refitting.

### Model Training (`src/components/model_trainer.py`)

- `ModelTrainerConfig`: holds the path for the trained model (`artifacts/model.pkl`).
- `initiate_model_trainer(train_array, test_array)`: takes the numpy arrays returned by `DataTransformation` (last column is the `math_score` target), slices each into `X`/`y`, then tunes and scores 8 candidate regressors — Random Forest, Decision Tree, Gradient Boosting, Linear Regression, K-Neighbors, XGBoost, CatBoost and AdaBoost — via `evaluate_models()` in `src/utils.py`. Each has a matching entry in a `params` dict of hyperparameter grids to search (an empty grid for `Linear Regression`, since it has nothing to tune).
- `evaluate_models()` runs `GridSearchCV(model, para, cv=3)` per model to find the best hyperparameters, then **refits that model on the full training split with those best params** (`model.set_params(**gs.best_params_)` then `model.fit(...)`) before scoring it against the test split with R², returning a `{model_name: test_r2}` report. Skipping that refit step is a common mistake — `GridSearchCV.fit()` only trains its own internal clones, not the original `model` object, so predicting straight after `gs.fit()` raises `NotFittedError` on the untrained estimator.
- The model with the highest test R² wins; if that score is still below `0.6`, `initiate_model_trainer()` raises a `CustomException` instead of shipping a weak model. Otherwise it logs the winner, pickles it to `artifacts/model.pkl` via `save_object()`, and returns its test R².

## End-to-End Flow

There's no standalone pipeline entry point yet, so running `python src/components/data_ingestion.py` is currently the way to exercise the whole thing — its `__main__` block chains straight through transformation and training:

1. **Ingestion** (`data_ingestion.py`): reads `notebook/data/stud.csv`, writes an untouched copy to `artifacts/data.csv`, splits it 80/20 (`random_state=42`) into `artifacts/train.csv` / `artifacts/test.csv`, and returns those two paths.
2. **Transformation** (`data_transformation.py`): loads the train/test CSVs from those paths, builds the `ColumnTransformer` preprocessor, fits it on the training features only, transforms both splits (with the `math_score` target reattached as the last column), pickles the fitted preprocessor to `artifacts/preprocessor.pkl`, and returns `(train_arr, test_arr, preprocessor_path)`.
3. **Training** (`model_trainer.py`): takes `train_arr`/`test_arr`, splits each into `X`/`y`, grid-searches hyperparameters for each candidate regressor, refits each on its best params, picks the best-scoring one, pickles it to `artifacts/model.pkl`, and returns its final R², which gets logged.

Every step logs through `src/logger.py` and wraps its body in `try/except` so failures surface as a `CustomException` with the originating file and line number. Once `src/pipeline/train_pipeline.py` is implemented, this same three-step wiring is expected to move there instead of living in `data_ingestion.py`'s `__main__` block.

## Next Steps

- Implement the training and prediction pipelines under `src/pipeline/`
- Add model evaluation reporting/metrics beyond the console log
