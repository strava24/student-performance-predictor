import os
import sys

import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    '''
    Util to save a python object as a binary file using dill
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    Util to evaluate multiple models and return their scores
    '''
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Refit with the best params found

            y_train_pred = model.predict(X_train)  # Predict on training data
            y_test_pred = model.predict(X_test)  # Predict on test data
            train_score = r2_score(y_train, y_train_pred)  # Calculate R2 score for training data
            test_score = r2_score(y_test, y_test_pred)  # Calculate R2 score for test data
            report[list(models.keys())[i]] = test_score  # Store test score in report

        return report

    except Exception as e:
        raise CustomException(e, sys)