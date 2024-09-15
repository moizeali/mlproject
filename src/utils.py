import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException

# Function to save an object to a file
def save_object(file_path, obj):
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate the trained models
# It accepts pre-trained models, evaluates their performance, and returns a report with R-squared scores
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Iterate over each model to evaluate
        for model_name, model in models.items():
            # Predict on the training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared score for training and testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the test score in the report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# Function to load an object from a file
def load_object(file_path):
    try:
        # Load the object using pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
