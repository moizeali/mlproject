import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Configuration class to store model-related file paths
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

# ModelTrainer class to handle the training of multiple models and selection of the best one
class ModelTrainer:
    def __init__(self):
        # Initializes model configuration with the location where the trained model will be saved
        self.model_trainer_config = ModelTrainerConfig()

    # Method to initiate model training and evaluation
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Splitting the input arrays into training features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # All columns except the last one as features
                train_array[:, -1],    # Last column as the target for training
                test_array[:, :-1],    # All columns except the last one as features for testing
                test_array[:, -1],     # Last column as the target for testing
            )
            
            # Dictionary of regression models to be evaluated
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters for the respective models
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of trees in the forest
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],  # Learning rate shrinks contribution of each tree
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],  # Proportion of samples used for fitting
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters for linear regression
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],  # Depth of the trees
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Dictionary to hold the best models after performing GridSearch
            best_models = {}

            # Perform GridSearchCV for each model
            for model_name, model in models.items():
                logging.info(f"Performing GridSearch for {model_name}")
                param_grid = params.get(model_name, {})  # Get the hyperparameters for the current model

                # If the model has hyperparameters defined in the params dictionary, use GridSearchCV
                if param_grid:
                    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0, scoring='r2')
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_  # Get the best model from GridSearchCV
                    logging.info(f"Best model for {model_name}: {grid_search.best_params_}")
                else:
                    # If no hyperparameters are defined, train the default model
                    model.fit(X_train, y_train)
                    best_model = model

                # Add the best model to the best_models dictionary
                best_models[model_name] = best_model

            # Evaluate models on training and test datasets using the best models found by GridSearchCV
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=best_models  # Pass the best models from GridSearch
            )
            
            # Get the best model's score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model based on the highest score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Retrieve the best model
            best_model = best_models[best_model_name]

            # If the best model's score is below a threshold (e.g., 0.6), raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict the target values for the test set using the best model
            predicted = best_model.predict(X_test)

            # Calculate R-squared value to evaluate the performance of the best model
            r2_square = r2_score(y_test, predicted)
            
            # Return the R-squared value as the final model performance metric
            return r2_square

        except Exception as e:
            # Handle any exception and raise a custom exception with detailed info
            raise CustomException(e, sys)