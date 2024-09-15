import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object
import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to the model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            # Logging and loading the model and preprocessor
            logging.info("Loading model and preprocessor objects")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Logging preprocessor transformation
            logging.info("Transforming input features using preprocessor")
            data_scaled = preprocessor.transform(features)

            # Making predictions using the loaded model
            logging.info("Making predictions with the loaded model")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str, 
                 test_preparation_course: str, reading_score: int, writing_score: int):
        # Initializing input data
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Creating a dictionary of input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Returning data as a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.error("Error occurred while creating input DataFrame")
            raise CustomException(e, sys)
