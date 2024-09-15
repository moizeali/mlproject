import os
import sys
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logging
import pandas as pd  # For data manipulation

from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from dataclasses import dataclass  # For creating configuration classes

from src.components.data_transformation import DataTransformation  # Data transformation component
from src.components.data_transformation import DataTransformationConfig  # Data transformation configuration
from src.components.model_trainer import ModelTrainerConfig  # Model trainer configuration
from src.components.model_trainer import ModelTrainer  # Model trainer component

# DataIngestionConfig: Configuration class for storing file paths for raw, training, and testing datasets
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path for training dataset
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path for testing dataset
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path for raw dataset

# DataIngestion: Class to handle the ingestion of data, including reading, saving, and splitting the dataset
class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    # Method to ingest data and split it into train and test sets
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from CSV file
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the directory if it doesn't exist and save the raw dataset
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split the dataset into training and testing sets (80% training, 20% testing)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training set to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths of the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Handle any exceptions that occur during data ingestion
        except Exception as e:
            raise CustomException(e, sys)

# Main block to execute the data ingestion, transformation, and model training
if __name__ == "__main__":
    # Create an instance of DataIngestion and initiate the ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Perform data transformation on the ingested train and test data
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of ModelTrainer and initiate model training with transformed data
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
