'''
In ML data ingestion means, getting data from different sources into your ML system 
so it can be processed and eventually used to train or make predictions

In the current approach we are reading the data from a csv file and then splitting it into train and test datasets
But in real time scenario we can get the data from different sources like database, API, web scraping etc. and then we can process it and store it in a csv file or database for further processing
'''
import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

'''
it is a good practice to create a config class for data ingestion
in python perspective since this class only holds class variables and no methods, 
it is a good practice to use dataclass decorator to create this class
'''
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

'''
since we are creating the main class with all the logic better to use the standard init approach instead of dataclass decorator
'''
class DataIngestion:

    # constructor to initialize the class variables
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') # read the dataset as dataframe
            logging.info('Read the dataset as dataframe')

            # create the directory if not available
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #export the dataframe to csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_data, test_data, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_data, test_data)
    logging.info(f"r2_score from the best model {r2_square}")