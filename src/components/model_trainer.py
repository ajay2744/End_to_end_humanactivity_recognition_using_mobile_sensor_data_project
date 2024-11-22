import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import model_evaluation

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    select_model_file_path=os.path.join("artifacts","best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,x_train,y_train):
        try:
            best_model_index,best_model=model_evaluation(x_train,y_train)
            logging.info("Returning best model for tuning further")
            return best_model_index,best_model

        except Exception as e:
            raise CustomException(sys,e)





