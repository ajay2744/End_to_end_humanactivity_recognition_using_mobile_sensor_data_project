from src.exception import CustomException
from src.logger import logging
from src.utils import load_model
import os
import sys

def predict(input):
    try:
        model=load_model("artifacts/final_model.pkl")
        logging.info("Returning the predicted output")
        return model.predict(input)
            
    except Exception as e:
        raise CustomException(e,sys)