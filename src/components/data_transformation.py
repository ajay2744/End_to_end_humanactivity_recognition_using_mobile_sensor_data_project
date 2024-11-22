from src.exception import CustomException
from src.logger import logging
import sys

from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass

def data_transformation(x,y=None):
    try:
        scaler=StandardScaler()
        scaler.fit_transform(x)
        logging.info("Returning data after transformation")
        return x
    except Exception as e:
        raise CustomException(e,sys)
