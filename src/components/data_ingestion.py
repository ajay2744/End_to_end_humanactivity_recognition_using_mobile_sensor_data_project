import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import data_transformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.components.model_tuner import model_tuner
from src.utils import load_model
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass


@dataclass
class DataIngestionCOnfig:
    train_data_path:str=os.path.join("artifacts","train.csv")
    train_smote_data_path:str=os.path.join("artifacts","train_smote.csv") 
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionCOnfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion component")

        try:
            df=pd.read_csv("notebook\data\Data.csv").drop(columns=['timestamp'],axis=1).drop_duplicates(ignore_index=True)
            logging.info("Read data as dataframe dropped irrelevent rows and columns")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Initiation of train test split")
            train_df,test_df=train_test_split(df,test_size=0.2,stratify=df['Activity'],random_state=42)
            smote=SMOTE(sampling_strategy=0.6,random_state=42)
            x_train=train_df.iloc[:,:-1]
            y_train=train_df.iloc[:,-1]
            x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)
            train_smote_df=pd.concat([x_train_smote,y_train_smote],axis=1)
            logging.info("Training data over sampled to balance minority class")
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            train_smote_df.to_csv(self.ingestion_config.train_smote_data_path,index=False,header=True)
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion part completed")
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path,
                    self.ingestion_config.train_smote_data_path
                    )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

    train_df=pd.read_csv("artifacts/train_smote.csv")
    x_train=train_df.drop(columns=["Activity"])
    y_train=train_df["Activity"]
    #x_train=data_transformation(x_train)

    model_trainer_obj=ModelTrainer()
    index,best_model=model_trainer_obj.initiate_model_trainer(x_train,y_train)

    final_model_path=model_tuner(index,best_model,x_train,y_train)
    final_model=load_model(final_model_path)

    test_df=pd.read_csv("artifacts/test.csv")
    x_test=test_df.drop(columns=["Activity"])
    y_test=test_df["Activity"]
    final_model.fit(x_train,y_train)

    y_test_pred=final_model.predict(x_test)
    print("F1-score on test set:",f1_score(y_test,y_test_pred))
    









    

