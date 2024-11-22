from src.exception import CustomException
from src.logger import logging

import os
import sys

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score,make_scorer,recall_score
import pickle

def model_evaluation(x_train,y_train):
    try:
        lg=LogisticRegression(random_state=42)
        sgd=SGDClassifier(random_state=42)
        knc=KNeighborsClassifier()
        sv=SVC(random_state=42)
        dt=DecisionTreeClassifier(random_state=42)
        rf=RandomForestClassifier(random_state=42)
        adb=AdaBoostClassifier(random_state=42)
        gdb=GradientBoostingClassifier(random_state=42)
        list_models=[lg,sgd,knc,sv,dt,rf,adb,gdb]
        models_str=['lg','sgd','knc','sv','dt','rf','adb','gdb']

        models_roc_auc={}
        for model in list_models:
            y_train_pred=cross_val_predict(model,x_train,y_train,cv=4)
            models_roc_auc[model]=roc_auc_score(y_train_pred,y_train)
        models_roc_auc_list=list(enumerate(models_roc_auc.items(),start=1))

        for i in range(len(models_roc_auc_list)):
            for j in range(i+1,len(models_roc_auc_list)):
                if models_roc_auc_list[i][1][1]>models_roc_auc_list[j][1][1]:
                    models_roc_auc_list[i],models_roc_auc_list[j]=models_roc_auc_list[j],models_roc_auc_list[i]

        return models_roc_auc_list[-1][0],models_roc_auc_list[-1][1][0]

    except Exception as e:
        raise CustomException(sys,e)
    
def balanced_recall(y_true, y_pred):
    # Sensitivity (Recall for positives)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)
    # Specificity (Recall for negatives)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    # Average of both
    return (sensitivity + specificity) / 2

def save_model(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)       
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Model saved")
    except Exception as e:
        CustomException(e,sys)

def load_model(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

    





