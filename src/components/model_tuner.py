import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from src.utils import balanced_recall,save_model

def model_tuner(index,model,x_train,y_train):
    param_gird_list=[[{"penalty":["l1","l2"]}],
                     [{"penalty":["l1","l2"]}],
                     [{"n_neighbors":[3,5,7]}],
                     [{"kernel":["rbf","poly"]}],
                     [{"max_depth":[None,5,10]}],
                     [{"n_estimators":[100,200,300],"max_depth":[5,10,15],"min_samples_split":[2,5,7]}],
                     [{"learning_rate":[0.01,0.1]}],
                     [{"learning_rate":[0.01,0.1]}]
                     ]
    
    scorer=make_scorer(balanced_recall)
    
    grid_cv=GridSearchCV(model,param_grid=param_gird_list[index-1],scoring=scorer,cv=3)
    grid_cv.fit(x_train,y_train)
    final_model=grid_cv.best_estimator_
    final_model.fit(x_train,y_train)

    final_model_path=os.path.join("artifacts/final_model.pkl")
    save_model(final_model_path,final_model)
    logging.info("returning final model path after hyper parameter tuning")

    return final_model_path


    