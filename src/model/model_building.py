import os
import pandas as pd
from src.entity import ModelBuildingConfig
from src.logging import logger
from src.constants.common import read_yaml
from src.configuration.configManager import ConfigurationManager
from sklearn.linear_model import LogisticRegression
import joblib

class ModelBuilding:
    def __init__(self,config:ModelBuildingConfig):
        self.config=config

    def trainer(self):
        logger.info("load into Dataframe")
        train_df = pd.read_csv(self.config.process_Train_data_path)
        # test_df = pd.read_csv(self.config.process_test_data)
        logger.info("splitting the  Dataframe")
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        logger.info("Intialzing model for training")
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logger.info('Model training completed')
        os.makedirs(os.path.dirname('models/clfLR.pkl'), exist_ok=True)
        joblib.dump(clf,'models/clfLR.pkl')
        logger.info('Model Saved')

if __name__ == "__main__":
    config=ConfigurationManager()
    getmdoelBuildingConfig = config.get_model_building()
    modelbuild = ModelBuilding(config=getmdoelBuildingConfig)
    modelbuild.trainer()        
