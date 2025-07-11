import os
import pandas as pd
from src.entity import FeatureEngineeringConfig
from src.logging import logger
from src.constants.common import read_yaml
from src.configuration.configManager import ConfigurationManager
from sklearn.feature_extraction.text import CountVectorizer
import joblib

class FeatureEngineering:
    def __init__(self,config:FeatureEngineeringConfig):
        self.config=config

    def vectorizor(self):
        params = read_yaml('params.yaml')
        max_features = params['max_features']
        vectorizer = CountVectorizer(max_features=max_features)
        logger.info("load into Dataframe")
        train_df = pd.read_csv(self.config.clean_Train_data_path)
        test_df = pd.read_csv(self.config.clean_test_data)

        train_rev = train_df['review'].values
        train_set = train_df['sentiment'].values
        test_rev = test_df['review'].values
        test_set = test_df['sentiment'].values
        logger.info("Converting into vector")

        X_train_bow = vectorizer.fit_transform(train_rev)
        X_test_bow = vectorizer.transform(test_rev)
        logger.info("Converting into Dataframe")
        traindf = pd.DataFrame(X_train_bow.toarray())
        traindf['label']=train_set
        testdf = pd.DataFrame(X_test_bow.toarray())
        testdf['label']=test_set
        logger.info("saving the vectorizer")
        os.makedirs(os.path.dirname('models/vectorizer.pkl'),exist_ok=True)
        joblib.dump(vectorizer,'models/vectorizer.pkl')
        logger.info("Saving the data ")
        traindf.to_csv(self.config.process_Train_data_path,index=False)
        testdf.to_csv(self.config.process_test_data,index=False)
        logger.info("Feature Engineering Completed Sucessfully")

if __name__ == "__main__":
    config=ConfigurationManager()
    getFeatureEngConfig = config.get_feature_Engineering()
    featEngg = FeatureEngineering(config=getFeatureEngConfig)
    featEngg.vectorizor()        

        

