import os
import pandas as pd
from src.constants.common import read_yaml
from src.entity import DataIngestionConfig
from src.logging import logger
from sklearn.model_selection import train_test_split
from src.configuration.configManager import ConfigurationManager
pd.set_option('future.no_silent_downcasting', True)


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
        
    
    def load_csv_file(self,filepath:str):
        try:
            logger.info("loading the data")
            df = pd.read_csv(filepath)
            logger.info("Data has been Extracted successfully")
            return df 
        except Exception as e:
            logger.info(f"Error file Extaction of csv file {e}")

    

    def preprocess_data(self,df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            logger.info("pre-processing...")
            final_df = df[df['sentiment'].isin(['positive', 'negative'])]
            final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0}).astype(int)
            logger.info('Data preprocessing completed')
            return final_df
        except KeyError as e:
            logger.error('Missing column in the dataframe: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error during preprocessing: %s', e)
            raise
    
    def main(self):
        try:
            param = read_yaml('param.yaml')
            test_size = param["test_size"]
            df = self.load_csv_file(self.config.data_source)
            final_df = self.preprocess_data(df=df)

            X = final_df['review']
            y = final_df['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=44)

            X_train.to_csv(self.config.Train_data_path)
            X_test.to_csv(self.config.test_data)
        except Exception as e:
            logger.info(f"Main Have a issue {e}")


if __name__ == "__main__":
    config=ConfigurationManager()
    getDataIngestionConfig = config.get_data_ingestion()
    dataingestion = DataIngestion(config=getDataIngestionConfig)
    dataingestion.main()
