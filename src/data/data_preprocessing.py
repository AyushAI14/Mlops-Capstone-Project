import os
import pandas as pd
from src.entity import DataProcessingConfig
from src.logging import logger
from src.configuration.configManager import ConfigurationManager
from text_prettifier import TextPrettifier


class DataPreprocessing:
    def __init__(self,config:DataProcessingConfig):
        self.config = config

    def clean_text(self,text):
        try:
            prettifier = TextPrettifier()

            text = text.lower()
            text = prettifier.remove_contractions(text)
            text = prettifier.remove_emojis(text)
            text = prettifier.remove_html_tags(text)
            text = prettifier.remove_urls(text)
            text = prettifier.remove_special_chars(text)
            text = prettifier.remove_stopwords(text)
            text = prettifier.remove_numbers(text)
            return text
        except Exception as e:
            logger.info(f"Error file cleaning text {e}")
    
    def main(self):
        try:
            logger.info("loaded the file")
            train_df = pd.read_csv(self.config.Train_data_path)
            test_df = pd.read_csv(self.config.test_data)

            logger.info("cleaning the reviews in data")
            train_df['review'] = train_df['review'].apply(self.clean_text)
            test_df['review'] = test_df['review'].apply(self.clean_text)

            train_df.to_csv(self.config.clean_Train_data_path)
            test_df.to_csv(self.config.clean_test_data)
        except Exception as e:
            logger.info(f'Error while transformaing the df {e}')



if __name__ == "__main__":
    config=ConfigurationManager()
    getDatatransConfig = config.get_data_transformation()
    datatrans = DataPreprocessing(config=getDatatransConfig)
    datatrans.main()
