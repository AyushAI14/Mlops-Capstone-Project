from src.utils import *
from src.constants.common import read_yaml
from src.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(self,config=CONFIG_FILE_PATH,
                 param = PARAM_FILE_PATH
                ):
        self.config=read_yaml(config)
        self.param=read_yaml(param)

    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        return DataIngestionConfig(
            root_dir=config["root_dir"],
            Train_data_path=config["Train_data_path"],
            test_data=config["test_data"],
            data_source=config["data_source"]
        )    
    