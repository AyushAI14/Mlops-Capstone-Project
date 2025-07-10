from src.utils import *
from src.constants.common import read_yaml
from src.entity import DataIngestionConfig,DataProcessingConfig,FeatureEngineeringConfig,ModelBuildingConfig,ModelEvalutionConfig,ModelRegistryConfig

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
    
    def get_data_transformation(self) -> DataProcessingConfig:
        config = self.config["data_transformation"]
        return DataProcessingConfig(
            root_dir=config["root_dir"],
            Train_data_path=config["Train_data_path"],
            test_data=config["test_data"],
            clean_Train_data_path=config["clean_Train_data_path"],
            clean_test_data=config["clean_test_data"],
        )
    def get_feature_Engineering(self) -> FeatureEngineeringConfig:
        config = self.config["feature_Engineering"]
        return FeatureEngineeringConfig(
            root_dir=config["root_dir"],
            process_Train_data_path=config["process_Train_data_path"],
            process_test_data=config["process_test_data"],
            clean_Train_data_path=config["clean_Train_data_path"],
            clean_test_data=config["clean_test_data"],
        )
    def get_model_building(self) -> ModelBuildingConfig:
        config = self.config["model_building"]
        return ModelBuildingConfig(
            # root_dir=config["root_dir"],
            process_Train_data_path=config["process_Train_data_path"],
            process_test_data=config["process_test_data"],
            # clean_Train_data_path=config["clean_Train_data_path"],
            # clean_test_data=config["clean_test_data"],
        )
    def get_model_Evalution(self) -> ModelEvalutionConfig:
        config = self.config["model_evalution"]
        return ModelEvalutionConfig(
            # root_dir=config["root_dir"],
            # process_Train_data_path=config["process_Train_data_path"],
            model_path=config["model_path"],
            process_test_data=config["process_test_data"],
            # clean_Train_data_path=config["clean_Train_data_path"],
            # clean_test_data=config["clean_test_data"],
        )
    def get_model_Registry(self) -> ModelRegistryConfig:
        config = self.config["model_registry"]
        return ModelRegistryConfig(
            json_file=config["json_file"]
        )