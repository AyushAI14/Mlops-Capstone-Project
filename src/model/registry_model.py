import json
import mlflow
from src.logging import logger
import dagshub
from src.entity import ModelRegistryConfig
from src.logging import logger
from src.configuration.configManager import ConfigurationManager


import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


mlflow.set_tracking_uri('https://dagshub.com/AyushAI14/Mlops-Capstone-Project.mlflow')
dagshub.init(repo_owner='AyushAI14', repo_name='Mlops-Capstone-Project', mlflow=True)

class ModelResistry:
    def __init__(self,config:ModelRegistryConfig):
        self.config=config

    def load_metrics_info(self,filepath:str):
        try:
            with open(filepath,'r') as f:
                model_info = json.load(f)
                logger.info("loaded the json file")
                return model_info
        except Exception as e:
            logger.error('Unexpected error occurred while loading the model info: %s', e)
            raise

    def register_Model(self,model_name: str, model_info: dict):
        """Register the model to the MLflow Model Registry."""
        try:
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            print(model_uri)
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Transition the model to "Staging" stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        except Exception as e:
            # logger.error('Error during model registration: %s', e)
            print(e)
            raise
    def main(self):
        try:
            model_info = self.load_metrics_info(self.config.json_file)
            model_name = "my_model"
            self.register_Model(model_name=model_name,model_info=model_info)
        except Exception as e:
            logger.error('Failed to complete the model registration process: %s', e)
            print(f"Error: {e}")

if __name__ == "__main__":
    config = ConfigurationManager()
    model_registry_config = config.get_model_Registry()
    modelregis = ModelResistry(config=model_registry_config)
    modelregis.main()