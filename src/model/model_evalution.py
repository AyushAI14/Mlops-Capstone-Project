import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
import joblib

from src.entity import ModelEvalutionConfig
from src.logging import logger
from src.constants.common import read_yaml
from src.configuration.configManager import ConfigurationManager
from dotenv import load_dotenv
import dagshub
import os

load_dotenv()

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# Set credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Init DagsHub MLflow integration
dagshub.init(
    repo_owner='AyushAI14',
    repo_name='Mlops-Capstone-Project',
    mlflow=True
)

# mlflow.set_tracking_uri('https://dagshub.com/AyushAI14/Mlops-Capstone-Project.mlflow')
# dagshub.init(repo_owner='AyushAI14', repo_name='Mlops-Capstone-Project', mlflow=True)

class ModelEvalution:
    def __init__(self, config: ModelEvalutionConfig):
        self.config = config

    def load_model(self, filepath: str):
        try:
            model = joblib.load(filepath)
            return model
        except Exception as e:
            logger.error(f"Error loading model file: {e}")
            raise

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info('Data loaded from %s', file_path)
            return df
        except pd.errors.ParserError as e:
            logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error occurred while loading the data: %s', e)
            raise

    def evaluate_model(self, clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and return the evaluation metrics."""
        try:
            y_pred = clf.predict(X_test)

            if hasattr(clf, "predict_proba"):
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
                logger.warning("Model does not support predict_proba. AUC will not be computed.")

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = None
                logger.warning("AUC skipped — either no predict_proba or y_test has only one class.")

            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }

            logger.info('Model evaluation metrics calculated')
            return metrics_dict
        except Exception as e:
            logger.error('Error during model evaluation: %s', e)
            raise

    def save_metrics(self, metrics: dict, file_path: str) -> None:
        """Save the evaluation metrics to a JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            logger.info('Metrics saved to %s', file_path)
        except Exception as e:
            logger.error('Error occurred while saving the metrics: %s', e)
            raise

    
    def save_model_info(self,run_id: str, model_path: str, file_path: str) -> None:
        """Save the model run ID and path to a JSON file."""
        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            with open(file_path, 'w') as file:
                json.dump(model_info, file, indent=4)
            logger.debug('Model info saved to %s', file_path)
        except Exception as e:
            logger.error('Error occurred while saving the model info: %s', e)
            raise

    def main(self):
        # mlflow.set_experiment("my-dvc-pipeline")
        EXPERIMENT_ID = "3"  # Replace with actual ID you got from DagsHub

        with mlflow.start_run(experiment_id=EXPERIMENT_ID) as run:
            try:
                clf = self.load_model(self.config.model_path)
                test_data = self.load_data(self.config.process_test_data)
                X_test = test_data.iloc[:, :-1].values
                y_test = test_data.iloc[:, -1].values

                metrics = self.evaluate_model(clf, X_test, y_test)
                os.makedirs(os.path.dirname('reports/metrics.json'), exist_ok=True)
                self.save_metrics(metrics, 'reports/metrics.json')

                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, metric_value)

                # Log model parameters to MLflow
                if hasattr(clf, 'get_params'):
                    params = clf.get_params()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)

                # Log model to MLflow
                mlflow.log_artifact(self.config.model_path)

                # Save model info
                os.makedirs(os.path.dirname('reports/experiment_info.json'), exist_ok=True)
                self.save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

                # Log the metrics file to MLflow
                mlflow.log_artifact('reports/metrics.json')

                logger.info(f"MLflow run completed successfully. Run ID: {run.info.run_id}")

            except Exception as e:
                logger.error('Failed to complete the model evaluation process: %s', e)
                print(f"Error: {e}")


if __name__ == "__main__":
    config = ConfigurationManager()
    model_eval_config = config.get_model_Evalution()
    modeleval = ModelEvalution(config=model_eval_config)
    modeleval.main()
