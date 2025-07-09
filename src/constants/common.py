import yaml
import os
from src.logging import logger

def read_yaml(filepath:str):
    try:
        with open(filepath,'r') as f:
            content = yaml.safe_load(f)
            logger.info("yaml file successfully extracted")
        return content
    except Exception as e:
        logger.info(f"Yaml file not extracted : {e}")


