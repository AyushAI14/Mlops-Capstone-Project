import os 
from pathlib import Path
import logging
import sys

log_file = 'logs'
log_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"
os.makedirs(log_file,exist_ok=True)
log_file_path = os.path.join(log_file,'loop.log')

try:
    logging.basicConfig(
    level=logging.INFO,
    format=log_str,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
    )
except Exception as e:
    print(e)

logger = logging.getLogger('Sentiment')