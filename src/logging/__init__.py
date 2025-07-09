import os 
from pathlib import Path
import logging
import sys
from datetime import datetime

logdir = 'logs'
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"
os.makedirs(logdir,exist_ok=True)
log_file_path = os.path.join(logdir,log_file)

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