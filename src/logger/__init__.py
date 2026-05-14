
import logging
import os
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

logs_dir_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir_path, LOG_FILE_NAME)

formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(file_handler)
logging.root.addHandler(stream_handler)
