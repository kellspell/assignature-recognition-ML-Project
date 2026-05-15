import os
import dill
import sys
import yaml
import base64
from typing import Any
from src.logger import logging
from src.exceptions import CustomException

def save_objects(file_path: str, obj: object) -> None:
    logging.info("Entered the saved_objects method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info('Existed the saved object mothod of utils')
    except Exception as e:
        raise CustomException(e, sys) from e   
    
def load_objects(file_path: str) -> Any:
    logging.info("Entered the load_objects method of utils")
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info('Existed the load object mothod of utils')
        return obj
    except Exception as e:
        raise CustomException(e, sys) from e  
    
def image_to_base64(image): 
    try:
        logging.info('Converting the image to base64 from the method utils')
        with open(image, 'rb') as img_file:
            my_string = base64.b64decode(img_file.read())
        logging.info('Existed the image object mothod of utils')
        return my_string
    except Exception as e:
        raise CustomException(e, sys) from e   
    
def read_yaml_file(file_path: str) -> dict:
    """Reads a yaml file and returns the content"""
    
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e             
    