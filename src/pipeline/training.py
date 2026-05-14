import os
import sys 
from src.logger import logging
from src.exceptions import CustomException
from src.components.data_injection import DataInjection
from src.artifacts.config_entry import DataInjectionConfig
from src.artifacts.artifects_entry import DataInjectionArtifacts

class TrainingPipeline: 
    def __init__(self):
        self.data_injection_config = DataInjectionConfig()
        
    def start_data_injection(self) -> DataInjectionArtifacts:
        logging.info("Entered the start_data_injection method of Training pipeline class")
        try:
            logging.info("Getting the dataset from Gcloud storage bucket")
            data_injection = DataInjection(
                data_injection_config=self.data_injection_config
            )
            data_injection_artifacts = data_injection.initiate_data_injection()
            logging.info("Got the dataset from Gcloud Storage")
            logging.info("Existed the start_data_injection method of Training pipeline class")
            return data_injection_artifacts
        
        except Exception as e:
            raise CustomException(e, sys) from e  
        
    def run_pipeline(self) -> None:
        logging.info("Entered the run pipeline method of Training pipeline class")
        try:
            self.data_injection_artifacts = self.start_data_injection()  
        except Exception as e:
            raise CustomException(e, sys) from e       
        
                