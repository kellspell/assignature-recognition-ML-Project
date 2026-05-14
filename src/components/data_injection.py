"""We Use this code to fecth the data from Google Cloud server"""

import os 
import sys
from zipfile import ZipFile
from src.logger import logging
from src.exceptions import CustomException
from src.configurations.gcloud_syncer import GCloudSync 
from src.artifacts.config_entry import DataInjectionConfig
from src.artifacts.artifects_entry import DataInjectionArtifacts

class DataInjection:
    def __init__(self, data_injection_config: DataInjectionConfig):
        self.data_injection_config = data_injection_config
        self.gcloud = GCloudSync()
        
    def get_data_from_gcloud(self)-> None:
        try:
            logging.info("Entered the get_data_from_cloud method of data injection class")
            os.makedirs(self.data_injection_config.DATA_INJECTION_ARTIFACTS_DIR, exist_ok=True)
            self.gcloud.sync_file_from_gcloud(self.data_injection_config.BUCKET_NAME,
                                              self.data_injection_config.ZIP_FILE_NAME,
                                              self.data_injection_config.DATA_INJECTION_ARTIFACTS_DIR)
            logging.info("Existed the get data from cloud method of data injection class")
        except Exception as e:
            raise CustomException(e, sys) from e    
        
    def unzip_and_clear(self) -> None:
        logging.info("Entered the unzip method of data injection class")
        try:
            with ZipFile(self.data_injection_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_injection_config.DATA_INJECTION_ARTIFACTS_DIR)
            logging.info("Existed the unzip method of data injection class")  
        except Exception as e:
            raise CustomException(e, sys) from e    
        
    def initiate_data_injection(self) -> DataInjectionArtifacts:
        logging.info("Entered the initialization method of data injection class")
        try:
            self.get_data_from_gcloud()
            logging.info("Fetched the zip file dataset from GCloud storage bucket")
            
            self.unzip_and_clear()
            logging.info("Unzip the Fetcheded the zip file dataset from GCloud storage bucket")
            
            logging.info("Deleting dataset.zip file")
            os.remove(os.path.join(self.data_injection_config.DATA_INJECTION_ARTIFACTS_DIR,
                                  self.data_injection_config.ZIP_FILE_NAME))
            
            data_injection_artifacts = DataInjectionArtifacts(
                dataset_path=self.data_injection_config.DATA_INJECTION_ARTIFACTS_DIR
            )
            logging.info(f"Data injection artifact: {data_injection_artifacts}")
            logging.info("Entered the initiate data injection method of data injection class")
            return data_injection_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e 
            
               
                 