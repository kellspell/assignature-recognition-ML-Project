import os 
import sys
from src.logger import logging
from src.exceptions import CustomException

class GCloudSync:
    
    def sync_file_from_gcloud(self, gcp_bucket_url, filename, destination):
        """ 
        params:
        gcp_bucket_url:
        filepath:
        destination:
        
        """
        try:
            command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/"
            # Command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename}{destination}/"
            
            os.system(command)
        except Exception as e:
            raise CustomException(e, sys) from e    


    def sync_file_to_gscloud(self, gcp_bucket_url, filename):
        """ 
        params:
        gcp_bucket_url:
        filepath:
        
        """
        try:
            command = f"gsutil cp {filename} gs://{gcp_bucket_url}/"
            # Command = f"gcloud storage cp {filename} gs://{gcp_bucket_url}/"
            
            os.system(command)
        except Exception as e:
            raise CustomException(e, sys) from e