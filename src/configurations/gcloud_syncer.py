import os
import sys
import shutil
import subprocess
from src.logger import logging
from src.exceptions import CustomException

GSUTIL_PATH = shutil.which("gsutil") or os.path.expanduser("~/google-cloud-sdk/bin/gsutil")

class GCloudSync:

    def sync_file_from_gcloud(self, gcp_bucket_url, filename, destination):
        """
        params:
        gcp_bucket_url:
        filepath:
        destination:

        """
        try:
            command = [GSUTIL_PATH, "cp", f"gs://{gcp_bucket_url}/{filename}", f"{destination}/"]
            subprocess.run(command, check=True)
        except Exception as e:
            raise CustomException(e, sys) from e


    def sync_file_to_gscloud(self, gcp_bucket_url, filename):
        """
        params:
        gcp_bucket_url:
        filepath:

        """
        try:
            command = [GSUTIL_PATH, "cp", filename, f"gs://{gcp_bucket_url}/"]
            subprocess.run(command, check=True)
        except Exception as e:
            raise CustomException(e, sys) from e