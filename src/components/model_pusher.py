import os 
import sys 
from src.logger import logging
from src.exceptions import CustomException
from src.configurations.gcloud_syncer import GCloudSync
from src.artifacts.config_entry import ModelPusherConfig
from src.artifacts.artifects_entry import ModelTrainerArtifacts, ModelPusherArtifacts

class ModelPusher:
    def __init__(self, 
                 model_pusher_config: ModelPusherConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.gcloud = GCloudSync()
        
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        logging.info('Entered at the initiale model pusher method of model training class')
        try:
            logging.info('Uploading the model to gcloud storage')
            self.gcloud.sync_file_to_gscloud(self.model_pusher_config.BUCKET_NAME, self.model_trainer_artifacts.trained_model_path)
            logging.info('Uploading the Best model to gcloud storage')
            logging.info('Saving the model pusher artifacts')
            model_pusher_artifacts = ModelPusherArtifacts(
                backet_name = self.model_pusher_config.BUCKET_NAME
            )
            logging.info('Exited the initiate model pusher method pf the model pusher class')
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e    