import os
import sys 
from src.logger import logging
from src.exceptions import CustomException
from src.components.data_injection import DataInjection
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.artifacts.config_entry import DataInjectionConfig, DataTransformationConfig, ModelTrainerConfig
from src.artifacts.artifects_entry import DataInjectionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts

class TrainingPipeline: 
    def __init__(self):
        self.data_injection_config = DataInjectionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        
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
        
    def start_data_transformation(self, data_injection_artifacts: DataInjectionArtifacts) -> DataTransformationArtifacts:
        logging.info("Entered the start_data_transformation method of Training pipeline class")
        try:
            data_transformation = DataTransformation(
                data_injection_artifacts=data_injection_artifacts,
                data_transformation_config = self.data_transformation_config,
            )
            
            data_transformation_artifacts = (data_transformation.initiate_data_transformation())
            logging.info("Exited the start_data_transformation method of Training pipeline class")
            return data_transformation_artifacts
            
        except Exception as e:
            raise CustomException(e, sys) from e  
        
    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("Entered the start_model_trainer method of Training pipeline class")
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifacts = (model_trainer.initiate_model_trainer())
            logging.info("Exited the start model trainer method of Training pipeline class")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e      
        
    def run_pipeline(self) -> None:
        logging.info("Entered the run pipeline method of Training pipeline class")
        try:
            self.data_injection_artifacts = self.start_data_injection()  
            
            self.data_transformation_artifacts = self.start_data_transformation(
                data_injection_artifacts=self.data_injection_artifacts
            )
            self.model_trainer_artifacts = self.start_model_trainer(
                data_transformation_artifacts=self.data_transformation_artifacts
            )
            
            
        except Exception as e:
            raise CustomException(e, sys) from e       
        
                