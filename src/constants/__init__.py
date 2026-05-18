import os 
import torch 
from datetime import datetime

# Common constant
use_cuda = torch.cuda.is_available()
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
DEVICE = torch.device("cuda:0" if use_cuda else "cpu")

# Outbound Connections 
APP_HOST = "0.0.0.0"
APP_PORT: int = 8080

# Data injection constants
DATA_INJECTION_ARTIFACTS_DIR = 'DataIngectionArtifacts'
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = 'train_transformed.pkl'
DATA_TRANSFORMATION_VALID_FILE_NAME = 'valid_transformed.pkl'
DATA_TRANSFORMATION_TEST_FILE_NAME = 'test_transformed.pkl'             

# model trainer constant
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_PATH = 'model.pt'

# Model Evaluation Constant
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = 'best_model'
MODEL_NAME ='model.pt'

# Prediction Pipeline 
LABEL_NAME = ['Forged', 'Original']
