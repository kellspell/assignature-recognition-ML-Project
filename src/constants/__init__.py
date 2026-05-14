import os 
import torch 
from datetime import datetime

# Common constant
use_cuda = torch.cuda.is_available()
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
DEVIDE = torch.device("cuda:0" if use_cuda else "cpu")

# Data injection constants
DATA_INJECTION_ARTIFACTS_DIR = 'DataIngectionArtifacts'

            