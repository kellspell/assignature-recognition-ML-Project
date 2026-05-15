import os 
import sys 
import torch 
from src.logger import logging
from src.exceptions import CustomException
from torchvision import datasets
from torchvision import transforms as Ts 
from src.utils.main_utils import save_objects
from src.artifacts.config_entry import DataTransformationConfig
from src.artifacts.artifects_entry import DataInjectionArtifacts, DataTransformationArtifacts

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_injection_artifacts: DataInjectionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_injection_artifects = data_injection_artifacts
        self.std = self.data_transformation_config.STD
        self.mean = self.data_transformation_config.MEAN
        self.img_size = self.data_transformation_config.IMG_SIZE
        self.degree_n = self.data_transformation_config.DEGREE_N
        self.degree_p = self.data_transformation_config.DEGREE_P
        self.train_ratio = self.data_transformation_config.TRAIN_RATIO
        self.valid_ratio = self.data_transformation_config.VALID_RATIO
    
    def get_transformation_data(self):
        try:
            logging.info('Entered the get transformation data method of data transformation class')    
            data_transform = Ts.Compose([
                Ts.Resize(size=(self.img_size, self.img_size)), # Risize img by 224 by 224
                Ts.RandomRotation(degrees=(self.degree_n, self.degree_p)), # Rotate the image +/- 20 degree
                Ts.ToTensor(), # Converting the dimention into tensor
                Ts.Normalize(self.mean, self.std), # Normalize by 3 means std of the image net, 3 channels 
            ])
            logging.info('Existed the get transformation method of the date transformation class')
            return data_transform
        except Exception as e:
                raise CustomException(e, sys) from e 
            
    def split_data(self, dataset, total_count):
        try:
            logging.info('Entered the split data method of data transformation class') 
            train_count = int(self.train_ratio * total_count)
            valid_count = int(self.valid_ratio * total_count)
            test_count = total_count - train_count - valid_count
            train_data, valid_data, test_data = torch.utils.data.random_split(dataset,(train_count, valid_count, test_count))
            logging.info('Existed the split data method of data transformation class')
            return train_data, valid_data, test_data
        except Exception as e:
                raise CustomException(e, sys) from e 
            
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
             logging.info('Entered on the initiate state data method of data transformation class')
             dataset = datasets.ImageFolder(self.data_injection_artifects.dataset_path, transform=self.get_transformation_data())
             
             total_count = len(dataset)
             logging.info(f'Total number of records: {total_count}')
             
             classes = len(os.listdir(self.data_injection_artifects.dataset_path))
             logging.info(f'Total number of classes: {classes}')
             
             train_dataset, valid_dataset, test_dataset = self.split_data(dataset, total_count)
             logging.info('Split dataset into train, valid and test')
             
             save_objects(self.data_transformation_config.TRAIN_TRANSFORMATION_OBJECT_FILE_PATH, train_dataset)
             save_objects(self.data_transformation_config.VALID_TRANSFORMATION_OBJECT_FILE_PATH, valid_dataset)
             save_objects(self.data_transformation_config.TEST_TRANSFORMATION_OBJECT_FILE_PATH, test_dataset)
             logging.info('Saved the train, valid and test transformed objects')
             
             data_transformation_artifacts = DataTransformationArtifacts(
                 train_transformed_object=self.data_transformation_config.TRAIN_TRANSFORMATION_OBJECT_FILE_PATH,
                 valid_transformed_object=self.data_transformation_config.VALID_TRANSFORMATION_OBJECT_FILE_PATH,
                 test_transformed_object=self.data_transformation_config.TEST_TRANSFORMATION_OBJECT_FILE_PATH,
                 classes=classes
             )
             logging.info('Existed the initiate data transformation method of data transformation class')
             return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
             
                    
        
            
                