import os
import sys 
import torch
import torch.nn as nn
from tqdm import tqdm
from src.constants import DEVICE
from torchvision import models 
from src.logger import logging
from src.exceptions import CustomException
from torch.utils.data import DataLoader
from src.utils.main_utils import load_objects 
from src.artifacts.config_entry import ModelTrainerConfig
from src.artifacts.artifects_entry import DataTransformationArtifacts, ModelTrainerArtifacts

class ModelTrainer:
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.learning_rate = self.model_trainer_config.LR
        self.epochs = self.model_trainer_config.EPOCHS
        self.batch_size = self.model_trainer_config.BATCH_SIZE
        self.num_workers = self.model_trainer_config.NUM_WORKERS
        
    def train(self, model, criterion, optimizer, train_dataloader, valid_dataloader):
        try:
            total_train_loss = 0
            total_test_loss = 0
            
            model.train()
            with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
                pbar.set_description(f'training...')
                for images, idxs in pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    idxs = idxs.to(DEVICE, non_blocking=True)
                    output = model(images)
                    
                    loss = criterion(output, idxs)
                    total_train_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
            model.eval()
            with torch.no_grad():
                with tqdm(valid_dataloader, unit='batch', leave=False) as pbar:
                    pbar.set_description(f'testing...')
                    for images, idxs in pbar:
                        images = images.to(DEVICE, non_blocking=True)
                        idxs = idxs.to(DEVICE, non_blocking=True)
                        output = model(images)

                        loss = criterion(output, idxs)
                        total_test_loss += loss.item()

            train_loss = total_train_loss / len(train_dataloader)
            valid_loss = total_test_loss / len(valid_dataloader)
            print(f"Train loss: {train_loss:.4f} Test loss: {valid_loss:.4f}  "
                  f"[raw: {total_train_loss:.4f}/{len(train_dataloader)} batches, "
                  f"{total_test_loss:.4f}/{len(valid_dataloader)} batches]")
        
        except Exception as e:
            raise CustomException(e, sys) from e  
        
    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info('Enterd the initiate model trainer method of model training class') 
            train_dataset = load_objects(self.data_transformation_artifacts.train_transformed_object) 
            valid_dataset = load_objects(self.data_transformation_artifacts.valid_transformed_object) 
            
            logging.info('Loaded dataset from data transformation artifects')
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers) 
            valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers) 
            logging.info('Loaded train and valid data loader')
            
            model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
            logging.info('Loaded pre-trained resnet34 model')
            
            model.fc = nn.Sequential(  # type: ignore[assignment]
                nn.Dropout(0.1),
                nn.Linear(model.fc.in_features, self.data_transformation_artifacts.classes)
            )
            logging.info('Updated the last layer of the pretrained model')
            
            model = model.to(DEVICE)
            
            criterion = torch.nn.CrossEntropyLoss()
            logging.info('Cross entropy loss function is used')
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
            logging.info('SGD optimizer is used')
            
            logging.info('Model trained started! ')
            for i in range(self.epochs):
                logging.info(f'Model training at epochs: {i+1}')
                print(f"Epoch {i + 1} / {self.epochs}")
                self.train(model, criterion, optimizer, train_loader, valid_loader)
            logging.info('Model trained Done! ')

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f'saved trained model at {self.model_trainer_config.TRAINED_MODEL_PATH}')

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f'Model trainer artifacts: {model_trainer_artifacts}')
            return model_trainer_artifacts
        except Exception as e:
                raise CustomException(e, sys) from e    
                        
                        