import os
import sys
import torch
from PIL import Image
from src.constants import CONFIG_PATH, DEVICE, LABEL_NAME
from src.logger import logging
from torchvision import transforms
from src.exceptions import CustomException
from src.utils.main_utils import read_yaml_file
from src.configurations.gcloud_syncer import GCloudSync

class PredictionPipeline:
    def __init__(self):
        self.gcloud = GCloudSync()
        self.config = read_yaml_file(CONFIG_PATH)
        self.img_size = self.config['data_transformation_config']['img_size']
        
    def image_loader(self, image_bytes):
        logging.info('Entered the image_loader method of prediction Pipeline class')
        try:
            logging.info('Load bytes image and save it to local')
            input_image = self.config['prediction_pipeline_config']['input_image']
            with open(input_image, 'wb' ) as image:
                image.write(image_bytes)
                image.close()
            path = os.path.join(os.getcwd(), input_image)
            image = Image.open(path)
            logging.info(f'Returns the saved image {image}')
            logging.info('Exited the image loader method of prediction pipeline')
            return image 
        except Exception as e:
            raise CustomException(e, sys) from e   
        
    def get_model_from_gcloud(self) -> str:
        logging.info('Entered the on get model from google cloud method of prediction Pipeline class')
        try:
            logging.info('Loading the best model from google cloud ')
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel")
            self.gcloud.sync_file_from_gcloud(self.config['prediction_pipeline_config']['bucket_name'], 
                                              self.config['prediction_pipeline_config']['model_name'], 
                                              predict_model_path)
            best_model_path = os.path.join(predict_model_path, self.config['prediction_pipeline_config']['model_name'])
            logging.info('Exited the get model from google cloud of predicted pipeline')
            return best_model_path
            
        except Exception as e:
            raise CustomException(e, sys) from e   
        
    def prediction(self, best_model_path: str, image) -> str:
        logging.info('Enterinf on the prediction method')
        try:
            logging.info('Loading best model')
            model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
            model.eval()
            logging.info('Load the image and preprocvess it')
            preprocess = transforms.Compose([
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.Grayscale(3),
                transforms.ToTensor()
            ])  
            image = preprocess(image)
            image = image[:3]
            
            logging.info('Converting the image to pytorch tensor and send it to device')
            image = image.unsqueeze(0).to(DEVICE)
            
            logging.info('Making predictions!!!')
            with torch.no_grad():
                logits = model(image)
                probs = torch.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1)
            
            logging.info(f'Predicted label: {pred_label.item()}')
            logging.info('Map the predicted label to the corresponding class name')
            predicted_class_name = LABEL_NAME[int(pred_label.item())]
            logging.info(f'Predicted class name: {predicted_class_name}')
            logging.info('Exiting the prediction method')
            return predicted_class_name
        except Exception as e:
            raise CustomException(e, sys) from e
        
        
        
    def run_pipeline(self, data):
        logging.info('Start to run the pipeline predition')
        try:
            image = self.image_loader(data)
            best_model_path: str = self.get_model_from_gcloud()
            detected_image = self.prediction(best_model_path, image)
            logging.info('Exiting the run pipeline method of Prediction pipeline')
            return detected_image
        except Exception as e:
            raise CustomException(e, sys) from e    
            
            
            
                
                
               
                