from dataclasses import dataclass

# Data injection artifacts

@dataclass
class DataInjectionArtifacts:
    dataset_path: str
    
    def to_dict(self):
        return self.__dict__
    
# Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    train_transformed_object: str 
    valid_transformed_object: str 
    test_transformed_object: str    
    classes: int 
    
    def to_dict(self):
        return self.__dict__
    
# Model training artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    
    def to_dict(self):
        return self.__dict__ 
    
# Model evaluation artifacts 
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool  
    
    def to_dict(self):
        return self.__dict__   
        
# Model pusher artifacts
@dataclass
class ModelPusherArtifacts:
    backet_name: str 
    
    def to_dict(self):
        return self.__dict__         