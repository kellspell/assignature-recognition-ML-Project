from dataclasses import dataclass

# Data injection artifacts

@dataclass
class DataInjectionArtifacts:
    dataset_path: str
    
    def to_dict(self):
        return self.__dict__