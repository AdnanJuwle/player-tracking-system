import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2
from typing import List, Dict

class ReIDModel(nn.Module):
    def __init__(self, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove classification layer
        self.model.eval()
        self.model.to(device)
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 128)),  # Standard ReID input size
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a player crop"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features.cpu().numpy().flatten()
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(features1, features2) / (norm1 * norm2)
    
    def batch_extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features from multiple crops efficiently"""
        batch = torch.stack([self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) 
                            for crop in crops]).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        return features.cpu().numpy()
