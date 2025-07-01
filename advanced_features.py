import cv2
import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with deep learning capabilities
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_color_features(self, image):
        """Extract comprehensive color features"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract histograms from different color spaces
        features = []
        
        # BGR histograms
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # HSV histograms
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # LAB histograms  
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features)
    
    def extract_texture_features(self, image):
        """Extract texture features using LBP and Gabor filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Local Binary Pattern (simplified version)
        lbp = self.local_binary_pattern(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        features.extend(lbp_hist.flatten())
        
        # Gabor filters
        gabor_responses = self.apply_gabor_filters(gray)
        features.extend(gabor_responses)
        
        return np.array(features)
    
    def local_binary_pattern(self, image, radius=1, n_points=8):
        """Simplified LBP implementation"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                code = 0
                
                # Sample points around the center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if 0 <= x < rows and 0 <= y < cols:
                        if image[x, y] >= center:
                            code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def apply_gabor_filters(self, image):
        """Apply Gabor filters for texture analysis"""
        features = []
        
        # Different orientations and frequencies
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.3]
        
        for orientation in orientations:
            for frequency in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 3, np.radians(orientation), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
        
        return features
    
    def extract_spatial_features(self, bbox, frame_shape):
        """Extract spatial position features"""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]
        
        # Normalize coordinates
        center_x = (x1 + x2) / (2 * frame_width)
        center_y = (y1 + y2) / (2 * frame_height)
        width_ratio = (x2 - x1) / frame_width
        height_ratio = (y2 - y1) / frame_height
        aspect_ratio = (x2 - x1) / (y2 - y1)
        
        return np.array([center_x, center_y, width_ratio, height_ratio, aspect_ratio])
