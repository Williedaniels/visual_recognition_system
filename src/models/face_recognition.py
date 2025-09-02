"""
Face Recognition and Advanced Visual Features
Built from scratch for Visual Recognition System

This module implements face detection, recognition, and advanced visual analysis
using traditional computer vision techniques and machine learning.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image_processing import ImageProcessor
from models.object_detector import BoundingBox


class FaceDetector:
    """Face detection using OpenCV cascade classifiers."""
    
    def __init__(self):
        """Initialize face detector with cascade classifiers."""
        self.processor = ImageProcessor()
        
        # Load pre-trained cascade classifiers
        try:
            # Try to load face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                self.face_cascade = None
                
        except Exception as e:
            print(f"Error loading cascade classifiers: {e}")
            self.face_cascade = None
            self.eye_cascade = None
            self.smile_cascade = None
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[BoundingBox]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image
            scale_factor: Scale factor for detection
            min_neighbors: Minimum neighbors for detection
            min_size: Minimum face size
            
        Returns:
            List of face bounding boxes
        """
        if self.face_cascade is None:
            return self._detect_faces_alternative(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        
        # Convert to BoundingBox objects
        face_boxes = []
        for (x, y, w, h) in faces:
            bbox = BoundingBox(x, y, w, h, 1.0, 'face')
            face_boxes.append(bbox)
        
        return face_boxes
    
    def _detect_faces_alternative(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Alternative face detection using simple heuristics.
        
        Args:
            image: Input image
            
        Returns:
            List of potential face regions
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        face_boxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size and aspect ratio (face-like characteristics)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if (0.7 <= aspect_ratio <= 1.3 and  # Roughly square
                area > 1000 and  # Minimum size
                w > 30 and h > 30):  # Minimum dimensions
                
                bbox = BoundingBox(x, y, w, h, 0.5, 'face_candidate')
                face_boxes.append(bbox)
        
        return face_boxes
    
    def detect_facial_features(self, image: np.ndarray, face_bbox: BoundingBox) -> Dict[str, List[BoundingBox]]:
        """
        Detect facial features within a face region.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Dictionary of detected features
        """
        features = {'eyes': [], 'smiles': []}
        
        if self.eye_cascade is None or self.smile_cascade is None:
            return features
        
        # Extract face region
        face_region = image[face_bbox.y:face_bbox.y2, face_bbox.x:face_bbox.x2]
        
        if face_region.size == 0:
            return features
        
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            # Convert coordinates to original image space
            abs_x = face_bbox.x + ex
            abs_y = face_bbox.y + ey
            eye_bbox = BoundingBox(abs_x, abs_y, ew, eh, 1.0, 'eye')
            features['eyes'].append(eye_bbox)
        
        # Detect smiles
        smiles = self.smile_cascade.detectMultiScale(gray_face, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            # Convert coordinates to original image space
            abs_x = face_bbox.x + sx
            abs_y = face_bbox.y + sy
            smile_bbox = BoundingBox(abs_x, abs_y, sw, sh, 1.0, 'smile')
            features['smiles'].append(smile_bbox)
        
        return features
    
    def extract_face_encoding(self, image: np.ndarray, face_bbox: BoundingBox) -> np.ndarray:
        """
        Extract face encoding/features for recognition.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Face feature vector
        """
        # Extract face region
        face_region = image[face_bbox.y:face_bbox.y2, face_bbox.x:face_bbox.x2]
        
        if face_region.size == 0:
            return np.array([])
        
        # Resize to standard size
        face_resized = cv2.resize(face_region, (64, 64))
        
        # Convert to grayscale
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
        
        # Extract features using multiple methods
        features = []
        
        # 1. Raw pixel values (normalized)
        pixel_features = face_gray.flatten() / 255.0
        features.extend(pixel_features)
        
        # 2. Histogram features
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        hist_features = hist.flatten() / (np.sum(hist) + 1e-7)
        features.extend(hist_features)
        
        # 3. LBP-like texture features
        # Simple gradient-based texture
        grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 75)
        ]
        features.extend(texture_features)
        
        return np.array(features)


class FaceRecognizer:
    """Face recognition system using traditional ML approaches."""
    
    def __init__(self, method: str = 'knn'):
        """
        Initialize face recognizer.
        
        Args:
            method: Recognition method ('knn', 'svm', 'pca')
        """
        self.method = method
        self.face_detector = FaceDetector()
        self.classifier = None
        self.scaler = StandardScaler()
        self.pca = None
        self.face_encodings = []
        self.face_labels = []
        self.label_to_name = {}
        self.is_trained = False
        
        # Initialize classifier
        if method == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=3)
        elif method == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True)
        elif method == 'pca':
            self.pca = PCA(n_components=50)
            self.classifier = KNeighborsClassifier(n_neighbors=3)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def add_person(self, name: str, images: List[np.ndarray]) -> int:
        """
        Add a person to the recognition database.
        
        Args:
            name: Person's name
            images: List of face images
            
        Returns:
            Number of face encodings added
        """
        print(f"Adding person: {name}")
        
        encodings_added = 0
        
        for i, image in enumerate(images):
            # Detect faces in image
            faces = self.face_detector.detect_faces(image)
            
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f.area)
                
                # Extract face encoding
                encoding = self.face_detector.extract_face_encoding(image, largest_face)
                
                if len(encoding) > 0:
                    self.face_encodings.append(encoding)
                    self.face_labels.append(name)
                    encodings_added += 1
                    
                    print(f"  Added encoding {encodings_added} for {name}")
            else:
                print(f"  No face detected in image {i+1}")
        
        # Update label mapping
        unique_labels = list(set(self.face_labels))
        self.label_to_name = {i: label for i, label in enumerate(unique_labels)}
        
        print(f"Added {encodings_added} encodings for {name}")
        return encodings_added
    
    def train(self) -> Dict[str, Any]:
        """
        Train the face recognition model.
        
        Returns:
            Training results
        """
        if not self.face_encodings:
            raise ValueError("No face encodings available for training")
        
        print("Training face recognition model...")
        
        X = np.array(self.face_encodings)
        y = np.array(self.face_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if specified
        if self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_predictions = self.classifier.predict(X_scaled)
        train_accuracy = np.mean(train_predictions == y)
        
        results = {
            'train_accuracy': train_accuracy,
            'n_samples': len(X),
            'n_features': X_scaled.shape[1],
            'n_people': len(set(y))
        }
        
        print(f"Training completed!")
        print(f"Samples: {results['n_samples']}, Features: {results['n_features']}")
        print(f"People: {results['n_people']}, Accuracy: {train_accuracy:.4f}")
        
        return results
    
    def recognize_face(self, image: np.ndarray, face_bbox: BoundingBox) -> Tuple[str, float]:
        """
        Recognize a face in an image.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Tuple of (person_name, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before recognition")
        
        # Extract face encoding
        encoding = self.face_detector.extract_face_encoding(image, face_bbox)
        
        if len(encoding) == 0:
            return "unknown", 0.0
        
        # Preprocess encoding
        encoding_scaled = self.scaler.transform(encoding.reshape(1, -1))
        
        if self.pca is not None:
            encoding_scaled = self.pca.transform(encoding_scaled)
        
        # Make prediction
        prediction = self.classifier.predict(encoding_scaled)[0]
        
        # Get confidence
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(encoding_scaled)[0]
            confidence = np.max(probabilities)
        else:
            # For KNN, use distance-based confidence
            distances, indices = self.classifier.kneighbors(encoding_scaled)
            confidence = 1.0 / (1.0 + np.mean(distances))
        
        return prediction, confidence
    
    def recognize_faces_in_image(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Recognize all faces in an image.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for recognition
            
        Returns:
            List of recognition results
        """
        results = []
        
        # Detect faces
        faces = self.face_detector.detect_faces(image)
        
        for face_bbox in faces:
            # Recognize face
            name, confidence = self.recognize_face(image, face_bbox)
            
            result = {
                'bbox': face_bbox,
                'name': name if confidence >= confidence_threshold else 'unknown',
                'confidence': confidence,
                'raw_name': name
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'method': self.method,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'pca': self.pca,
            'face_encodings': self.face_encodings,
            'face_labels': self.face_labels,
            'label_to_name': self.label_to_name,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Face recognition model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.face_encodings = model_data['face_encodings']
        self.face_labels = model_data['face_labels']
        self.label_to_name = model_data['label_to_name']
        self.is_trained = model_data['is_trained']
        
        print(f"Face recognition model loaded from {filepath}")


class AdvancedVisualAnalyzer:
    """Advanced visual analysis features."""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    def calculate_image_similarity(self, image1: np.ndarray, image2: np.ndarray, 
                                 method: str = 'histogram') -> float:
        """
        Calculate similarity between two images.
        
        Args:
            image1: First image
            image2: Second image
            method: Similarity method ('histogram', 'structural', 'features')
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if method == 'histogram':
            return self._histogram_similarity(image1, image2)
        elif method == 'structural':
            return self._structural_similarity(image1, image2)
        elif method == 'features':
            return self._feature_similarity(image1, image2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _histogram_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate histogram-based similarity."""
        # Resize images to same size
        size = (256, 256)
        img1_resized = cv2.resize(image1, size)
        img2_resized = cv2.resize(image2, size)
        
        # Calculate histograms
        hist1 = self.processor.compute_histogram(img1_resized, bins=64)
        hist2 = self.processor.compute_histogram(img2_resized, bins=64)
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1.astype(np.float32), 
                                    hist2.astype(np.float32), 
                                    cv2.HISTCMP_CORREL)
        
        return max(0, correlation)  # Ensure non-negative
    
    def _structural_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate structural similarity (simplified SSIM)."""
        # Convert to grayscale and resize
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # Resize to same size
        size = (256, 256)
        gray1 = cv2.resize(gray1, size)
        gray2 = cv2.resize(gray2, size)
        
        # Convert to float
        img1_f = gray1.astype(np.float64)
        img2_f = gray2.astype(np.float64)
        
        # Calculate means
        mu1 = np.mean(img1_f)
        mu2 = np.mean(img2_f)
        
        # Calculate variances and covariance
        var1 = np.var(img1_f)
        var2 = np.var(img2_f)
        cov = np.mean((img1_f - mu1) * (img2_f - mu2))
        
        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        
        ssim = numerator / denominator if denominator > 0 else 0
        return max(0, min(1, ssim))  # Clamp to [0, 1]
    
    def _feature_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate feature-based similarity using keypoints."""
        # Extract features
        kp1, desc1 = self.processor.extract_orb_features(image1, n_features=100)
        kp2, desc2 = self.processor.extract_orb_features(image2, n_features=100)
        
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Calculate similarity based on matches
        if len(matches) == 0:
            return 0.0
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate similarity score
        good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
        similarity = len(good_matches) / max(len(desc1), len(desc2))
        
        return min(1.0, similarity)
    
    def analyze_scene_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze scene composition and visual properties.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of scene analysis results
        """
        analysis = {}
        
        # Basic image properties
        h, w = image.shape[:2]
        analysis['dimensions'] = {'width': w, 'height': h, 'aspect_ratio': w/h}
        
        # Color analysis
        if len(image.shape) == 3:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Dominant colors (simplified)
            pixels = image.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_color = unique_colors[dominant_idx].tolist()
            
            # Color statistics
            analysis['color'] = {
                'dominant_color_bgr': dominant_color,
                'mean_hue': np.mean(hsv[:, :, 0]),
                'mean_saturation': np.mean(hsv[:, :, 1]),
                'mean_brightness': np.mean(hsv[:, :, 2]),
                'color_variance': np.var(pixels)
            }
        
        # Texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture measures
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        analysis['texture'] = {
            'edge_density': edge_density,
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'roughness': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        }
        
        # Composition analysis
        # Rule of thirds analysis
        third_h, third_w = h // 3, w // 3
        
        # Divide image into 9 regions
        regions = []
        for i in range(3):
            for j in range(3):
                region = gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                regions.append(np.mean(region))
        
        analysis['composition'] = {
            'rule_of_thirds_variance': np.var(regions),
            'center_brightness': regions[4],  # Center region
            'corner_brightness': np.mean([regions[0], regions[2], regions[6], regions[8]]),
            'brightness_balance': np.std(regions)
        }
        
        # Complexity analysis
        # Calculate image entropy as a measure of complexity
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        analysis['complexity'] = {
            'entropy': entropy,
            'unique_intensities': len(np.unique(gray)),
            'contrast': np.max(gray) - np.min(gray)
        }
        
        return analysis
    
    def detect_image_quality_issues(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect common image quality issues.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of quality assessment results
        """
        issues = {}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        issues['blur'] = {
            'laplacian_variance': laplacian_var,
            'is_blurry': laplacian_var < 100,  # Threshold for blur detection
            'blur_level': 'high' if laplacian_var < 50 else 'medium' if laplacian_var < 100 else 'low'
        }
        
        # Brightness analysis
        mean_brightness = np.mean(gray)
        issues['brightness'] = {
            'mean_brightness': mean_brightness,
            'is_too_dark': mean_brightness < 50,
            'is_too_bright': mean_brightness > 200,
            'brightness_level': 'dark' if mean_brightness < 85 else 'bright' if mean_brightness > 170 else 'normal'
        }
        
        # Contrast analysis
        contrast = np.std(gray)
        issues['contrast'] = {
            'contrast_std': contrast,
            'is_low_contrast': contrast < 30,
            'contrast_level': 'low' if contrast < 30 else 'high' if contrast > 80 else 'normal'
        }
        
        # Noise estimation (simplified)
        # Use high-frequency content as noise indicator
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_response = cv2.filter2D(gray, -1, kernel)
        noise_level = np.std(noise_response)
        
        issues['noise'] = {
            'noise_level': noise_level,
            'is_noisy': noise_level > 20,
            'noise_category': 'high' if noise_level > 30 else 'medium' if noise_level > 15 else 'low'
        }
        
        # Overall quality score (0-1, higher is better)
        quality_factors = [
            1.0 if not issues['blur']['is_blurry'] else 0.3,
            1.0 if not issues['brightness']['is_too_dark'] and not issues['brightness']['is_too_bright'] else 0.5,
            1.0 if not issues['contrast']['is_low_contrast'] else 0.4,
            1.0 if not issues['noise']['is_noisy'] else 0.6
        ]
        
        issues['overall_quality'] = {
            'score': np.mean(quality_factors),
            'grade': 'excellent' if np.mean(quality_factors) > 0.9 else 
                    'good' if np.mean(quality_factors) > 0.7 else
                    'fair' if np.mean(quality_factors) > 0.5 else 'poor'
        }
        
        return issues


def create_sample_face_dataset(output_dir: str, n_people: int = 3, n_images_per_person: int = 5):
    """
    Create a sample face dataset for testing.
    
    Args:
        output_dir: Directory to save sample images
        n_people: Number of different people
        n_images_per_person: Number of images per person
        
    Returns:
        Dictionary mapping person names to image lists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = {}
    
    print(f"Creating sample face dataset in {output_dir}...")
    
    for person_id in range(n_people):
        person_name = f"person_{person_id + 1}"
        person_images = []
        
        person_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        for img_id in range(n_images_per_person):
            # Create synthetic face-like image
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Background
            bg_color = np.random.randint(50, 150, 3)
            img[:] = bg_color
            
            # Face (ellipse)
            center = (100, 100)
            axes = (60 + person_id * 5, 80 + person_id * 3)  # Slightly different for each person
            face_color = np.random.randint(150, 220, 3).tolist()
            cv2.ellipse(img, center, axes, 0, 0, 360, face_color, -1)
            
            # Eyes
            eye_color = [0, 0, 0]  # Black eyes
            cv2.circle(img, (80, 85), 8, eye_color, -1)
            cv2.circle(img, (120, 85), 8, eye_color, -1)
            
            # Nose (small triangle)
            nose_pts = np.array([[100, 100], [95, 110], [105, 110]], np.int32)
            cv2.fillPoly(img, [nose_pts], [100, 100, 100])
            
            # Mouth
            cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, [50, 50, 50], 2)
            
            # Add some variation
            if img_id > 0:
                # Slight rotation
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (200, 200))
            
            # Add noise
            noise = np.random.randint(-20, 20, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            filename = f"{person_name}_{img_id:02d}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, img)
            
            person_images.append(img)
        
        dataset[person_name] = person_images
        print(f"Created {len(person_images)} images for {person_name}")
    
    return dataset


def test_face_recognition():
    """Test function for face recognition system."""
    print("Testing Face Recognition System...")
    
    # Test face detector
    print("\n--- Testing Face Detector ---")
    detector = FaceDetector()
    
    # Create a test image with face-like structure
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    test_img[:] = [100, 100, 100]  # Gray background
    
    # Draw a face-like structure
    cv2.ellipse(test_img, (150, 150), (60, 80), 0, 0, 360, (200, 180, 160), -1)  # Face
    cv2.circle(test_img, (130, 130), 8, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_img, (170, 130), 8, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(test_img, (150, 180), (15, 8), 0, 0, 180, (50, 50, 50), 2)  # Mouth
    
    faces = detector.detect_faces(test_img)
    print(f"Detected {len(faces)} faces in test image")
    
    # Test face recognition
    print("\n--- Testing Face Recognition ---")
    
    # Create sample dataset
    dataset_dir = "/tmp/face_dataset"
    face_dataset = create_sample_face_dataset(dataset_dir, n_people=3, n_images_per_person=4)
    
    # Test different recognition methods
    methods = ['knn', 'svm']
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} Face Recognition ---")
        
        recognizer = FaceRecognizer(method=method)
        
        # Add people to database
        for person_name, images in face_dataset.items():
            # Use first 3 images for training, last 1 for testing
            train_images = images[:3]
            recognizer.add_person(person_name, train_images)
        
        # Train recognizer
        if recognizer.face_encodings:
            results = recognizer.train()
            
            # Test recognition on remaining images
            correct_predictions = 0
            total_predictions = 0
            
            for person_name, images in face_dataset.items():
                test_image = images[-1]  # Last image for testing
                
                recognition_results = recognizer.recognize_faces_in_image(test_image)
                
                for result in recognition_results:
                    predicted_name = result['name']
                    confidence = result['confidence']
                    
                    print(f"  True: {person_name}, Predicted: {predicted_name}, Confidence: {confidence:.3f}")
                    
                    if predicted_name == person_name:
                        correct_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"  Recognition accuracy: {accuracy:.3f}")
            
            # Save and load model
            model_path = f"/tmp/face_recognizer_{method}.pkl"
            recognizer.save_model(model_path)
            
            new_recognizer = FaceRecognizer(method=method)
            new_recognizer.load_model(model_path)
            print(f"  Model saved and loaded successfully")
    
    # Test advanced visual analyzer
    print("\n--- Testing Advanced Visual Analyzer ---")
    
    analyzer = AdvancedVisualAnalyzer()
    
    # Test image similarity
    img1 = face_dataset['person_1'][0]
    img2 = face_dataset['person_1'][1]  # Same person
    img3 = face_dataset['person_2'][0]  # Different person
    
    similarity_same = analyzer.calculate_image_similarity(img1, img2, method='histogram')
    similarity_diff = analyzer.calculate_image_similarity(img1, img3, method='histogram')
    
    print(f"Similarity (same person): {similarity_same:.3f}")
    print(f"Similarity (different person): {similarity_diff:.3f}")
    
    # Test scene analysis
    scene_analysis = analyzer.analyze_scene_composition(test_img)
    print(f"Scene entropy: {scene_analysis['complexity']['entropy']:.3f}")
    print(f"Edge density: {scene_analysis['texture']['edge_density']:.3f}")
    
    # Test quality assessment
    quality_issues = analyzer.detect_image_quality_issues(test_img)
    print(f"Image quality grade: {quality_issues['overall_quality']['grade']}")
    print(f"Quality score: {quality_issues['overall_quality']['score']:.3f}")
    
    print("\nFace Recognition System test completed successfully!")


if __name__ == "__main__":
    test_face_recognition()

