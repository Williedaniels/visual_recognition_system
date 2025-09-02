"""
Object Detection Model
Built from scratch for Visual Recognition System

This module implements object detection using sliding window approach
and traditional computer vision techniques, demonstrating core concepts.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import json

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image_processing import ImageProcessor
from models.image_classifier import FeatureExtractor


class BoundingBox:
    """Represents a bounding box with coordinates and confidence."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 confidence: float = 0.0, class_name: str = ""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_name = class_name
    
    @property
    def x2(self) -> int:
        """Right edge of bounding box."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge of bounding box."""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box.
        
        Args:
            other: Another bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'class_name': self.class_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        """Create from dictionary representation."""
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            confidence=data.get('confidence', 0.0),
            class_name=data.get('class_name', '')
        )


class SlidingWindowDetector:
    """Object detector using sliding window approach."""
    
    def __init__(self, window_sizes: List[Tuple[int, int]] = None, 
                 step_size: int = 16, scale_factor: float = 1.2):
        """
        Initialize the sliding window detector.
        
        Args:
            window_sizes: List of (width, height) tuples for detection windows
            step_size: Step size for sliding window
            scale_factor: Factor for multi-scale detection
        """
        self.window_sizes = window_sizes or [(64, 64), (96, 96), (128, 128)]
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        self.is_trained = False
        self.classes = None
        
    def extract_windows(self, image: np.ndarray, window_size: Tuple[int, int]) -> List[Tuple[np.ndarray, int, int]]:
        """
        Extract sliding windows from image.
        
        Args:
            image: Input image
            window_size: Size of sliding window (width, height)
            
        Returns:
            List of (window_image, x, y) tuples
        """
        windows = []
        h, w = image.shape[:2]
        win_w, win_h = window_size
        
        for y in range(0, h - win_h + 1, self.step_size):
            for x in range(0, w - win_w + 1, self.step_size):
                window = image[y:y + win_h, x:x + win_w]
                windows.append((window, x, y))
        
        return windows
    
    def create_image_pyramid(self, image: np.ndarray, min_size: Tuple[int, int] = (100, 100)) -> List[Tuple[np.ndarray, float]]:
        """
        Create image pyramid for multi-scale detection.
        
        Args:
            image: Input image
            min_size: Minimum size for pyramid levels
            
        Returns:
            List of (scaled_image, scale) tuples
        """
        pyramid = []
        scale = 1.0
        
        while True:
            # Calculate new dimensions
            new_h = int(image.shape[0] / scale)
            new_w = int(image.shape[1] / scale)
            
            # Stop if image becomes too small
            if new_h < min_size[1] or new_w < min_size[0]:
                break
            
            # Resize image
            scaled_image = cv2.resize(image, (new_w, new_h))
            pyramid.append((scaled_image, scale))
            
            # Update scale
            scale *= self.scale_factor
        
        return pyramid
    
    def prepare_training_data(self, positive_samples: List[Tuple[np.ndarray, List[BoundingBox]]], 
                            negative_samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from positive and negative samples.
        
        Args:
            positive_samples: List of (image, bounding_boxes) tuples
            negative_samples: List of negative images (no objects)
            
        Returns:
            Feature matrix and labels
        """
        print("Preparing training data...")
        
        features = []
        labels = []
        
        # Process positive samples
        for image, bboxes in positive_samples:
            for bbox in bboxes:
                # Extract object region
                obj_region = image[bbox.y:bbox.y2, bbox.x:bbox.x2]
                
                if obj_region.size > 0:
                    # Resize to standard size
                    obj_region = cv2.resize(obj_region, (64, 64))
                    
                    # Extract features
                    obj_features = self.feature_extractor.extract_all_features(obj_region)
                    features.append(obj_features)
                    labels.append(1)  # Positive class
        
        # Process negative samples
        for image in negative_samples:
            # Extract random windows as negative samples
            windows = self.extract_windows(image, (64, 64))
            
            # Sample a subset of windows
            n_samples = min(len(windows), 20)
            sampled_windows = np.random.choice(len(windows), n_samples, replace=False)
            
            for idx in sampled_windows:
                window, _, _ = windows[idx]
                window_features = self.feature_extractor.extract_all_features(window)
                features.append(window_features)
                labels.append(0)  # Negative class
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
        
        return X, y
    
    def train(self, positive_samples: List[Tuple[np.ndarray, List[BoundingBox]]], 
              negative_samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Train the object detector.
        
        Args:
            positive_samples: List of (image, bounding_boxes) tuples
            negative_samples: List of negative images
            
        Returns:
            Training results
        """
        print("Training object detector...")
        
        # Prepare training data
        X, y = self.prepare_training_data(positive_samples, negative_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train classifier
        self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        self.classes = ['background', 'object']
        
        # Evaluate
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_positive': np.sum(y == 1),
            'n_negative': np.sum(y == 0)
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5, 
                      nms_threshold: float = 0.3) -> List[BoundingBox]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Threshold for non-maximum suppression
            
        Returns:
            List of detected bounding boxes
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        detections = []
        
        # Create image pyramid for multi-scale detection
        pyramid = self.create_image_pyramid(image)
        
        for scaled_image, scale in pyramid:
            # Extract windows at current scale
            for window_size in self.window_sizes:
                windows = self.extract_windows(scaled_image, window_size)
                
                for window, x, y in windows:
                    # Extract features
                    features = self.feature_extractor.extract_all_features(window)
                    features = features.reshape(1, -1)
                    
                    # Predict
                    prediction = self.classifier.predict(features)[0]
                    confidence = self.classifier.predict_proba(features)[0][1]  # Probability of positive class
                    
                    if prediction == 1 and confidence >= confidence_threshold:
                        # Convert coordinates back to original scale
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(window_size[0] * scale)
                        orig_h = int(window_size[1] * scale)
                        
                        bbox = BoundingBox(orig_x, orig_y, orig_w, orig_h, 
                                         confidence, 'object')
                        detections.append(bbox)
        
        # Apply non-maximum suppression
        if detections:
            detections = self.non_maximum_suppression(detections, nms_threshold)
        
        return detections
    
    def non_maximum_suppression(self, detections: List[BoundingBox], 
                               threshold: float = 0.3) -> List[BoundingBox]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of detected bounding boxes
            threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of bounding boxes
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_detections = []
        
        while detections:
            # Take the detection with highest confidence
            best_detection = detections.pop(0)
            filtered_detections.append(best_detection)
            
            # Remove detections with high IoU
            remaining_detections = []
            for detection in detections:
                if best_detection.iou(detection) < threshold:
                    remaining_detections.append(detection)
            
            detections = remaining_detections
        
        return filtered_detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[BoundingBox], 
                           title: str = "Object Detections") -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detected bounding boxes
            title: Plot title
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (detection.x, detection.y), 
                         (detection.x2, detection.y2), 
                         (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(result_image, 
                         (detection.x, detection.y - label_size[1] - 10),
                         (detection.x + label_size[0], detection.y), 
                         (0, 255, 0), -1)
            
            cv2.putText(result_image, label, 
                       (detection.x, detection.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image
    
    def save_model(self, filepath: str):
        """Save the trained detector."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained detector")
        
        model_data = {
            'classifier': self.classifier,
            'window_sizes': self.window_sizes,
            'step_size': self.step_size,
            'scale_factor': self.scale_factor,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Detector saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained detector."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.window_sizes = model_data['window_sizes']
        self.step_size = model_data['step_size']
        self.scale_factor = model_data['scale_factor']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        
        print(f"Detector loaded from {filepath}")


class TemplateMatchingDetector:
    """Simple template matching detector for comparison."""
    
    def __init__(self):
        self.templates = {}
    
    def add_template(self, name: str, template: np.ndarray):
        """Add a template for detection."""
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.templates[name] = template
    
    def detect_template(self, image: np.ndarray, template_name: str, 
                       threshold: float = 0.8) -> List[BoundingBox]:
        """
        Detect template in image using template matching.
        
        Args:
            image: Input image
            template_name: Name of template to detect
            threshold: Matching threshold
            
        Returns:
            List of detected bounding boxes
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Perform template matching
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations above threshold
        locations = np.where(result >= threshold)
        
        detections = []
        h, w = template.shape
        
        for pt in zip(*locations[::-1]):  # Switch x and y
            confidence = result[pt[1], pt[0]]
            bbox = BoundingBox(pt[0], pt[1], w, h, confidence, template_name)
            detections.append(bbox)
        
        return detections


def create_sample_detection_dataset(output_dir: str, n_samples: int = 50):
    """
    Create a sample dataset for object detection testing.
    
    Args:
        output_dir: Directory to save sample images
        n_samples: Number of samples to create
        
    Returns:
        Tuple of (positive_samples, negative_samples)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    positive_samples = []
    negative_samples = []
    
    print(f"Creating sample detection dataset in {output_dir}...")
    
    # Create positive samples (images with objects)
    for i in range(n_samples):
        # Create image with objects
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Random background
        bg_color = np.random.randint(0, 100, 3)
        img[:] = bg_color
        
        # Add 1-3 objects
        n_objects = np.random.randint(1, 4)
        bboxes = []
        
        for j in range(n_objects):
            # Random object properties
            obj_size = np.random.randint(30, 80)
            x = np.random.randint(0, 300 - obj_size)
            y = np.random.randint(0, 300 - obj_size)
            color = np.random.randint(100, 255, 3).tolist()
            
            # Draw object (circle)
            center = (x + obj_size // 2, y + obj_size // 2)
            cv2.circle(img, center, obj_size // 2, color, -1)
            
            # Create bounding box
            bbox = BoundingBox(x, y, obj_size, obj_size, 1.0, 'circle')
            bboxes.append(bbox)
        
        # Save image
        img_path = os.path.join(output_dir, f"positive_{i:03d}.jpg")
        cv2.imwrite(img_path, img)
        
        positive_samples.append((img, bboxes))
    
    # Create negative samples (images without objects)
    for i in range(n_samples // 2):
        # Create image with just background patterns
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Random background with patterns
        bg_color = np.random.randint(0, 150, 3)
        img[:] = bg_color
        
        # Add some random lines/rectangles (not circular objects)
        for j in range(np.random.randint(0, 5)):
            pt1 = (np.random.randint(0, 300), np.random.randint(0, 300))
            pt2 = (np.random.randint(0, 300), np.random.randint(0, 300))
            color = np.random.randint(0, 255, 3).tolist()
            cv2.line(img, pt1, pt2, color, np.random.randint(1, 5))
        
        # Save image
        img_path = os.path.join(output_dir, f"negative_{i:03d}.jpg")
        cv2.imwrite(img_path, img)
        
        negative_samples.append(img)
    
    print(f"Created {len(positive_samples)} positive and {len(negative_samples)} negative samples")
    return positive_samples, negative_samples


def test_object_detector():
    """Test function for the object detector."""
    print("Testing Object Detector...")
    
    # Create sample dataset
    dataset_dir = "/tmp/detection_dataset"
    positive_samples, negative_samples = create_sample_detection_dataset(dataset_dir, n_samples=30)
    
    # Test sliding window detector
    print("\n--- Testing Sliding Window Detector ---")
    
    detector = SlidingWindowDetector(
        window_sizes=[(32, 32), (48, 48), (64, 64)],
        step_size=8,
        scale_factor=1.3
    )
    
    # Train detector
    results = detector.train(positive_samples, negative_samples)
    
    # Test detection on a sample image
    test_image, test_bboxes = positive_samples[0]
    detections = detector.detect_objects(test_image, confidence_threshold=0.3)
    
    print(f"Ground truth objects: {len(test_bboxes)}")
    print(f"Detected objects: {len(detections)}")
    
    # Visualize results
    result_image = detector.visualize_detections(test_image, detections)
    
    # Save visualization
    output_path = "/tmp/detection_result.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Detection result saved to {output_path}")
    
    # Save and load model
    model_path = "/tmp/object_detector.pkl"
    detector.save_model(model_path)
    
    new_detector = SlidingWindowDetector()
    new_detector.load_model(model_path)
    
    # Test loaded detector
    detections2 = new_detector.detect_objects(test_image, confidence_threshold=0.3)
    print(f"Loaded detector found {len(detections2)} objects")
    
    # Test template matching detector
    print("\n--- Testing Template Matching Detector ---")
    
    template_detector = TemplateMatchingDetector()
    
    # Create a simple template
    template = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(template, (20, 20), 15, 255, -1)
    template_detector.add_template('circle', template)
    
    # Test template detection
    template_detections = template_detector.detect_template(test_image, 'circle', threshold=0.3)
    print(f"Template matching found {len(template_detections)} objects")
    
    print("Object Detector test completed successfully!")


if __name__ == "__main__":
    test_object_detector()

