"""
Image Classification Model
Built from scratch for Visual Recognition System

This module implements image classification using traditional computer vision
techniques and simple neural networks, demonstrating core concepts.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.image_processing import ImageProcessor


class FeatureExtractor:
    """Extract various features from images for classification."""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            image: Input image
            bins: Number of histogram bins per channel
            
        Returns:
            Flattened histogram features
        """
        if len(image.shape) == 3:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
            
            # Concatenate histograms
            features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [bins * 3], [0, 256])
            features = hist.flatten()
        
        # Normalize
        features = features / (np.sum(features) + 1e-7)
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns (LBP) approximation.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Texture feature vector
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP-like texture analysis
        # Calculate gradients in different directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Create texture features
        features = []
        
        # Gradient magnitude statistics
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 75)
        ])
        
        # Gradient direction histogram
        direction_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        direction_hist = direction_hist / (np.sum(direction_hist) + 1e-7)
        features.extend(direction_hist)
        
        # Edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return np.array(features)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shape features from image.
        
        Args:
            image: Input image
            
        Returns:
            Shape feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Extent (ratio of contour area to bounding rectangle area)
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Compactness
            compactness = (perimeter ** 2) / area if area > 0 else 0
            
            features = [area, perimeter, aspect_ratio, extent, solidity, compactness]
        else:
            features = [0, 0, 0, 0, 0, 0]
        
        return np.array(features)
    
    def extract_keypoint_features(self, image: np.ndarray, method: str = 'orb') -> np.ndarray:
        """
        Extract keypoint-based features.
        
        Args:
            image: Input image
            method: Feature extraction method ('orb', 'sift')
            
        Returns:
            Aggregated keypoint features
        """
        if method == 'orb':
            keypoints, descriptors = self.processor.extract_orb_features(image, n_features=100)
        else:
            keypoints, descriptors = self.processor.extract_sift_features(image)
        
        if descriptors is not None and len(descriptors) > 0:
            # Aggregate descriptors using statistics
            features = []
            features.extend(np.mean(descriptors, axis=0))
            features.extend(np.std(descriptors, axis=0))
            features.extend(np.max(descriptors, axis=0))
            features.extend(np.min(descriptors, axis=0))
            
            # Add number of keypoints
            features.append(len(keypoints))
            
            return np.array(features)
        else:
            # Return zero features if no keypoints found
            feature_size = 32 * 4 + 1 if method == 'orb' else 128 * 4 + 1
            return np.zeros(feature_size)
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all types of features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Combined feature vector
        """
        # Resize image for consistent feature extraction
        image_resized = cv2.resize(image, (224, 224))
        
        # Extract different types of features
        color_features = self.extract_color_histogram(image_resized)
        texture_features = self.extract_texture_features(image_resized)
        shape_features = self.extract_shape_features(image_resized)
        keypoint_features = self.extract_keypoint_features(image_resized)
        
        # Combine all features
        all_features = np.concatenate([
            color_features,
            texture_features,
            shape_features,
            keypoint_features
        ])
        
        return all_features


class ImageClassifier:
    """Image classifier using traditional machine learning approaches."""
    
    def __init__(self, classifier_type: str = 'svm'):
        """
        Initialize the image classifier.
        
        Args:
            classifier_type: Type of classifier ('svm', 'rf', 'knn')
        """
        self.classifier_type = classifier_type
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        self.classes = None
        self.is_trained = False
        
        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def prepare_dataset(self, image_paths: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset by extracting features from images.
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            
        Returns:
            Feature matrix and label array
        """
        print(f"Extracting features from {len(image_paths)} images...")
        
        features_list = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_all_features(image)
                features_list.append(features)
                valid_labels.append(label)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid images found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the image classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results dictionary
        """
        print(f"Training {self.classifier_type.upper()} classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Store classes
        self.classes = np.unique(y)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)
        
        # Predictions for detailed evaluation
        y_pred = self.classifier.predict(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_predictions': y_pred,
            'test_labels': y_test,
            'classes': self.classes
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict the class of an image.
        
        Args:
            image: Input image
            
        Returns:
            Predicted class and confidence
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(image)
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.classifier.predict(features)[0]
        
        # Get confidence if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return prediction, confidence
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        for image in images:
            prediction, confidence = self.predict(image)
            results.append((prediction, confidence))
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.classifier_type = model_data['classifier_type']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def visualize_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualize classification results.
        
        Args:
            results: Results dictionary from training
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=results['classes'], 
                   yticklabels=results['classes'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Accuracy by Class
        report = results['classification_report']
        classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[0, 1].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[0, 1].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[0, 1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Classification Metrics by Class')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overall Metrics
        overall_metrics = ['precision', 'recall', 'f1-score']
        macro_scores = [report['macro avg'][metric] for metric in overall_metrics]
        weighted_scores = [report['weighted avg'][metric] for metric in overall_metrics]
        
        x = np.arange(len(overall_metrics))
        axes[1, 0].bar(x - 0.2, macro_scores, 0.4, label='Macro Average', alpha=0.8)
        axes[1, 0].bar(x + 0.2, weighted_scores, 0.4, label='Weighted Average', alpha=0.8)
        
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Overall Performance Metrics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(overall_metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training vs Test Accuracy
        accuracies = ['Training', 'Testing']
        accuracy_values = [results['train_accuracy'], results['test_accuracy']]
        
        bars = axes[1, 1].bar(accuracies, accuracy_values, alpha=0.8, 
                             color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Training vs Testing Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracy_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results visualization saved to {save_path}")
        
        plt.show()


def create_sample_dataset(output_dir: str, n_samples_per_class: int = 50):
    """
    Create a sample dataset with synthetic images for testing.
    
    Args:
        output_dir: Directory to save the sample images
        n_samples_per_class: Number of samples per class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    classes = ['circles', 'rectangles', 'triangles']
    image_paths = []
    labels = []
    
    print(f"Creating sample dataset in {output_dir}...")
    
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(n_samples_per_class):
            # Create synthetic image
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Add random background
            bg_color = np.random.randint(0, 100, 3)
            img[:] = bg_color
            
            # Add shape based on class
            if class_name == 'circles':
                center = (np.random.randint(50, 150), np.random.randint(50, 150))
                radius = np.random.randint(20, 40)
                color = np.random.randint(100, 255, 3).tolist()
                cv2.circle(img, center, radius, color, -1)
                
            elif class_name == 'rectangles':
                pt1 = (np.random.randint(20, 100), np.random.randint(20, 100))
                pt2 = (pt1[0] + np.random.randint(40, 80), pt1[1] + np.random.randint(40, 80))
                color = np.random.randint(100, 255, 3).tolist()
                cv2.rectangle(img, pt1, pt2, color, -1)
                
            elif class_name == 'triangles':
                pts = np.array([
                    [np.random.randint(50, 150), np.random.randint(30, 70)],
                    [np.random.randint(30, 90), np.random.randint(130, 170)],
                    [np.random.randint(110, 170), np.random.randint(130, 170)]
                ], np.int32)
                color = np.random.randint(100, 255, 3).tolist()
                cv2.fillPoly(img, [pts], color)
            
            # Add some noise
            noise = np.random.randint(-20, 20, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            filename = f"{class_name}_{i:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, img)
            
            image_paths.append(filepath)
            labels.append(class_name)
    
    print(f"Sample dataset created with {len(image_paths)} images")
    return image_paths, labels


def test_image_classifier():
    """Test function for the image classifier."""
    print("Testing Image Classifier...")
    
    # Create sample dataset
    dataset_dir = "/tmp/sample_dataset"
    image_paths, labels = create_sample_dataset(dataset_dir, n_samples_per_class=30)
    
    # Test different classifiers
    classifiers = ['svm', 'rf', 'knn']
    
    for clf_type in classifiers:
        print(f"\n--- Testing {clf_type.upper()} Classifier ---")
        
        # Create classifier
        classifier = ImageClassifier(classifier_type=clf_type)
        
        # Prepare dataset
        X, y = classifier.prepare_dataset(image_paths, labels)
        
        # Train classifier
        results = classifier.train(X, y, test_size=0.3)
        
        # Test single prediction
        test_image = cv2.imread(image_paths[0])
        prediction, confidence = classifier.predict(test_image)
        print(f"Sample prediction: {prediction} (confidence: {confidence:.3f})")
        
        # Save model
        model_path = f"/tmp/{clf_type}_classifier.pkl"
        classifier.save_model(model_path)
        
        # Test loading
        new_classifier = ImageClassifier(classifier_type=clf_type)
        new_classifier.load_model(model_path)
        
        # Verify loaded model works
        prediction2, confidence2 = new_classifier.predict(test_image)
        print(f"Loaded model prediction: {prediction2} (confidence: {confidence2:.3f})")
        
        print(f"{clf_type.upper()} classifier test completed!")
    
    print("\nImage Classifier test completed successfully!")


if __name__ == "__main__":
    test_image_classifier()

