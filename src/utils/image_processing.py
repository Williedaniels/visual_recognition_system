"""
Core Image Processing Utilities
Built from scratch for Visual Recognition System

This module implements fundamental image processing operations
including filtering, edge detection, feature extraction, and preprocessing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
import math


class ImageProcessor:
    """
    Core image processing class with fundamental computer vision operations.
    """
    
    def __init__(self):
        """Initialize the image processor."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def load_image(self, image_path: str, color_mode: str = 'BGR') -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            color_mode: Color mode ('BGR', 'RGB', 'GRAY')
            
        Returns:
            Loaded image as numpy array
        """
        if color_mode == 'GRAY':
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if color_mode == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image
            size: Target size (width, height)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        return cv2.resize(image, size, interpolation=interpolation)
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            Normalized image
        """
        image_float = image.astype(np.float32)
        
        if method == 'minmax':
            # Scale to [0, 1]
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val)
            else:
                normalized = image_float
                
        elif method == 'zscore':
            # Zero mean, unit variance
            mean = np.mean(image_float)
            std = np.std(image_float)
            if std > 0:
                normalized = (image_float - mean) / std
            else:
                normalized = image_float - mean
                
        elif method == 'unit':
            # Scale to [-1, 1]
            normalized = (image_float / 127.5) - 1.0
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, 
                           sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian kernel
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def detect_edges_canny(self, image: np.ndarray, low_threshold: int = 50, 
                          high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detector.
        
        Args:
            image: Input image (grayscale)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Binary edge image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def detect_corners_harris(self, image: np.ndarray, block_size: int = 2, 
                             ksize: int = 3, k: float = 0.04) -> np.ndarray:
        """
        Detect corners using Harris corner detector.
        
        Args:
            image: Input image (grayscale)
            block_size: Size of neighborhood for corner detection
            ksize: Aperture parameter for Sobel operator
            k: Harris detector free parameter
            
        Returns:
            Corner response image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_float = np.float32(image)
        corners = cv2.cornerHarris(image_float, block_size, ksize, k)
        
        return corners
    
    def extract_sift_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT features from image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def extract_orb_features(self, image: np.ndarray, n_features: int = 500) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features from image.
        
        Args:
            image: Input image
            n_features: Maximum number of features to extract
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def compute_histogram(self, image: np.ndarray, bins: int = 256, 
                         channels: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute color histogram of image.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            channels: Channels to compute histogram for
            
        Returns:
            Histogram array
        """
        if len(image.shape) == 3:
            if channels is None:
                channels = [0, 1, 2]
            hist = cv2.calcHist([image], channels, None, [bins] * len(channels), 
                              [0, 256] * len(channels))
        else:
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        
        return hist.flatten()
    
    def apply_morphological_operations(self, image: np.ndarray, operation: str, 
                                     kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to binary image.
        
        Args:
            image: Input binary image
            operation: Type of operation ('erosion', 'dilation', 'opening', 'closing')
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
            
        Returns:
            Processed image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erosion':
            result = cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilation':
            result = cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'opening':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        return result
    
    def segment_image_kmeans(self, image: np.ndarray, k: int = 3) -> np.ndarray:
        """
        Segment image using K-means clustering.
        
        Args:
            image: Input image
            k: Number of clusters
            
        Returns:
            Segmented image
        """
        # Reshape image to 2D array of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and reshape to original image shape
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        return segmented_image
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_eq')
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
        elif method == 'histogram_eq':
            enhanced_l = cv2.equalizeHist(l_channel)
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
        
        if len(image.shape) == 3:
            lab[:, :, 0] = enhanced_l
            enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced_image = enhanced_l
        
        return enhanced_image
    
    def visualize_features(self, image: np.ndarray, keypoints: List, 
                          title: str = "Feature Points") -> None:
        """
        Visualize detected features on image.
        
        Args:
            image: Input image
            keypoints: Detected keypoints
            title: Plot title
        """
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                             color=(0, 255, 0), 
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(12, 8))
        if len(img_with_keypoints.shape) == 3:
            plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"{title} ({len(keypoints)} features)")
        plt.axis('off')
        plt.show()


def test_image_processor():
    """Test function for the image processor."""
    print("Testing Image Processor...")
    
    # Create a test image
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(test_image, (100, 100), 30, (0, 0, 255), -1)
    
    processor = ImageProcessor()
    
    # Test basic operations
    print("Testing basic operations...")
    resized = processor.resize_image(test_image, (100, 100))
    print(f"Original size: {test_image.shape}, Resized: {resized.shape}")
    
    normalized = processor.normalize_image(test_image, method='minmax')
    print(f"Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
    
    # Test edge detection
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    edges = processor.detect_edges_canny(gray_image)
    print(f"Edge pixels detected: {np.sum(edges > 0)}")
    
    # Test feature extraction
    keypoints, descriptors = processor.extract_orb_features(test_image)
    print(f"ORB features extracted: {len(keypoints)}")
    
    # Test histogram
    hist = processor.compute_histogram(test_image)
    print(f"Histogram computed with {len(hist)} bins")
    
    print("Image Processor test completed successfully!")
    
    return processor


if __name__ == "__main__":
    test_image_processor()

