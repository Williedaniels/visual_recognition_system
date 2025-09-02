"""
Visual Recognition System Web Application
Flask Backend API

This module provides a web API for the visual recognition system,
including endpoints for image classification, object detection, and face recognition.
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import os
import sys
from PIL import Image
import json
from typing import Dict, List, Any
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our models
from models.image_classifier import ImageClassifier, FeatureExtractor
from models.object_detector import SlidingWindowDetector, TemplateMatchingDetector
from models.face_recognition import FaceRecognizer, FaceDetector, AdvancedVisualAnalyzer
from utils.image_processing import ImageProcessor

app = Flask(__name__)
# Enable CORS for all routes, which is suitable for development.
CORS(app)

# Global variables for models
image_classifier = None
object_detector = None
face_recognizer = None
face_detector = FaceDetector()
visual_analyzer = AdvancedVisualAnalyzer()
image_processor = ImageProcessor()

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
MODELS_FOLDER = '/tmp/models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array.
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        Image as numpy array
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded image string
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'image_classifier': image_classifier is not None,
            'object_detector': object_detector is not None,
            'face_recognizer': face_recognizer is not None
        }
    })


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """
    Classify an uploaded image.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Use a simple feature-based classifier if no trained model
        if image_classifier is None:
            # Create a simple classifier for demonstration
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_all_features(image)
            
            # Simple heuristic classification based on features
            color_features = feature_extractor.extract_color_histogram(image)
            texture_features = feature_extractor.extract_texture_features(image)
            
            # Analyze dominant colors and textures
            if np.mean(color_features[:32]) > 0.1:  # High red/warm colors
                prediction = "warm_toned"
                confidence = 0.75
            elif np.mean(texture_features) > 10:  # High texture
                prediction = "textured"
                confidence = 0.70
            else:
                prediction = "smooth"
                confidence = 0.65
        else:
            prediction, confidence = image_classifier.predict(image)
        
        # Analyze image properties
        scene_analysis = visual_analyzer.analyze_scene_composition(image)
        quality_analysis = visual_analyzer.detect_image_quality_issues(image)
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'scene_analysis': {
                'dimensions': scene_analysis['dimensions'],
                'complexity': {
                    'entropy': float(scene_analysis['complexity']['entropy']),
                    'contrast': float(scene_analysis['complexity']['contrast'])
                },
                'quality': {
                    'score': float(quality_analysis['overall_quality']['score']),
                    'grade': quality_analysis['overall_quality']['grade']
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/detect_objects', methods=['POST'])
def detect_objects():
    """
    Detect objects in an uploaded image.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Use template matching for demonstration if no trained detector
        if object_detector is None:
            template_detector = TemplateMatchingDetector()
            
            # Create simple templates
            circle_template = np.zeros((40, 40), dtype=np.uint8)
            cv2.circle(circle_template, (20, 20), 15, 255, -1)
            template_detector.add_template('circle', circle_template)
            
            # Detect circles
            detections = template_detector.detect_template(image, 'circle', threshold=0.3)
        else:
            detections = object_detector.detect_objects(image, confidence_threshold=0.3)
        
        # Convert detections to JSON-serializable format
        detection_results = []
        for detection in detections:
            detection_results.append({
                'x': detection.x,
                'y': detection.y,
                'width': detection.width,
                'height': detection.height,
                'confidence': float(detection.confidence),
                'class_name': detection.class_name
            })
        
        # Create visualization
        result_image = image.copy()
        for detection in detections:
            cv2.rectangle(result_image, 
                         (detection.x, detection.y), 
                         (detection.x + detection.width, detection.y + detection.height), 
                         (0, 255, 0), 2)
            
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(result_image, label, 
                       (detection.x, detection.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Encode result image
        result_image_base64 = encode_image_to_base64(result_image)
        
        return jsonify({
            'detections': detection_results,
            'result_image': result_image_base64,
            'num_detections': len(detection_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/recognize_faces', methods=['POST'])
def recognize_faces():
    """
    Recognize faces in an uploaded image.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Recognize faces if model is available
        recognition_results = []
        if face_recognizer is not None and face_recognizer.is_trained:
            for face_bbox in faces:
                name, confidence = face_recognizer.recognize_face(image, face_bbox)
                recognition_results.append({
                    'bbox': {
                        'x': face_bbox.x,
                        'y': face_bbox.y,
                        'width': face_bbox.width,
                        'height': face_bbox.height
                    },
                    'name': name,
                    'confidence': float(confidence)
                })
        else:
            # Just return detected faces without recognition
            for i, face_bbox in enumerate(faces):
                recognition_results.append({
                    'bbox': {
                        'x': face_bbox.x,
                        'y': face_bbox.y,
                        'width': face_bbox.width,
                        'height': face_bbox.height
                    },
                    'name': f'face_{i+1}',
                    'confidence': 0.8
                })
        
        # Create visualization
        result_image = image.copy()
        for result in recognition_results:
            bbox = result['bbox']
            cv2.rectangle(result_image, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         (255, 0, 0), 2)
            
            label = f"{result['name']}: {result['confidence']:.2f}"
            cv2.putText(result_image, label, 
                       (bbox['x'], bbox['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Encode result image
        result_image_base64 = encode_image_to_base64(result_image)
        
        return jsonify({
            'faces': recognition_results,
            'result_image': result_image_base64,
            'num_faces': len(recognition_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    """
    Perform comprehensive image analysis.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Perform various analyses
        scene_analysis = visual_analyzer.analyze_scene_composition(image)
        quality_analysis = visual_analyzer.detect_image_quality_issues(image)
        
        # Extract features
        feature_extractor = FeatureExtractor()
        color_features = feature_extractor.extract_color_histogram(image)
        texture_features = feature_extractor.extract_texture_features(image)
        shape_features = feature_extractor.extract_shape_features(image)
        
        # Detect faces and objects for comprehensive analysis
        faces = face_detector.detect_faces(image)
        
        return jsonify({
            'scene_composition': {
                'dimensions': scene_analysis['dimensions'],
                'color_analysis': scene_analysis.get('color', {}),
                'texture_analysis': {
                    'edge_density': float(scene_analysis['texture']['edge_density']),
                    'gradient_mean': float(scene_analysis['texture']['gradient_mean']),
                    'roughness': float(scene_analysis['texture']['roughness'])
                },
                'composition_analysis': {
                    'rule_of_thirds_variance': float(scene_analysis['composition']['rule_of_thirds_variance']),
                    'brightness_balance': float(scene_analysis['composition']['brightness_balance'])
                },
                'complexity': {
                    'entropy': float(scene_analysis['complexity']['entropy']),
                    'contrast': float(scene_analysis['complexity']['contrast']),
                    'unique_intensities': int(scene_analysis['complexity']['unique_intensities'])
                }
            },
            'quality_assessment': {
                'overall_score': float(quality_analysis['overall_quality']['score']),
                'overall_grade': quality_analysis['overall_quality']['grade'],
                'blur_analysis': {
                    'is_blurry': quality_analysis['blur']['is_blurry'],
                    'blur_level': quality_analysis['blur']['blur_level']
                },
                'brightness_analysis': {
                    'mean_brightness': float(quality_analysis['brightness']['mean_brightness']),
                    'brightness_level': quality_analysis['brightness']['brightness_level']
                },
                'contrast_analysis': {
                    'contrast_level': quality_analysis['contrast']['contrast_level'],
                    'is_low_contrast': quality_analysis['contrast']['is_low_contrast']
                },
                'noise_analysis': {
                    'noise_category': quality_analysis['noise']['noise_category'],
                    'is_noisy': quality_analysis['noise']['is_noisy']
                }
            },
            'feature_summary': {
                'color_diversity': float(np.std(color_features)),
                'texture_complexity': float(np.mean(texture_features)),
                'shape_regularity': float(np.mean(shape_features))
            },
            'content_detection': {
                'num_faces': len(faces),
                'has_faces': len(faces) > 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/compare_images', methods=['POST'])
def compare_images():
    """
    Compare two images for similarity.
    """
    try:
        data = request.get_json()
        
        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Two images required'}), 400
        
        # Decode images
        image1 = decode_base64_image(data['image1'])
        image2 = decode_base64_image(data['image2'])
        
        # Calculate similarities using different methods
        histogram_similarity = visual_analyzer.calculate_image_similarity(
            image1, image2, method='histogram'
        )
        structural_similarity = visual_analyzer.calculate_image_similarity(
            image1, image2, method='structural'
        )
        feature_similarity = visual_analyzer.calculate_image_similarity(
            image1, image2, method='features'
        )
        
        # Overall similarity (weighted average)
        overall_similarity = (
            0.4 * histogram_similarity + 
            0.4 * structural_similarity + 
            0.2 * feature_similarity
        )
        
        return jsonify({
            'similarities': {
                'histogram': float(histogram_similarity),
                'structural': float(structural_similarity),
                'features': float(feature_similarity),
                'overall': float(overall_similarity)
            },
            'similarity_level': (
                'very_high' if overall_similarity > 0.8 else
                'high' if overall_similarity > 0.6 else
                'medium' if overall_similarity > 0.4 else
                'low' if overall_similarity > 0.2 else
                'very_low'
            )
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/process_image', methods=['POST'])
def process_image():
    """
    Apply image processing operations.
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        # Get processing options
        operations = data.get('operations', [])
        
        processed_image = image.copy()
        
        for operation in operations:
            op_type = operation.get('type')
            params = operation.get('params', {})
            
            if op_type == 'blur':
                kernel_size = params.get('kernel_size', 5)
                sigma = params.get('sigma', 1.0)
                processed_image = image_processor.apply_gaussian_blur(
                    processed_image, kernel_size, sigma
                )
            
            elif op_type == 'edge_detection':
                low_threshold = params.get('low_threshold', 50)
                high_threshold = params.get('high_threshold', 150)
                edges = image_processor.detect_edges_canny(
                    processed_image, low_threshold, high_threshold
                )
                # Convert single channel to 3-channel for consistency
                processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            elif op_type == 'enhance_contrast':
                method = params.get('method', 'clahe')
                processed_image = image_processor.enhance_contrast(processed_image, method)
            
            elif op_type == 'resize':
                width = params.get('width', 256)
                height = params.get('height', 256)
                processed_image = image_processor.resize_image(processed_image, (width, height))
            
            elif op_type == 'normalize':
                method = params.get('method', 'minmax')
                normalized = image_processor.normalize_image(processed_image, method)
                # Convert back to uint8
                processed_image = (normalized * 255).astype(np.uint8)
        
        # Encode result
        result_image_base64 = encode_image_to_base64(processed_image)
        
        return jsonify({
            'processed_image': result_image_base64,
            'operations_applied': len(operations)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("Starting Visual Recognition System Web Application...")
    print("Available endpoints:")
    print("  GET  /                    - Main interface")
    print("  GET  /api/health          - Health check")
    print("  POST /api/classify        - Image classification")
    print("  POST /api/detect_objects  - Object detection")
    print("  POST /api/recognize_faces - Face recognition")
    print("  POST /api/analyze_image   - Comprehensive image analysis")
    print("  POST /api/compare_images  - Image similarity comparison")
    print("  POST /api/process_image   - Image processing operations")
    print()
    
    port = int(os.environ.get("PORT", 5000))
    host = '0.0.0.0'
    print(f"Starting server on http://{'localhost' if host == '0.0.0.0' else host}:{port}")
    
    app.run(host=host, port=port, debug=True)
