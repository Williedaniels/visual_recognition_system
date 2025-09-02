# Visual Recognition System

This project is a comprehensive Visual Recognition System built from scratch, demonstrating a wide range of computer vision and machine learning capabilities. It includes modules for image classification, object detection, face recognition, and advanced visual analysis, all accessible through a modern web interface.

## Features

- **Image Classification**: Classify images into different categories using traditional machine learning models (SVM, Random Forest, KNN) with comprehensive feature extraction (color, texture, shape, keypoints).
- **Object Detection**: Detect objects in images using a sliding window approach with a trained SVM classifier and non-maximum suppression.
- **Face Recognition**: Detect faces, recognize individuals, and identify facial features (eyes, smiles) using cascade classifiers and machine learning models.
- **Advanced Visual Analysis**: Perform in-depth analysis of images, including scene composition, quality assessment (blur, brightness, contrast, noise), and similarity comparison.
- **Web Interface**: A modern, responsive web interface built with React and Flask that allows users to upload images and interact with all the visual recognition features in real-time.
- **Modular Architecture**: The system is designed with a modular architecture, making it easy to extend and customize with new models and features.

## Project Structure

```sh
visual_recognition_system/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_classifier.py
│   │   ├── object_detector.py
│   │   ├── face_recognition.py
│   │   └── neural_network.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── image_processing.py
├── web/
│   ├── app.py
│   └── visual-recognition-frontend/
│       ├── public/
│       ├── src/
│       ├── package.json
│       └── ...
├── requirements.txt
└── README.md
```

## How to Install and Run

### Prerequisites

- Python 3.8+
- Node.js 14+
- `pip` and `npm` (or `pnpm`)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/visual-recognition-system.git
   cd visual-recognition-system
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**

   ```bash
   cd web/visual-recognition-frontend
   npm install
   ```

### Running the Application

1. **Start the Flask backend:**
   Open a terminal and run:

   ```bash
   cd visual-recognition-system
   python3 web/app.py
   ```

   The backend will be running at `http://localhost:5000`.

2. **Start the React frontend:**
   Open another terminal and run:

   ```bash
   cd visual-recognition-system/web/visual-recognition-frontend
   npm run dev
   ```

   The frontend will be running at `http://localhost:5173`.

3. **Access the application:**
   Open your browser and navigate to `http://localhost:5173` to use the Visual Recognition System.
