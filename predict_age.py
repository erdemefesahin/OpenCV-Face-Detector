"""
Age Prediction Module
=====================
This module loads the trained CNN age model and provides a prediction function
for integration with the main face detection script.

Usage in face detection script:
    from predict_age import predict_age
    
    face_roi = ...  # Extract face region from image
    age_group = predict_age(face_roi)
    print(f"Predicted age group: {age_group}")

Author: Erdem Efe Sahin
Date: November 2025
"""

import numpy as np
import cv2
import json
import os
from tensorflow import keras


# ===========================
# Global Variables
# ===========================
_MODEL = None
_LABEL_MAPPING = None
_MODEL_PATH = 'age_model.h5'
_LABELS_PATH = 'age_labels.json'
_IMG_SIZE = (200, 200)


# ===========================
# Model Loading
# ===========================
def load_age_model(model_path=None, labels_path=None):
    """
    Load the trained age classification model and label mapping.
    
    Args:
        model_path: Path to the .h5 model file (default: 'age_model.h5')
        labels_path: Path to the labels JSON file (default: 'age_labels.json')
        
    Returns:
        model: Loaded Keras model
        label_mapping: Dictionary mapping class indices to age group names
        
    Raises:
        FileNotFoundError: If model or labels file not found
    """
    global _MODEL, _LABEL_MAPPING
    
    # Use default paths if not specified
    if model_path is None:
        model_path = _MODEL_PATH
    if labels_path is None:
        labels_path = _LABELS_PATH
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please train the model first using train_age_model.py"
        )
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Please ensure age_labels.json is in the same directory as the model"
        )
    
    # Load model
    print(f"Loading age model from {model_path}...")
    _MODEL = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    # Load label mapping
    with open(labels_path, 'r') as f:
        _LABEL_MAPPING = json.load(f)
    
    # Convert string keys to integers
    _LABEL_MAPPING = {int(k): v for k, v in _LABEL_MAPPING.items()}
    
    print(f"Label mapping loaded: {_LABEL_MAPPING}")
    
    return _MODEL, _LABEL_MAPPING


# ===========================
# Preprocessing
# ===========================
def preprocess_face(face_roi, target_size=_IMG_SIZE):
    """
    Preprocess a face ROI for model prediction.
    
    This function applies the same preprocessing steps used during training:
    1. Convert BGR to RGB (if needed)
    2. Resize to target size
    3. Normalize pixel values to [0, 1]
    4. Add batch dimension
    
    Args:
        face_roi: Input face image (numpy array)
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        Preprocessed face ready for model input (shape: 1, height, width, 3)
    """
    # Make a copy to avoid modifying original
    face = face_roi.copy()
    
    # Ensure RGB format (OpenCV uses BGR by default)
    if len(face.shape) == 3 and face.shape[2] == 3:
        # Assume BGR, convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Resize to model's expected input size
    face = cv2.resize(face, target_size)
    
    # Normalize to [0, 1]
    face = face.astype('float32') / 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    face = np.expand_dims(face, axis=0)
    
    return face


# ===========================
# Prediction Functions
# ===========================
def predict_age(face_roi, return_confidence=False):
    """
    Predict the age group of a face.
    
    This is the main function to use for integration with face detection scripts.
    
    Args:
        face_roi: Face region of interest (numpy array, BGR or RGB format)
        return_confidence: If True, also return confidence scores for all classes
        
    Returns:
        If return_confidence is False:
            age_group: Predicted age group as string (e.g., "25-32")
        If return_confidence is True:
            (age_group, confidence, all_predictions): Tuple of:
                - age_group: Predicted age group (string)
                - confidence: Confidence score for predicted class (0-1)
                - all_predictions: Dictionary of {age_group: confidence} for all classes
        
    Raises:
        RuntimeError: If model not loaded
        ValueError: If face_roi is invalid
    """
    global _MODEL, _LABEL_MAPPING
    
    # Lazy loading: load model on first call if not already loaded
    if _MODEL is None or _LABEL_MAPPING is None:
        try:
            load_age_model()
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot predict age: {e}")
    
    # Validate input
    if face_roi is None or face_roi.size == 0:
        raise ValueError("Invalid face_roi: empty or None")
    
    if len(face_roi.shape) not in [2, 3]:
        raise ValueError(f"Invalid face_roi shape: {face_roi.shape}. Expected 2D or 3D array.")
    
    # Preprocess face
    try:
        face_processed = preprocess_face(face_roi)
    except Exception as e:
        raise ValueError(f"Error preprocessing face: {e}")
    
    # Make prediction
    predictions = _MODEL.predict(face_processed, verbose=0)
    
    # Get predicted class index
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Map index to age group label
    age_group = _LABEL_MAPPING.get(predicted_class_idx, "Unknown")
    
    if return_confidence:
        # Create dictionary of all predictions
        all_predictions = {
            _LABEL_MAPPING[i]: float(predictions[0][i])
            for i in range(len(predictions[0]))
        }
        return age_group, float(confidence), all_predictions
    else:
        return age_group


def predict_age_batch(face_rois):
    """
    Predict age groups for multiple faces in batch.
    
    More efficient than calling predict_age() multiple times when you have many faces.
    
    Args:
        face_rois: List of face ROIs (numpy arrays)
        
    Returns:
        List of predicted age groups (strings)
    """
    global _MODEL, _LABEL_MAPPING
    
    # Load model if needed
    if _MODEL is None or _LABEL_MAPPING is None:
        load_age_model()
    
    # Preprocess all faces
    faces_processed = np.array([preprocess_face(face)[0] for face in face_rois])
    
    # Batch prediction
    predictions = _MODEL.predict(faces_processed, verbose=0)
    
    # Map predictions to age groups
    predicted_classes = np.argmax(predictions, axis=1)
    age_groups = [_LABEL_MAPPING.get(idx, "Unknown") for idx in predicted_classes]
    
    return age_groups


def get_age_distribution(face_roi):
    """
    Get probability distribution over all age groups for a face.
    
    Useful for understanding model confidence and visualizing age predictions.
    
    Args:
        face_roi: Face region of interest (numpy array)
        
    Returns:
        Dictionary mapping age groups to probabilities
    """
    _, _, all_predictions = predict_age(face_roi, return_confidence=True)
    return all_predictions


# ===========================
# Utility Functions
# ===========================
def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information
    """
    global _MODEL, _LABEL_MAPPING
    
    if _MODEL is None:
        return {"status": "not_loaded"}
    
    return {
        "status": "loaded",
        "input_shape": _MODEL.input_shape,
        "output_shape": _MODEL.output_shape,
        "num_classes": len(_LABEL_MAPPING) if _LABEL_MAPPING else 0,
        "age_groups": list(_LABEL_MAPPING.values()) if _LABEL_MAPPING else []
    }


def test_prediction(image_path):
    """
    Test the prediction on a single image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Predicted age group
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    # Predict
    age_group, confidence, all_predictions = predict_age(img, return_confidence=True)
    
    # Print results
    print(f"\nPrediction for {image_path}:")
    print(f"  Predicted age group: {age_group}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"\nAll predictions:")
    for age, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {age}: {prob*100:.2f}%")
    
    return age_group


# ===========================
# Main (for testing)
# ===========================
if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("Age Prediction Module - Test Mode")
    print("="*60)
    
    # Test model loading
    try:
        load_age_model()
        print("\n✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"\n✗ Error loading model: {e}")
        sys.exit(1)
    
    # Print model info
    info = get_model_info()
    print(f"\nModel Information:")
    print(f"  Status: {info['status']}")
    print(f"  Input shape: {info['input_shape']}")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"  Age groups: {info['age_groups']}")
    
    # Test with image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            test_prediction(image_path)
        except Exception as e:
            print(f"\nError testing prediction: {e}")
    else:
        print("\nTo test on an image, run:")
        print("  python predict_age.py <path_to_image>")
    
    print("\n" + "="*60)
    print("Integration example:")
    print("="*60)
    print("""
from predict_age import predict_age
import cv2

# Load your face image
face_roi = cv2.imread('face.jpg')

# Get age prediction
age_group = predict_age(face_roi)
print(f"Age: {age_group}")

# Get detailed predictions
age_group, confidence, all_preds = predict_age(face_roi, return_confidence=True)
print(f"Age: {age_group} (confidence: {confidence*100:.1f}%)")
""")
