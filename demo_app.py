 import os
import uuid
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from config import (
    MODEL_PATH, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, 
    MAX_FILE_SIZE, CLASS_NAMES, IMG_SIZE
)
from utils.face_detector import FaceDetector
from utils.recommender import HairstyleRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# Initialize components
logger.info("Loading face detection model...")
face_detector = FaceDetector()
recommender = HairstyleRecommender()

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Detect and crop face
    face = face_detector.detect_and_crop(img)
    if face is None:
        return None
    
    # Prepare for model (normalize to 0-1)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_face():
    """
    Analyze uploaded face image and recommend hairstyles
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'
        }), 400
    
    try:
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = UPLOAD_FOLDER / filename
        
        # Save uploaded file
        file.save(filepath)
        
        # Preprocess image
        face_input = preprocess_image(str(filepath))
        if face_input is None:
            # Clean up file
            os.remove(filepath)
            return jsonify({
                'error': 'No face detected in the image. Please upload a clear front-facing photo.'
            }), 400
        
        # Run prediction
        predictions = model.predict(face_input, verbose=0)[0]
        
        # Get top 2 predictions
        indices = np.argsort(predictions)[::-1][:2]
        top2 = [(CLASS_NAMES[i], float(predictions[i])) for i in indices]
        
        # Get all probabilities
        all_probs = {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Get recommendations
        recommendations = recommender.recommend(top2)
        
        # Clean up file (optional - keep for debugging)
        # os.remove(filepath)
        
        return jsonify({
            'success': True,
            'face_shape': {
                'primary': top2[0][0],
                'confidence': top2[0][1],
                'secondary': top2[1][0],
                'all_probabilities': all_probs
            },
            'recommendations': recommendations,
            'image_url': f'/static/uploads/{filename}'
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({
            'error': f'Processing error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'error': f'File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

if __name__ == '__main__':
    logger.info("Starting Face Shape Detection API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
