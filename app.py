from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import time
from threading import Lock

app = Flask(__name__)

class MaskDetector:
    def __init__(self, model_path='face_mask.h5'):
        """Initialize the mask detector with heavy optimizations"""
        print("Loading model...")
        
        # Optimize TensorFlowcls
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        
        # Load model
        self.model = load_model(model_path, compile=False)
        
        # Warmup the model with dummy data
        print("Warming up model...")
        # dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
        # dummy_input = self.preprocess_face(dummy_face)
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        for _ in range(3):
            _ = self.model.predict(dummy_input, verbose=0)
        
        print("Model loaded and warmed up!")
        
        # Initialize face detector with DNN (faster than Haar Cascade)
        self.use_dnn = True
        try:
            # Try to use DNN face detector (much faster)
            model_file = "res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "deploy.prototxt"
            self.face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("Using DNN face detector (fast)")
        except:
            # Fallback to Haar Cascade
            self.use_dnn = False
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("Using Haar Cascade face detector (fallback)")
        
        # Class labels
        self.class_labels = ['with_mask', 'without_mask', 'incorrect_mask']
        self.class_colors = {
            'with_mask': (0, 255, 0),
            'without_mask': (0, 0, 255),
            'incorrect_mask': (0, 165, 255)
        }
        
        # Caching and frame skipping
        self.last_frame_time = 0
        self.min_frame_interval = 0.1  # Process max 10 fps
        self.last_detections = []
        self.detection_lock = Lock()
        
        # Performance tracking
        self.detection_times = []
        
    def detect_faces_dnn(self, frame):
        """Fast face detection using DNN"""
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append((x, y, x2 - x, y2 - y))
        
        return faces
    
    def detect_faces_haar(self, frame):
        """Fallback face detection using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(60, 60),
            maxSize=(400, 400)
        )
        
        return faces
    
    def preprocess_face(self, face_image):
        """Optimized preprocessing"""
        # Use cv2.resize with INTER_AREA for downscaling (faster)
        # face_resized = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_AREA)
        face_resized = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_AREA)
        # Normalize efficiently
        face_normalized = face_resized.astype('float32') * (1.0 / 255.0)
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    
    def predict_mask(self, face_image):
        """Predict mask status"""
        processed_face = self.preprocess_face(face_image)
        predictions = self.model.predict(processed_face, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_labels[predicted_class_idx]
        return predicted_class, confidence
    
    def process_frame(self, frame):
        """Process frame with aggressive optimizations"""
        current_time = time.time()
        
        # Frame skipping - return cached results if too soon
        with self.detection_lock:
            if current_time - self.last_frame_time < self.min_frame_interval:
                return self.last_detections
            
            self.last_frame_time = current_time
        
        start_time = time.time()
        
        # Aggressive resizing for faster processing
        height, width = frame.shape[:2]
        original_width = width
        
        if width > 480:
            scale = 480 / width
            new_width = 480
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scale_back = original_width / 480
        else:
            scale_back = 1.0
        
        # Detect faces
        if self.use_dnn:
            faces = self.detect_faces_dnn(frame)
        else:
            faces = self.detect_faces_haar(frame)
        
        detections = []
        
        # Process only the first face for maximum speed
        # Or process up to 2 faces if needed
        max_faces = 1  # Change to 2 if you need to detect multiple faces
        
        if len(faces) > 0:
            # Sort by area (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            for (x, y, w, h) in faces[:max_faces]:
                # Add some margin
                margin = int(0.1 * min(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)
                
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    predicted_class, confidence = self.predict_mask(face_region)
                    
                    # Scale coordinates back
                    if scale_back != 1.0:
                        x = int(x * scale_back)
                        y = int(y * scale_back)
                        w = int(w * scale_back)
                        h = int(h * scale_back)
                    
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h),
                        'class': predicted_class,
                        'confidence': confidence,
                        'color': self.class_colors[predicted_class]
                    })
        
        # Cache results
        with self.detection_lock:
            self.last_detections = detections
        
        # Performance tracking
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        if len(self.detection_times) > 30:
            self.detection_times.pop(0)
            avg_time = np.mean(self.detection_times)
            print(f"Avg detection: {avg_time:.3f}s, FPS: {1/avg_time:.1f}")
        
        return detections

# Global detector instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = MaskDetector()
    return detector

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect masks in uploaded frame"""
    try:
        # Initialize detector lazily
        det = get_detector()
        
        # Get image data
        data = request.json
        image_data = data['image']
        
        # Decode base64 - optimized
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        
        # Decode with OpenCV (faster than PIL)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'})
        
        # Process frame
        detections = det.process_frame(frame)
        
        return jsonify({
            'status': 'success',
            'detections': detections
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Pre-load detector at startup
    print("Initializing detector...")
    get_detector()
    print("Ready!")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)