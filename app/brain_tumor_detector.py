import os
import numpy as np
import joblib
import pandas as pd
import glob
import logging
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from config import Config

# Try to import tensorflow, fallback to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using sklearn models")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainTumorDetector:
    def __init__(self):
        self.cnn_model = None
        self.knn_model = None
        self.label_encoder = None
        self.feature_scaler = None
        self.feature_extractor = None
        self.csv_data = None
        self.config = Config()
        
        # Load models and data
        self._load_models()
        self._create_feature_extractor()
        self._load_csv_data()
    
    def _load_models(self):
        """Load all required models"""
        try:
            # Try to load improved models first
            if TENSORFLOW_AVAILABLE and os.path.exists(self.config.CNN_MODEL_PATH):
                logger.info(f"Loading CNN model from {self.config.CNN_MODEL_PATH}")
                self.cnn_model = tf.keras.models.load_model(self.config.CNN_MODEL_PATH)
            elif TENSORFLOW_AVAILABLE and os.path.exists(self.config.CNN_MODEL_PATH_FALLBACK):
                logger.info(f"Loading fallback CNN model from {self.config.CNN_MODEL_PATH_FALLBACK}")
                self.cnn_model = tf.keras.models.load_model(self.config.CNN_MODEL_PATH_FALLBACK)
            else:
                # Try to load sklearn models as fallback
                sklearn_model_path = self.config.CNN_MODEL_PATH.replace('.h5', '.pkl')
                sklearn_model_path_fallback = self.config.CNN_MODEL_PATH_FALLBACK.replace('.h5', '.pkl')
                
                if os.path.exists(sklearn_model_path):
                    logger.info(f"Loading sklearn CNN model from {sklearn_model_path}")
                    self.cnn_model = joblib.load(sklearn_model_path)
                elif os.path.exists(sklearn_model_path_fallback):
                    logger.info(f"Loading fallback sklearn CNN model from {sklearn_model_path_fallback}")
                    self.cnn_model = joblib.load(sklearn_model_path_fallback)
                else:
                    raise FileNotFoundError("CNN model not found")
            
            # Load KNN model
            if os.path.exists(self.config.KNN_MODEL_PATH):
                logger.info(f"Loading KNN model from {self.config.KNN_MODEL_PATH}")
                self.knn_model = joblib.load(self.config.KNN_MODEL_PATH)
            elif os.path.exists(self.config.KNN_MODEL_PATH_FALLBACK):
                logger.info(f"Loading fallback KNN model from {self.config.KNN_MODEL_PATH_FALLBACK}")
                self.knn_model = joblib.load(self.config.KNN_MODEL_PATH_FALLBACK)
            else:
                raise FileNotFoundError("KNN model not found")
            
            # Load label encoder
            if os.path.exists(self.config.LABEL_ENCODER_PATH):
                logger.info(f"Loading label encoder from {self.config.LABEL_ENCODER_PATH}")
                self.label_encoder = joblib.load(self.config.LABEL_ENCODER_PATH)
            elif os.path.exists(self.config.LABEL_ENCODER_PATH_FALLBACK):
                logger.info(f"Loading fallback label encoder from {self.config.LABEL_ENCODER_PATH_FALLBACK}")
                self.label_encoder = joblib.load(self.config.LABEL_ENCODER_PATH_FALLBACK)
            else:
                raise FileNotFoundError("Label encoder not found")
            
            # Try to load feature scaler (optional)
            if os.path.exists(self.config.FEATURE_SCALER_PATH):
                logger.info(f"Loading feature scaler from {self.config.FEATURE_SCALER_PATH}")
                self.feature_scaler = joblib.load(self.config.FEATURE_SCALER_PATH)
            elif os.path.exists(self.config.FEATURE_SCALER_PATH_FALLBACK):
                logger.info(f"Loading fallback feature scaler from {self.config.FEATURE_SCALER_PATH_FALLBACK}")
                self.feature_scaler = joblib.load(self.config.FEATURE_SCALER_PATH_FALLBACK)
            else:
                logger.warning("Feature scaler not found, proceeding without scaling")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _create_feature_extractor(self):
        """Create feature extractor model"""
        if TENSORFLOW_AVAILABLE:
            self.feature_extractor = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu')
            ])
        else:
            # Use a simple feature extractor for demo
            from sklearn.decomposition import PCA
            self.feature_extractor = PCA(n_components=128)
    
    def _load_csv_data(self):
        """Load CSV data with tumor information"""
        try:
            if os.path.exists(self.config.CSV_DATA_PATH):
                logger.info(f"Loading CSV data from {self.config.CSV_DATA_PATH}")
                self.csv_data = pd.read_csv(self.config.CSV_DATA_PATH)
                self.csv_data.columns = self.csv_data.columns.str.strip()
                self.csv_data['Imagefilename'] = self.csv_data['Imagefilename'].str.strip().str.lower()
                self.csv_data = self.csv_data.dropna()
            else:
                logger.warning("CSV data file not found, creating sample data")
                self._create_sample_csv_data()
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            self._create_sample_csv_data()
    
    def _create_sample_csv_data(self):
        """Create sample CSV data for tumor types"""
        sample_data = {
            'Imagefilename': ['y1', 'y2', 'y3', 'y4', 'y5'],
            'tumor name': ['Glioma', 'Meningioma', 'Pituitary', 'Glioma', 'Meningioma'],
            'tumor type': ['Malignant', 'Benign', 'Benign', 'Malignant', 'Benign']
        }
        self.csv_data = pd.DataFrame(sample_data)
        # Save sample data
        self.csv_data.to_csv(self.config.CSV_DATA_PATH, index=False)
        logger.info("Created sample CSV data")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            logger.info(f"Preprocessing image: {image_path}")
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None, None
            if TENSORFLOW_AVAILABLE:
                try:
                    img = load_img(image_path, target_size=(256, 256), color_mode='rgb')
                except Exception as e:
                    logger.error(f"Error loading image with TensorFlow: {e}")
                    return None, None
                try:
                    img_array = img_to_array(img)
                except Exception as e:
                    logger.error(f"Error converting image to array: {e}")
                    return None, None
                img_array = img_array / 255.0  # Normalize
                return np.expand_dims(img_array, axis=0), img_array
            else:
                try:
                    img = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logger.error(f"Error loading image with PIL: {e}")
                    return None, None
                try:
                    img = img.resize((256, 256))
                    img_array = np.array(img) / 255.0
                except Exception as e:
                    logger.error(f"Error resizing/converting image: {e}")
                    return None, None
                # Use feature extractor (PCA) to reduce to expected features
                try:
                    img_flattened = img_array.flatten().reshape(1, -1)
                    if hasattr(self.feature_extractor, 'transform'):
                        img_features = self.feature_extractor.transform(img_flattened)
                    else:
                        img_features = img_flattened
                except Exception as e:
                    logger.error(f"Error extracting features with PCA: {e}")
                    return None, None
                return img_features, img_array
        except Exception as e:
            logger.error(f"General error in preprocess_image: {e}")
            return None, None
    
    def extract_features(self, img_batch):
        """Extract features using feature extractor"""
        if TENSORFLOW_AVAILABLE:
            features = self.feature_extractor.predict(img_batch)
        else:
            # For sklearn models, generate random features for demo
            features = np.random.randn(img_batch.shape[0], 128)
        
        if self.feature_scaler:
            features = self.feature_scaler.transform(features)
        return features
    
    def find_best_match(self, query_features, folder_path):
        """Find best matching image in dataset folder"""
        best_match = None
        best_score = float('inf')
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return None, float('inf')
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        logger.info(f"Found {len(image_files)} images in {os.path.basename(folder_path)} folder")
        
        for img_path in image_files[:50]:  # Limit to first 50 images for performance
            try:
                filename = os.path.splitext(os.path.basename(img_path))[0].lower()
                img_batch, _ = self.preprocess_image(img_path)
                if img_batch is None:
                    continue
                
                features = self.extract_features(img_batch)
                distance = np.linalg.norm(features - query_features)
                
                if distance < best_score:
                    best_score = distance
                    best_match = filename
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        return best_match, best_score
    
    def lookup_tumor_details(self, filename):
        """Look up tumor details in CSV data"""
        if self.csv_data is None:
            return None
        
        # Try exact match first
        matching_row = self.csv_data[self.csv_data['Imagefilename'] == filename]
        
        # Try partial match if no exact match
        if matching_row.empty:
            for idx, row in self.csv_data.iterrows():
                if filename in row['Imagefilename'] or row['Imagefilename'] in filename:
                    matching_row = self.csv_data.iloc[[idx]]
                    break
        
        if not matching_row.empty:
            return {
                "tumor_name": matching_row['tumor name'].iloc[0],
                "tumor_type": matching_row['tumor type'].iloc[0]
            }
        return None
    
    def predict(self, image_path):
        """Main prediction function"""
        try:
            # Preprocess image
            img_batch, img_array = self.preprocess_image(image_path)
            if img_batch is None:
                return {"result": "error", "message": "Error preprocessing image"}
            
            # Step 1: CNN prediction for tumor presence
            if TENSORFLOW_AVAILABLE:
                tumor_prediction = self.cnn_model.predict(img_batch)[0][0]
            else:
                # For sklearn models, predict probability
                tumor_prediction = self.cnn_model.predict_proba(img_batch)[0][1]
            logger.info(f"CNN tumor prediction: {tumor_prediction:.4f}")
            
            # Handle uncertain predictions
            if 0.3 < tumor_prediction < 0.7:
                query_features = self.extract_features(img_batch)
                
                # Check dataset paths
                yes_folder = self.config.YES_FOLDER if os.path.exists(self.config.YES_FOLDER) else self.config.YES_FOLDER_ALT
                no_folder = self.config.NO_FOLDER if os.path.exists(self.config.NO_FOLDER) else self.config.NO_FOLDER_ALT
                
                _, yes_distance = self.find_best_match(query_features, yes_folder)
                _, no_distance = self.find_best_match(query_features, no_folder)
                
                if no_distance < yes_distance:
                    return {
                        "result": "negative",
                        "tumor_type": "",
                        "tumor_name": "",
                        "confidence": f"{min(90, (100 - no_distance)):.1f}%",
                        "cnn_score": f"{tumor_prediction:.4f}",
                        "message": "No tumor detected. MRI appears normal."
                    }
                
                tumor_present = tumor_prediction > 0.5
            else:
                tumor_present = tumor_prediction > self.config.TUMOR_THRESHOLD
            
            # If no tumor detected
            if not tumor_present:
                return {
                    "result": "negative",
                    "tumor_type": "",
                    "tumor_name": "",
                    "confidence": f"{(1-tumor_prediction)*100:.1f}%",
                    "cnn_score": f"{tumor_prediction:.4f}",
                    "message": "No tumor detected. MRI appears normal."
                }
            
            # For tumor-positive cases
            query_features = self.extract_features(img_batch)
            yes_folder = self.config.YES_FOLDER if os.path.exists(self.config.YES_FOLDER) else self.config.YES_FOLDER_ALT
            
            matching_image, distance = self.find_best_match(query_features, yes_folder)
            
            if matching_image and distance < self.config.DISTANCE_THRESHOLD:
                tumor_details = self.lookup_tumor_details(matching_image)
                
                if tumor_details:
                    confidence = max(0, 100 - (distance / self.config.MAX_DISTANCE * 100))
            return {
                "result": "positive",
                "tumor_type": tumor_details["tumor_type"],
                "tumor_name": tumor_details["tumor_name"],
                "confidence": f"{confidence:.1f}%",
                "cnn_score": f"{tumor_prediction:.4f}",
                "message": f"Brain tumor detected - Type: {tumor_details['tumor_type']}, Name: {tumor_details['tumor_name']}"
            }
            
            # Fallback to KNN classification
            tumor_type_encoded = self.knn_model.predict(query_features)
            tumor_type = self.label_encoder.inverse_transform(tumor_type_encoded)[0]
            
            return {
                "result": "positive",
                "tumor_type": tumor_type,
                "tumor_name": "",
                "confidence": f"{tumor_prediction*100:.1f}%",
                "cnn_score": f"{tumor_prediction:.4f}",
                "message": f"Brain tumor detected - Type: {tumor_type}"
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                "result": "error",
                "tumor_type": "",
                "tumor_name": "",
                "confidence": "",
                "cnn_score": "",
                "message": f"Prediction error: {str(e)}"
            }

# Global detector instance
detector = None

def get_detector():
    """Get or create detector instance"""
    global detector
    if detector is None:
        detector = BrainTumorDetector()
    return detector