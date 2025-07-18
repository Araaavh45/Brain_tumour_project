from flask import Flask, render_template, request
import os
import base64
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
# Add this import at the top of your script
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder for temporary uploads

# Define absolute path for the dataset folders - CONSIDER USING RELATIVE PATHS
yes_folder_path = r"C:\Users\bnava\project\brain tumor\Brain Tummor\DATASETS\yes"
app.config['YES_FOLDER'] = yes_folder_path

no_folder_path = r"C:\Users\bnava\project\brain tumor\Brain Tummor\DATASETS\no"
app.config['NO_FOLDER'] = no_folder_path

# Make sure the folders exist
if not os.path.exists(app.config['YES_FOLDER']):
    logger.warning(f"YES folder not found at {app.config['YES_FOLDER']}")
if not os.path.exists(app.config['NO_FOLDER']):
    logger.warning(f"NO folder not found at {app.config['NO_FOLDER']}")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models - CONSIDER USING RELATIVE PATHS
MODEL_PATH = r"C:\Users\bnava\project\brain tumor\Brain Tummor\model.h5"
KNN_MODEL_PATH = r"C:\Users\bnava\project\brain tumor\Brain Tummor\knn_model.pkl"
LABEL_ENCODER_PATH = r"C:\Users\bnava\project\brain tumor\Brain Tummor\label_encoder.pkl"
CSV_PATH = r"C:\Users\bnava\OneDrive\Documents\Brain Tumor2.csv"

# Debug flag
DEBUG = True

# MODIFIED THRESHOLDS - Lower threshold for tumor detection and higher for distance
TUMOR_THRESHOLD = 0.55  # Decreased from 0.75 to reduce false negatives
DISTANCE_THRESHOLD = 50.0  # More lenient threshold for feature matching
MAX_DISTANCE = 100.0  # Maximum reasonable distance for confidence calculation

try:
    # Load CNN model for tumor detection
    logger.info(f"Loading CNN model from {MODEL_PATH}")
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load KNN model for tumor classification
    logger.info(f"Loading KNN model from {KNN_MODEL_PATH}")
    knn_model = joblib.load(KNN_MODEL_PATH)
    
    # Load label encoder for decoding predictions
    logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # Load CSV data containing tumor information
    logger.info(f"Loading CSV data from {CSV_PATH}")
    csv_data = pd.read_csv(CSV_PATH)

    # Clean CSV data
    csv_data.columns = csv_data.columns.str.strip()
    csv_data['Imagefilename'] = csv_data['Imagefilename'].str.strip().str.lower()
    csv_data_clean = csv_data.dropna()
    
    if DEBUG:
        logger.info(f"CSV data loaded with {len(csv_data_clean)} valid entries")
        logger.info(f"Sample filenames: {csv_data_clean['Imagefilename'].head().tolist()}")
        
except FileNotFoundError as e:
    logger.error(f"Error loading models or CSV: {e}")
    exit()

# Define feature extractor model based on CNN layers
def create_feature_extractor():
    """Create a feature extractor from CNN model layers that outputs 128 features"""
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        # Add a Dense layer to reduce dimensions to exactly 128 features
        tf.keras.layers.Dense(128)
    ])
    return feature_extractor

# Create feature extractor once
feature_extractor = create_feature_extractor()

def normalize_image(img_array):
    """Normalize image values to range [0,1]"""
    return img_array / 255.0

def preprocess_image(image_path):
    """Load, resize, and normalize image for model input"""
    try:
        # Load and resize image to 256x256
        img = load_img(image_path, target_size=(256, 256), color_mode='rgb')
        # Convert to numpy array
        img_array = img_to_array(img)
        # Normalize pixel values
        img_array = normalize_image(img_array)
        # Add batch dimension
        return np.expand_dims(img_array, axis=0), img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

def extract_features(img_batch):
    """Extract CNN features from image batch"""
    return feature_extractor.predict(img_batch)

def extract_filename_without_extension(path):
    """Extract just the filename without extension from a path"""
    return os.path.splitext(os.path.basename(path))[0].lower()

def find_best_feature_match(query_features, folder_path):
    """
    Find the best match for query features among images in the specified folder
    using feature vector similarity
    """
    best_match = None
    best_score = float('inf')  # Lower is better (using Euclidean distance)
    match_details = []
    
    # Verify folder exists
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"Folder not found at {folder.absolute()}")
        return None, float('inf'), []
    
    # Get all image files with common extensions
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if DEBUG:
        logger.info(f"Found {len(image_files)} image files in {os.path.basename(folder_path)} folder")
    
    # If no image files found
    if not image_files:
        logger.warning(f"No image files found in {folder_path}")
        return None, float('inf'), []
    
    # Compare features with each image in the folder
    for img_path in image_files:
        try:
            # Get filename for database matching
            filename = extract_filename_without_extension(img_path)
            
            # Process image and extract features
            img_batch, _ = preprocess_image(img_path)
            if img_batch is None:
                continue
                
            # Extract feature vector
            features = extract_features(img_batch)
            
            # Calculate Euclidean distance between feature vectors
            distance = np.linalg.norm(features - query_features)
            
            # Save match details
            match_details.append({
                'filename': filename,
                'distance': float(distance)  # Convert to Python float for JSON serialization
            })
            
            # Update best match if distance is lower
            if distance < best_score:
                best_score = distance
                best_match = filename
                
        except Exception as e:
            logger.error(f"Error processing comparison image {img_path}: {e}")
            continue
    
    # Sort match details by distance (ascending)
    match_details.sort(key=lambda x: x['distance'])
    
    if DEBUG and match_details:
        logger.info(f"Best match in {os.path.basename(folder_path)} folder: {best_match} with distance {best_score}")
        logger.info(f"Top 3 matches: {match_details[:3]}")
    
    return best_match, best_score, match_details

def lookup_tumor_details(filename):
    """Look up tumor details in CSV file based on filename"""
    if DEBUG:
        logger.info(f"Looking up filename: '{filename}' in CSV")
        
    # Look for exact match
    matching_row = csv_data_clean[csv_data_clean['Imagefilename'] == filename]
    
    # If no exact match, try to find a partial match
    if matching_row.empty:
        if DEBUG:
            logger.info(f"No exact match for '{filename}', trying partial matches")
        
        # Try to find filenames that contain our search term
        for idx, row in csv_data_clean.iterrows():
            if filename in row['Imagefilename'] or row['Imagefilename'] in filename:
                matching_row = csv_data_clean.iloc[[idx]]
                if DEBUG:
                    logger.info(f"Found partial match: '{row['Imagefilename']}'")
                break
    
    if not matching_row.empty:
        if DEBUG:
            logger.info(f"Match found! Details: {matching_row[['tumor name', 'tumor type']].iloc[0].to_dict()}")
        return {
            "tumor_name": matching_row['tumor name'].iloc[0],
            "tumor_type": matching_row['tumor type'].iloc[0]
        }
    else:
        if DEBUG:
            logger.info("No match found in CSV")
        return None

def predict_tumor(image_path):
    """Main prediction function"""
    # Preprocess image
    img_batch, img_array = preprocess_image(image_path)
    if img_batch is None:
        return {"result": "error", "message": "Error preprocessing image."}

    # Get image filename without extension for logging
    img_filename = extract_filename_without_extension(image_path)
    if DEBUG:
        logger.info(f"Processing image: {img_filename}")

    # STEP 1: Use CNN to predict if tumor is present
    tumor_prediction = cnn_model.predict(img_batch)[0][0]
    
    if DEBUG:
        logger.info(f"CNN tumor prediction: {tumor_prediction:.4f} (Threshold: {TUMOR_THRESHOLD})")
    
    # MODIFIED: Simplified logic for uncertain cases and REDUCED BIAS TOWARD NEGATIVE CLASS
    if 0.3 < tumor_prediction < 0.7:  # Wider uncertain range
        if DEBUG:
            logger.info("Prediction in uncertain range, performing additional verification")
        
        # Extract features from the image
        query_features = extract_features(img_batch)
        
        # Compare with both YES and NO samples to determine tumor presence
        _, yes_distance, _ = find_best_feature_match(query_features, app.config['YES_FOLDER'])
        _, no_distance, _ = find_best_feature_match(query_features, app.config['NO_FOLDER'])
        
        if DEBUG:
            logger.info(f"Distance to closest tumor sample: {yes_distance}")
            logger.info(f"Distance to closest normal sample: {no_distance}")
        
        # MODIFIED: Removed the bias toward negative class
        # Now it's a direct comparison between which sample is closer
        if no_distance < yes_distance:
            return {
                "result": "negative",
                "message": "No tumor detected. MRI appears normal.",
                "confidence": f"{min(90, (100 - no_distance)):.1f}%",
                "cnn_score": f"{tumor_prediction:.4f}"
            }
        
        # Otherwise, continue with tumor detection but use a moderate threshold
        tumor_present = tumor_prediction > 0.5  # Neutral threshold for uncertain cases
    else:
        # For very clear predictions, use the standard threshold
        tumor_present = tumor_prediction > TUMOR_THRESHOLD
    
    # If not tumor present, return negative result
    if not tumor_present:
        return {
            "result": "negative",
            "message": "No tumor detected. MRI appears normal.",
            "confidence": f"{(1-tumor_prediction)*100:.1f}%",
            "cnn_score": f"{tumor_prediction:.4f}"
        }
    
    # For tumor-positive images, continue with feature extraction and matching
    query_features = extract_features(img_batch)
    
    # Step 3: Find the best match in the 'yes' folder based on feature similarity
    matching_image, distance, matches = find_best_feature_match(query_features, app.config['YES_FOLDER'])
    
    # MODIFIED: More lenient distance threshold
    if matching_image and distance < DISTANCE_THRESHOLD:
        # Step 4: Look up tumor details in CSV file
        tumor_details = lookup_tumor_details(matching_image)
        
        if tumor_details:
            # Calculate confidence score (inverse of distance)
            # Convert distance to confidence percentage (lower distance = higher confidence)
            confidence = max(0, 100 - (distance / MAX_DISTANCE * 100))
            
            return {
                "result": "positive",
                "tumor_name": tumor_details["tumor_name"],
                "tumor_type": tumor_details["tumor_type"],
                "confidence": f"{confidence:.1f}%",
                "matched_image": matching_image,
                "distance": float(distance),
                "cnn_score": f"{tumor_prediction:.4f}",
                "message": f"Brain tumor detected - Type: {tumor_details['tumor_type']}, Name: {tumor_details['tumor_name']} (Match confidence: {confidence:.1f}%)"
            }
    
    # Fallback: Use KNN model if no good match found
    tumor_type_encoded = knn_model.predict(query_features)
    tumor_type = label_encoder.inverse_transform(tumor_type_encoded)[0]
    
    # Include information about closest match if available
    match_info = ""
    if matching_image:
        match_info = f" (Closest database match: {matching_image}, but confidence too low)"
    
    return {
        "result": "positive",
        "tumor_type": tumor_type,
        "confidence": f"{tumor_prediction*100:.1f}%",
        "cnn_score": f"{tumor_prediction:.4f}",
        "message": f"Brain tumor detected - Type: {tumor_type}{match_info}"
    }

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    prediction = None
    selected_image = None
    debug_info = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No file part")
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")
        
        if file:
            # Process the uploaded file
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # Base64 encode the image for displaying on the frontend
                with open(filepath, "rb") as image_file:
                    selected_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Get prediction
                result = predict_tumor(filepath)
                prediction = result.get("message", "Error during prediction")
                
                # Add debug info
                if DEBUG and isinstance(result, dict):
                    debug_info = {
                        "result_type": result.get("result", ""),
                        "confidence": result.get("confidence", ""),
                        "cnn_score": result.get("cnn_score", "")
                    }
                    
                    if "matched_image" in result:
                        debug_info.update({
                            "matched_image": result.get("matched_image", ""),
                            "distance": result.get("distance", 0),
                            "tumor_name": result.get("tumor_name", ""),
                            "tumor_type": result.get("tumor_type", "")
                        })
                
                # Add raw prediction values to debug info
                debug_info["tumor_threshold"] = TUMOR_THRESHOLD
                debug_info["distance_threshold"] = DISTANCE_THRESHOLD
                
                # Clean up uploaded file
                os.remove(filepath)
                
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}", exc_info=True)
                prediction = f"An error occurred during prediction: {str(e)}"
                import traceback
                debug_info = {"error": str(e), "traceback": traceback.format_exc()}

    return render_template('index.html', 
                          prediction=prediction, 
                          selected_image=selected_image,
                          debug_info=debug_info)

# Debug routes for testing
@app.route('/debug/yes_folder', methods=['GET'])
def debug_yes_folder():
    """List all files in the yes folder"""
    yes_folder = Path(app.config['YES_FOLDER'])
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(app.config['YES_FOLDER'], ext)))
    
    return {
        "yes_folder_exists": yes_folder.exists(),
        "yes_folder_path": str(yes_folder.absolute()),
        "image_count": len(image_files),
        "images": [os.path.basename(f) for f in image_files]
    }

@app.route('/debug/no_folder', methods=['GET'])
def debug_no_folder():
    """List all files in the no folder"""
    no_folder = Path(app.config['NO_FOLDER'])
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(app.config['NO_FOLDER'], ext)))
    
    return {
        "no_folder_exists": no_folder.exists(),
        "no_folder_path": str(no_folder.absolute()),
        "image_count": len(image_files),
        "images": [os.path.basename(f) for f in image_files]
    }

@app.route('/debug/csv', methods=['GET'])
def debug_csv():
    """Show CSV data"""
    return {
        "total_rows": len(csv_data_clean),
        "columns": csv_data_clean.columns.tolist(),
        "samples": csv_data_clean[['Imagefilename', 'tumor name', 'tumor type']].head(10).to_dict('records')
    }

@app.route('/debug/thresholds', methods=['GET'])
def debug_thresholds():
    """Show current threshold settings"""
    return {
        "tumor_threshold": TUMOR_THRESHOLD,
        "distance_threshold": DISTANCE_THRESHOLD,
        "max_distance": MAX_DISTANCE
    }

@app.route('/debug/prediction/<float:score>', methods=['GET'])
def debug_prediction(score):
    """Test what prediction would be made with a given CNN score"""
    result = "POSITIVE" if score > TUMOR_THRESHOLD else "NEGATIVE"
    uncertain = "Yes" if 0.3 < score < 0.7 else "No"
    return {
        "score": score,
        "result": result,
        "in_uncertain_range": uncertain,
        "tumor_threshold": TUMOR_THRESHOLD
    }

if __name__ == '__main__':
     app.run(debug=True, port=5001)