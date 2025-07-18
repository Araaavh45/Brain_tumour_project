#!/usr/bin/env python3
"""
Create demo models for the Brain Tumor Detection System
This script creates placeholder models that can be used for demonstration
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up paths
BASE_DIR = Path(__file__).parent
BRAIN_TUMOR_DIR = BASE_DIR / "Brain Tummor"
DATASETS_PATH = BRAIN_TUMOR_DIR / "DATASETS"

print(f"Base directory: {BASE_DIR}")
print(f"Brain Tumor directory: {BRAIN_TUMOR_DIR}")
print(f"Datasets path: {DATASETS_PATH}")

def create_sample_csv_data():
    """Create sample CSV data for tumor classification"""
    sample_data = [
        {'Imagefilename': 'y1', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y2', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y3', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y4', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y5', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y6', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y7', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y8', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y9', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y10', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y11', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y12', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y13', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y14', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y15', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y16', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y17', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y18', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y19', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y20', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
    ]
    
    df = pd.DataFrame(sample_data)
    csv_path = BRAIN_TUMOR_DIR / "tumor_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV data at: {csv_path}")
    return df

def create_demo_knn_model():
    """Create a demo KNN model for tumor classification"""
    print("Creating demo KNN model...")
    
    # Create sample feature data (128 features to match expected input)
    np.random.seed(42)
    n_samples = 100
    n_features = 128
    
    # Generate synthetic features
    features = np.random.randn(n_samples, n_features)
    
    # Create labels for different tumor types
    tumor_types = ['Glioma', 'Meningioma', 'Pituitary']
    labels = np.random.choice(tumor_types, n_samples)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )
    
    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_model.fit(X_train, y_train)
    
    # Test accuracy
    train_accuracy = knn_model.score(X_train, y_train)
    test_accuracy = knn_model.score(X_test, y_test)
    
    print(f"KNN Model - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return knn_model, label_encoder

def create_demo_cnn_model():
    """Create a demo CNN model using RandomForest as a substitute"""
    print("Creating demo CNN model (using RandomForest as substitute)...")
    
    # Create sample image-like features (flattened 256x256x3 would be too large, so we use smaller)
    np.random.seed(42)
    n_samples = 200
    n_features = 1000  # Reduced feature size for demo
    
    # Generate synthetic features
    features = np.random.randn(n_samples, n_features)
    
    # Create binary labels (0 = no tumor, 1 = tumor)
    labels = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 40% positive cases
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Train RandomForest as CNN substitute
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Test accuracy
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    
    print(f"CNN Model (RandomForest) - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return rf_model

def save_models(knn_model, label_encoder, cnn_model):
    """Save all models to files"""
    print("Saving models...")
    
    # Save KNN model
    knn_path = BRAIN_TUMOR_DIR / "knn_model.pkl"
    joblib.dump(knn_model, knn_path)
    print(f"KNN model saved to: {knn_path}")
    
    # Save improved KNN model (same model, different name)
    knn_improved_path = BRAIN_TUMOR_DIR / "improved_knn_model_85.pkl"
    joblib.dump(knn_model, knn_improved_path)
    print(f"Improved KNN model saved to: {knn_improved_path}")
    
    # Save label encoder
    encoder_path = BRAIN_TUMOR_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Save improved label encoder (same encoder, different name)
    encoder_improved_path = BRAIN_TUMOR_DIR / "label_encoder_85.pkl"
    joblib.dump(label_encoder, encoder_improved_path)
    print(f"Improved label encoder saved to: {encoder_improved_path}")
    
    # Save CNN model (RandomForest substitute)
    cnn_path = BRAIN_TUMOR_DIR / "model.pkl"
    joblib.dump(cnn_model, cnn_path)
    print(f"CNN model (RandomForest) saved to: {cnn_path}")
    
    # Save improved CNN model
    cnn_improved_path = BRAIN_TUMOR_DIR / "improved_model_85.pkl"
    joblib.dump(cnn_model, cnn_improved_path)
    print(f"Improved CNN model saved to: {cnn_improved_path}")
    
    # Create a simple feature scaler (identity scaler for demo)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit on dummy data
    dummy_data = np.random.randn(100, 128)
    scaler.fit(dummy_data)
    
    scaler_path = BRAIN_TUMOR_DIR / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Feature scaler saved to: {scaler_path}")
    
    scaler_improved_path = BRAIN_TUMOR_DIR / "feature_scaler_85.pkl"
    joblib.dump(scaler, scaler_improved_path)
    print(f"Improved feature scaler saved to: {scaler_improved_path}")

def main():
    """Main function"""
    print("Creating demo models for Brain Tumor Detection System...")
    
    # Create directories if they don't exist
    BRAIN_TUMOR_DIR.mkdir(exist_ok=True)
    
    # Create sample CSV data
    csv_data = create_sample_csv_data()
    
    # Create demo models
    knn_model, label_encoder = create_demo_knn_model()
    cnn_model = create_demo_cnn_model()
    
    # Save models
    save_models(knn_model, label_encoder, cnn_model)
    
    print("\nDemo models created successfully!")
    print("The system is now ready to run with these placeholder models.")
    print("Note: These are demo models for testing purposes. For production use,")
    print("you should train proper CNN models with TensorFlow/Keras.")

if __name__ == "__main__":
    main()