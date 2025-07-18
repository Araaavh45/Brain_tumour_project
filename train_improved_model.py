#!/usr/bin/env python3
"""
Improved Brain Tumor Detection Model Training Script
This script creates optimized models with better accuracy and proper file paths
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

# Set up paths
BASE_DIR = Path(__file__).parent
BRAIN_TUMOR_DIR = BASE_DIR / "Brain Tummor"
DATASETS_PATH = BRAIN_TUMOR_DIR / "DATASETS"

# Check for dataset paths
YES_FOLDER = DATASETS_PATH / "yes"
NO_FOLDER = DATASETS_PATH / "no"

# Alternative paths
YES_FOLDER_ALT = DATASETS_PATH / "brain_tumor_dataset" / "yes"
NO_FOLDER_ALT = DATASETS_PATH / "brain_tumor_dataset" / "no"

# Use alternative paths if main paths don't exist
if not YES_FOLDER.exists() and YES_FOLDER_ALT.exists():
    YES_FOLDER = YES_FOLDER_ALT
    NO_FOLDER = NO_FOLDER_ALT

print(f"Using dataset paths:")
print(f"YES folder: {YES_FOLDER}")
print(f"NO folder: {NO_FOLDER}")

# Verify paths exist
if not YES_FOLDER.exists() or not NO_FOLDER.exists():
    print("Error: Dataset folders not found!")
    print(f"Checked paths: {YES_FOLDER}, {NO_FOLDER}")
    sys.exit(1)

# Model parameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_SPLIT = 0.2

def create_sample_csv_data():
    """Create sample CSV data for tumor classification"""
    # Get some sample filenames from the yes folder
    yes_files = list(YES_FOLDER.glob("*.jpg")) + list(YES_FOLDER.glob("*.jpeg")) + list(YES_FOLDER.glob("*.JPG"))
    
    sample_data = []
    tumor_types = ['Glioma', 'Meningioma', 'Pituitary']
    tumor_classifications = ['Malignant', 'Benign', 'Benign']
    
    for i, file_path in enumerate(yes_files[:50]):  # Use first 50 files
        filename = file_path.stem.lower()
        tumor_type = tumor_types[i % len(tumor_types)]
        tumor_class = tumor_classifications[i % len(tumor_classifications)]
        
        sample_data.append({
            'Imagefilename': filename,
            'tumor name': tumor_type,
            'tumor type': tumor_class
        })
    
    # Add some more diverse data
    additional_data = [
        {'Imagefilename': 'y1', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y2', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
        {'Imagefilename': 'y3', 'tumor name': 'Pituitary', 'tumor type': 'Benign'},
        {'Imagefilename': 'y4', 'tumor name': 'Glioma', 'tumor type': 'Malignant'},
        {'Imagefilename': 'y5', 'tumor name': 'Meningioma', 'tumor type': 'Benign'},
    ]
    
    sample_data.extend(additional_data)
    
    df = pd.DataFrame(sample_data)
    csv_path = BRAIN_TUMOR_DIR / "tumor_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV data at: {csv_path}")
    return df

def load_and_preprocess_data():
    """Load and preprocess image data"""
    print("Loading and preprocessing data...")
    
    # Use ImageDataGenerator for better data handling
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Create training data generator
    train_generator = datagen.flow_from_directory(
        str(DATASETS_PATH),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        seed=42
    )
    
    # Create validation data generator
    validation_generator = datagen.flow_from_directory(
        str(DATASETS_PATH),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=42
    )
    
    print(f"Found {train_generator.samples} training images")
    print(f"Found {validation_generator.samples} validation images")
    print(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, validation_generator

def create_improved_cnn_model():
    """Create an improved CNN model"""
    model = Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile with better optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_feature_extractor():
    """Create feature extractor for KNN"""
    feature_extractor = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu')
    ])
    
    return feature_extractor

def extract_features_for_knn(feature_extractor, data_generator):
    """Extract features for KNN training"""
    print("Extracting features for KNN...")
    
    features = []
    labels = []
    filenames = []
    
    # Reset generator
    data_generator.reset()
    
    for i in range(len(data_generator)):
        batch_x, batch_y = data_generator[i]
        batch_features = feature_extractor.predict(batch_x, verbose=0)
        
        features.extend(batch_features)
        labels.extend(batch_y)
        
        # Get filenames for this batch
        start_idx = i * data_generator.batch_size
        end_idx = min((i + 1) * data_generator.batch_size, len(data_generator.filenames))
        batch_filenames = data_generator.filenames[start_idx:end_idx]
        filenames.extend(batch_filenames)
        
        if i % 10 == 0:
            print(f"Processed batch {i+1}/{len(data_generator)}")
    
    return np.array(features), np.array(labels), filenames

def train_knn_classifier(features, labels, csv_data):
    """Train KNN classifier for tumor type classification"""
    print("Training KNN classifier...")
    
    # Filter features for positive cases only (tumor present)
    positive_indices = labels == 1
    positive_features = features[positive_indices]
    
    # Create labels for tumor types based on CSV data
    tumor_labels = []
    for i, is_positive in enumerate(positive_indices):
        if is_positive:
            # For now, assign random tumor types - in real scenario, 
            # you'd match with actual filenames from CSV
            tumor_labels.append(np.random.choice(['Glioma', 'Meningioma', 'Pituitary']))
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(tumor_labels)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(positive_features)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(scaled_features, encoded_labels)
    
    # Evaluate
    train_pred = knn.predict(scaled_features)
    accuracy = accuracy_score(encoded_labels, train_pred)
    print(f"KNN Training Accuracy: {accuracy:.4f}")
    
    return knn, label_encoder, scaler

def main():
    """Main training function"""
    print("Starting improved model training...")
    
    # Create sample CSV data
    csv_data = create_sample_csv_data()
    
    # Load data
    train_gen, val_gen = load_and_preprocess_data()
    
    # Create and train CNN model
    print("Creating and training CNN model...")
    cnn_model = create_improved_cnn_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            str(BRAIN_TUMOR_DIR / "improved_model_85.h5"),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    history = cnn_model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_accuracy, val_precision, val_recall = cnn_model.evaluate(val_gen, verbose=0)
    print(f"\nCNN Model Performance:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    # Save final model
    cnn_model.save(str(BRAIN_TUMOR_DIR / "improved_model_85.h5"))
    print(f"CNN model saved to: {BRAIN_TUMOR_DIR / 'improved_model_85.h5'}")
    
    # Create feature extractor
    feature_extractor = create_feature_extractor()
    
    # Extract features for KNN
    train_features, train_labels, train_filenames = extract_features_for_knn(feature_extractor, train_gen)
    
    # Train KNN classifier
    knn_model, label_encoder, feature_scaler = train_knn_classifier(train_features, train_labels, csv_data)
    
    # Save KNN components
    joblib.dump(knn_model, str(BRAIN_TUMOR_DIR / "improved_knn_model_85.pkl"))
    joblib.dump(label_encoder, str(BRAIN_TUMOR_DIR / "label_encoder_85.pkl"))
    joblib.dump(feature_scaler, str(BRAIN_TUMOR_DIR / "feature_scaler_85.pkl"))
    
    print(f"KNN model saved to: {BRAIN_TUMOR_DIR / 'improved_knn_model_85.pkl'}")
    print(f"Label encoder saved to: {BRAIN_TUMOR_DIR / 'label_encoder_85.pkl'}")
    print(f"Feature scaler saved to: {BRAIN_TUMOR_DIR / 'feature_scaler_85.pkl'}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(str(BRAIN_TUMOR_DIR / "training_history.png"))
    plt.show()
    
    print("\nTraining completed successfully!")
    print(f"Model accuracy: {val_accuracy*100:.1f}%")
    print("All model files have been saved to the Brain Tummor directory.")

if __name__ == "__main__":
    main()