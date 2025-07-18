import os
from datetime import timedelta

class Config:
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'brain-tumor-detection-secret-key-2024'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///brain_tumor_system.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Model paths (relative to project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BRAIN_TUMOR_DIR = os.path.join(BASE_DIR, 'Brain Tummor')
    
    # Model file paths
    CNN_MODEL_PATH = os.path.join(BRAIN_TUMOR_DIR, 'improved_model_85.h5')
    KNN_MODEL_PATH = os.path.join(BRAIN_TUMOR_DIR, 'improved_knn_model_85.pkl')
    LABEL_ENCODER_PATH = os.path.join(BRAIN_TUMOR_DIR, 'label_encoder_85.pkl')
    FEATURE_SCALER_PATH = os.path.join(BRAIN_TUMOR_DIR, 'feature_scaler_85.pkl')
    
    # Fallback model paths
    CNN_MODEL_PATH_FALLBACK = os.path.join(BRAIN_TUMOR_DIR, 'model.h5')
    KNN_MODEL_PATH_FALLBACK = os.path.join(BRAIN_TUMOR_DIR, 'knn_model.pkl')
    LABEL_ENCODER_PATH_FALLBACK = os.path.join(BRAIN_TUMOR_DIR, 'label_encoder.pkl')
    FEATURE_SCALER_PATH_FALLBACK = os.path.join(BRAIN_TUMOR_DIR, 'feature_scaler.pkl')
    
    # Dataset paths
    DATASETS_PATH = os.path.join(BRAIN_TUMOR_DIR, 'DATASETS')
    YES_FOLDER = os.path.join(DATASETS_PATH, 'yes')
    NO_FOLDER = os.path.join(DATASETS_PATH, 'no')
    
    # Alternative dataset paths
    YES_FOLDER_ALT = os.path.join(DATASETS_PATH, 'brain_tumor_dataset', 'yes')
    NO_FOLDER_ALT = os.path.join(DATASETS_PATH, 'brain_tumor_dataset', 'no')
    
    # CSV data path (you can create this or use existing data)
    CSV_DATA_PATH = os.path.join(BRAIN_TUMOR_DIR, 'tumor_data.csv')
    
    # Prediction thresholds
    TUMOR_THRESHOLD = 0.55
    DISTANCE_THRESHOLD = 50.0
    MAX_DISTANCE = 100.0
    
    # Email configuration (for notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Pagination
    POSTS_PER_PAGE = 10
    USERS_PER_PAGE = 20
    
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}