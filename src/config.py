"""Configuration settings for PhishGuard application."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Model files
MODEL_FILE = MODEL_DIR / "phishing_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

# Flask configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MODEL_PATH = str(MODEL_FILE)
    VECTORIZER_PATH = str(VECTORIZER_FILE)
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
