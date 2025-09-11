"""Utility functions for text preprocessing and model operations."""

import re
import string
import joblib
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess text for phishing detection.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Cleaned text ready for vectorization
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", "", text)
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Strip whitespace
    return text.strip()

def load_model(model_path: str, vectorizer_path: str) -> Tuple[object, object]:
    """
    Load the trained model and vectorizer.
    
    Args:
        model_path: Path to the saved model file
        vectorizer_path: Path to the saved vectorizer file
        
    Returns:
        Tuple of (model, vectorizer)
        
    Raises:
        FileNotFoundError: If model or vectorizer files don't exist
        Exception: If there's an error loading the files
    """
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def predict_message(model, vectorizer, message: str) -> Tuple[int, float]:
    """
    Predict if a message is phishing or legitimate.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        message: Message to classify
        
    Returns:
        Tuple of (prediction, confidence_score)
    """
    try:
        # Preprocess the message
        cleaned_message = preprocess_text(message)
        
        # Vectorize the message
        vectorized = vectorizer.transform([cleaned_message])
        
        # Get prediction
        prediction = model.predict(vectorized)[0]
        
        # Get confidence score (probability)
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(vectorized)[0].max()
        else:
            confidence = 1.0  # Fallback if model doesn't support probabilities
        
        return int(prediction), float(confidence)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
