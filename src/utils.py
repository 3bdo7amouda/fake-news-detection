"""
Utility functions for the fake news detection project.
This module contains helper functions used across the project.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the project."""
    
    # File paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    NOTEBOOKS_DIR = "notebooks"
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 5000
    
    # Dataset URLs (for reference)
    KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets"

def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [Config.DATA_DIR, Config.MODELS_DIR, Config.NOTEBOOKS_DIR]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def save_model(model: Any, filename: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model to save
        filename: Name of the file to save (without extension)
    """
    filepath = os.path.join(Config.MODELS_DIR, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {filepath}")

def load_model(filename: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filename: Name of the file to load (without extension)
        
    Returns:
        The loaded model
    """
    filepath = os.path.join(Config.MODELS_DIR, f"{filename}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded: {filepath}")
    return model

def calculate_confidence(probabilities: np.ndarray) -> float:
    """
    Calculate confidence percentage from model probabilities.
    
    Args:
        probabilities: Array of class probabilities
        
    Returns:
        Confidence percentage (0-100)
    """
    max_prob = np.max(probabilities)
    confidence = max_prob * 100
    return round(confidence, 2)

def format_text_for_display(text: str, max_length: int = 500) -> str:
    """
    Format text for display by truncating if too long.
    
    Args:
        text: Input text
        max_length: Maximum length for display
        
    Returns:
        Formatted text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return {k: round(v, 4) for k, v in metrics.items()}

def print_model_performance(model_name: str, metrics: Dict[str, float]) -> None:
    """
    Print model performance metrics in a formatted way.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of performance metrics
    """
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("-" * 40)

def validate_data_file(filepath: str) -> bool:
    """
    Validate that the data file exists and is readable.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        return False
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.error("Data file is empty")
            return False
        logger.info(f"Data file validated: {filepath} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.error(f"Error reading data file: {e}")
        return False