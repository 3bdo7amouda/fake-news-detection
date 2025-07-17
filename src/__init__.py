"""
Fake News Detection Project - Source Package
This package contains all the core modules for fake news detection.
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"
__description__ = "AI-powered fake news detection system"

# Import main classes for easy access
from .prediction import FakeNewsDetector, ModelComparator
from .model_training import ModelTrainer
from .data_preprocessing import DataPreprocessor
from .utils import Config

__all__ = [
    'FakeNewsDetector',
    'ModelComparator', 
    'ModelTrainer',
    'DataPreprocessor',
    'Config'
]