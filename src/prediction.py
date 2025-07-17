"""
Prediction module for fake news detection.
This module provides the main interface for making predictions on new text.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any
import os

from .utils import Config, logger, load_model, calculate_confidence, format_text_for_display
from .data_preprocessing import DataPreprocessor

class FakeNewsDetector:
    """
    Main prediction class for fake news detection.
    Provides easy-to-use interface for making predictions.
    """
    
    def __init__(self, model_name: str = 'logistic_regression'):
        """
        Initialize the fake news detector.
        
        Args:
            model_name: Name of the model to use for predictions
        """
        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
    def load_model(self, model_name: str = None) -> None:
        """
        Load a trained model and preprocessor.
        
        Args:
            model_name: Name of the model to load (optional)
        """
        if model_name:
            self.model_name = model_name
            
        try:
            # Load the model
            self.model = load_model(self.model_name)
            
            # Load the preprocessor
            self.preprocessor = load_model('preprocessor')
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except FileNotFoundError as e:
            logger.error(f"Model or preprocessor not found: {e}")
            logger.info("Please train the models first by running: python main.py --train")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Predict whether a news article is fake or real.
        
        Args:
            text: The news article text
            title: The news article title (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            self.load_model()
        
        # Combine title and text
        combined_text = f"{title} {text}".strip()
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess_text(combined_text)
        
        # Extract features
        features = self.preprocessor.extract_features([processed_text], fit=False)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Calculate confidence
        confidence = calculate_confidence(probabilities)
        
        # Prepare result
        result = {
            'text': format_text_for_display(combined_text),
            'prediction': int(prediction),
            'is_real': bool(prediction),
            'is_fake': not bool(prediction),
            'confidence': confidence,
            'probabilities': {
                'fake': round(probabilities[0] * 100, 2),
                'real': round(probabilities[1] * 100, 2)
            },
            'model_used': self.model_name
        }
        
        return result
    
    def predict_batch(self, texts: List[str], titles: List[str] = None) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of news article texts
            titles: List of news article titles (optional)
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            self.load_model()
        
        if titles is None:
            titles = [""] * len(texts)
        
        results = []
        for text, title in zip(texts, titles):
            result = self.predict(text, title)
            results.append(result)
        
        return results
    
    def get_prediction_summary(self, result: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of the prediction.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Formatted summary string
        """
        status = "REAL" if result['is_real'] else "FAKE"
        confidence = result['confidence']
        
        summary = f"""
Prediction Summary:
==================
Status: {status}
Confidence: {confidence}%
Model Used: {result['model_used']}

Probability Breakdown:
- Real News: {result['probabilities']['real']}%
- Fake News: {result['probabilities']['fake']}%

Text Preview:
{result['text']}
"""
        return summary
    
    def evaluate_url(self, url: str) -> Dict[str, Any]:
        """
        Evaluate a news article from a URL.
        
        Args:
            url: URL of the news article
            
        Returns:
            Prediction result
        """
        # This would require web scraping implementation
        # For now, return a placeholder
        return {
            'error': 'URL evaluation not implemented yet',
            'message': 'Please copy and paste the article text directly'
        }

class ModelComparator:
    """
    Class for comparing predictions from multiple models.
    """
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize the model comparator.
        
        Args:
            model_names: List of model names to compare
        """
        if model_names is None:
            model_names = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
        
        self.model_names = model_names
        self.detectors = {}
        
        # Initialize detectors for each model
        for model_name in model_names:
            self.detectors[model_name] = FakeNewsDetector(model_name)
    
    def compare_predictions(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Compare predictions from multiple models.
        
        Args:
            text: The news article text
            title: The news article title (optional)
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for model_name, detector in self.detectors.items():
            try:
                result = detector.predict(text, title)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Calculate consensus
        predictions = [r.get('prediction', 0) for r in results.values() if 'prediction' in r]
        confidences = [r.get('confidence', 0) for r in results.values() if 'confidence' in r]
        
        if predictions:
            consensus = {
                'majority_vote': int(np.mean(predictions) > 0.5),
                'average_confidence': np.mean(confidences),
                'agreement_rate': len([p for p in predictions if p == predictions[0]]) / len(predictions)
            }
        else:
            consensus = {'error': 'No valid predictions'}
        
        return {
            'individual_results': results,
            'consensus': consensus,
            'text_preview': format_text_for_display(f"{title} {text}".strip())
        }
    
    def get_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """
        Get a formatted summary of model comparison.
        
        Args:
            comparison: Comparison results
            
        Returns:
            Formatted summary string
        """
        individual = comparison['individual_results']
        consensus = comparison['consensus']
        
        summary = f"""
Model Comparison Summary:
========================
Text Preview: {comparison['text_preview']}

Individual Model Results:
"""
        
        for model_name, result in individual.items():
            if 'error' in result:
                summary += f"- {model_name}: ERROR - {result['error']}\n"
            else:
                status = "REAL" if result['is_real'] else "FAKE"
                confidence = result['confidence']
                summary += f"- {model_name}: {status} ({confidence}% confidence)\n"
        
        if 'error' not in consensus:
            majority_status = "REAL" if consensus['majority_vote'] else "FAKE"
            summary += f"""
Consensus:
- Majority Vote: {majority_status}
- Average Confidence: {consensus['average_confidence']:.2f}%
- Model Agreement: {consensus['agreement_rate']:.2f}
"""
        
        return summary