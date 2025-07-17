"""
Model training module for fake news detection.
This module implements multiple machine learning models for classification.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .utils import Config, logger, save_model, get_model_metrics, print_model_performance
from .data_preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Model training class for fake news detection.
    Implements multiple ML algorithms and provides comparison.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'svm': SVC(
                random_state=Config.RANDOM_STATE,
                probability=True,
                kernel='linear'
            ),
            'naive_bayes': MultinomialNB()
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.preprocessor = None
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of trained models and performances
        """
        logger.info("Starting model training...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = get_model_metrics(y_test, y_pred)
            
            # Store results
            self.trained_models[model_name] = model
            self.model_performances[model_name] = metrics
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
            
            # Print performance
            print_model_performance(model_name, metrics)
            
            # Save model
            save_model(model, model_name)
            
        logger.info("Model training completed!")
        return results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on F1-score.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.model_performances:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        best_model_name = max(
            self.model_performances.keys(),
            key=lambda x: self.model_performances[x]['f1_score']
        )
        
        best_model = self.trained_models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} "
                   f"(F1-score: {self.model_performances[best_model_name]['f1_score']:.4f})")
        
        return best_model_name, best_model
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Plot comparison of model performances.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.model_performances:
            raise ValueError("No models trained yet. Call train_models() first.")
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.model_performances).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(metrics_df.index, metrics_df[metric], color=colors[i])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str, y_true: np.ndarray, 
                            y_pred: np.ndarray, save_path: str = None) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_classification_report(self, model_name: str, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> None:
        """
        Generate and print classification report.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
        """
        print(f"\nClassification Report - {model_name}")
        print("=" * 50)
        print(classification_report(y_true, y_pred, 
                                  target_names=['Fake', 'Real']))
    
    def train_full_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """
        Complete training pipeline from data loading to model evaluation.
        
        Args:
            data_path: Path to the dataset (optional, uses sample data if None)
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting full training pipeline...")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Load data
        if data_path and validate_data_file(data_path):
            df = self.preprocessor.load_data(data_path)
        else:
            logger.info("Using sample data for demonstration")
            df = self.preprocessor.get_sample_data()
        
        # Prepare dataset
        df_processed = self.preprocessor.prepare_dataset(df)
        
        # Extract features
        X = self.preprocessor.extract_features(df_processed['processed_text'].tolist())
        y = df_processed['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train models
        results = self.train_models(X_train, y_train, X_test, y_test)
        
        # Save preprocessor
        save_model(self.preprocessor, 'preprocessor')
        
        # Get best model
        best_model_name, best_model = self.get_best_model()
        
        # Plot comparison
        self.plot_model_comparison()
        
        # Generate detailed report for best model
        y_pred_best = results[best_model_name]['predictions']
        self.generate_classification_report(best_model_name, y_test, y_pred_best)
        self.plot_confusion_matrix(best_model_name, y_test, y_pred_best)
        
        return {
            'best_model': best_model_name,
            'results': results,
            'preprocessor': self.preprocessor
        }

def validate_data_file(filepath: str) -> bool:
    """Import the validation function from utils."""
    from .utils import validate_data_file as validate_func
    return validate_func(filepath)