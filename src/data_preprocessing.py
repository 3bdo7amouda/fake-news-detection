"""
Data preprocessing module for fake news detection.
This module handles data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging

from .utils import Config, logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class DataPreprocessor:
    """
    Data preprocessing class for fake news detection.
    Handles data loading, cleaning, and feature extraction.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the fake news dataset from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the dataset for training.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['title'] = df_processed['title'].fillna('')
        df_processed['text'] = df_processed['text'].fillna('')
        df_processed['subject'] = df_processed['subject'].fillna('')
        
        # Combine title and text for better feature extraction
        df_processed['combined_text'] = (
            df_processed['title'] + ' ' + df_processed['text']
        )
        
        # Preprocess the combined text
        logger.info("Preprocessing text data...")
        df_processed['processed_text'] = df_processed['combined_text'].apply(
            self.preprocess_text
        )
        
        # Remove rows with empty processed text
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        logger.info(f"Dataset prepared: {len(df_processed)} rows after preprocessing")
        return df_processed
    
    def extract_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from text data.
        
        Args:
            texts: List of text strings
            fit: Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            Feature matrix
        """
        if fit:
            features = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            logger.info(f"Features extracted: {features.shape}")
        else:
            if not self.is_fitted:
                raise ValueError("Vectorizer not fitted. Call with fit=True first.")
            features = self.vectorizer.transform(texts)
        
        return features.toarray()
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Check if dataset is too small for stratified splitting
        min_samples_per_class = 2
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # If any class has fewer than min_samples_per_class, don't stratify
        if any(count < min_samples_per_class for count in class_counts) or len(y) < 10:
            logger.warning(f"Dataset too small for stratified splitting ({len(y)} samples). Using simple random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=max(0.2, 1/len(y)),  # Ensure at least 1 test sample
                random_state=Config.RANDOM_STATE
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=Config.TEST_SIZE, 
                random_state=Config.RANDOM_STATE,
                stratify=y
            )
        
        logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for testing when dataset is not available.
        
        Returns:
            Sample DataFrame
        """
        sample_data = {
            'title': [
                'Scientists discover new treatment for cancer',
                'SHOCKING: Aliens found in government facility',
                'Stock market reaches new high',
                'Miracle cure discovered by local mom',
                'Government announces new policy changes'
            ],
            'text': [
                'Researchers at major university have developed a promising new treatment for cancer that shows significant results in clinical trials.',
                'Government officials deny but sources confirm that extraterrestrial beings are being held at secret facility.',
                'The stock market experienced significant gains today as investors responded positively to economic indicators.',
                'Local mother discovers amazing cure that doctors hate using this one simple trick from her kitchen.',
                'The government has announced several new policy changes that will affect citizens starting next month.'
            ],
            'subject': ['Health', 'Conspiracy', 'Business', 'Health', 'Politics'],
            'label': [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
        }
        
        return pd.DataFrame(sample_data)