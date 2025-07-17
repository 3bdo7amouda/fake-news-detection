"""
Simple Fake News Detection using one model
"""

import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import argparse
import os

# Download required NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_text(self, text):
        """Complete text preprocessing"""
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def load_data(self, filepath):
        """Load the dataset"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Return sample data if file not found
            return self.get_sample_data()
    
    def get_sample_data(self):
        """Create sample data for testing"""
        data = {
            'title': [
                'Scientists Discover Breakthrough Cancer Treatment',
                'SHOCKING: Government Hiding Alien Contact',
                'Stock Market Reaches Record High',
                'Miracle Weight Loss Pill Doctors Hate',
                'New Research Shows Climate Change Impact',
                'BREAKING: Celebrity Found Dead in Home',
                'Local School Wins National Award',
                'Secret Government Mind Control Program Exposed',
                'Technology Company Announces New Product',
                'Incredible Home Remedy Cures Everything'
            ],
            'text': [
                'Medical researchers have made significant progress in cancer treatment through rigorous clinical trials.',
                'Anonymous sources claim the government has been hiding evidence of extraterrestrial contact for decades.',
                'Financial markets showed strong performance today as investors responded to positive economic indicators.',
                'This amazing pill will help you lose weight fast without diet or exercise, doctors dont want you to know.',
                'Climate scientists publish peer-reviewed study showing measurable impacts of global warming.',
                'Police are investigating the suspicious death of a well-known celebrity at their residence.',
                'Local high school students achieved excellence in national academic competition.',
                'Whistleblower reveals top-secret program designed to control peoples thoughts and behaviors.',
                'Tech giant unveils innovative product following years of development and testing.',
                'This one weird trick from your kitchen will solve all your health problems instantly.'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
        }
        
        return pd.DataFrame(data)
    
    def train(self, data_path=None):
        """Train the fake news detection model"""
        logger.info("Starting model training...")
        
        # Load data
        df = self.load_data(data_path) if data_path else self.get_sample_data()
        
        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        # Combine title and text
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess text
        logger.info("Preprocessing text...")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['label']
        
        logger.info(f"Training data: {len(X)} samples")
        logger.info(f"Real news: {sum(y)} samples")
        logger.info(f"Fake news: {len(y) - sum(y)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Save model
        self.save_model()
        logger.info("Model training completed!")
        
        return accuracy
    
    def predict(self, text, title=""):
        """Predict if news is fake or real"""
        if not self.model or not self.vectorizer:
            self.load_model()
        
        # Combine title and text
        combined_text = title + ' ' + text
        
        # Preprocess
        processed_text = self.preprocess_text(combined_text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        # Calculate confidence
        confidence = max(probability) * 100
        
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': confidence,
            'is_real': prediction == 1,
            'probabilities': {
                'fake': probability[0] * 100,
                'real': probability[1] * 100
            }
        }
        
        return result
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        # Save model and vectorizer
        with open('models/fake_news_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info("Model saved successfully!")
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('models/fake_news_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("Model loaded successfully!")
        except FileNotFoundError:
            logger.error("Model files not found. Please train the model first.")
            raise

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Interactive prediction')
    parser.add_argument('--demo', action='store_true', help='Run demo examples')
    parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    detector = FakeNewsDetector()
    
    if args.train:
        print("🤖 Training Fake News Detection Model")
        print("=" * 50)
        detector.train(args.data)
        
    elif args.predict:
        print("🔍 Fake News Detection - Interactive Mode")
        print("=" * 50)
        
        try:
            detector.load_model()
            print("✅ Model loaded successfully!")
            
            while True:
                print("\nEnter news article (or 'quit' to exit):")
                title = input("Title: ").strip()
                if title.lower() == 'quit':
                    break
                
                text = input("Text: ").strip()
                if text.lower() == 'quit':
                    break
                
                result = detector.predict(text, title)
                
                print(f"\n🔍 Prediction: {result['prediction']}")
                print(f"📊 Confidence: {result['confidence']:.1f}%")
                print(f"📈 Probabilities: Real={result['probabilities']['real']:.1f}%, Fake={result['probabilities']['fake']:.1f}%")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please train the model first with: python main.py --train")
    
    elif args.demo:
        print("📚 Fake News Detection - Demo Examples")
        print("=" * 50)
        
        try:
            detector.load_model()
            
            examples = [
                {
                    'title': 'Scientists Discover New Treatment for Cancer',
                    'text': 'Researchers at major university have developed promising new treatment',
                    'expected': 'Real'
                },
                {
                    'title': 'SHOCKING: Aliens Found in Government Facility',
                    'text': 'Government officials deny but sources confirm extraterrestrial beings held',
                    'expected': 'Fake'
                },
                {
                    'title': 'Miracle Cure Discovered by Local Mom',
                    'text': 'This amazing trick will cure all diseases doctors hate this simple method',
                    'expected': 'Fake'
                }
            ]
            
            for i, example in enumerate(examples, 1):
                result = detector.predict(example['text'], example['title'])
                
                print(f"\nExample {i}:")
                print(f"Title: {example['title']}")
                print(f"Expected: {example['expected']}")
                print(f"Predicted: {result['prediction']} ({result['confidence']:.1f}% confidence)")
                
                if result['prediction'] == example['expected']:
                    print("✅ Correct!")
                else:
                    print("❌ Incorrect!")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please train the model first with: python main.py --train")
    
    else:
        print("Fake News Detection System")
        print("Usage:")
        print("  python main.py --train [--data path/to/dataset.csv]")
        print("  python main.py --predict")
        print("  python main.py --demo")

if __name__ == "__main__":
    main()