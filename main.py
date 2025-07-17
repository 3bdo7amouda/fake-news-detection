"""
Simple Fake News Detection
"""
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Clean text
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_kaggle_datasets(self, fake_csv_path, true_csv_path):
        """Load real Kaggle datasets with proper balancing"""
        try:
            # Load fake news dataset
            fake_df = pd.read_csv(fake_csv_path)
            fake_df['label'] = 0  # Fake = 0
            
            # Load true news dataset  
            true_df = pd.read_csv(true_csv_path)
            true_df['label'] = 1  # Real = 1
            
            # Balance the datasets to prevent bias
            min_size = min(len(fake_df), len(true_df))
            fake_df = fake_df.sample(n=min_size, random_state=42)
            true_df = true_df.sample(n=min_size, random_state=42)
            
            # Combine datasets
            combined_df = pd.concat([fake_df, true_df], ignore_index=True)
            
            # Shuffle the combined dataset
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"Loaded {len(fake_df)} fake and {len(true_df)} real articles (balanced)")
            print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
            
            return combined_df
            
        except Exception as e:
            print(f"Error loading Kaggle datasets: {e}")
            print("Please ensure the Kaggle dataset files are available:")
            print(f"- Fake news CSV: {fake_csv_path}")
            print(f"- True news CSV: {true_csv_path}")
            raise Exception("Kaggle datasets are required for training")

    def train(self, fake_csv_path=None, true_csv_path=None):
        """Train the model with Kaggle datasets only"""
        if not fake_csv_path or not true_csv_path:
            print("Error: Both fake_csv_path and true_csv_path are required")
            print("Please provide paths to the Kaggle dataset files:")
            print("- Fake.csv (fake news dataset)")
            print("- True.csv (real news dataset)")
            raise ValueError("Kaggle dataset paths are required")
        
        df = self.load_kaggle_datasets(fake_csv_path, true_csv_path)
        
        # Prepare data
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        X = df['processed_text']
        y = df['label']
        
        print(f"Training with {len(X)} articles")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Use improved TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Use Logistic Regression with balanced class weights
        self.model = LogisticRegression(
            class_weight='balanced',  # This helps with any remaining class imbalance
            random_state=42,
            max_iter=1000,
            C=1.0  # Regularization parameter
        )
        
        self.model.fit(X_train_tfidf, y_train)
        
        # Test the model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model trained and saved!")
    
    def predict(self, text, title=""):
        """Predict if news is fake or real"""
        # Load model if needed
        if not self.model:
            try:
                with open('models/model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open('models/vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            except:
                print("Training model...")
                self.train()
        
        # Make prediction
        combined_text = str(title) + ' ' + str(text)
        processed_text = self.preprocess_text(combined_text)
        
        if not processed_text:
            return {
                'prediction': 'Unknown',
                'confidence': 50.0,
                'is_real': False
            }
        
        text_tfidf = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        confidence = max(probability) * 100
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': confidence,
            'is_real': prediction == 1
        }

def demo():
    """Run demo with example articles"""
    detector = FakeNewsDetector()
    
    examples = [
        ("Scientists Discover Cancer Treatment", "Researchers develop new treatment"),
        ("SHOCKING: Government Hiding Aliens", "Anonymous sources claim evidence"),
        ("Miracle Weight Loss Pill", "Amazing pill helps lose weight fast"),
        ("Stock Market Hits Record High", "Markets showed strong performance"),
        ("Secret Mind Control Program", "Whistleblower reveals program"),
        ("Medical Journal Research", "Journal publishes breakthrough research")
    ]
    
    print("🔍 Fake News Detection Demo")
    print("=" * 40)
    
    for title, text in examples:
        result = detector.predict(text, title)
        print(f"'{title}' → {result['prediction']} ({result['confidence']:.1f}%)")

def detect():
    """Detect fake news in user input"""
    detector = FakeNewsDetector()
    
    print("🔍 Fake News Detection")
    print("=" * 40)
    print("Enter your news article:")
    
    title = input("Title: ").strip()
    text = input("Text: ").strip()
    
    if text:
        result = detector.predict(text, title)
        print(f"\n📰 Result: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.1f}%")
    else:
        print("Please enter some text to analyze")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Demo (see examples)")
    print("2. Detect (analyze your own text)")
    
    try:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            demo()
        elif choice == "2":
            detect()
        else:
            print("Invalid choice. Running demo...")
            demo()
    except:
        demo()