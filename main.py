"""
Fake News Detection System
"""
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Normalize patterns that might bias the model
        text = re.sub(r'http\S+|www\S+|https\S+', ' weblink ', text)
        text = re.sub(r'\b(reuters|associated press|ap news|cnn|bbc|fox news)\b', 'newsource', text)
        text = re.sub(r'\b(washington|new york|london)\b', 'majorcity', text)
        text = re.sub(r'\b(president|senator|congressman|politician)\b', 'official', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords and short tokens
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def load_datasets(self):
        """Load and balance the training datasets"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fake_df = pd.read_csv(os.path.join(current_dir, 'Fake.csv'))
        true_df = pd.read_csv(os.path.join(current_dir, 'True.csv'))
        
        fake_df['label'] = 0
        true_df['label'] = 1
        
        # Balance datasets
        min_size = min(len(fake_df), len(true_df))
        fake_df = fake_df.sample(n=min_size, random_state=42)
        true_df = true_df.sample(n=min_size, random_state=42)
        
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(fake_df)} fake and {len(true_df)} real articles")
        return combined_df

    def train(self):
        """Train the fake news detection model"""
        df = self.load_datasets()
        
        # Prepare text data
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 50]  # Filter short texts
        
        X, y = df['processed_text'], df['label']
        print(f"Training with {len(X)} articles")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', ngram_range=(1, 2),
            min_df=5, max_df=0.7, sublinear_tf=True
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(
            class_weight='balanced', random_state=42, max_iter=1000, C=0.1, solver='liblinear'
        )
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model saved!")
    
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
        
        # Process and predict
        combined_text = str(title) + ' ' + str(text)
        processed_text = self.preprocess_text(combined_text)
        
        if not processed_text:
            return {'prediction': 'Unknown', 'confidence': 50.0, 'is_real': False}
        
        text_tfidf = self.vectorizer.transform([processed_text])
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Apply calibration - higher threshold for fake classification
        fake_prob, real_prob = probabilities[0], probabilities[1]
        
        if fake_prob > 0.65:  # Calibrated threshold
            final_prediction, confidence = 0, fake_prob * 100
        else:
            final_prediction, confidence = 1, real_prob * 100
        
        # Check for legitimate news patterns
        original_text = combined_text.lower()
        legitimate_patterns = [
            r'\b(reuters|associated press|ap)\b', r'\bpublished in.*journal\b',
            r'\bearnings report\b', r'\bstock market\b', r'\bweather service\b',
            r'\buniversity\b.*\bresearch\b', r'\bgovernment official\b', r'\baccording to.*study\b'
        ]
        
        legitimate_score = sum(1 for pattern in legitimate_patterns if re.search(pattern, original_text))
        
        # Adjust prediction if legitimate patterns found
        if legitimate_score >= 2 and final_prediction == 0 and confidence < 70:
            final_prediction, confidence = 1, 60.0
        
        return {
            'prediction': 'Real' if final_prediction == 1 else 'Fake',
            'confidence': confidence,
            'is_real': final_prediction == 1
        }

def demo():
    """Run demo with example articles"""
    detector = FakeNewsDetector()
    examples = [
        ("Scientists Discover Cancer Treatment", "Researchers develop new treatment"),
        ("SHOCKING: Government Hiding Aliens", "Anonymous sources claim evidence"),
        ("Stock Market Hits Record High", "Markets showed strong performance"),
        ("Secret Mind Control Program", "Whistleblower reveals program")
    ]
    
    print("🔍 Fake News Detection Demo")
    print("=" * 40)
    for title, text in examples:
        result = detector.predict(text, title)
        print(f"'{title}' → {result['prediction']} ({result['confidence']:.1f}%)")

def detect():
    """Interactive fake news detection"""
    detector = FakeNewsDetector()
    print("🔍 Fake News Detection")
    print("=" * 40)
    
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
            print("Running demo...")
            demo()
    except:
        demo()
