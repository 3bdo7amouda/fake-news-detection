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
import os

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
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def get_sample_data(self):
        """Sample training data"""
        data = {
            'title': [
                'Scientists Discover Cancer Treatment', 'SHOCKING: Government Hiding Aliens',
                'Stock Market Hits Record High', 'Miracle Weight Loss Pill',
                'Climate Change Research Published', 'Celebrity Found Dead',
                'School Wins National Award', 'Secret Mind Control Program',
                'Tech Company New Product', 'Home Remedy Cures Everything',
                'Vaccine Study Results', 'Amazing Weight Loss Trick',
                'Economic Report Job Growth', 'Moon Landing Conspiracy',
                'Medical Journal Research', 'Doctors Hate This Trick',
                'Government New Policy', 'Celebrity Scandal Exposed',
                'Nobel Prize Winners', 'Kitchen Ingredient Miracle Cure'
            ],
            'text': [
                'Researchers develop treatment through clinical trials',
                'Anonymous sources claim government hiding alien evidence',
                'Markets showed strong performance with positive indicators',
                'Amazing pill helps lose weight without diet or exercise',
                'Scientists publish study on climate change impacts',
                'Police investigating celebrity death at residence',
                'Students achieved excellence in national competition',
                'Whistleblower reveals mind control program',
                'Company unveils product after years of development',
                'Kitchen trick solves all health problems instantly',
                'Study confirms vaccine safety and effectiveness',
                'Weight loss secret trainers dont want you to know',
                'Report shows job growth across multiple sectors',
                'Evidence suggests moon landing was fake',
                'Journal publishes breakthrough heart disease research',
                'Discover trick doctors hate pharmaceutical companies',
                'Government announces healthcare reform policy',
                'Celebrity scandal reveals secret affairs',
                'Scientists receive Nobel Prize for quantum research',
                'Kitchen ingredient treats cancer diabetes instantly'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        return pd.DataFrame(data)
    
    def train(self):
        """Train the model"""
        df = self.get_sample_data()
        
        # Prepare data
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        X = df['processed_text']
        y = df['label']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_tfidf, y_train)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model trained!")
    
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
        combined_text = title + ' ' + text
        processed_text = self.preprocess_text(combined_text)
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