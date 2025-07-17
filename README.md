# 🔍 Fake News Detection System

A machine learning system that classifies news articles as fake or real using natural language processing and logistic regression.

## 📋 Features

- **High Accuracy**: 97%+ accuracy on test data
- **Balanced Training**: Uses calibrated thresholds to reduce false positives
- **Web Interface**: Easy-to-use Streamlit web app
- **Command Line**: Interactive CLI for quick testing
- **Pattern Recognition**: Detects legitimate news patterns

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
git clone <repository-url>
cd fake-news-detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Command Line:**
```bash
python main.py
```

## 📦 Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- streamlit

## 🎯 Usage

### Web Interface
1. **Demo Tab**: See example predictions on sample articles
2. **Detect Tab**: Analyze your own news articles

### Command Line
1. Choose option 1 for demo examples
2. Choose option 2 to analyze custom text

## 🧠 How It Works

1. **Data Loading**: Uses balanced Kaggle datasets (Fake.csv, True.csv)
2. **Text Processing**: Cleans and normalizes text, removes bias patterns
3. **Feature Extraction**: TF-IDF vectorization with unigrams and bigrams
4. **Classification**: Logistic regression with calibrated thresholds
5. **Pattern Detection**: Recognizes legitimate news source patterns

## 📁 Project Structure

```
fake-news-detection/
├── main.py           # Core detection logic
├── app.py           # Streamlit web interface
├── requirements.txt # Python dependencies
├── Fake.csv        # Fake news dataset
├── True.csv        # Real news dataset
├── models/         # Trained model files (auto-generated)
└── README.md       # This file
```

## 🔧 Installation on New VM

1. **Install Python 3.7+**
2. **Create project directory**
3. **Copy all files** (main.py, app.py, requirements.txt, Fake.csv, True.csv)
4. **Follow Quick Start steps above**

The model will automatically train on first run using the included datasets.

## 📊 Performance

- **Accuracy**: 97%+
- **Balanced**: Equal precision for fake and real news
- **Calibrated**: Reduced false positive rate for legitimate news
- **Fast**: Predictions in milliseconds

## 🔍 Example Results

- ✅ "Stock Market Update" → Real (85.2%)
- ✅ "University Research Published" → Real (78.4%)
- ❌ "SHOCKING: Secret Government Plot" → Fake (92.1%)
- ❌ "Miracle Weight Loss Pill" → Fake (89.3%)
