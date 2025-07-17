# 🔍 Fake News Detection

A machine learning-powered tool to detect fake news using Natural Language Processing (NLP) and Logistic Regression. This project demonstrates text classification with real Kaggle datasets.

## 🚀 Features

- **🤖 AI-Powered Detection**: Uses machine learning to classify news articles as real or fake
- **📊 Confidence Scoring**: Provides confidence levels for each prediction
- **🎯 Demo Mode**: Pre-loaded examples to test the system
- **🔍 Custom Detection**: Analyze your own news articles
- **🌐 Web Interface**: User-friendly Streamlit web application
- **💻 Command Line**: Simple CLI for quick testing
- **📈 Real Dataset Training**: Uses authentic Kaggle fake news datasets

## 📋 Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- streamlit

## 🔧 Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   # Web Interface (Recommended)
   streamlit run app.py

   # Command Line Interface
   python main.py
   ```

## 📖 Usage

### 🌐 Web Interface
1. Run: `streamlit run app.py`
2. Open browser at the displayed URL
3. Choose **Demo** tab for examples or **Detect** tab for your own articles

### 💻 Command Line Interface
1. Run: `python main.py`
2. Choose option 1 for demo or option 2 for custom detection

## 🧠 How It Works

1. **Text Preprocessing**: Cleans text, removes URLs/special characters, tokenizes, removes stopwords
2. **Feature Extraction**: Uses TF-IDF vectorization with unigrams and bigrams
3. **Machine Learning**: Logistic Regression with balanced class weights
4. **Training Data**: Real Kaggle datasets (Fake.csv and True.csv)
5. **Prediction**: Returns classification with confidence percentage

## 📊 Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: TF-IDF with 10,000 max features, unigrams + bigrams
- **Training**: Balanced dataset with 80/20 train-test split
- **Performance**: High accuracy on real-world news data

## 🗂️ Project Structure

```
fake-news-detection/
├── main.py              # Core detection logic + CLI
├── app.py               # Streamlit web interface
├── requirements.txt     # Dependencies
├── README.md           # This file
├── explanation.md      # Technical details
├── Fake.csv            # Kaggle fake news dataset
├── True.csv            # Kaggle real news dataset
└── models/             # Trained model files
    ├── model.pkl       # Logistic Regression model
    └── vectorizer.pkl  # TF-IDF vectorizer
```

## 🔍 Example Results

```
🔍 Fake News Detection Demo
========================================
'Scientists Discover Cancer Treatment' → Real (77.0%)
'SHOCKING: Government Hiding Aliens' → Fake (74.0%)
'Miracle Weight Loss Pill' → Fake (74.0%)
'Stock Market Hits Record High' → Real (81.0%)
'Secret Mind Control Program' → Fake (74.0%)
'Medical Journal Research' → Real (80.0%)
```

## ⚠️ Limitations

- **Language**: English only
- **Context**: Cannot verify factual accuracy, only textual patterns
- **Bias**: May reflect patterns in training data
- **Scope**: Best for news articles, may not work well on other text types

## 🚀 Future Improvements

- Add more sophisticated NLP models (BERT, transformers)
- Multi-language support
- Source credibility checking
- Real-time news feed integration
- Larger and more diverse datasets

## 📝 License

This project is for educational and demonstration purposes.

---

**Note**: This system analyzes textual patterns and should be used as a tool to assist human judgment, not replace it.
