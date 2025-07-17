# 🔍 Fake News Detection

A simple yet effective AI-powered tool to detect fake news using machine learning. This project demonstrates text classification using Natural Language Processing (NLP) and Random Forest algorithms.

## 🚀 Features

- **🤖 AI-Powered Detection**: Uses machine learning to classify news articles as real or fake
- **📊 Confidence Scoring**: Provides confidence levels for each prediction
- **🎯 Demo Mode**: Pre-loaded examples to test the system
- **🔍 Custom Detection**: Analyze your own news articles
- **🌐 Web Interface**: User-friendly Streamlit web application
- **💻 Command Line**: Simple CLI for quick testing
- **📈 Real-time Processing**: Instant analysis of news articles

## 📋 Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- streamlit

## 🔧 Installation

1. **Clone or download the project:**
    ```bash
    git clone <repository-url>
    cd fake-news-detection
    ```

2. **Create virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    # Web Interface (Recommended)
    streamlit run app.py

    # Command Line Interface
    python main.py
    ```

## 📖 Usage

### 🌐 Web Interface

1. **Start the web application:**
    ```bash
    streamlit run app.py
    ```

2. **Open your browser** and go to the displayed URL (usually `http://localhost:8501`)

3. **Choose your option:**
   - **Demo Tab**: Click "Run Demo" to see example predictions
   - **Detect Tab**: Enter your own news article to analyze

### 💻 Command Line Interface

1. **Run the CLI:**
    ```bash
    python main.py
    ```

2. **Choose an option:**
   - **Option 1**: Demo (see 6 example predictions)
   - **Option 2**: Detect (analyze your own text)

## 🎯 Demo Examples

The system comes with pre-loaded examples that demonstrate different types of news:

- **Real News**: Scientific discoveries, market reports, government announcements
- **Fake News**: Miracle cures, conspiracy theories, clickbait headlines
- **Uncertain**: Articles that are harder to classify

## 🧠 How It Works

### 1. **Text Preprocessing**
- Converts text to lowercase
- Removes URLs, special characters, and numbers
- Tokenizes text into individual words
- Removes stopwords and short words

### 2. **Feature Extraction**
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Converts text into numerical features that machine learning can understand
- Focuses on the most important 5,000 words

### 3. **Machine Learning Model**
- **Algorithm**: Random Forest Classifier
- **Training Data**: 20 sample articles (10 real, 10 fake)
- **Features**: Title + Article text combined
- **Output**: Binary classification (Real/Fake) with confidence score

### 4. **Prediction Process**
- Combines title and article text
- Preprocesses the combined text
- Converts to numerical features using the trained vectorizer
- Makes prediction using the trained model
- Returns result with confidence percentage

## 📊 Model Performance

- **Accuracy**: ~75% on test data
- **Training Data**: 20 balanced samples
- **Confidence Range**: 50-85% typical range
- **Processing Time**: Near-instant for single articles

## 🗂️ Project Structure

```
fake-news-detection/
├── main.py              # Core detection logic + CLI
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── explanation.md      # Detailed project explanation
├── models/             # Trained model files
│   ├── model.pkl       # Random Forest model
│   └── vectorizer.pkl  # TF-IDF vectorizer
└── __pycache__/        # Python cache files
```

## 🔍 Example Results

```
🔍 Fake News Detection Demo
========================================
'Scientists Discover Cancer Treatment' → Real (77.0%)
'SHOCKING: Government Hiding Aliens' → Real (57.0%)
'Miracle Weight Loss Pill' → Fake (74.0%)
'Stock Market Hits Record High' → Real (81.0%)
'Secret Mind Control Program' → Fake (74.0%)
'Medical Journal Research' → Real (80.0%)
```

## ⚠️ Limitations

- **Training Data**: Uses sample data, not real-world datasets
- **Scope**: Designed for demonstration purposes
- **Accuracy**: Limited by small training dataset
- **Language**: English only
- **Context**: Cannot verify factual accuracy, only textual patterns

## 🚀 Future Improvements

- Train with larger, real-world datasets
- Add more sophisticated NLP models (BERT, GPT)
- Include source credibility checking
- Multi-language support
- Real-time news feed integration
- Fact-checking API integration

## 🛠️ Technical Details

- **Framework**: scikit-learn for machine learning
- **NLP**: NLTK for text processing
- **Web Framework**: Streamlit for user interface
- **Model**: Random Forest with 100 estimators
- **Vectorization**: TF-IDF with 5,000 features
- **Cross-validation**: 80/20 train-test split

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📧 Contact

For questions or suggestions, please create an issue in the repository.

---

**Note**: This is a demonstration project. For production use, please train with larger, verified datasets and implement additional validation measures.
