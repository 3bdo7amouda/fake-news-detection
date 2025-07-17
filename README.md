# 🔍 Fake News Detection

A simple AI-powered tool to detect fake news using machine learning.

## Features

- **Demo**: See example predictions on sample news articles
- **Detection**: Analyze your own news articles
- **Web Interface**: Easy-to-use Streamlit web app
- **Command Line**: Simple CLI for quick testing

## Quick Start

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the application:**

    **Web Interface (Recommended):**
    ```bash
    streamlit run app.py
    ```

    **Command Line:**
    ```bash
    python main.py
    ```

## Usage

### Web Interface
- Open your browser and go to the provided URL
- Choose **Demo** tab to see example predictions
- Choose **Detect** tab to analyze your own articles

### Command Line
- Choose option 1 for demo examples
- Choose option 2 to analyze your own text

## How It Works

The system uses a Random Forest classifier trained on sample data to classify news articles as real or fake. It processes text by:
1. Cleaning and normalizing text
2. Removing stopwords
3. Converting to numerical features using TF-IDF
4. Making predictions with confidence scores

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- streamlit

## Project Structure

```
fake-news-detection/
├── main.py          # Core detection logic + CLI
├── app.py           # Web interface
├── requirements.txt # Dependencies
└── README.md        # This file
```

## Note

This is a demo project using sample training data. For production use, train with a larger, real-world dataset.
