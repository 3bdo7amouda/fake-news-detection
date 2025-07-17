# Fake News Detection Project

A professional Python AI project that detects fake news and provides confidence percentages using machine learning.

## Features

- **AI-Powered Detection**: Uses machine learning models to classify news as real or fake
- **Confidence Scores**: Provides percentage confidence for each prediction
- **Multiple Models**: Implements various ML algorithms for comparison
- **Data Visualization**: Comprehensive analysis and visualizations
- **Easy to Use**: Simple interface for testing news articles
- **Professional Documentation**: Well-commented and structured code

## Dataset

This project uses the Fake News Detection dataset from Kaggle:
- Dataset: [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- Features: Title, Text, Subject, Date
- Labels: Real (1) vs Fake (0) news

## Project Structure

```
fake-news-detection/
├── data/                          # Dataset files
├── models/                        # Trained model files
├── notebooks/                     # Jupyter notebooks for analysis
├── src/                          # Source code
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── model_training.py          # Model training and evaluation
│   ├── prediction.py              # Prediction functions
│   └── utils.py                   # Utility functions
├── app.py                         # Streamlit web application
├── main.py                        # Main script to run the project
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/3bdo7amouda/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` folder

## Usage

### Quick Start
```bash
python main.py
```

### Web Interface
```bash
streamlit run app.py
```

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## Model Performance

The project implements multiple machine learning models:
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust ensemble method
- **Support Vector Machine**: Effective for text classification
- **Naive Bayes**: Simple yet effective for text data

## Example Usage

```python
from src.prediction import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Predict news authenticity
result = detector.predict("Your news article text here")
print(f"Prediction: {'Real' if result['is_real'] else 'Fake'}")
print(f"Confidence: {result['confidence']:.2f}%")
```
