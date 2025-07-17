# 🔍 Fake News Detection AI Project

A comprehensive Python AI project that uses machine learning to detect fake news articles with confidence percentages. This project implements multiple ML algorithms and provides both command-line and web interfaces for easy usage.

## 🎯 Features

- **AI-Powered Detection**: Uses multiple ML algorithms for accurate classification
- **Confidence Scores**: Provides percentage confidence for each prediction
- **Model Comparison**: Compare predictions from different models
- **Web Interface**: User-friendly Streamlit web application
- **Command Line Interface**: Easy-to-use CLI for training and predictions
- **Batch Processing**: Analyze multiple articles at once
- **Interactive Jupyter Notebook**: Comprehensive data analysis and visualization
- **Professional Architecture**: Well-structured, documented, and modular code

## 🤖 Machine Learning Models

- **Logistic Regression**: Fast and interpretable linear model
- **Random Forest**: Robust ensemble method with feature importance
- **Support Vector Machine**: Effective for high-dimensional text data
- **Naive Bayes**: Simple yet effective probabilistic model

## 📊 Dataset

This project is designed to work with the Fake News Detection dataset from Kaggle:
- **Source**: [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- **Format**: CSV file with columns: `title`, `text`, `subject`, `label`
- **Labels**: 0 = Fake News, 1 = Real News

## 🛠️ Technology Stack

- **Python**: Programming language
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Jupyter**: Interactive development environment

## 📁 Project Structure

```
fake-news-detection/
├── app.py                          # Streamlit web application
├── main.py                         # Command-line interface
├── setup.sh                        # Project setup script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── data/                          # Dataset storage
├── models/                        # Trained models storage
├── notebooks/                     # Jupyter notebooks
│   └── fake_news_analysis.ipynb  # Data analysis notebook
└── src/                          # Source code modules
    ├── __init__.py               # Package initialization
    ├── data_preprocessing.py     # Data preprocessing utilities
    ├── model_training.py         # Model training logic
    ├── prediction.py             # Prediction interface
    └── utils.py                  # Utility functions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd fake-news-detection

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Train Models

```bash
# Train with sample data (for demonstration)
python main.py --train

# Train with your own dataset
python main.py --train --data path/to/your/dataset.csv
```

### 4. Make Predictions

```bash
# Interactive prediction mode
python main.py --predict

# Demo with example articles
python main.py --demo

# Web interface
streamlit run app.py
```

## 💻 Usage Examples

### Command Line Interface

```bash
# Train all models
python main.py --train

# Interactive prediction mode
python main.py --predict

# Demo with examples
python main.py --demo

# Help
python main.py --help
```

### Web Interface

```bash
# Start the web application
streamlit run app.py

# Access at http://localhost:8501
```

### Python API

```python
from src.prediction import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Make prediction
result = detector.predict(
    text="Your news article text here",
    title="Article title"
)

print(f"Prediction: {'REAL' if result['is_real'] else 'FAKE'}")
print(f"Confidence: {result['confidence']}%")
```

## 🔧 Configuration

The project includes several configurable parameters in `src/utils.py`:

```python
class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MAX_FEATURES = 5000
    MODEL_DIR = "models"
    LOG_LEVEL = "INFO"
```

## 📈 Model Performance

The project provides comprehensive model evaluation including:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Model Comparison**: Side-by-side performance analysis

## 🌐 Web Interface Features

The Streamlit web application includes:

1. **Single Prediction**: Analyze individual articles
2. **Model Comparison**: Compare predictions across models
3. **Batch Analysis**: Process multiple articles from CSV
4. **Training Interface**: Train models through the web UI
5. **Interactive Visualizations**: Charts and graphs
6. **Model Performance**: Detailed metrics and comparisons

## 📊 Data Analysis

The Jupyter notebook (`notebooks/fake_news_analysis.ipynb`) provides:

- **Data Exploration**: Statistical analysis and visualizations
- **Text Analysis**: Word clouds and frequency analysis
- **Model Training**: Step-by-step model development
- **Performance Evaluation**: Detailed metrics and comparisons
- **Interactive Testing**: Test custom examples

## 🔍 How It Works

1. **Data Preprocessing**: Clean and normalize text data
2. **Feature Extraction**: Convert text to numerical features using TF-IDF
3. **Model Training**: Train multiple ML models on the processed data
4. **Prediction**: Use trained models to classify new articles
5. **Confidence Calculation**: Provide probability-based confidence scores

## 🎯 Key Features Explained

### Text Preprocessing
- URL and email removal
- HTML tag cleaning
- Special character removal
- Tokenization and lemmatization
- Stop word removal

### Feature Engineering
- TF-IDF vectorization
- N-gram extraction (1-2 grams)
- Feature scaling and normalization
- Dimensionality control

### Model Ensemble
- Multiple algorithm comparison
- Consensus-based predictions
- Performance benchmarking
- Best model selection

## 🚨 Important Notes

- **Dataset**: For best results, use a comprehensive dataset with thousands of examples
- **Performance**: Sample data performance is limited due to small dataset size
- **Scalability**: The project is designed to handle large datasets efficiently
- **Customization**: Easy to modify and extend for specific use cases

## 🔧 Troubleshooting

### Common Issues

1. **NLTK Data Error**: Run `python -c "import nltk; nltk.download('punkt_tab')"`
2. **Memory Issues**: Reduce `MAX_FEATURES` in configuration
3. **Model Not Found**: Ensure models are trained first with `python main.py --train`

### Performance Tips

- Use a larger, high-quality dataset for better accuracy
- Experiment with different feature extraction methods
- Try hyperparameter tuning for specific models
- Consider ensemble methods for improved performance
