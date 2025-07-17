#!/bin/bash

# Fake News Detection Project Setup Script
# This script sets up the project environment and installs dependencies

echo "🚀 Setting up Fake News Detection Project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud jupyter streamlit plotly beautifulsoup4 requests

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create directories
echo "📁 Creating project directories..."
mkdir -p data models notebooks

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Train the models: python main.py --train"
echo "3. Test predictions: python main.py --predict"
echo "4. Run web interface: streamlit run app.py"
echo "5. Open Jupyter notebook: jupyter notebook notebooks/fake_news_analysis.ipynb"
echo ""
echo "📊 To use the Kaggle dataset:"
echo "- Download from: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets"
echo "- Place the CSV file in the data/ folder"
echo "- Train with: python main.py --train --data data/your_dataset.csv"
echo ""
echo "Happy fake news detection! 🔍"