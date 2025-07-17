"""
Streamlit web application for fake news detection.
This provides a user-friendly web interface for the fake news detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.prediction import FakeNewsDetector, ModelComparator
from src.model_training import ModelTrainer
from src.utils import create_directories

# Page configuration
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-news {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .fake-news {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_confidence_color(confidence):
    """Return CSS class based on confidence level."""
    if confidence >= 80:
        return "confidence-high"
    elif confidence >= 60:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_prediction_result(result):
    """Display prediction result in a formatted way."""
    if result['is_real']:
        box_class = "real-news"
        status_emoji = "✅"
        status_text = "REAL NEWS"
    else:
        box_class = "fake-news"
        status_emoji = "❌"
        status_text = "FAKE NEWS"
    
    confidence_class = get_confidence_color(result['confidence'])
    
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h3>{status_emoji} {status_text}</h3>
        <p><strong>Confidence:</strong> <span class="{confidence_class}">{result['confidence']}%</span></p>
        <p><strong>Model Used:</strong> {result['model_used']}</p>
        <p><strong>Probability Breakdown:</strong></p>
        <ul>
            <li>Real News: {result['probabilities']['real']}%</li>
            <li>Fake News: {result['probabilities']['fake']}%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_probability_chart(result):
    """Create a probability visualization chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Real News', 'Fake News'],
            y=[result['probabilities']['real'], result['probabilities']['fake']],
            marker_color=['green', 'red']
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Category",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    
    return fig

def create_comparison_chart(comparison_results):
    """Create a comparison chart for multiple models."""
    models = list(comparison_results['individual_results'].keys())
    real_probs = []
    fake_probs = []
    
    for model in models:
        result = comparison_results['individual_results'][model]
        if 'probabilities' in result:
            real_probs.append(result['probabilities']['real'])
            fake_probs.append(result['probabilities']['fake'])
        else:
            real_probs.append(0)
            fake_probs.append(0)
    
    fig = go.Figure(data=[
        go.Bar(name='Real News', x=models, y=real_probs, marker_color='green'),
        go.Bar(name='Fake News', x=models, y=fake_probs, marker_color='red')
    ])
    
    fig.update_layout(
        title="Model Comparison - Prediction Probabilities",
        xaxis_title="Models",
        yaxis_title="Probability (%)",
        barmode='group',
        height=500
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detection AI</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Single Prediction", "Model Comparison", "Batch Analysis", "Training", "About"]
    )
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Model Comparison":
        model_comparison_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Training":
        training_page()
    elif page == "About":
        about_page()

def single_prediction_page():
    """Single prediction page."""
    st.header("📰 Single Article Analysis")
    
    # Model selection
    model_options = ['logistic_regression', 'random_forest', 'svm', 'naive_bayes']
    selected_model = st.selectbox(
        "Select Model:",
        model_options,
        index=0
    )
    
    # Input fields
    col1, col2 = st.columns([1, 2])
    
    with col1:
        title = st.text_input("Article Title (optional):", placeholder="Enter the news title...")
    
    with col2:
        text = st.text_area(
            "Article Text:",
            height=200,
            placeholder="Paste the news article text here..."
        )
    
    if st.button("🔍 Analyze Article", type="primary"):
        if not text.strip():
            st.error("Please enter some text to analyze.")
            return
        
        try:
            # Initialize detector
            detector = FakeNewsDetector(selected_model)
            
            with st.spinner("Analyzing article..."):
                result = detector.predict(text, title)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                display_prediction_result(result)
            
            with col2:
                chart = create_probability_chart(result)
                st.plotly_chart(chart, use_container_width=True)
            
            # Show text preview
            st.subheader("📄 Text Preview")
            st.info(result['text'])
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure the models are trained first. Go to the Training page to train the models.")

def model_comparison_page():
    """Model comparison page."""
    st.header("🔬 Multi-Model Comparison")
    st.write("Compare predictions from multiple models to get a consensus view.")
    
    # Input fields
    title = st.text_input("Article Title (optional):", placeholder="Enter the news title...")
    text = st.text_area(
        "Article Text:",
        height=200,
        placeholder="Paste the news article text here..."
    )
    
    if st.button("🔍 Compare Models", type="primary"):
        if not text.strip():
            st.error("Please enter some text to analyze.")
            return
        
        try:
            # Initialize comparator
            comparator = ModelComparator()
            
            with st.spinner("Comparing models..."):
                comparison = comparator.compare_predictions(text, title)
            
            # Display consensus
            if 'error' not in comparison['consensus']:
                st.subheader("📊 Consensus Results")
                consensus = comparison['consensus']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    majority_vote = "REAL" if consensus['majority_vote'] else "FAKE"
                    st.metric("Majority Vote", majority_vote)
                
                with col2:
                    st.metric("Average Confidence", f"{consensus['average_confidence']:.1f}%")
                
                with col3:
                    st.metric("Model Agreement", f"{consensus['agreement_rate']:.1f}")
            
            # Display individual results
            st.subheader("🤖 Individual Model Results")
            
            for model_name, result in comparison['individual_results'].items():
                if 'error' not in result:
                    with st.expander(f"{model_name.replace('_', ' ').title()}"):
                        display_prediction_result(result)
            
            # Create comparison chart
            st.subheader("📈 Model Comparison Chart")
            chart = create_comparison_chart(comparison)
            st.plotly_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please make sure the models are trained first. Go to the Training page to train the models.")

def batch_analysis_page():
    """Batch analysis page."""
    st.header("📊 Batch Analysis")
    st.write("Upload a CSV file or enter multiple articles for batch analysis.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Assume the CSV has 'title' and 'text' columns
            if 'text' in df.columns:
                if st.button("🔍 Analyze Batch", type="primary"):
                    detector = FakeNewsDetector()
                    
                    with st.spinner("Analyzing articles..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, row in df.iterrows():
                            title = row.get('title', '')
                            text = row.get('text', '')
                            
                            if text:
                                result = detector.predict(text, title)
                                results.append({
                                    'title': title,
                                    'prediction': 'REAL' if result['is_real'] else 'FAKE',
                                    'confidence': result['confidence'],
                                    'real_prob': result['probabilities']['real'],
                                    'fake_prob': result['probabilities']['fake']
                                })
                            
                            progress_bar.progress((i + 1) / len(df))
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("📋 Batch Analysis Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        real_count = len(results_df[results_df['prediction'] == 'REAL'])
                        st.metric("Real News", real_count)
                    
                    with col2:
                        fake_count = len(results_df[results_df['prediction'] == 'FAKE'])
                        st.metric("Fake News", fake_count)
                    
                    with col3:
                        avg_confidence = results_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fake_news_analysis_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error("CSV file must contain a 'text' column.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Upload a CSV file with news articles to perform batch analysis.")

def training_page():
    """Training page."""
    st.header("🎓 Model Training")
    st.write("Train machine learning models on your dataset.")
    
    # Dataset upload
    st.subheader("📤 Dataset Upload")
    uploaded_file = st.file_uploader("Upload training dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset preview:")
            st.dataframe(df.head())
            
            # Save uploaded file
            dataset_path = "data/uploaded_dataset.csv"
            df.to_csv(dataset_path, index=False)
            st.success("Dataset uploaded successfully!")
            
        except Exception as e:
            st.error(f"Error uploading dataset: {str(e)}")
    
    # Training options
    st.subheader("⚙️ Training Options")
    
    use_sample_data = st.checkbox("Use sample data for demonstration", value=True)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Create directories
                create_directories()
                
                # Initialize trainer
                trainer = ModelTrainer()
                
                # Train models
                data_path = None if use_sample_data else "data/uploaded_dataset.csv"
                results = trainer.train_full_pipeline(data_path)
                
                st.success("✅ Training completed successfully!")
                st.write(f"Best model: {results['best_model']}")
                
                # Display model performances
                st.subheader("📊 Model Performance")
                
                performance_data = []
                for model_name, metrics in trainer.model_performances.items():
                    performance_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}"
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def about_page():
    """About page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🔍 Fake News Detection AI
    
    This is a professional Python AI project that uses machine learning to detect fake news articles 
    and provide confidence percentages for predictions.
    
    ### 🎯 Features
    - **AI-Powered Detection**: Uses multiple ML algorithms for accurate classification
    - **Confidence Scores**: Provides percentage confidence for each prediction
    - **Model Comparison**: Compare predictions from different models
    - **Batch Analysis**: Process multiple articles at once
    - **Web Interface**: User-friendly Streamlit interface
    - **Professional Code**: Well-documented and structured codebase
    
    ### 🤖 Machine Learning Models
    - **Logistic Regression**: Fast and interpretable linear model
    - **Random Forest**: Robust ensemble method
    - **Support Vector Machine**: Effective for text classification
    - **Naive Bayes**: Simple yet effective probabilistic model
    
    ### 📊 Dataset
    This project uses the Fake News Detection dataset from Kaggle:
    [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
    
    ### 🛠️ Technology Stack
    - **Python**: Programming language
    - **Scikit-learn**: Machine learning library
    - **NLTK**: Natural language processing
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation
    - **Plotly**: Interactive visualizations
    
    ### 📈 How It Works
    1. **Text Preprocessing**: Clean and normalize the input text
    2. **Feature Extraction**: Convert text to numerical features using TF-IDF
    3. **Model Prediction**: Use trained ML models to classify the text
    4. **Confidence Calculation**: Provide probability-based confidence scores
    5. **Result Display**: Show predictions with detailed analysis
    
    ### 👨‍💻 Usage
    1. **Training**: Train models on your dataset or use sample data
    2. **Single Prediction**: Analyze individual news articles
    3. **Model Comparison**: Compare predictions from multiple models
    4. **Batch Analysis**: Process multiple articles from CSV files
    
    ### 🔧 Installation
    ```bash
    pip install -r requirements.txt
    python main.py --train
    streamlit run app.py
    ```
    
    ### 📝 License
    This project is open source and available under the MIT License.
    """)

if __name__ == "__main__":
    main()