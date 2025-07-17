"""
Simple Streamlit Web Interface for Fake News Detection
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add current directory to path so we can import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FakeNewsDetector

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = FakeNewsDetector()

def main():
    st.title("🔍 Fake News Detection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Options")
    
    # Check if model exists
    model_exists = os.path.exists('models/fake_news_model.pkl')
    
    if not model_exists:
        st.sidebar.error("⚠️ Model not trained yet!")
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                st.session_state.detector.train()
                st.success("Model trained successfully!")
                st.rerun()
    else:
        st.sidebar.success("✅ Model ready!")
    
    # Model training section
    st.sidebar.markdown("### Train New Model")
    uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=['csv'])
    
    if uploaded_file is not None:
        if st.sidebar.button("Train with uploaded data"):
            with st.spinner("Training model with your data..."):
                # Save uploaded file temporarily
                with open("temp_dataset.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Train with uploaded data
                st.session_state.detector.train("temp_dataset.csv")
                
                # Remove temporary file
                os.remove("temp_dataset.csv")
                
                st.success("Model trained with your data!")
    
    # Main content
    if model_exists or st.sidebar.button("Continue with demo"):
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "Demo Examples"])
        
        with tab1:
            st.header("📰 Analyze Single Article")
            
            # Input fields
            title = st.text_input("Article Title:", placeholder="Enter the news article title...")
            text = st.text_area("Article Text:", height=200, placeholder="Enter the full article text...")
            
            if st.button("🔍 Analyze Article", type="primary"):
                if text.strip():
                    try:
                        # Load model if not already loaded
                        if not st.session_state.detector.model:
                            st.session_state.detector.load_model()
                        
                        # Make prediction
                        result = st.session_state.detector.predict(text, title)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if result['is_real']:
                                st.success(f"✅ **REAL NEWS**")
                            else:
                                st.error(f"❌ **FAKE NEWS**")
                        
                        with col2:
                            st.info(f"📊 **Confidence: {result['confidence']:.1f}%**")
                        
                        # Show probabilities
                        st.markdown("### Probability Breakdown")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Real News", f"{result['probabilities']['real']:.1f}%")
                        
                        with col2:
                            st.metric("Fake News", f"{result['probabilities']['fake']:.1f}%")
                        
                        # Progress bars
                        st.markdown("### Probability Distribution")
                        st.progress(result['probabilities']['real'] / 100)
                        st.text(f"Real: {result['probabilities']['real']:.1f}%")
                        
                        st.progress(result['probabilities']['fake'] / 100)
                        st.text(f"Fake: {result['probabilities']['fake']:.1f}%")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.info("Please train the model first using the sidebar.")
                else:
                    st.warning("Please enter some text to analyze.")
        
        with tab2:
            st.header("📊 Batch Analysis")
            
            # Sample data for batch analysis
            sample_articles = [
                {
                    'title': 'Scientists Discover New Cancer Treatment',
                    'text': 'Researchers at major university have developed promising new treatment through clinical trials'
                },
                {
                    'title': 'SHOCKING: Government Hiding Alien Contact',
                    'text': 'Anonymous sources claim government has been hiding evidence of extraterrestrial contact'
                },
                {
                    'title': 'Stock Market Reaches Record High',
                    'text': 'Financial markets showed strong performance as investors responded to positive indicators'
                },
                {
                    'title': 'Miracle Weight Loss Pill Doctors Hate',
                    'text': 'This amazing pill will help you lose weight fast without diet or exercise'
                }
            ]
            
            if st.button("🔍 Analyze Sample Articles"):
                try:
                    if not st.session_state.detector.model:
                        st.session_state.detector.load_model()
                    
                    results = []
                    for article in sample_articles:
                        result = st.session_state.detector.predict(article['text'], article['title'])
                        results.append({
                            'Title': article['title'],
                            'Prediction': result['prediction'],
                            'Confidence': f"{result['confidence']:.1f}%",
                            'Real_Prob': f"{result['probabilities']['real']:.1f}%",
                            'Fake_Prob': f"{result['probabilities']['fake']:.1f}%"
                        })
                    
                    # Display results in a table
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Summary
                    fake_count = sum(1 for r in results if r['Prediction'] == 'Fake')
                    real_count = len(results) - fake_count
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real News", real_count)
                    with col2:
                        st.metric("Fake News", fake_count)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Please train the model first.")
            
            # Custom batch upload
            st.markdown("### Upload Your Articles")
            uploaded_batch = st.file_uploader("Upload CSV with 'title' and 'text' columns", type=['csv'])
            
            if uploaded_batch is not None:
                df_batch = pd.read_csv(uploaded_batch)
                
                if 'title' in df_batch.columns and 'text' in df_batch.columns:
                    if st.button("Analyze Uploaded Articles"):
                        try:
                            if not st.session_state.detector.model:
                                st.session_state.detector.load_model()
                            
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, row in df_batch.iterrows():
                                result = st.session_state.detector.predict(row['text'], row['title'])
                                results.append({
                                    'Title': row['title'],
                                    'Prediction': result['prediction'],
                                    'Confidence': f"{result['confidence']:.1f}%"
                                })
                                progress_bar.progress((i + 1) / len(df_batch))
                            
                            st.success(f"Analyzed {len(results)} articles!")
                            df_results = pd.DataFrame(results)
                            st.dataframe(df_results, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.error("CSV must contain 'title' and 'text' columns")
        
        with tab3:
            st.header("🧪 Demo Examples")
            
            examples = [
                {
                    'title': 'Scientists Discover New Treatment for Cancer',
                    'text': 'Researchers at major university have developed promising new treatment',
                    'expected': 'Real'
                },
                {
                    'title': 'SHOCKING: Aliens Found in Government Facility',
                    'text': 'Government officials deny but sources confirm extraterrestrial beings held',
                    'expected': 'Fake'
                },
                {
                    'title': 'Miracle Cure Discovered by Local Mom',
                    'text': 'This amazing trick will cure all diseases doctors hate this simple method',
                    'expected': 'Fake'
                },
                {
                    'title': 'Stock Market Reaches New High',
                    'text': 'Financial markets showed strong performance today as investors responded positively',
                    'expected': 'Real'
                }
            ]
            
            if st.button("🔍 Run Demo Examples"):
                try:
                    if not st.session_state.detector.model:
                        st.session_state.detector.load_model()
                    
                    for i, example in enumerate(examples, 1):
                        result = st.session_state.detector.predict(example['text'], example['title'])
                        
                        st.markdown(f"### Example {i}")
                        st.markdown(f"**Title:** {example['title']}")
                        st.markdown(f"**Expected:** {example['expected']}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if result['is_real']:
                                st.success(f"Predicted: **{result['prediction']}**")
                            else:
                                st.error(f"Predicted: **{result['prediction']}**")
                        
                        with col2:
                            st.info(f"Confidence: **{result['confidence']:.1f}%**")
                        
                        with col3:
                            if result['prediction'] == example['expected']:
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect!")
                        
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Please train the model first.")
    
    else:
        st.info("👆 Please train the model first using the sidebar.")

if __name__ == "__main__":
    main()