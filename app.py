"""
Streamlit Web Interface for Fake News Detection
"""

import streamlit as st
from main import FakeNewsDetector

st.set_page_config(page_title="Fake News Detection", page_icon="🔍")

# Initialize detector
if 'detector' not in st.session_state:
    st.session_state.detector = FakeNewsDetector()

st.title("🔍 Fake News Detection")
st.markdown("Analyze news articles to check if they're fake or real using AI")

# Create tabs
tab1, tab2 = st.tabs(["🧪 Demo", "🔍 Detect"])

with tab1:
    st.header("Demo Examples")
    if st.button("🧪 Run Demo", type="primary"):
        examples = [
            ("Scientists Win Nobel Prize", "Researchers develop breakthrough treatment"),
            ("SHOCKING: Government Conspiracy", "Anonymous sources reveal secrets"),
            ("Stock Market Update", "Markets showed strong performance today"),
            ("Secret Mind Control", "Whistleblower reveals program")
        ]
        
        for i, (title, text) in enumerate(examples, 1):
            result = st.session_state.detector.predict(text, title)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {title}**")
            with col2:
                if result['is_real']:
                    st.success(f"✅ {result['prediction']} ({result['confidence']:.1f}%)")
                else:
                    st.error(f"❌ {result['prediction']} ({result['confidence']:.1f}%)")

with tab2:
    st.header("Analyze Your Article")
    title = st.text_input("📰 Title:", placeholder="Enter article title...")
    text = st.text_area("📝 Content:", height=150, placeholder="Enter article content...")
    
    if st.button("🔍 Analyze", type="primary"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = st.session_state.detector.predict(text, title)
                
                col1, col2 = st.columns(2)
                with col1:
                    if result['is_real']:
                        st.success(f"✅ **{result['prediction']} News**")
                    else:
                        st.error(f"❌ **{result['prediction']} News**")
                with col2:
                    st.info(f"📊 **Confidence: {result['confidence']:.1f}%**")
                
                # Confidence interpretation
                if result['confidence'] > 80:
                    st.info("🎯 High confidence")
                elif result['confidence'] > 60:
                    st.warning("⚠️ Medium confidence")
                else:
                    st.error("❓ Low confidence")
        else:
            st.warning("⚠️ Please enter some text to analyze")