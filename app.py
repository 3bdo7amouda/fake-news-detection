"""
Simple Web Demo for Fake News Detection
"""

import streamlit as st
from main import FakeNewsDetector

st.set_page_config(
    page_title="Fake News Demo",
    page_icon="🔍"
)

# Initialize detector
if 'detector' not in st.session_state:
    st.session_state.detector = FakeNewsDetector()

st.title("🔍 Fake News Detection")

# Add description
st.markdown("""
**Choose what you want to do:**
- **Demo**: See example predictions on sample news articles
- **Detect**: Analyze your own news article to check if it's fake or real
""")

# Create tabs for demo and detection
tab1, tab2 = st.tabs(["🧪 Demo", "🔍 Detect"])

with tab1:
    st.header("Demo Examples")
    st.markdown("Click the button below to see how the AI classifies different types of news:")
    
    # Demo button
    if st.button("🧪 Run Demo", type="primary"):
        examples = [
            ("Scientists Discover Cancer Treatment", "Researchers develop new treatment"),
            ("SHOCKING: Government Hiding Aliens", "Anonymous sources claim evidence"),
            ("Miracle Weight Loss Pill", "Amazing pill helps lose weight fast"),
            ("Stock Market Hits Record High", "Markets showed strong performance"),
            ("Secret Mind Control Program", "Whistleblower reveals program"),
            ("Medical Journal Research", "Journal publishes breakthrough research")
        ]
        
        st.subheader("Demo Results:")
        
        for i, (title, text) in enumerate(examples, 1):
            result = st.session_state.detector.predict(text, title)
            if result:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {title}**")
                with col2:
                    if result['is_real']:
                        st.success(f"✅ {result['prediction']} ({result['confidence']:.1f}%)")
                    else:
                        st.error(f"❌ {result['prediction']} ({result['confidence']:.1f}%)")

with tab2:
    st.header("Analyze Your News Article")
    st.markdown("Enter any news article below to check if it's fake or real:")
    
    # Input fields
    title = st.text_input("📰 News Title:", placeholder="Enter the article title...")
    text = st.text_area("📝 News Text:", height=150, placeholder="Enter the article content...")
    
    # Analyze button
    if st.button("🔍 Analyze Article", type="primary"):
        if text.strip():
            with st.spinner("Analyzing article..."):
                result = st.session_state.detector.predict(text, title)
                if result:
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['is_real']:
                            st.success(f"✅ **{result['prediction']} News**")
                        else:
                            st.error(f"❌ **{result['prediction']} News**")
                    
                    with col2:
                        st.info(f"📊 **Confidence: {result['confidence']:.1f}%**")
                    
                    # Show interpretation
                    if result['confidence'] > 80:
                        st.info("🎯 High confidence prediction")
                    elif result['confidence'] > 60:
                        st.warning("⚠️ Medium confidence prediction")
                    else:
                        st.error("❓ Low confidence prediction - result may be uncertain")
        else:
            st.warning("⚠️ Please enter some text to analyze")
    
    # Add example suggestions
    st.markdown("---")
    st.markdown("**Need ideas? Try these example headlines:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try: 'Scientists win Nobel Prize'"):
            st.rerun()
    with col2:
        if st.button("Try: 'SHOCKING: One weird trick'"):
            st.rerun()