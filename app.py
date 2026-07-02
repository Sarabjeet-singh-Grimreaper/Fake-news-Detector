import streamlit as st
import pickle
from src.preprocessing import full_preprocess_pipeline

# Configure Page Layout and Style
st.set_page_config(
    page_title="Veritas AI | Fake News Detection Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium styling (Vanilla CSS Injection)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Header Card */
    .hero-container {
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent), 
                    radial-gradient(circle at bottom left, rgba(236, 72, 153, 0.05), transparent),
                    #111827;
        padding: 3rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
        letter-spacing: -0.025em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 700px;
        margin: 0 auto;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Interactive Card container */
    .card {
        background-color: #111827;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Badge styling */
    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .badge-real {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Glassmorphism sidebar */
    .css-163ttbj, [data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model assets from disk
@st.cache_resource
def load_assets():
    assets = {}
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        assets["vectorizer"] = pickle.load(f)
        
    models = ["knn", "logreg", "random_forest", "neuralnet"]
    for m in models:
        try:
            with open(f"models/{m}_model.pkl", "rb") as f:
                assets[m] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file models/{m}_model.pkl not found. Please run training first.")
    return assets

# Load models and vectorizer
assets = load_assets()

# Sidebar Styling
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <img src='https://img.icons8.com/nolan/96/security-shield.png' width='70'/>
    <h3 style='margin-top: 10px; font-weight: 700; color: #f1f5f9;'>Veritas AI Engine</h3>
    <p style='color: #64748b; font-size: 0.85rem;'>Academic NLP Framework v1.0.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Classifier Suite")
model_option = st.sidebar.selectbox(
    "Select Model Architecture:",
    ["Logistic Regression", "Random Forest", "Neural Network (MLP)", "K-Nearest Neighbors (KNN)"]
)

model_key_map = {
    "Logistic Regression": "logreg",
    "Random Forest": "random_forest",
    "Neural Network (MLP)": "neuralnet",
    "K-Nearest Neighbors (KNN)": "knn"
}
selected_model_key = model_key_map[model_option]

st.sidebar.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='background-color: #111827; padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
    <h4 style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; color: #818cf8;'>Strict Rule Constraints</h4>
    <p style='font-size: 0.8rem; color: #94a3b8; line-height: 1.4; margin: 0;'>
        This system is built without high-level NLP tokenizers (NLTK/SpaCy wrappers). Everything uses custom regex and standard scikit-learn.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Dashboard layout
st.markdown("""
<div class='hero-container'>
    <div class='hero-title'>VERITAS AI</div>
    <div class='hero-subtitle'>
        Academic Research Portal for real-time validation and linguistic auditing of online publications. 
        Engineered with a scratch-built preprocessing pipeline and custom ML classifiers.
    </div>
</div>
""", unsafe_allow_html=True)

col_input, col_results = st.columns([1.8, 1.2], gap="large")

with col_input:
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <h3 style='font-weight: 700; font-size: 1.5rem; color: #f1f5f9;'>Linguistic Analysis Workspace</h3>
        <p style='color: #64748b; font-size: 0.9rem;'>Enter the body text of the article below for full vector analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    article_text = st.text_area(
        "",
        height=320,
        placeholder="Paste full news article content here to scan for credibility tags...",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("Scan Article Legitimacy", type="primary", use_container_width=True)

with col_results:
    st.markdown("""
    <div>
        <h3 style='font-weight: 700; font-size: 1.5rem; color: #f1f5f9;'>Real-time Diagnostic Output</h3>
        <p style='color: #64748b; font-size: 0.9rem;'>Telemetry and prediction confidence metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if analyze_btn:
        if not article_text.strip():
            st.error("Please insert a valid text sample to analyze.")
        else:
            with st.spinner("Decoding language vectors..."):
                # Clean and Tokenize
                cleaned_text = full_preprocess_pipeline(article_text)
                
                # Metric Calculations
                original_word_count = len(article_text.split())
                cleaned_word_count = len(cleaned_text.split())
                removed_words = original_word_count - cleaned_word_count
                
                # Transform & Predict
                vectorizer = assets["vectorizer"]
                vectorized_input = vectorizer.transform([cleaned_text])
                
                model = assets[selected_model_key]
                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                
                # Visual Verdict Card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("<div class='verdict-badge badge-real'>🛡️ Verified Real</div>", unsafe_allow_html=True)
                    confidence = probabilities[1] * 100
                else:
                    st.markdown("<div class='verdict-badge badge-fake'>⚠️ Flagged Fake</div>", unsafe_allow_html=True)
                    confidence = probabilities[0] * 100
                
                st.markdown(f"<h3 style='margin: 0; font-weight: 700; font-size: 2.2rem;'>{confidence:.1f}% <span style='font-size: 1rem; color: #64748b;'>confidence</span></h3>", unsafe_allow_html=True)
                st.progress(confidence / 100.0)
                st.caption(f"Evaluated via {model_option}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Diagnostics Card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0; margin-bottom:1rem; font-weight:600; font-size:1.1rem; color: #818cf8;'>Linguistic Diagnostic Telemetry</h4>", unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Raw Word Count", original_word_count)
                with metric_col2:
                    st.metric("Filtered Out (Stopwords)", removed_words)
                    
                with st.expander("Show Tokenized Input Stream"):
                    st.write(cleaned_text if cleaned_text else "*[All text stripped as stopwords or empty]*")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Default placeholder card
        st.markdown("""
        <div style='background-color: #111827; border-radius: 20px; padding: 3rem 2rem; border: 1px dashed rgba(255, 255, 255, 0.1); text-align: center; color: #64748b;'>
            <img src='https://img.icons8.com/nolan/64/combo-chart.png' style='opacity: 0.5; margin-bottom: 1rem;'/>
            <p style='margin: 0; font-size: 0.95rem;'>Awaiting text input stream to initialize classification models...</p>
        </div>
        """, unsafe_allow_html=True)
