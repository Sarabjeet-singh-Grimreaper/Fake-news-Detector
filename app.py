import streamlit as st
import pickle
import pandas as pd
import numpy as np
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
    
    /* Model Verdict Card Grid */
    .model-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .model-verdict-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .model-verdict-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .verdict-badge-small {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    .badge-real-small {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-fake-small {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Tags styling for keywords */
    .keyword-tag {
        display: inline-block;
        background-color: rgba(129, 140, 248, 0.15);
        color: #a5b4fc;
        padding: 0.35rem 0.85rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(129, 140, 248, 0.2);
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
    <p style='color: #64748b; font-size: 0.85rem;'>Academic NLP Framework v1.1.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background-color: #111827; padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 1.5rem;'>
    <h4 style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; color: #818cf8;'>Multi-Model Verdicts</h4>
    <p style='font-size: 0.8rem; color: #94a3b8; line-height: 1.4; margin: 0;'>
        This updated build processes input text through <strong>all 4 trained classifiers concurrently</strong> to visualize consensus and divergence.
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='background-color: #111827; padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
    <h4 style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; color: #f472b6;'>Scratch Restrictions</h4>
    <p style='font-size: 0.8rem; color: #94a3b8; line-height: 1.4; margin: 0;'>
        No NLTK or SpaCy wrappers are active. Preprocessing relies purely on compiled module-level regexes for <strong>max performance</strong>.
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

col_input, col_results = st.columns([1.6, 1.4], gap="large")

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
    
    analyze_btn = st.button("Audit and Verify Legitimacy", type="primary", use_container_width=True)

with col_results:
    st.markdown("""
    <div>
        <h3 style='font-weight: 700; font-size: 1.5rem; color: #f1f5f9;'>Diagnostic Audit Suite</h3>
        <p style='color: #64748b; font-size: 0.9rem;'>Consensus statistics across the classifier portfolio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if analyze_btn:
        if not article_text.strip():
            st.error("Please insert a valid text sample to analyze.")
        else:
            with st.spinner("Decoding language vectors..."):
                # 1. Preprocess using optimized module regexes
                cleaned_text = full_preprocess_pipeline(article_text)
                
                # Metric Calculations
                original_word_count = len(article_text.split())
                cleaned_word_count = len(cleaned_text.split())
                removed_words = original_word_count - cleaned_word_count
                
                # 2. TF-IDF Vectorization
                vectorizer = assets["vectorizer"]
                vectorized_input = vectorizer.transform([cleaned_text])
                
                # Compute top local terms based on TF-IDF weights in this document
                feature_names = np.array(vectorizer.get_feature_names_out())
                feature_indices = vectorized_input.tocoo().col
                feature_values = vectorized_input.tocoo().data
                
                # Match tokens to scores
                top_tokens = []
                if len(feature_values) > 0:
                    sorted_indices = np.argsort(feature_values)[::-1]
                    top_tokens = [(feature_names[feature_indices[idx]], feature_values[idx]) for idx in sorted_indices[:8]]
                
                # 3. Model Predict Suite (Run predictions on all 4 models)
                model_configs = {
                    "Logistic Regression": "logreg",
                    "Random Forest": "random_forest",
                    "MLP Neural Net": "neuralnet",
                    "K-Nearest Neighbors": "knn"
                }
                
                predictions_summary = []
                for name, key in model_configs.items():
                    model = assets[key]
                    pred = model.predict(vectorized_input)[0]
                    probs = model.predict_proba(vectorized_input)[0]
                    confidence = probs[pred] * 100
                    predictions_summary.append((name, pred, confidence))
                
                # UI Layout - Consensus Grid
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0; margin-bottom:1rem; font-weight:600; font-size:1.1rem; color: #c084fc;'>Model Consensus Panel</h4>", unsafe_allow_html=True)
                
                grid_cols = st.columns(2)
                for idx, (name, pred, conf) in enumerate(predictions_summary):
                    col = grid_cols[idx % 2]
                    with col:
                        # Render each model's report inside a small container
                        badge_class = "badge-real-small" if pred == 1 else "badge-fake-small"
                        badge_label = "Real" if pred == 1 else "Fake"
                        
                        st.markdown(f"""
                        <div style='background-color: #1e293b; padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 0.75rem; text-align: center;'>
                            <div style='font-size: 0.85rem; font-weight:600; color: #94a3b8;'>{name}</div>
                            <div class='verdict-badge-small {badge_class}'>{badge_label}</div>
                            <div style='font-size: 0.95rem; font-weight: 700; margin-top: 0.25rem; color: #f1f5f9;'>{conf:.1f}% conf</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # UI Layout - Key Predictive Features
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0; margin-bottom:0.75rem; font-weight:600; font-size:1.1rem; color: #818cf8;'>Strongest TF-IDF Vocabulary Weights</h4>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-top:-0.5rem; margin-bottom:1rem;'>Terms in this article that carried the highest mathematical variance.</p>", unsafe_allow_html=True)
                
                if top_tokens:
                    tags_html = "".join([f"<span class='keyword-tag'>{word} ({score:.2f})</span>" for word, score in top_tokens])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='font-size: 0.85rem; color: #64748b;'>No significant feature matches found in text.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # UI Layout - Diagnostic Telemetry
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top:0; margin-bottom:1rem; font-weight:600; font-size:1.1rem; color: #f472b6;'>Linguistic Telemetry</h4>", unsafe_allow_html=True)
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Raw Words", original_word_count)
                with metric_col2:
                    st.metric("Stopwords Removed", removed_words)
                    
                with st.expander("Show Tokenized Input Stream"):
                    st.write(cleaned_text if cleaned_text else "*[All text stripped as stopwords or empty]*")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Default placeholder card
        st.markdown("""
        <div style='background-color: #111827; border-radius: 20px; padding: 4rem 2rem; border: 1px dashed rgba(255, 255, 255, 0.1); text-align: center; color: #64748b; min-height: 400px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <img src='https://img.icons8.com/nolan/64/combo-chart.png' style='opacity: 0.5; margin-bottom: 1.5rem;'/>
            <h4 style='color: #94a3b8; font-weight:600; margin-bottom:0.5rem;'>Workspace Awaiting Audits</h4>
            <p style='margin: 0; font-size: 0.9rem; max-width: 320px;'>Input a text stream in the workspace and select "Audit and Verify" to generate diagnostic stats.</p>
        </div>
        """, unsafe_allow_html=True)
