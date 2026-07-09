import streamlit as st
import pickle
import pandas as pd
import numpy as np
from src.preprocessing import full_preprocess_pipeline
from src.scraper import scrape_article

# Configure Page Layout and Style
st.set_page_config(
    page_title="Veritas AI | Fake News Detection Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium styling (Vanilla CSS Injection)
# Custom Premium styling (Vanilla CSS Injection with Animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.2);
        }
        50% {
            box-shadow: 0 0 25px rgba(129, 140, 248, 0.4);
            border-color: rgba(129, 140, 248, 0.5);
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    /* Global styles */
    .stApp {
        background-color: #080b11;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }
    
    /* Header Card (Animated Hero) */
    .hero-container {
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.2), transparent), 
                    radial-gradient(circle at bottom left, rgba(244, 114, 182, 0.08), transparent),
                    #0e131f;
        padding: 3.5rem 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.5);
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .hero-title {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
        letter-spacing: -0.03em;
        text-shadow: 0px 10px 20px rgba(99, 102, 241, 0.15);
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Premium Glassmorphism Cards */
    .card {
        background: rgba(14, 19, 31, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.8rem;
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .card:hover {
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 20px 35px -10px rgba(99, 102, 241, 0.15);
        transform: translateY(-2px);
    }
    
    /* Model Verdict Card Grid & Hover Effects */
    .model-verdict-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(8px);
        padding: 1.5rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .model-verdict-card:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(30, 41, 59, 0.9);
        box-shadow: 0 12px 20px -5px rgba(99, 102, 241, 0.25);
    }
    
    .verdict-badge-small {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 0.65rem;
        letter-spacing: 0.05em;
        transition: all 0.2s ease;
    }
    
    .badge-real-small {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.1);
    }
    
    .badge-fake-small {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 0 10px rgba(248, 113, 113, 0.1);
    }
    
    /* Custom Tags with Pulse / Glow */
    .keyword-tag {
        display: inline-block;
        background-color: rgba(99, 102, 241, 0.08);
        color: #c7d2fe;
        padding: 0.4rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        transition: all 0.2s ease;
    }
    
    .keyword-tag:hover {
        background-color: rgba(99, 102, 241, 0.2);
        color: #ffffff;
        transform: scale(1.05);
        border-color: rgba(99, 102, 241, 0.5);
    }

    /* Interactive Inputs & Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    }

    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model assets from disk
@st.cache_resource
def load_assets():
    assets = {}
    try:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            assets["vectorizer"] = pickle.load(f)
    except FileNotFoundError:
        st.error("Vectorizer 'models/tfidf_vectorizer.pkl' not found. Please run the training pipeline first.")
        assets["vectorizer"] = None
        
    models = ["knn", "logreg", "random_forest", "neuralnet"]
    for m in models:
        try:
            with open(f"models/{m}_model.pkl", "rb") as f:
                assets[m] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file 'models/{m}_model.pkl' not found. This model will be unavailable.")
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

# Initialize session state for article text
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

col_input, col_results = st.columns([1.6, 1.4], gap="large")

with col_input:
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <h3 style='font-weight: 700; font-size: 1.5rem; color: #f1f5f9;'>Linguistic Analysis Workspace</h3>
        <p style='color: #64748b; font-size: 0.9rem;'>Provide the article content below manually or fetch it live from the web.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_tab1, input_tab2 = st.tabs(["✍️ Manual Text Input", "🌐 Real-Time URL Fetcher"])
    
    with input_tab1:
        article_text = st.text_area(
            "Paste Article Text",
            value=st.session_state["article_text"],
            height=320,
            placeholder="Paste full news article content here to scan for credibility tags...",
            label_visibility="collapsed"
        )
        st.session_state["article_text"] = article_text
        
        # --- Example Buttons ---
        st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 1rem;'>Or, try one of our pre-loaded examples:</p>", unsafe_allow_html=True)
        example_cols = st.columns(2)
        with example_cols[0]:
            if st.button("Load REAL News Example", use_container_width=True):
                st.session_state["article_text"] = """WASHINGTON (Reuters) - The U.S. economy maintained a solid pace of growth in the fourth quarter, with the Federal Reserve's aggressive interest rate hikes appearing to have had little impact on consumer spending. Gross domestic product increased at a 2.9% annualized rate last quarter, the Commerce Department said in its advance fourth-quarter GDP estimate on Thursday. While that was a step down from the 3.2% pace logged in the third quarter, it was faster than the 2.6% rate that economists polled by Reuters had forecast. The report painted a picture of an economy that remains resilient in the face of the central bank's fastest rate-hiking cycle since the 1980s."""
                st.rerun()
        with example_cols[1]:
            if st.button("Load FAKE News Example", use_container_width=True):
                st.session_state["article_text"] = """A sensational report has just been leaked from a top-secret government facility, revealing a shocking conspiracy. Sources claim that for decades, world leaders have been secretly replaced by look-alike actors trained in an undisclosed location in the Swiss Alps. The report, which appeared on the website 'TruthUnleashed.info', alleges that this shadow government is controlling global events for their own benefit. "You won't believe what they've been hiding," an anonymous insider was quoted as saying. The article urges readers to share this information widely before it gets taken down. Read more via the link in the description."""
                st.rerun()
                
    with input_tab2:
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Enter any news website article URL to fetch and analyze its contents in real-time:</p>", unsafe_allow_html=True)
        news_url = st.text_input("News Article URL", placeholder="https://www.reuters.com/article-example...")
        
        if st.button("Fetch & Load Article", type="secondary", use_container_width=True):
            if news_url.strip():
                with st.spinner("Fetching live page content..."):
                    result = scrape_article(news_url.strip())
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Successfully fetched article: '{result['title']}'")
                        st.session_state["article_text"] = f"TITLE: {result['title']}\n\n{result['text']}"
                        st.rerun()
            else:
                st.warning("Please enter a valid URL.")
                
        if st.session_state["article_text"]:
            with st.expander("Preview of Loaded Content"):
                st.write(st.session_state["article_text"])
                
    analyze_btn = st.button("Audit and Verify Legitimacy", type="primary", use_container_width=True)

with col_results:
    st.markdown(f"""
    <div>
        <h3 style='font-weight: 700; font-size: 1.5rem; color: #f1f5f9;'>Diagnostic Audit Suite</h3>
        <p style='color: #64748b; font-size: 0.9rem;'>Consensus statistics across the classifier portfolio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if analyze_btn:
        # Check if vectorizer and at least one model is loaded
        if not assets.get("vectorizer") or len(assets) <= 1:
            st.error("Core model assets are missing. Please run the training pipeline via `python run_pipeline.py --train` and restart the app.")
        elif not st.session_state["article_text"].strip():
            st.error("Please insert a valid text sample to analyze.")
        else:
            with st.spinner("Decoding language vectors..."):
                # 1. Preprocess using optimized module regexes
                cleaned_text = full_preprocess_pipeline(st.session_state["article_text"])
                
                # Metric Calculations
                original_word_count = len(st.session_state["article_text"].split())
                cleaned_word_count = len(cleaned_text.split())
                removed_words = original_word_count - cleaned_word_count
                
                # 2. TF-IDF Vectorization
                vectorizer = assets["vectorizer"]
                vectorized_input = vectorizer.transform([cleaned_text])
                
                # Compute top local terms based on TF-IDF weights
                feature_names = np.array(vectorizer.get_feature_names_out())
                # Get the single row of the sparse matrix
                row = vectorized_input.tocoo()
                sorted_indices = np.argsort(row.data)[::-1]
                top_tokens = [(feature_names[row.col[i]], row.data[i]) for i in sorted_indices[:8]]
                
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
                    if not model: continue # Skip if model failed to load

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
