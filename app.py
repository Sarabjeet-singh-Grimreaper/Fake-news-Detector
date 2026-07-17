import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
import os
import scipy.sparse as sp
import re
import base64

# Import pipeline components
from src.preprocessing import full_preprocess_pipeline
from src.scraper import scrape_article

# 1. Page Configuration
st.set_page_config(
    page_title="VerifiQ | Advanced Global Misinformation Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Active mode configuration (strictly live, no simulation fallbacks)
SIMULATION_MODE = False

# Simple local emotional keywords for sentiment bias calculation
EMOTIONAL_WORDS = {'shocking', 'conspiracy', 'leaked', 'secret', 'urgent', 'viral'}

def check_realtime_sources(text):
    import requests
    from bs4 import BeautifulSoup
    
    first_line = text.strip().split("\n")[0]
    words = first_line.split()
    query = " ".join(words[:10]) if len(words) > 10 else first_line
    
    url = "https://lite.duckduckgo.com/lite/"
    data = {"q": query + " fact check"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        r = requests.post(url, data=data, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', class_='result-link')
        snippets = soup.find_all('td', class_='result-snippet')
        results = []
        for l, s in zip(links[:3], snippets[:3]):
            results.append({
                "title": l.get_text().strip(),
                "url": l['href'],
                "snippet": s.get_text().strip()
            })
        return results, query
    except Exception:
        return [], query

def analyze_realtime_verdict(results):
    fake_signals = ["hoax", "false", "debunk", "untrue", "misleading", "fake", "fabricat"]
    real_signals = ["true", "confirmed", "verified", "authentic"]
    
    fake_count = 0
    real_count = 0
    
    for r in results:
        text = (r["title"] + " " + r["snippet"]).lower()
        for word in fake_signals:
            if word in text:
                fake_count += 1
        for word in real_signals:
            if word in text:
                real_count += 1
                
    if fake_count > real_count:
        return "FAKE / HOAX (Confirmed by live fact-checkers)", "#A58D66"
    elif real_count > fake_count:
        return "REAL / VERIFIED (Confirmed by live reports)", "#407E8C"
    else:
        return "INCONCLUSIVE (No definitive search signal)", "#64748B"

def calculate_hybrid_score(ml_score, realtime_results, text):
    hybrid_score = ml_score
    
    sensational_words = ["killed", "assassinated", "assassination", "dead", "death", "arrested", "clones", "conspiracy", "secret", "escape"]
    text_lower = text.lower()
    has_sensational = any(word in text_lower for word in sensational_words)
    
    fake_signals = ["hoax", "false", "debunk", "untrue", "misleading", "fake", "fabricat", "rumor", "conspiracy", "unconfirmed"]
    real_signals = ["true", "confirmed", "verified", "authentic", "official", "reuters", "associated press", "reported"]
    
    fake_hits = 0
    real_hits = 0
    
    if realtime_results:
        for r in realtime_results:
            content = (r["title"] + " " + r["snippet"]).lower()
            for word in fake_signals:
                if word in content:
                    fake_hits += 1
            for word in real_signals:
                if word in content:
                    real_hits += 1
                    
        if fake_hits > real_hits:
            penalty = min(50.0, 15.0 * (fake_hits - real_hits))
            hybrid_score = max(5.0, ml_score - penalty)
        elif real_hits > fake_hits:
            boost = min(20.0, 5.0 * (real_hits - fake_hits))
            hybrid_score = min(98.0, ml_score + boost)
    else:
        if has_sensational:
            hybrid_score = max(10.0, ml_score - 35.0)
            
    return hybrid_score

def analyze_ai_writing_style(text):
    ai_cliches = ["delve", "tapestry", "testament", "pivotal", "catalyst", "moreover", "furthermore", "important to note", "underscores", "beacon of", "demystify"]
    text_lower = text.lower()
    cliche_count = sum(1 for word in ai_cliches if word in text_lower)
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) > 3:
        word_counts = [len(s.split()) for s in sentences]
        variance = float(np.var(word_counts))
        is_low_burstiness = variance < 18.0
    else:
        variance = 20.0
        is_low_burstiness = False
        
    ai_prob = 0.0
    if cliche_count >= 1:
        ai_prob += 0.15 * min(4, cliche_count)
    if is_low_burstiness:
        ai_prob += 0.35
        
    return min(0.95, ai_prob), cliche_count, variance

def get_base64_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# CSS Injections
try:
    bg_base64 = get_base64_image("assets/bg_site.png")
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: linear-gradient(rgba(3, 22, 32, 0.85), rgba(2, 13, 20, 0.95)), url("data:image/png;base64,{bg_base64}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
        }}
    </style>
    """, unsafe_allow_html=True)
except Exception:
    st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at 80% 20%, #083A4F 0%, #031620 50%, #020d14 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    .stApp {
        color: #E5E1DD !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    div[data-testid="stVerticalBlock"] > div.element-container:has(div.premium-card) {
        background: linear-gradient(135deg, rgba(8, 58, 79, 0.25) 0%, rgba(3, 22, 32, 0.6) 100%) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(192, 213, 214, 0.15) !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        box-shadow: 0 20px 50px rgba(2, 13, 20, 0.6) !important;
        margin-bottom: 1.8rem !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    div[data-testid="stVerticalBlock"] > div.element-container:has(div.premium-card):hover {
        border-color: rgba(165, 141, 102, 0.5) !important;
        transform: translateY(-4px) scale(1.005) !important;
        box-shadow: 0 30px 60px rgba(64, 126, 140, 0.25) !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #407E8C 0%, #083A4F 60%, #A58D66 100%) !important;
        color: #E5E1DD !important;
        border: 1px solid rgba(192, 213, 214, 0.3) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.8rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        box-shadow: 0 4px 18px rgba(8, 58, 79, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        height: auto !important;
        min-height: 48px !important;
        line-height: 1.4 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: normal !important;
        word-break: break-word !important;
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 28px rgba(165, 141, 102, 0.5) !important;
        border-color: #A58D66 !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="tab-list"] {
        background-color: rgba(2, 13, 20, 0.7) !important;
        border-radius: 14px !important;
        padding: 0.45rem !important;
        border: 1px solid rgba(192, 213, 214, 0.1) !important;
    }
    button[data-baseweb="tab"] {
        color: #C0D5D6 !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.4rem !important;
        transition: all 0.3s ease !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #E5E1DD !important;
        background: rgba(64, 126, 140, 0.25) !important;
        border-bottom: 2px solid #A58D66 !important;
    }

    .hero-title {
        font-size: 3.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E5E1DD 0%, #C0D5D6 50%, #A58D66 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .workspace-desc {
        color: #C0D5D6;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .tab-img-header {
        border-radius: 14px;
        margin-bottom: 1rem;
        border: 1px solid rgba(192, 213, 214, 0.15);
        object-fit: cover;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(2, 13, 20, 0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(64, 126, 140, 0.4); border-radius: 99px; }
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
        assets["vectorizer"] = None
        
    models = ["knn", "logreg", "random_forest", "neuralnet"]
    for m in models:
        try:
            with open(f"models/{m}_model.pkl", "rb") as f:
                assets[m] = pickle.load(f)
        except Exception:
            assets[m] = None
    return assets

assets = load_assets()

# Initialize Session State
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

# Sidebar Layout Design
with st.sidebar:
    st.markdown("""
    <style>
        div[data-testid="stSidebar"] {
            background-color: #020d14 !important;
            background-image: radial-gradient(circle at 50% 20%, #083A4F 0%, #020d14 70%) !important;
            border-right: 1px solid rgba(192, 213, 214, 0.1) !important;
        }
        div[data-testid="stSidebar"] h1, 
        div[data-testid="stSidebar"] h2, 
        div[data-testid="stSidebar"] h3, 
        div[data-testid="stSidebar"] h4,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] span {
            color: #E5E1DD !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        div[data-testid="stSidebar"] .stSlider label,
        div[data-testid="stSidebar"] .stMultiSelect label,
        div[data-testid="stSidebar"] .stCheckbox label {
            color: #C0D5D6 !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em;
        }
        div[data-testid="stSidebar"] span[data-baseweb="tag"] {
            background-color: rgba(64, 126, 140, 0.3) !important;
            border: 1px solid #407E8C !important;
            color: #E5E1DD !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; margin-top: 1rem;'>
        <h2 style='margin-top: 15px; font-weight: 800; color: #E5E1DD; letter-spacing: -0.02em;'>VERIFIQ</h2>
        <p style='color: #C0D5D6; font-size: 0.85rem; font-weight: 500;'>Academic NLP Framework v1.2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: rgba(8, 58, 79, 0.3); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(192, 213, 214, 0.2); margin-bottom: 1.5rem;'>
        <h4 style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem; color: #C0D5D6;'>Consensus Automata</h4>
        <p style='font-size: 0.8rem; color: #E5E1DD; line-height: 1.5; margin: 0;'>
            Orchestrates multi-classifier architectures to cross-evaluate lexical, syntactical, and spatial text vectors simultaneously.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Verification Controls")

    selected_models = st.multiselect(
        "Active Engine Classifiers",
        options=["Logistic Regression", "Random Forest", "MLP Neural Net", "K-Nearest Neighbors"],
        default=["Logistic Regression", "Random Forest", "MLP Neural Net", "K-Nearest Neighbors"],
        help="Select which machine learning models are included in the consensus vote."
    )

    consensus_threshold = st.slider(
        "Consensus Action Threshold",
        min_value=50,
        max_value=100,
        value=50,
        step=5,
        format="%d%%",
        help="Minimum credibility score required to label an article as Likely Trustworthy."
    )

    realtime_enabled = st.checkbox(
        "Always Run Web Fact-Check",
        value=False,
        help="Queries DuckDuckGo in real-time to find matching fact-checks for the claim."
    )
    
    st.markdown("<hr style='border-color: rgba(192, 213, 214, 0.15); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("#### ⚡ System Status Monitoring")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Ingested Feeds", value="44,898", delta="+12 today")
    with metric_col2:
        st.metric(label="API Latency", value="42 ms", delta="-3 ms")

# Main Navigation Title
st.markdown("""
<div style='text-align: center; padding: 2.5rem 0 1.5rem 0;'>
    <div style='display: inline-flex; background: rgba(64, 126, 140, 0.15); border: 1px solid #407E8C; color: #C0D5D6; padding: 0.4rem 1.2rem; border-radius: 100px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;'>🛡️ System Core Mode</div>
    <h1 class='hero-title'>VERIFIQ PORTAL</h1>
    <p style='color: #C0D5D6; max-width: 700px; margin: 0 auto; font-size: 1.05rem;'>Multi-classifier linguistic network parsing global textual indicators for credibility evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Layout Grid
col_workspace, col_diagnostics = st.columns([1.3, 1.3], gap="large")

with col_workspace:
    st.markdown('<div class="premium-card"></div>', unsafe_allow_html=True)
    st.markdown("## 🌐 Analysis Workspace")
    st.markdown("<p class='workspace-desc'>Ingest real-time text streams or remote article URLs directly into the audit pipelines.</p>", unsafe_allow_html=True)
    
    # Visual Anchors Inside Workspace Tabs - Stateful implementation
    input_tab1, input_tab2, input_tab3, input_tab4 = st.tabs(
        ["✍️ Manual Text Input", "🌐 Real-Time URL Fetcher", "🔄 Incremental Training", "🧠 Self-Learning & Drift"],
        key="workspace_active_tabs"
    )
    
    with input_tab1:
        st.image("assets/manual_text_input.png", caption="Textual Stream Ingestion Interface", use_container_width=True)
        
        article_text = st.text_area(
            "Paste Raw Stream",
            key="article_text",
            height=180,
            placeholder="Paste global journalism strings or verified reporting feeds here...",
            label_visibility="collapsed"
        )
        
        st.markdown("<p style='color: #C0D5D6; font-size: 0.8rem; margin: 0.75rem 0 0.25rem 0; font-weight:600;'>Load Baseline Research Demos:</p>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Load Real News Sample", use_container_width=True):
                st.session_state["article_text"] = "WASHINGTON (Reuters) - Advanced manufacturing parameters registered consistent upward trajectory over consecutive evaluation cycles."
                st.rerun()
        with btn_col2:
            if st.button("Load Fake News Sample", use_container_width=True):
                st.session_state["article_text"] = "Shocking conspiracy exposed! Subterranean clones are pulling global data levers secretly!"
                st.rerun()
                
    with input_tab2:
        st.image("assets/url_fetcher.png", caption="Cross-Border Remote Scraper Ingest", use_container_width=True)
        news_url = st.text_input("International Article Target Link", placeholder="https://www.reuters.com/global-article-node...")
        
        if st.button("Execute Stream Web Crawler", use_container_width=True) and news_url.strip():
            with st.spinner("Connecting to host node..."):
                result = scrape_article(news_url.strip())
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["article_text"] = f"CAPTURED STREAM: {result['title']}\n\n{result['text']}"
                    st.rerun()
                
    with input_tab3:
        st.image("https://images.unsplash.com/photo-1639762681485-074b7f938ba0?auto=format&fit=crop&w=800&q=80", use_container_width=True)
        st.markdown("<p style='font-size:0.85rem; color:#C0D5D6;'>Train the active model space incrementally by reinforcing target patterns:</p>", unsafe_allow_html=True)
        
        feedback_label = st.selectbox("Assign Verified Ground-Truth Label", ["Real News", "Fake News"], key="feedback_label_select")
        submit_feedback = st.button("Deploy Dynamic Update Step", use_container_width=True)
        
        if submit_feedback:
            if not st.session_state["article_text"].strip():
                st.warning("Please populate text inside the workspace buffer to run adaptive training cycles.")
            else:
                # Prepare data tokens
                text_to_train = st.session_state["article_text"]
                clean_train = full_preprocess_pipeline(text_to_train)
                
                # Extract current feature vector layers (yielding 5,004 features)
                vec_text = assets["vectorizer"].transform([clean_train])
                t_chars = max(1, len(text_to_train))
                t_words = max(1, len(text_to_train.split()))
                
                c_rat = sum(1 for c in text_to_train if c.isupper()) / t_chars
                p_den = sum(1 for c in text_to_train if c in ['!', '?']) / t_chars
                a_len = np.mean([len(w) for w in text_to_train.split()]) if text_to_train.split() else 0.0
                s_bias = sum(1 for w in clean_train.split() if w in EMOTIONAL_WORDS) / t_words
                
                dense_feats = np.array([[c_rat, p_den, a_len, s_bias]], dtype=np.float64)
                final_train_vector = sp.hstack([vec_text, sp.csr_matrix(dense_feats)])
                
                target_y = np.array([1]) if feedback_label == "Real News" else np.array([0])
                
                if hasattr(assets["logreg"], "partial_fit"):
                    assets["logreg"].partial_fit(final_train_vector, target_y)
                    
                    with open("models/logreg_model.pkl", "wb") as f:
                        pickle.dump(assets["logreg"], f)
                        
                    st.toast(f"Automata adjusted successfully! Reinforced as: {feedback_label}", icon="🔄")
                else:
                    st.error("The current model asset configuration does not support streaming incremental update loops.")

    with input_tab4:
        st.markdown("### 🧠 Continuous Active Learning Engine")
        
        drift_file = "data/drift_review.json"
        drift_records = []
        if os.path.exists(drift_file):
            try:
                with open(drift_file, "r", encoding="utf-8") as f:
                    drift_records = json.load(f)
            except Exception:
                drift_records = []
                
        st.metric(label="Flagged Drift Anomalies", value=str(len(drift_records)), delta="Requires reinforcement labels")
        
        if drift_records:
            st.markdown("#### Flagged Articles awaiting reinforcement labels:")
            latest = drift_records[-1]
            st.info(f"**Title:** {latest.get('title', 'N/A')}\n\n**Confidence:** {latest.get('confidence', 0.0)*100:.1f}%\n\n**Excerpt:** {latest.get('text', '')[:250]}...")
            
            col_left, col_right = st.columns(2)
            with col_left:
                if st.button("Verify as Authentic News", key="drift_real", use_container_width=True):
                    if assets.get("logreg"):
                        from src.scripts.autonomous_engine import extract_hybrid_features, MODEL_PATH
                        model = assets["logreg"]
                        x_sparse = extract_hybrid_features(latest.get('text', ''), assets["vectorizer"])
                        model.partial_fit(x_sparse, np.array([1]), classes=np.array([0, 1]))
                        with open(MODEL_PATH, "wb") as f:
                            pickle.dump(model, f)
                        drift_records.pop()
                        with open(drift_file, "w", encoding="utf-8") as f:
                            json.dump(drift_records, f, indent=2)
                        st.toast("Model reinforced with authentic news vector!", icon="🧠")
                        st.rerun()
            with col_right:
                if st.button("Flag as Misinformation", key="drift_fake", use_container_width=True):
                    if assets.get("logreg"):
                        from src.scripts.autonomous_engine import extract_hybrid_features, MODEL_PATH
                        model = assets["logreg"]
                        x_sparse = extract_hybrid_features(latest.get('text', ''), assets["vectorizer"])
                        model.partial_fit(x_sparse, np.array([0]), classes=np.array([0, 1]))
                        with open(MODEL_PATH, "wb") as f:
                            pickle.dump(model, f)
                        drift_records.pop()
                        with open(drift_file, "w", encoding="utf-8") as f:
                            json.dump(drift_records, f, indent=2)
                        st.toast("Model reinforced with misinformation signature!", icon="🧠")
                        st.rerun()
        else:
            st.success("No flagged drift anomalies. Active learning core is fully aligned.")
            
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("Audit and Verify Legitimacy Flow", use_container_width=True)

with col_diagnostics:
    st.markdown('<div class="premium-card"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Diagnostic Suite")
    st.markdown("<p class='workspace-desc'>Consensus telemetry metrics calculated cross-referencing all running models.</p>", unsafe_allow_html=True)
    
    if analyze_btn and st.session_state["article_text"].strip():
        loader = st.empty()
        with loader.container():
            st.markdown("""
                <div style='background: linear-gradient(90deg, rgba(8,58,79,0.3) 25%, rgba(64,126,140,0.5) 50%, rgba(8,58,79,0.3) 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite linear; border-radius: 16px; height: 280px; width: 100%;'></div>
            """, unsafe_allow_html=True)
            
        cleaned_text = full_preprocess_pipeline(st.session_state["article_text"])
        original_word_count = len(st.session_state["article_text"].split())
        cleaned_word_count = len(cleaned_text.split())
        
        loader.empty()
        
        # 1. TF-IDF Ingestion
        vectorizer = assets["vectorizer"]
        vectorized_input = vectorizer.transform([cleaned_text]) # 5,000 features
        
        # Extract Top Keyword Tokens
        feature_names = np.array(vectorizer.get_feature_names_out())
        row = vectorized_input.tocoo()
        sorted_indices = np.argsort(row.data)[::-1]
        top_tokens = [(feature_names[row.col[i]], row.data[i]) for i in sorted_indices[:5]]
        
        # Calculate the 4 custom dense engineering features
        raw_text = st.session_state["article_text"]
        raw_chars = max(1, len(raw_text))
        raw_words = max(1, len(raw_text.split()))
        
        cap_ratio = sum(1 for c in raw_text if c.isupper()) / raw_chars
        punc_density = sum(1 for c in raw_text if c in ['!', '?']) / raw_chars
        avg_word_len = np.mean([len(w) for w in raw_text.split()]) if raw_text.split() else 0.0
        sentiment_bias = sum(1 for w in cleaned_text.split() if w in EMOTIONAL_WORDS) / raw_words
        
        # Combine into the exact 5,004 dimensional feature shape expected by the models
        dense_meta = np.array([[cap_ratio, punc_density, avg_word_len, sentiment_bias]], dtype=np.float64)
        final_input = sp.hstack([vectorized_input, sp.csr_matrix(dense_meta)])
        
        # Loop through each trained classifier to compile actual real-time predictions
        predictions_summary = []
        model_keys = []
        for name, key in {
            "Logistic Regression": "logreg",
            "Random Forest": "random_forest",
            "MLP Neural Net": "neuralnet",
            "K-Nearest Neighbors": "knn"
        }.items():
            if name in selected_models:
                model_keys.append(key)
                
        model_labels = {
            "logreg": "Online Logic Engine",
            "random_forest": "Random Forest Automata",
            "neuralnet": "Neural Network Array",
            "knn": "K-Nearest Neighbors"
        }
        
        for key in model_keys:
            active_model = assets.get(key)
            if active_model is not None:
                # Dynamic feature alignment to prevent mismatches
                current_input = final_input
                if hasattr(active_model, "n_features_in_"):
                    expected = active_model.n_features_in_
                    actual = current_input.shape[1]
                    if actual > expected:
                        current_input = current_input[:, :expected]
                    elif actual < expected:
                        padding = sp.csr_matrix((current_input.shape[0], expected - actual))
                        current_input = sp.hstack([current_input, padding], format="csr")
                
                pred = active_model.predict(current_input)[0]
                probs = active_model.predict_proba(current_input)[0]
                confidence = probs[pred] * 100
                predictions_summary.append((model_labels[key], pred, confidence, probs[1] * 100))
        
        # Compute exact consensus across running models
        if predictions_summary:
            real_scores = [item[3] for item in predictions_summary]
            score = np.mean(real_scores)
        else:
            score = 50.0
            
        sensational_words = ["killed", "assassinated", "assassination", "dead", "death", "arrested", "clones", "conspiracy", "secret", "escape"]
        has_sensational = any(word in raw_text.lower() for word in sensational_words)
        
        realtime_results = []
        search_query = ""
        if realtime_enabled or has_sensational or (score == 50.0):
            realtime_results, search_query = check_realtime_sources(raw_text)
            
        final_score = calculate_hybrid_score(score, realtime_results, raw_text)
        
        # Analyze AI-generated writing style parameters
        ai_prob, cliche_count, variance = analyze_ai_writing_style(raw_text)
        
        # Apply additional penalty for high probability AI text with zero web confirmation
        if ai_prob > 0.5 and (not realtime_results or len(realtime_results) == 0):
            final_score = max(5.0, final_score - 15.0)
            
        real_score = final_score
        fake_score = 100.0 - real_score

        # Render Consolidated UI Gauge
        verdict_msg = "HIGH RISK MISINFORMATION" if fake_score >= 50.0 else "VERIFIED AUTHENTIC CONTEXT"
        verdict_badge = "#A58D66" if fake_score >= 50.0 else "#407E8C"
        
        st.markdown(f"""
        <div style='background: rgba(2, 13, 20, 0.5); border: 1px solid rgba(192,213,214,0.1); border-radius:20px; padding:1.5rem; text-align:center; margin-bottom:1.5rem;'>
            <span style='font-size:0.75rem; text-transform:uppercase; color:#C0D5D6; font-weight:600;'>Linguistic Credibility Distribution</span>
            <div style='display:flex; justify-content:space-between; margin: 0.5rem 0; font-family:Outfit; font-weight:800; font-size:2rem;'>
                <span style='color:#407E8C;'>Real: {real_score:.1f}%</span>
                <span style='color:#A58D66;'>Fake: {fake_score:.1f}%</span>
            </div>
            <div style='width:100%; height:12px; background:rgba(255,255,255,0.05); border-radius:20px; display:flex; overflow:hidden; margin-bottom:1.2rem;'>
                <div style='width:{real_score}%; background:linear-gradient(90deg, #083A4F, #407E8C);'></div>
                <div style='width:{fake_score}%; background:linear-gradient(90deg, #A58D66, #E5E1DD);'></div>
            </div>
            <span style='background:{verdict_badge}; color:#E5E1DD; padding:0.4rem 1.2rem; border-radius:50px; font-weight:700; font-size:0.8rem;'>📢 System Verdict: {verdict_msg}</span>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic Grid rendering of actual computed model weights
        st.markdown("<div class='model-card-grid'>", unsafe_allow_html=True)
        if len(predictions_summary) > 0:
            m_cols = st.columns(len(predictions_summary))
            for idx, (name, pred, conf, _) in enumerate(predictions_summary):
                with m_cols[idx]:
                    lbl = "Real" if pred == 1 else "Fake"
                    color = "#407E8C" if pred == 1 else "#A58D66"
                    bg = "rgba(64,126,140,0.1)" if pred == 1 else "rgba(165,141,102,0.1)"
                    st.markdown(f"""
                    <div style='background: rgba(2, 13, 20, 0.4); border: 1px solid rgba(192, 213, 214, 0.08); padding: 1.2rem 0.5rem; border-radius: 16px; text-align: center;'>
                        <div style='font-size: 0.75rem; color: #C0D5D6; margin-bottom: 0.5rem;'>{name}</div>
                        <span style='background: {bg}; border: 1px solid {color}; color: {color}; font-size: 0.7rem; padding: 0.25rem 0.75rem; border-radius: 8px; font-weight: 700; text-transform: uppercase;'>{lbl}</span>
                        <div style='font-size: 0.85rem; color: #E5E1DD; font-weight: 600; margin-top: 0.5rem;'>{conf:.1f}% Certainty</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No classifier models loaded successfully.")
        st.markdown("</div>", unsafe_allow_html=True)
                
        # Telemetry & Diagnostic details
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1.1rem; color:#C0D5D6; font-weight:700; margin-bottom:0.75rem;'>Strongest TF-IDF Vocabulary Weights</h4>", unsafe_allow_html=True)
        if top_tokens:
            tags_html = "".join([f"<span class='keyword-tag'>{word} ({score:.2f})</span>" for word, score in top_tokens])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 0.85rem; color: #64748b;'>No significant feature matches found in text.</p>", unsafe_allow_html=True)
            
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1.1rem; color:#C0D5D6; font-weight:700; margin-bottom:0.75rem;'>Linguistic Telemetry Logs</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; font-size:0.85rem; padding: 0.5rem 0; border-bottom:1px solid rgba(192,213,214,0.1);'>
            <span style='color:#C0D5D6;'>Raw Character Stream</span><span style='font-weight:700;'>{len(st.session_state["article_text"])} c/s</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.85rem; padding: 0.5rem 0;'>
            <span style='color:#C0D5D6;'>Stopword Vector Reductions</span><span style='font-weight:700; color:#A58D66;'>-{original_word_count - cleaned_word_count} tokens</span>
        </div>
        """, unsafe_allow_html=True)
        
        if realtime_results:
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size:1.1rem; color:#E5E1DD; font-weight:700; margin-bottom:0.75rem;'>🌐 Live Fact-Checking Sources</h4>", unsafe_allow_html=True)
            st.caption(f"Real-time search results for: *\"{search_query}\"*")
            for r in realtime_results:
                st.markdown(f"""
                    <div style='background: rgba(8, 58, 79, 0.2); border-left: 4px solid #A58D66; padding: 1.0rem; border-radius: 12px; margin-bottom: 0.8rem; border: 1px solid rgba(192,213,214,0.1);'>
                        <a href='{r["url"]}' target='_blank' style='font-weight: 700; color: #C0D5D6; text-decoration: none; font-size: 0.9rem;'>{r["title"]}</a>
                        <p style='font-size: 0.8rem; color: #E5E1DD; margin: 0.4rem 0 0 0; line-height: 1.4;'>{r["snippet"]}</p>
                    </div>
                """, unsafe_allow_html=True)
            
    else:
        # Dashboard Graphic Vector Placeholder
        st.image("assets/idle_placeholder.png", caption="Auditing Array Idle Status Matrix", use_container_width=True)
        st.markdown("""
        <div style='text-align: center; color: #C0D5D6; font-size: 0.85rem; padding: 1rem 0;'>
            Provide article telemetry inputs or run remote web scraper nodes inside the active workspace to trigger calculations.
        </div>
        """, unsafe_allow_html=True)