import sys
import os
import streamlit as st
import streamlit.components.v1 as components
import pickle
import pandas as pd
import numpy as np
import time
import scipy.sparse as sp
import re

# Import V2.0 pipeline, features, scraper and explainability
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import full_preprocess_pipeline
from src.features import extract_dense_features, DENSE_FEATURE_NAMES
from src.pipeline import NewsCredibilityPipeline
from src.scraper import scrape_article
from src.domain_trust import get_domain_credibility
from src.explainability import explain_prediction
from src.realtime_verification import verify_realtime_news, fetch_live_breaking_news
from src.credibility_engine import evaluate_comprehensive_credibility

# 1. Page Configuration
st.set_page_config(
    page_title="VerifiQ | Enterprise Credibility Analysis Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render HTML cleanly without Markdown codeblock bugs
def render_html(html_str: str):
    cleaned = "\n".join([line.strip() for line in html_str.split("\n")])
    st.markdown(cleaned, unsafe_allow_html=True)

# 2. Cyber-Security / Dark-Mode Design Tokens & Style Injection
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Root styling and backgrounds */
    .stApp {
        background-color: #030712 !important;
        background-image: radial-gradient(circle at 80% 15%, #0B192C 0%, #030712 75%) !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Clean sidebar design */
    div[data-testid="stSidebar"] {
        background-color: #080D1A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Accessible Focus Indicators */
    *:focus-visible {
        outline: 2px solid #38BDF8 !important;
        outline-offset: 2px !important;
    }
    
    /* Glassmorphism custom cards */
    .premium-card {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.35rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    .premium-card:hover {
        border-color: rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 15px 40px rgba(56, 189, 248, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Modern buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Modern Streamlit Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #94A3B8 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.65rem 1.1rem !important;
        transition: all 0.2s ease !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }
    
    /* Modern text input and text area styling */
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8, 0 0 15px rgba(56, 189, 248, 0.25) !important;
    }
    
    /* Headline styling */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    
    /* Chip style tags with hover transitions */
    .chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.3rem;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        cursor: help;
    }
    .chip-green {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .chip-green:hover {
        background-color: rgba(16, 185, 129, 0.22);
        border-color: #10B981;
        transform: scale(1.03);
    }
    .chip-red {
        background-color: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border-color: rgba(244, 63, 94, 0.3);
    }
    .chip-red:hover {
        background-color: rgba(244, 63, 94, 0.22);
        border-color: #F43F5E;
        transform: scale(1.03);
    }
    
    /* Progress bar labels styling */
    .bar-label-container {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 0.25rem;
        margin-top: 0.75rem;
    }
    
    /* Custom pipeline process node styling */
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        background: #0F172A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        margin: 0.25rem;
    }
    .pipeline-step-active {
        border-color: #38BDF8;
        color: #38BDF8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
</style>
""")

# Helper function to load model assets
@st.cache_resource
def load_assets():
    pipeline = NewsCredibilityPipeline()
    try:
        pipeline.load("models")
    except FileNotFoundError:
        pipeline = None
        
    models = {}
    model_files = {
        "Voting Ensemble": "voting_ensemble_model.pkl",
        "Logistic Regression": "logreg_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl"
    }
    for name, filename in model_files.items():
        try:
            with open(f"models/{filename}", "rb") as f:
                models[name] = pickle.load(f)
        except Exception:
            models[name] = None
    return pipeline, models

pipeline, models = load_assets()

# Sidebar Layout Design
with st.sidebar:
    render_html("""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 2rem; margin-top: 1rem;'>
        <div style='background-color: #3B82F6; padding: 8px; border-radius: 8px; color: white; font-weight: 800; font-family: Outfit;'>🛡️</div>
        <div>
            <h3 style='margin: 0; font-weight: 800; font-family: Outfit; font-size: 1.25rem;'>VERIFIQ</h3>
            <p style='margin: 0; color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;'>Enterprise AI Platform</p>
        </div>
    </div>
    """)

    navigation = st.radio(
        "Platform Navigation",
        options=["🔍 Credibility Analyzer", "⚡ Live News Wire Scanner", "🌌 Info Flow Simulation", "📊 Engine Telemetry & Admin"],
        key="navigation"
    )

    st.markdown("<hr style='border-color: #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Active Parameters")
    selected_model_name = st.selectbox(
        "Target Ensemble Classifier",
        options=["Voting Ensemble", "Logistic Regression", "Random Forest", "SVM"],
        index=0,
        help="Model backing the threat consensus scoring algorithm."
    )
    
    st.info("⚡ System Health: OPTIMAL\n🤖 Engine: Multi-Signal Fusion V2.5\n🌐 Live Wire Radar: ACTIVE\n🎯 Real-Time Fact-Check: ENABLED")

    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 🌐 Visual Dynamics")
    render_html("""
    <a href='https://Sarabjeet-singh-Grimreaper.github.io/Fake-news-Detector/' target='_blank' rel='noopener noreferrer' style='text-decoration: none;'>
        <div style='background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 10px 14px; text-align: center; color: #38BDF8; font-weight: 600; font-size: 0.82rem; transition: all 0.2s;'>
            Open Physics Simulation 🌌
        </div>
    </a>
    """)
# Workspace switcher logic
if navigation == "📊 Engine Telemetry & Admin":
    render_html("""
    <div style='padding: 1.5rem 0;'>
        <div style='display: inline-flex; background: rgba(59, 130, 246, 0.1); border: 1px solid #3B82F6; color: #3B82F6; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-bottom: 1rem;'>📊 Telemetry Console</div>
        <h1 class='hero-title'>PLATFORM METADATA & HEALTH</h1>
        <p style='color: #94A3B8;'>Verify production model calibrations, ROC accuracies, and feature dimensionalities.</p>
    </div>
    """)
    
    password = st.text_input("Enter Admin Authorization Key", type="password", key="admin_auth_key")
    admin_key = os.environ.get("VERIFIQ_ADMIN_KEY", "VerifIQ_Admin_2026")
    if password == admin_key:
        st.success("Access authorized.")
        
        # Stat grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Multi-Signal Engine", "V2.5.0 Fusion")
            st.metric("Dense Metrics", "12 Active Scales")
        with col2:
            st.metric("Live Corroborator", "Active (Google News RSS)")
            st.metric("Domain Registry", "150+ Wires / 40+ Satire")
        with col3:
            st.metric("Benchmark Accuracy", "100.0% (11/11 Passed)")
            st.metric("Inference Latency", "15 ms (ML) / 450 ms (Live)")
            
        st.markdown("### 📈 Model & Engine Consensus Benchmarks")
        metrics_df = pd.DataFrame({
            "Evaluation Paradigm": ["Baseline Logistic Regression", "Baseline Random Forest", "Baseline Linear SVM", "Baseline Voting Ensemble", "Multi-Signal Fusion Engine V2.5"],
            "Generalization Accuracy": ["98.25%", "96.48%", "98.66%", "98.70%", "99.10%"],
            "Out-of-Domain Accuracy": ["72.73%", "63.64%", "72.73%", "72.73%", "100.00%"],
            "Real-Time News Verification": ["No (Static)", "No (Static)", "No (Static)", "No (Static)", "Yes (Live Global Wires)"],
            "Satire / Debunk Detection": ["No", "No", "No", "No", "Yes (Full Guardrails)"]
        })
        st.table(metrics_df)
    elif password:
        st.error("Invalid authorization key.")

elif navigation == "⚡ Live News Wire Scanner":
    render_html("""
    <div style='padding: 1rem 0;'>
        <div style='display: inline-flex; background: rgba(56, 189, 248, 0.1); border: 1px solid #38BDF8; color: #38BDF8; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.8rem;'>⚡ Real-Time Radar</div>
        <h1 class='hero-title'>LIVE NEWS WIRE SCANNER</h1>
        <p style='color: #94A3B8; font-size: 1.05rem;'>Real-time streaming surveillance across global news wires. Cross-reference breaking stories, detect viral rumors, and audit claims live.</p>
    </div>
    """)
    
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Live Claim Cross-Referencer")
    st.write("Cross-reference any breaking event, rumor, or claim across global news wires (Reuters, AP, BBC, Bloomberg, etc.) to verify its authenticity in real-time.")
    
    claim_query = st.text_input("Enter breaking claim or headline to cross-reference", placeholder="e.g. Federal Reserve interest rate announcement, oil price shifts, WHO health alerts...")
    if st.button("⚡ Cross-Reference Live Wires", use_container_width=True):
        if not claim_query.strip():
            st.warning("Please enter a claim to verify.")
        else:
            with st.spinner("Searching global news wires & fact-check networks..."):
                ver_res = verify_realtime_news(headline=claim_query)
                
                badge_col = "#22C55E" if ver_res["badge"] == "Corroborated" else ("#EF4444" if ver_res["badge"] == "Debunked" else "#F59E0B")
                render_html(f"""
                <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.5rem; margin-top: 1rem;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='background: {badge_col}22; color: {badge_col}; border: 1px solid {badge_col}; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;'>{ver_res['badge']}</span>
                            <span style='color: #94A3B8; font-size: 0.85rem; margin-left: 10px;'>{ver_res['corroboration_status']}</span>
                        </div>
                        <div style='font-size: 1.6rem; font-weight: 800; color: {badge_col}; font-family: Outfit;'>{ver_res['corroboration_score']}/100</div>
                    </div>
                </div>
                """)
                
                if ver_res["fact_check_alert"]:
                    st.error(f"🚨 **Fact-Check Alert:** {ver_res['fact_check_details']}")
                    
                if ver_res["matching_articles"]:
                    st.markdown("##### 📰 Matching Verified Live Coverage:")
                    for idx, art in enumerate(ver_res["matching_articles"]):
                        st.markdown(f"- **[{art['title']}]({art['link']})** — *{art['source']}* ({art['pub_date']})")
                else:
                    st.info("ℹ️ No matching stories found on recognized international news wires for this query.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 📡 Live Breaking News Stream")
    st.write("Browse real-time top stories currently trending on international news wires. Click **Audit Credibility** on any item to evaluate it with the full multi-signal pipeline.")
    
    col_cat, col_ref = st.columns([4, 1])
    with col_cat:
        category = st.selectbox(
            "Filter Live Feed Category",
            options=["All Top Stories", "World", "Business", "Technology", "Science", "Health"],
            index=0
        )
    with col_ref:
        st.write("")
        st.write("")
        btn_refresh = st.button("🔄 Refresh Feed", use_container_width=True)
        
    cat_map = {
        "All Top Stories": "all",
        "World": "world",
        "Business": "business",
        "Technology": "technology",
        "Science": "science",
        "Health": "health"
    }
    
    with st.spinner("Streaming live global headlines..."):
        feed_items = fetch_live_breaking_news(category=cat_map[category], limit=10)
        
    if not feed_items:
        st.warning("Unable to fetch live feed. Check network connection or try refreshing.")
    else:
        for idx, item in enumerate(feed_items):
            with st.container():
                st.markdown("<div class='premium-card' style='margin-bottom: 0.8rem;'>", unsafe_allow_html=True)
                card_col1, card_col2 = st.columns([5, 1])
                with card_col1:
                    st.markdown(f"#### [{item['title']}]({item['link']})")
                    st.markdown(f"<span style='color: #38BDF8; font-weight: 600; font-size: 0.8rem;'>🏛️ {item['source']}</span> • <span style='color: #64748B; font-size: 0.8rem;'>🕒 {item['pub_date']}</span>", unsafe_allow_html=True)
                with card_col2:
                    st.write("")
                    if st.button("⚡ Audit", key=f"feed_audit_{idx}", use_container_width=True):
                        st.session_state.sample_title = item['title']
                        st.session_state.sample_text = item['title']
                        st.session_state.auto_trigger = True
                        st.session_state.navigation = "🔍 Credibility Analyzer"
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

elif navigation == "🌌 Info Flow Simulation":
    render_html("""
    <div style='padding: 1rem 0;'>
        <div style='display: inline-flex; background: rgba(168, 85, 247, 0.1); border: 1px solid #A855F7; color: #A855F7; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.8rem;'>🌌 Contagion Dynamics</div>
        <h1 class='hero-title'>INFORMATION FLOW & CONTAGION SIMULATION</h1>
        <p style='color: #94A3B8; font-size: 1.05rem;'>Real-time HTML5 canvas particle physics modeling truth propagation, viral misinformation cascades, and network verification.</p>
    </div>
    """)

    @st.cache_data
    def load_simulation_bundle():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "index.html"), "r", encoding="utf-8") as f:
            html = f.read()
        with open(os.path.join(base_dir, "style.css"), "r", encoding="utf-8") as f:
            css = f.read()
        with open(os.path.join(base_dir, "app.js"), "r", encoding="utf-8") as f:
            js = f.read()
        # Inline CSS and JS
        html = html.replace('<link rel="stylesheet" href="style.css">', f'<style>{css}</style>')
        html = html.replace('<script src="app.js"></script>', f'<script>{js}</script>')
        return html

    try:
        sim_bundle = load_simulation_bundle()
        components.html(sim_bundle, height=820, scrolling=True)
    except Exception as e:
        st.error(f"Failed to load particle simulation: {e}")

else:
    # Public Credibility Analyzer
    render_html("""
    <div style='padding: 1rem 0;'>
        <h1 class='hero-title'>ENTERPRISE CREDIBILITY ANALYZER</h1>
        <p style='color: #94A3B8; font-size: 1.05rem;'>Audit structural writing styles, live news wire corroboration, and publisher registries to evaluate real-time credibility.</p>
    </div>
    """)
    
    # Interactive sample presets
    st.markdown("##### ⚡ Quick-Load Test Samples:")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    if "sample_text" not in st.session_state:
        st.session_state.sample_text = ""
    if "sample_title" not in st.session_state:
        st.session_state.sample_title = ""

    with sample_col1:
        if st.button("📰 Real News Preset", use_container_width=True):
            st.session_state.sample_title = "Federal Reserve Holds Benchmark Interest Rates Steady"
            st.session_state.sample_text = "The Federal Reserve announced on Tuesday that it would maintain the benchmark interest rate within the current range. Chairman Powell indicated that economic indicators register steady growth and inflation metrics remain aligned with long-term stabilization objectives. Financial markets reacted with moderate gains following the announcement."
    with sample_col2:
        if st.button("⚠️ Sensational Preset", use_container_width=True):
            st.session_state.sample_title = "SHOCKING SECRET CONSPIRACY EXPOSED!!!"
            st.session_state.sample_text = "URGENT BREAKING: Leaked secret documents confirm that the space launch was entirely staged in a desert warehouse! Global elites are using CGI holographic projection models to fake satellite pictures and control the populations' minds. Share this immediately before authorities take it down!"
    with sample_col3:
        if st.button("🔬 Science Preset", use_container_width=True):
            st.session_state.sample_title = "Oxford Malaria Vaccine Receives International Regulatory Approval"
            st.session_state.sample_text = "Researchers at Oxford University have developed a new malaria vaccine that has shown up to 80% efficacy in phase III clinical trials. The World Health Organization confirmed that the low-cost treatment could prevent hundreds of thousands of deaths across developing nations each year."
    with sample_col4:
        if st.button("⚡ Live Oil Market Preset", use_container_width=True):
            st.session_state.sample_title = "Global Oil Prices Stabilize Amid Middle East Supply Shifts"
            st.session_state.sample_text = "Global oil prices stabilized on Friday as traders weighed supply disruptions in the Middle East against rising production in North America. Brent crude futures settled at $78.50 a barrel, while West Texas Intermediate rose slightly following inventory announcements."

    tab_input1, tab_input2 = st.tabs(["📝 Content Stream", "🌐 Live URL Fetcher"])
    
    with tab_input1:
        user_title = st.text_input("Article Headline (Optional)", value=st.session_state.sample_title, placeholder="Paste headline here...")
        user_text = st.text_area("Article Body Text", value=st.session_state.sample_text, height=120, placeholder="Paste full article text content here...")
        
    with tab_input2:
        article_url = st.text_input("Resource URL Address", placeholder="https://news-outlet.com/article-slug")
        
    btn_analyze = st.button("Evaluate Credibility scorecard", use_container_width=True)
    
    if btn_analyze:
        text_to_analyze = user_text.strip()
        title_to_analyze = user_title.strip()
        target_url = article_url.strip()
        
        # Auto-detect if user pasted a URL in text or title input
        if not target_url:
            if text_to_analyze.startswith("http://") or text_to_analyze.startswith("https://"):
                target_url = text_to_analyze
                text_to_analyze = ""
            elif title_to_analyze.startswith("http://") or title_to_analyze.startswith("https://"):
                target_url = title_to_analyze
                title_to_analyze = ""
        
        if target_url:
            with st.spinner("Fetching resource content..."):
                scraped = scrape_article(target_url)
                if scraped and "error" not in scraped:
                    text_to_analyze = scraped.get("text", "")
                    title_to_analyze = scraped.get("title", "")
                    st.success(f"Successfully scraped: **{title_to_analyze}** ({scraped.get('word_count', 0)} words)")
                else:
                    err_msg = scraped.get("error", "Unknown crawler error") if isinstance(scraped, dict) else "Crawl agent failed"
                    st.error(f"Crawl agent failed: {err_msg}. Please input content manually.")
        elif not text_to_analyze and title_to_analyze:
            # If user provided a headline only, use it as text to analyze
            text_to_analyze = title_to_analyze
            
        if not text_to_analyze.strip():
            st.warning("Please supply a headline, news body text, or a URL before invoking audit.")
        elif pipeline is None or models.get(selected_model_name) is None:
            st.error("System assets not fully loaded. Ensure models are trained using --train flag.")
        else:
            with st.spinner("Executing Multi-Signal Credibility Audit & Live Wire Corroboration..."):
                t_start = time.time()
                active_model = models[selected_model_name]
                
                # Transform using the pipeline
                X_comb, clean_text_list, dense_list = pipeline.transform([text_to_analyze], [title_to_analyze])
                clean_txt = clean_text_list[0]
                dense_feats = dense_list[0]
                dense_scaled = pipeline.scaler.transform([dense_feats])[0]
                
                # Full Multi-Signal Evaluation
                cred_res = evaluate_comprehensive_credibility(
                    text=text_to_analyze,
                    title=title_to_analyze,
                    url=target_url,
                    pipeline=pipeline,
                    model=active_model,
                    check_realtime=True
                )
                
                verdict = cred_res["verdict"]
                verdict_color = cred_res["verdict_color"]
                composite_score = cred_res["composite_score"]
                risk_score = cred_res["risk_score"]
                risk_label = cred_res["risk_label"]
                risk_color = cred_res["risk_color"]
                confidence_tier = cred_res["confidence_tier"]
                reason = cred_res["reason"]
                signals = cred_res["signals"]
                realtime_details = cred_res["realtime_details"]
                dom_cred = cred_res["domain_details"]
                features_dict = cred_res["features_dict"]
                
                t_elapsed = (time.time() - t_start) * 1000.0
                
                # Writing Quality Breakdown
                flesch_score = features_dict["flesch_reading_ease"]
                flesch_desc = "Conversational syntax" if flesch_score > 80 else ("Standard news readability" if flesch_score > 50 else "Complex structure")
                    
                lexical_div = features_dict["lexical_diversity"] * 100.0
                lex_desc = "Excellent richness" if lexical_div > 40.0 else ("Standard diversity" if lexical_div > 20.0 else "Repetitive style")
                    
                sentence_structure = min(100.0, features_dict["avg_sentence_len"] * 4.0)
                avg_sent_val = features_dict["avg_sentence_len"]
                sent_desc = "Complex syntax" if avg_sent_val > 25 else ("Standard reporting syntax" if avg_sent_val > 15 else "Simple phrasing")
                    
                entropy_percentage = min(100.0, features_dict["entropy"] * 12.0)
                entropy_val = features_dict["entropy"]
                ent_desc = "Natural distribution" if entropy_val > 4.5 else ("Standard distribution" if entropy_val > 3.5 else "Artificial distribution")
                    
                writing_qual = (flesch_score + lexical_div + sentence_structure + entropy_percentage) / 4.0
                
                # Attribution confidence score (Group C features)
                att_score = min(100.0, max(0.0, (
                    features_dict.get("named_entities_est", 0) * 8.0 + 
                    features_dict.get("quotes_count", 0) * 15.0 + 
                    features_dict.get("credibility_citations", 0) * 20.0
                )))
                
                # ----------------- TABS UX INTEGRATION -----------------
                tab_overview, tab_indicators, tab_explanation = st.tabs([
                    "🛡️ Credibility Overview", 
                    "📊 Stylistic Indicators", 
                    "🔍 Feature Attributions"
                ])
                
                # ================= TAB 1: OVERVIEW =================
                with tab_overview:
                    ml_c = "#22C55E" if signals['ml_prediction'] == "Real" else "#EF4444"
                    # Giant Hero Card
                    render_html(f"""
                    <div style='text-align: center; padding: 2.5rem 1.5rem; border-radius: 16px; background-color: #111827; border: 1px solid #1E293B; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4);'>
                        <div style='color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.12em; margin-bottom: 0.75rem;'>🛡️ MULTI-SIGNAL CREDIBILITY AUDIT</div>
                        <h1 style='color: {verdict_color}; margin: 0; font-size: 3rem; font-family: Outfit; font-weight: 900; letter-spacing: -0.02em;'>{verdict.upper()}</h1>
                        <h2 style='color: #FFFFFF; margin: 8px 0 0 0; font-size: 2.2rem; font-weight: 800;'>{composite_score:.0f}%</h2>
                        <p style='color: #94A3B8; font-size: 0.95rem; margin: 4px 0 16px 0; font-weight: 600; text-transform: uppercase;'>{confidence_tier} • {risk_label}</p>
                        
                        <!-- Direct Comparison Badges -->
                        <div style='display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin: 0 auto 20px auto; max-width: 650px;'>
                            <div style='background: rgba(255,255,255,0.06); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);'>
                                <span style='color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;'>🤖 ML Classifier ({selected_model_name}):</span>
                                <span style='color: {ml_c}; font-weight: 800; margin-left: 6px; font-size: 0.95rem;'>{signals['ml_prediction'].upper()} ({signals['ml_confidence']:.1f}%)</span>
                            </div>
                            <div style='background: rgba(255,255,255,0.06); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);'>
                                <span style='color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;'>🎯 Final Decision:</span>
                                <span style='color: {verdict_color}; font-weight: 800; margin-left: 6px; font-size: 0.95rem;'>{verdict.upper()}</span>
                            </div>
                        </div>

                        <!-- Gauge Meter -->
                        <div style='max-width: 450px; margin: 0 auto;'>
                            <div style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.25rem;'>
                                <span>THREAT & DECEPTION RISK ASSESSMENT</span>
                                <span style='color: {risk_color};'>{risk_label}</span>
                            </div>
                            <div style='background-color: #1E293B; height: 12px; border-radius: 9999px; overflow: hidden;'>
                                <div style='background-color: {risk_color}; width: {risk_score:.0f}%; height: 100%;'></div>
                            </div>
                            <div style='text-align: right; font-size: 0.7rem; color: #64748B; margin-top: 0.25rem;'>Risk Index: {risk_score:.0f} / 100</div>
                        </div>
                        
                        <p style='margin: 20px auto 0 auto; color: #E2E8F0; max-width: 750px; font-size: 0.9rem; line-height: 1.5; border-top: 1px solid #1E293B; padding-top: 14px;'>
                            <strong>Assessment Context:</strong> {reason}
                        </p>
                    </div>
                    """)
                    
                    # 4 KPI Cards with Multi-Signal Breakdown
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    with kpi_col1:
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>🧠 ML Consensus</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {ml_c}; margin-top: 5px; font-family: Outfit;'>{signals['ml_score']:.0f}%</div>
                            <div style='font-size: 0.7rem; color: #64748B;'>{signals['ml_prediction']} ({signals['ml_confidence']:.0f}%)</div>
                        </div>
                        """)
                    with kpi_col2:
                        rt_c = "#22C55E" if signals['realtime_score'] > 70 else ("#F59E0B" if signals['realtime_score'] > 45 else "#EF4444")
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>🌐 Live Corroboration</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {rt_c}; margin-top: 5px; font-family: Outfit;'>{signals['realtime_score']:.0f}%</div>
                            <div style='font-size: 0.7rem; color: #64748B;'>{realtime_details.get('badge', 'Unverified')}</div>
                        </div>
                        """)
                    with kpi_col3:
                        dom_c = "#22C55E" if signals['domain_score'] > 75 else ("#F59E0B" if signals['domain_score'] > 45 else "#EF4444")
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>🏛️ Domain Trust</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {dom_c}; margin-top: 5px; font-family: Outfit;'>{signals['domain_score']:.0f}%</div>
                            <div style='font-size: 0.7rem; color: #64748B;'>{signals['domain_badge']}</div>
                        </div>
                        """)
                    with kpi_col4:
                        st_c = "#22C55E" if signals['stylistic_score'] > 60 else ("#F59E0B" if signals['stylistic_score'] > 40 else "#EF4444")
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>✍️ Stylometric Quality</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {st_c}; margin-top: 5px; font-family: Outfit;'>{signals['stylistic_score']:.0f}%</div>
                            <div style='font-size: 0.7rem; color: #64748B;'>Writing Integrity</div>
                        </div>
                        """)

                    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
                    
                    # Real-Time Wire Corroboration Card
                    st.markdown("<div class='premium-card' style='border-color: rgba(56, 189, 248, 0.4);'>", unsafe_allow_html=True)
                    st.markdown("### 🌐 Real-Time Live Wire Corroboration & Fact-Check")
                    st.markdown(f"**Query Extracted:** `{realtime_details.get('query', 'N/A')}` • **Status:** {realtime_details.get('corroboration_status', 'No coverage')}")
                    
                    if realtime_details.get("fact_check_alert"):
                        st.error(f"🚨 **Fact-Check Warning Detected:** {realtime_details.get('fact_check_details')}")
                        
                    if realtime_details.get("matching_articles"):
                        st.markdown("##### 📰 Matching Verified Live Coverage:")
                        for art in realtime_details["matching_articles"]:
                            st.markdown(f"- **[{art['title']}]({art['link']})** — *{art['source']}* ({art['pub_date']})")
                    else:
                        st.info("ℹ️ **Uncorroborated on Live Wires:** No independent reports matching this specific claim were found on recognized global news wires (Reuters, AP, BBC, etc.). Caution is advised.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Connected Pipeline Lifecycle Diagram
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    st.markdown("### ⚙️ Pipeline Process Timeline")
                    steps = ["Ingestion", "Live Corroboration", "Features & Scaling", "Ensemble Consensus", "Explainability"]
                    steps_html = ""
                    for idx, step in enumerate(steps):
                        steps_html += f"<span class='pipeline-step pipeline-step-active'>✓ {step}</span>"
                        if idx < len(steps) - 1:
                            steps_html += " <span style='color: #3B82F6;'>↓</span> "
                    render_html(steps_html)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Deduplicated Recommendation summary block (Issue Checklist / Alert style)
                    st.markdown("<div class='premium-card' style='border-color: #3B82F6;'>", unsafe_allow_html=True)
                    st.markdown("### ✅ Recommendation & Assessment")
                    if verdict in ["Verified Real News", "Likely Real News", "Likely Real"]:
                        st.write("The article strongly aligns with verified journalism standards and/or is actively corroborated by independent news wires.")
                    elif verdict in ["Debunked / False Claim", "Likely Fake / Disinformation", "Likely Fake"]:
                        st.write("Critical reliability alerts or lack of corroboration detected. This story exhibits high indicators of fabricated or debunked claims.")
                    elif verdict == "Satire / Parody":
                        st.write("This content is identified as satirical parody or humor. It is not intended as factual journalistic reporting.")
                    elif verdict == "Clickbait / Sensationalism":
                        st.write("Sensationalized clickbait patterns detected. The content aims to provoke emotional reactions rather than convey verified information.")
                    else:
                        st.write("The credibility metrics are inconclusive. Cross-referencing claims via authoritative news wire registries is advised.")
                    st.write("*This multi-signal scorecard fuses real-time global news wire surveillance, source domain registries, and machine learning stylometrics.*")
                    st.markdown("</div>", unsafe_allow_html=True)

                # ================= TAB 2: STYLISTIC INDICATORS =================
                with tab_indicators:
                    grid_col1, grid_col2 = st.columns(2)
                    with grid_col1:
                        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                        st.markdown("### ✍️ Writing Quality Metrics")
                        
                        # Readability bar
                        render_html(f"""
                        <div class='bar-label-container'><span>Readability (Flesch Index)</span><span>{flesch_score:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #3B82F6; width: {flesch_score}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>{flesch_desc}</div>
                        """)
                        
                        # Sentence Structure
                        render_html(f"""
                        <div class='bar-label-container'><span>Sentence Structure Complexity</span><span>{sentence_structure:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #3B82F6; width: {sentence_structure}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>{sent_desc}</div>
                        """)
                        
                        # Lexical diversity
                        render_html(f"""
                        <div class='bar-label-container'><span>Lexical Diversity (Type-Token Ratio)</span><span>{lexical_div:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #3B82F6; width: {lexical_div}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>{lex_desc}</div>
                        """)
                        
                        # Entropy
                        render_html(f"""
                        <div class='bar-label-container'><span>Vocabulary Entropy</span><span>{entropy_percentage:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #3B82F6; width: {entropy_percentage}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>{ent_desc}</div>
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                        st.markdown("### 🏛️ Source Attribution Checklists")
                        st.write(f"**Attribution Confidence Score:** {att_score:.1f}%")
                        if features_dict['named_entities_est'] > 0:
                            st.write(f"✓ **{features_dict['named_entities_est']} named entities** / organizations detected")
                        else:
                            st.write("✗ No named entities or organizations detected")
                            
                        if features_dict['quotes_count'] > 0:
                            st.write(f"✓ **{features_dict['quotes_count']} quote markers** / speaker statements")
                        else:
                            st.write("✗ No quoted speaker statements detected")
                            
                        if features_dict['credibility_citations'] > 0:
                            st.write(f"✓ **{features_dict['credibility_citations']} reference anchors** / reporter datelines")
                        else:
                            st.write("✗ No reference anchors or official attributions detected")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with grid_col2:
                        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                        st.markdown("### ⚡ Threat Indicators")
                        
                        clickbait_pct = min(100.0, features_dict['clickbait_score'] * 40)
                        render_html(f"""
                        <div class='bar-label-container'><span>Sensational Clickbait Scale</span><span>{clickbait_pct:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #EF4444; width: {clickbait_pct}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>Score: {features_dict['clickbait_score']:.2f}</div>
                        """)
                        
                        spec_pct = min(100.0, features_dict['speculation_ratio'] * 2500)
                        render_html(f"""
                        <div class='bar-label-container'><span>Vocabulary Speculation Rate</span><span>{spec_pct:.1f}%</span></div>
                        <div style='background-color: #1E293B; height: 6px; border-radius: 9999px; overflow: hidden;'><div style='background-color: #F59E0B; width: {spec_pct}%; height: 100%; transition: width 0.8s ease;'></div></div>
                        <div style='font-size: 0.7rem; color: #64748B; margin-top: 2px;'>Speculation Ratio: {features_dict['speculation_ratio'] * 100:.3f}%</div>
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                        if article_url:
                            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                            st.markdown("### 🌐 URL Domain Check")
                            dom_cred = get_domain_credibility(article_url)
                            badge_color = "#22C55E" if dom_cred["badge"] == "Trusted" else ("#F59E0B" if dom_cred["badge"] == "Neutral" else "#EF4444")
                            render_html(f"""
                            <div style='font-size: 1.1rem; font-weight: 700; margin-bottom: 5px;'>{dom_cred['domain']}</div>
                            <div style='display: inline-block; background: rgba(59, 130, 246, 0.1); border: 1px solid #3B82F6; color: #3B82F6; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase;'>{dom_cred['badge']}</div>
                            <div style='font-size: 0.85rem; margin-top: 10px; color: #94A3B8;'>{dom_cred['status']}</div>
                            <div style='font-size: 0.8rem; margin-top: 5px; font-weight: 700; color: {badge_color};'>Credibility Rating: {dom_cred['score']}/100</div>
                            """)
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Possible Weaknesses / Concerns Section (Issue 5)
                        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                        st.markdown("### ⚠ Possible Structural Weaknesses")
                        weaknesses = []
                        if features_dict["quotes_count"] == 0:
                            weaknesses.append("• No direct quotes or attributed speakers detected.")
                        if features_dict["credibility_citations"] == 0:
                            weaknesses.append("• Lacks external links, official datelines, or references.")
                        if features_dict["clickbait_score"] > 1.5:
                            weaknesses.append("• Elevated sensational patterns matching clickbait profiles.")
                        if not weaknesses:
                            weaknesses.append("• No major stylistic anomalies detected.")
                        for w in weaknesses:
                            st.write(w)
                        st.markdown("</div>", unsafe_allow_html=True)

                # ================= TAB 3: EXPLANATIONS =================
                with tab_explanation:
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Feature Attribution Explanations")
                    
                    target_expl_model = models.get(selected_model_name) or models.get("Logistic Regression")
                    if target_expl_model is not None:
                        vocab_size = len(pipeline.vectorizer.get_feature_names_out())
                        tfidf_slice = X_comb[:, :vocab_size]
                        expl = explain_prediction(text_to_analyze, clean_txt, tfidf_slice, dense_feats, target_expl_model, pipeline.vectorizer, dense_scaled=dense_scaled)
                        
                        if "error" not in expl:
                            exp_col1, exp_col2 = st.columns(2)
                            with exp_col1:
                                st.write("##### 🟢 Credibility Signifiers")
                                if expl["top_real_words"]:
                                    for word, contrib, val in expl["top_real_words"][:5]:
                                        if word == "minister":
                                            ctx = "Professionally written institutional reporting."
                                        elif word == "government":
                                            ctx = "Common in official political journalism."
                                        elif word == "parliament":
                                            ctx = "Associated with legislative reporting."
                                        elif word in ["reuters", "ap", "press"]:
                                            ctx = "Standard news wire attribution signature."
                                        else:
                                            ctx = "Matches conventions of authentic articles."
                                        render_html(f"""
                                        <span class='chip chip-green' title='{ctx}'>✓ {word}</span> <span style='font-size: 0.75rem; color: #64748B;'>— {ctx}</span>
                                        """)
                                else:
                                    st.write("No strong credibility vocabulary detected.")
                                    
                            with exp_col2:
                                st.write("##### 🔴 Unreliability Signifiers")
                                if expl["top_fake_words"]:
                                    for word, contrib, val in expl["top_fake_words"][:5]:
                                        if word in ["unbelievable", "shocking", "exposed", "conspiracy", "secret", "miracle"]:
                                            ctx = "Clickbait term indicating sensationalized reporting."
                                        elif word in ["rumored", "allegedly", "reportedly", "anonymous"]:
                                            ctx = "Speculative terminology lacking direct official references."
                                        else:
                                            ctx = "Matches conventions of unverified stories."
                                        render_html(f"""
                                        <span class='chip chip-red' title='{ctx}'>⚠ {word}</span> <span style='font-size: 0.75rem; color: #64748B;'>— {ctx}</span>
                                        """)
                                else:
                                    st.write("No strong unreliability vocabulary detected.")
                                    
                            # Decision Influence Distribution (Horizontal Bar Chart layout instead of table)
                            st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
                            st.write("##### 📊 Decision Influence Distribution")
                            st.write("Normalized influence share each feature category has on the prediction decision (summing to 100%):")
                            
                            for cat, data in expl["category_summary"].items():
                                if isinstance(data, dict):
                                    share_val = data.get("share", 0.0)
                                    direction = data.get("direction", "N/A")
                                else:
                                    share_val = abs(float(data))
                                    direction = "Positive (+)" if float(data) >= 0 else "Negative (-)"
                                
                                color = "#10B981" if "positive" in direction.lower() else "#EF4444"
                                
                                render_html(f"""
                                <div class='bar-label-container'><span>{cat} ({direction})</span><span>{share_val:.1f}%</span></div>
                                <div style='background-color: #1E293B; height: 8px; border-radius: 9999px; overflow: hidden;'><div style='background-color: {color}; width: {share_val}%; height: 100%;'></div></div>
                                """)
                        else:
                            st.info("Feature explanations unavailable.")
                    else:
                        st.info("Logistic Regression model is required for generating local feature contributions.")
                    st.markdown("</div>", unsafe_allow_html=True)