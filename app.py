import sys
import os
import streamlit as st
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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Root styling and backgrounds */
    .stApp {
        background-color: #0B1220 !important;
        background-image: radial-gradient(circle at 80% 20%, #111E30 0%, #0B1220 70%) !important;
        color: #E2E8F0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Clean sidebar design */
    div[data-testid="stSidebar"] {
        background-color: #080D1A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Glassmorphism custom cards with reduced padding */
    .premium-card {
        background: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid #1E293B !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    .premium-card:hover {
        border-color: #3B82F6 !important;
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Modern buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 15px rgba(29, 78, 216, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
        border-color: #3B82F6 !important;
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
        background-color: rgba(34, 197, 94, 0.1);
        color: #4ADE80;
        border-color: rgba(34, 197, 94, 0.25);
    }
    .chip-green:hover {
        background-color: rgba(34, 197, 94, 0.2);
        border-color: #4ADE80;
        transform: scale(1.03);
    }
    .chip-red {
        background-color: rgba(239, 68, 68, 0.1);
        color: #F87171;
        border-color: rgba(239, 68, 68, 0.25);
    }
    .chip-red:hover {
        background-color: rgba(239, 68, 68, 0.2);
        border-color: #F87171;
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
        background: #111827;
        border: 1px solid #1E293B;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        margin: 0.25rem;
    }
    .pipeline-step-active {
        border-color: #3B82F6;
        color: #3B82F6;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.2);
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
        options=["🔍 Credibility Analyzer", "📊 Engine Telemetry & Admin"],
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
    
    st.info("⚡ System Health: OPTIMAL\n🤖 Active pipeline: V2.0\n🎯 Feature mapping: Classical Groups A-F")

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
    if password == "VerifIQ_Admin_2026":
        st.success("Access authorized.")
        
        # Stat grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pipeline Version", "2.0.0")
            st.metric("Dense Metrics", "12 Active Scales")
        with col2:
            st.metric("Dataset Split", "ISOT + Unseen V2.0")
            st.metric("Model Calibration", "Calibrated Sigmoid Soft Ensemble")
        with col3:
            st.metric("Total Vocabulary Size", "4000 fields")
            st.metric("Inference Avg Latency", "12 ms")
            
        st.markdown("### 📈 Model Consensus Benchmarks")
        metrics_df = pd.DataFrame({
            "Classifier Model": ["Logistic Regression", "Random Forest", "SVM", "Voting Ensemble"],
            "Generalization Accuracy": ["98.25%", "96.48%", "98.66%", "98.70%"],
            "Out-of-Domain Accuracy": ["72.73%", "63.64%", "72.73%", "72.73%"],
            "Calibration Score (Brier)": ["0.015", "0.024", "0.014", "0.012"]
        })
        st.table(metrics_df)
    elif password:
        st.error("Invalid authorization key.")

else:
    # Public Credibility Analyzer
    render_html("""
    <div style='padding: 1rem 0;'>
        <h1 class='hero-title'>ENTERPRISE CREDIBILITY ANALYZER</h1>
        <p style='color: #94A3B8; font-size: 1.05rem;'>Audit structural writing styles, speculation rates, and attributions to evaluate reliability metrics.</p>
    </div>
    """)
    
    tab_input1, tab_input2 = st.tabs(["📝 Content Stream", "🌐 Live URL Fetcher"])
    
    with tab_input1:
        user_title = st.text_input("Article Headline (Optional)", placeholder="Paste headline here...")
        user_text = st.text_area("Article Body Text", height=120, placeholder="Paste full article text content here...")
        
    with tab_input2:
        article_url = st.text_input("Resource URL Address", placeholder="https://news-outlet.com/article-slug")
        
    btn_analyze = st.button("Evaluate Credibility scorecard", use_container_width=True)
    
    if btn_analyze:
        text_to_analyze = ""
        title_to_analyze = ""
        
        if article_url:
            with st.spinner("Fetching resource content..."):
                scraped = scrape_article(article_url)
                if scraped and "error" not in scraped:
                    text_to_analyze = scraped.get("text", "")
                    title_to_analyze = scraped.get("title", "")
                    st.success(f"Successfully scraped: **{title_to_analyze}**")
                else:
                    st.error("Crawl agent failed to fetch resource. Please input content manually.")
        else:
            text_to_analyze = user_text
            title_to_analyze = user_title
            
        if not text_to_analyze.strip():
            st.warning("Please supply news body text or a URL before invoking audit.")
        elif pipeline is None or models.get(selected_model_name) is None:
            st.error("System assets not fully loaded. Ensure models are trained using --train flag.")
        else:
            with st.spinner("Auditing features and mapping vectors..."):
                t_start = time.time()
                
                # Transform using the pipeline
                X_comb, clean_text_list, dense_list = pipeline.transform([text_to_analyze], [title_to_analyze])
                clean_txt = clean_text_list[0]
                dense_feats = dense_list[0]
                
                # Predict
                active_model = models[selected_model_name]
                pred = active_model.predict(X_comb)[0]
                probs = active_model.predict_proba(X_comb)[0]
                
                raw_confidence = probs[pred] * 100.0
                t_elapsed = (time.time() - t_start) * 1000.0
                
                # Risk score calculations
                risk_score = (1.0 - probs[1]) * 100.0
                if risk_score > 70:
                    risk_label = "HIGH RISK"
                    risk_color = "#EF4444"
                elif risk_score > 30:
                    risk_label = "MODERATE RISK"
                    risk_color = "#F59E0B"
                else:
                    risk_label = "LOW RISK"
                    risk_color = "#22C55E"
                
                # Confidence Tiers
                if raw_confidence < 55.0:
                    verdict = "Uncertain"
                    confidence_tier = "Inconclusive"
                    verdict_color = "#94A3B8"
                else:
                    verdict = "Likely Real" if pred == 1 else "Likely Fake"
                    verdict_color = "#22C55E" if pred == 1 else "#EF4444"
                    if raw_confidence >= 90.0:
                        confidence_tier = "High confidence"
                    elif raw_confidence >= 75.0:
                        confidence_tier = "Moderate confidence"
                    else:
                        confidence_tier = "Low confidence"
                
                # Extract dense features dictionary
                features_dict, _ = extract_dense_features(text_to_analyze, clean_txt, title_to_analyze)
                
                # Dynamic calculations for credibility groups
                ling_score = probs[1] * 100.0
                
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
                
                # ----------------- TABS UX INTEGRATION -----------------
                tab_overview, tab_indicators, tab_explanation = st.tabs([
                    "🛡️ Credibility Overview", 
                    "📊 Stylistic Indicators", 
                    "🔍 Feature Attributions"
                ])
                
                # ================= TAB 1: OVERVIEW =================
                with tab_overview:
                    # Giant Hero Card (Using render_html to strip all indentation and fix HTML text bug)
                    render_html(f"""
                    <div style='text-align: center; padding: 3rem 1.5rem; border-radius: 16px; background-color: #111827; border: 1px solid #1E293B; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4);'>
                        <div style='color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.12em; margin-bottom: 0.75rem;'>🛡️ NEWS CREDIBILITY ANALYSIS</div>
                        <h1 style='color: {verdict_color}; margin: 0; font-size: 3.4rem; font-family: Outfit; font-weight: 900; letter-spacing: -0.02em;'>{verdict.upper()}</h1>
                        <h2 style='color: #FFFFFF; margin: 10px 0 0 0; font-size: 2.2rem; font-weight: 800;'>{raw_confidence:.0f}%</h2>
                        <p style='color: #94A3B8; font-size: 0.95rem; margin: 5px 0 20px 0; font-weight: 600; text-transform: uppercase;'>{confidence_tier} • {risk_label}</p>
                        
                        <!-- Gauge Meter -->
                        <div style='max-width: 450px; margin: 0 auto;'>
                            <div style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.25rem;'>
                                <span>SECURITY RISK ASSESSMENT</span>
                                <span style='color: {risk_color};'>{risk_label}</span>
                            </div>
                            <div style='background-color: #1E293B; height: 12px; border-radius: 9999px; overflow: hidden;'>
                                <div style='background-color: {risk_color}; width: {risk_score:.0f}%; height: 100%;'></div>
                            </div>
                            <div style='text-align: right; font-size: 0.7rem; color: #64748B; margin-top: 0.25rem;'>Risk Score: {risk_score:.0f} / 100</div>
                        </div>
                        
                        <p style='margin: 20px auto 0 auto; color: #94A3B8; max-width: 750px; font-size: 0.85rem; line-height: 1.5; border-top: 1px solid #1E293B; padding-top: 10px;'>
                            <strong>Operational Context:</strong> This classifier estimates linguistic behavior resembling trained datasets. Factual verification requires checking independent journalistic sources.
                        </p>
                    </div>
                    """)
                    
                    # 4 KPI Cards with Emojis
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    with kpi_col1:
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>🧠 Credibility</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {verdict_color}; margin-top: 5px; font-family: Outfit;'>{ling_score:.1f}%</div>
                        </div>
                        """)
                    with kpi_col2:
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>⚠ Risk</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {risk_color}; margin-top: 5px; font-family: Outfit;'>{risk_label.split()[0]}</div>
                        </div>
                        """)
                    with kpi_col3:
                        writing_color = "#22C55E" if writing_qual > 60 else ("#F59E0B" if writing_qual > 40 else "#EF4444")
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>✍ Writing</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {writing_color}; margin-top: 5px; font-family: Outfit;'>{writing_qual:.0f}%</div>
                        </div>
                        """)
                    with kpi_col4:
                        att_score = min(100.0, (features_dict['quotes_count']*10 + features_dict['credibility_citations']*15))
                        att_color = "#22C55E" if att_score > 50 else ("#F59E0B" if att_score > 20 else "#EF4444")
                        render_html(f"""
                        <div style='background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; text-align: center;'>
                            <div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;'>🏛 Sources</div>
                            <div style='font-size: 1.8rem; font-weight: 800; color: {att_color}; margin-top: 5px; font-family: Outfit;'>{att_score:.0f}%</div>
                        </div>
                        """)

                    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
                    
                    # Connected Pipeline Lifecycle Diagram
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    st.markdown("### ⚙️ Pipeline Process Timeline")
                    steps = ["Ingestion", "Cleaning", "Features", "Prediction", "Explanation"]
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
                    if verdict == "Likely Real":
                        st.write("The article strongly resembles conventions of professional reporting. No structural indicators of fabrication were detected.")
                    elif verdict == "Likely Fake":
                        st.write("Stylistic anomalies linked to fabricated reporting were detected. Independent manual verification is recommended.")
                    else:
                        st.write("The stylistic metrics are inconclusive. Cross-referencing claims via authoritative news wire registries is advised.")
                    st.write("*This assessment evaluates linguistic credibility, rather than factual truth. Independent verification is recommended for critical claims.*")
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
                    
                    lr_model = models.get("Logistic Regression")
                    if lr_model is not None:
                        expl = explain_prediction(text_to_analyze, clean_txt, pipeline.vectorizer.transform([clean_txt]), dense_feats, lr_model, pipeline.vectorizer)
                        
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