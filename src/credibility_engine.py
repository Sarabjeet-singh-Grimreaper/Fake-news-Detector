import re
import time
from typing import Dict, Any, Optional
from src.preprocessing import full_preprocess_pipeline
from src.features import extract_dense_features
from src.domain_trust import get_domain_credibility
from src.realtime_verification import verify_realtime_news

SATIRE_KEYWORDS = [
    r'\bofficially conceded\b', r'\bconceded today\b', r'\btoo difficult to keep\b',
    r'\bfalling off the edge\b', r'\bscientists confirm for the first time that everything\b',
    r'\bstudy finds 100%\b', r'\bannounced plans to cancel\b', r'\bbaffled experts\b',
    r'\bconfirmed that nothing\b', r'\bsatire\b', r'\bparody\b', r'\bhumor\b'
]

def check_satirical_patterns(text: str) -> float:
    """
    Evaluates whether text contains classic rhetorical parody/satire tropes.
    Returns a satire likelihood score between 0.0 and 1.0.
    """
    text_lower = text.lower()
    matches = 0
    for pat in SATIRE_KEYWORDS:
        if re.search(pat, text_lower):
            matches += 1
            
    if matches >= 2:
        return 0.95
    elif matches == 1:
        return 0.70
    return 0.0

def evaluate_comprehensive_credibility(
    text: str,
    title: str = "",
    url: str = "",
    pipeline = None,
    model = None,
    check_realtime: bool = True
) -> Dict[str, Any]:
    """
    Fuses multi-signal intelligence:
    1. Machine Learning Linguistic & Stylometric Consensus
    2. Real-Time Live News Wire Corroboration (Google News RSS & Fact-Check alerts)
    3. Source Domain Reputation (150+ news agencies, fact checkers, satire, hoax sites)
    4. Stylometric and Clickbait Heuristics
    5. Satire and Debunk Guards
    """
    clean_txt = full_preprocess_pipeline(text)
    features_dict, dense_feats_list = extract_dense_features(text, clean_txt, title)
    
    # 1. Machine Learning Model Signal
    ml_score = 50.0
    ml_pred = 1
    ml_confidence = 50.0
    if pipeline is not None and model is not None and pipeline.is_fitted:
        try:
            X_comb, _, _ = pipeline.transform([text], [title])
            probs = model.predict_proba(X_comb)[0]
            ml_pred = int(model.predict(X_comb)[0])
            # Probability of being Real (class 1)
            ml_score = float(probs[1]) * 100.0
            ml_confidence = float(probs[ml_pred]) * 100.0
        except Exception:
            pass
            
    # 2. Source Domain Trust Signal
    dom_result = get_domain_credibility(url) if url else {"domain": "N/A", "score": 50, "status": "No URL provided", "badge": "Neutral"}
    domain_score = float(dom_result["score"])
    
    # 3. Real-Time Live News Wire Corroboration Signal
    if check_realtime:
        realtime_result = verify_realtime_news(headline=title, body_text=text, domain=dom_result.get("domain", ""))
    else:
        realtime_result = {
            "query": "",
            "corroboration_score": 50,
            "corroboration_status": "Real-time check skipped",
            "badge": "Neutral",
            "articles_found": 0,
            "trusted_matches": 0,
            "fact_check_alert": False,
            "fact_check_details": "",
            "matching_articles": []
        }
    corroboration_score = float(realtime_result.get("corroboration_score", 50.0))
    
    # 4. Stylometric & Linguistic Integrity Signal
    flesch = features_dict["flesch_reading_ease"]
    lex_div = min(100.0, features_dict["lexical_diversity"] * 120.0)
    spec_ratio = features_dict["speculation_ratio"]
    clickbait = features_dict["clickbait_score"]
    quotes_citations = min(100.0, features_dict["quotes_count"] * 15 + features_dict["credibility_citations"] * 20)
    
    # Stylometric score (higher = resembles credible journalism)
    stylistic_score = (flesch * 0.3 + lex_div * 0.3 + quotes_citations * 0.4)
    if clickbait > 1.5:
        stylistic_score = max(0.0, stylistic_score - 25.0)
    if spec_ratio > 0.05:
        stylistic_score = max(0.0, stylistic_score - 20.0)
    stylistic_score = max(5.0, min(95.0, stylistic_score))
    
    # 5. Satire and Parody Detection
    satire_prob = check_satirical_patterns(text) or (check_satirical_patterns(title) if title else 0.0)
    is_satire_domain = dom_result.get("badge") == "Satire / Parody"
    
    # --- FUSION ENGINE DECISION LOGIC ---
    # Case A: Explicit Fact-Check Debunk Alert
    if realtime_result.get("fact_check_alert"):
        composite_score = 12.0
        verdict = "Debunked / False Claim"
        verdict_color = "#EF4444"
        risk_label = "CRITICAL RISK"
        risk_color = "#EF4444"
        confidence_tier = "High Confidence"
        reason = f"Explicit fact-check debunk detected on live news wires ({realtime_result.get('fact_check_details')})."
        
    # Case B: Satire or Parody
    elif is_satire_domain or (satire_prob >= 0.7 and corroboration_score <= 60):
        composite_score = 15.0
        verdict = "Satire / Parody"
        verdict_color = "#A855F7"  # Purple for satire
        risk_label = "HUMOR / SATIRE"
        risk_color = "#A855F7"
        confidence_tier = "High Confidence"
        reason = "Content exhibits recognized satirical phrasing or originates from a known humor/satire publisher."
        
    # Case C: High Clickbait Sensationalism
    elif clickbait >= 1.8:
        composite_score = 18.0
        verdict = "Clickbait / Sensationalism"
        verdict_color = "#EF4444"
        risk_label = "HIGH RISK"
        risk_color = "#EF4444"
        confidence_tier = "High Confidence"
        reason = "Severe sensational clickbait patterns detected (exaggerated emotional headlines, urgency lures, excessive punctuation)."

    # Case D: Known Low-Trust / Disinformation Domain
    elif dom_result.get("badge") == "Low Trust":
        composite_score = min(25.0, (ml_score * 0.3 + domain_score * 0.7))
        verdict = "Likely Fake / Disinformation"
        verdict_color = "#EF4444"
        risk_label = "HIGH RISK"
        risk_color = "#EF4444"
        confidence_tier = "High Confidence"
        reason = f"Publisher domain ({dom_result.get('domain')}) has a documented record of unverified or fabricated claims."

    # Case E: High-Confidence ML Fake with Zero Corroborating Wires
    elif ml_score <= 40.0 and realtime_result.get("trusted_matches", 0) == 0 and realtime_result.get("articles_found", 0) <= 1:
        composite_score = min(32.0, ml_score * 0.6 + corroboration_score * 0.4)
        verdict = "Likely Fake"
        verdict_color = "#EF4444"
        risk_label = "HIGH RISK"
        risk_color = "#EF4444"
        confidence_tier = "High Confidence" if ml_score <= 30 else "Moderate Confidence"
        reason = "Stylistic markers strongly resemble fabricated reports, and zero corroborating reports were found on global news wires."

    # Case F: Strong Live Corroboration by Verified News Wires (e.g. Reuters, AP, BBC)
    elif realtime_result.get("trusted_matches", 0) >= 2 or (realtime_result.get("trusted_matches", 0) >= 1 and dom_result.get("badge") == "Trusted"):
        composite_score = max(88.0, 0.40 * corroboration_score + 0.30 * domain_score + 0.20 * ml_score + 0.10 * stylistic_score)
        verdict = "Verified Real News"
        verdict_color = "#22C55E"
        risk_label = "LOW RISK"
        risk_color = "#22C55E"
        confidence_tier = "High Confidence"
        reason = f"Verified in real-time across {realtime_result.get('trusted_matches')} recognized global news wires."
        
    # Case G: Moderate Live Corroboration or Single Tier-1 Wire
    elif realtime_result.get("trusted_matches", 0) == 1:
        composite_score = max(76.0, 0.45 * corroboration_score + 0.30 * ml_score + 0.15 * domain_score + 0.10 * stylistic_score)
        verdict = "Likely Real News"
        verdict_color = "#22C55E"
        risk_label = "LOW RISK"
        risk_color = "#22C55E"
        confidence_tier = "Moderate Confidence"
        reason = "Corroborated by active global news wire reports matching this topic."
        
    # Case H: General Multi-Source Weighted Fusion
    else:
        has_domain = bool(url and dom_result.get("badge") != "Neutral")
        has_wires = bool(realtime_result.get("articles_found", 0) > 0)
        
        if has_domain and has_wires:
            w_ml, w_realtime, w_domain, w_style = 0.35, 0.30, 0.25, 0.10
            composite_score = (
                w_ml * ml_score +
                w_realtime * corroboration_score +
                w_domain * domain_score +
                w_style * stylistic_score
            )
        elif has_domain:
            w_ml, w_domain, w_style = 0.60, 0.30, 0.10
            composite_score = (
                w_ml * ml_score +
                w_domain * domain_score +
                w_style * stylistic_score
            )
        elif has_wires:
            w_ml, w_realtime, w_style = 0.55, 0.35, 0.10
            composite_score = (
                w_ml * ml_score +
                w_realtime * corroboration_score +
                w_style * stylistic_score
            )
        else:
            # Standalone text / claim without external wire matches or known URL domain
            # Directly powered by the ML classifier consensus & linguistic integrity
            w_ml, w_style = 0.85, 0.15
            composite_score = w_ml * ml_score + w_style * stylistic_score
            
        # Penalties for severe clickbait or high speculation
        if clickbait > 1.4:
            composite_score -= 15.0
        if spec_ratio > 0.08:
            composite_score -= 12.0
            
        composite_score = max(0.0, min(100.0, composite_score))
        
        # DECISIVE THRESHOLDS: Deliver clear Real vs Fake decisions
        if composite_score >= 52.0:
            if composite_score >= 78.0:
                verdict = "Verified Real News"
                confidence_tier = "High Confidence"
            else:
                verdict = "Likely Real"
                confidence_tier = "Moderate Confidence"
            verdict_color = "#22C55E"
            risk_label = "LOW RISK"
            risk_color = "#22C55E"
            reason = "Linguistic markers, writing structure, and model consensus indicate credible reporting."
        elif composite_score <= 48.0:
            if composite_score <= 28.0:
                verdict = "High-Risk Disinformation"
                confidence_tier = "High Confidence"
            else:
                verdict = "Likely Fake"
                confidence_tier = "Moderate Confidence"
            verdict_color = "#EF4444"
            risk_label = "HIGH RISK"
            risk_color = "#EF4444"
            reason = "Stylistic irregularities, sensational markers, or model consensus indicate fabricated content."
        else:
            verdict = "Uncertain / Unverified"
            verdict_color = "#F59E0B"
            risk_label = "MODERATE RISK"
            risk_color = "#F59E0B"
            confidence_tier = "Inconclusive"
            reason = "Borderline classification: article exhibits mixed linguistic signals."
            

            
    risk_score = 100.0 - composite_score
    
    return {
        "composite_score": round(composite_score, 1),
        "verdict": verdict,
        "verdict_color": verdict_color,
        "risk_score": round(risk_score, 1),
        "risk_label": risk_label,
        "risk_color": risk_color,
        "confidence_tier": confidence_tier,
        "reason": reason,
        "clean_text": clean_txt,
        "features_dict": features_dict,
        "dense_feats_list": dense_feats_list,
        "signals": {
            "ml_score": round(ml_score, 1),
            "ml_prediction": "Real" if ml_pred == 1 else "Fake",
            "ml_confidence": round(ml_confidence, 1),
            "realtime_score": round(corroboration_score, 1),
            "realtime_status": realtime_result.get("corroboration_status", ""),
            "domain_score": round(domain_score, 1),
            "domain_status": dom_result.get("status", ""),
            "domain_badge": dom_result.get("badge", "Neutral"),
            "stylistic_score": round(stylistic_score, 1)
        },
        "realtime_details": realtime_result,
        "domain_details": dom_result
    }
