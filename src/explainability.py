import numpy as np
import scipy.sparse as sp

DENSE_FEATURE_NAMES = [
    "Average Sentence Length", "Lexical Diversity", "Entropy", "Flesch Reading Ease", # Group B
    "Quotes Count", "Named Entities Est", "Credibility Citations",                   # Group C
    "Speculation Ratio",                                                            # Group D
    "Clickbait Score",                                                              # Group E
    "Polarity", "Subjectivity", "Emotional Intensity"                               # Group F
]

def explain_prediction(text_raw: str, clean_str: str, vectorized_tfidf, dense_feats_list, model, vectorizer) -> dict:
    """
    Computes local feature contribution explanations (normalized decision influence shares)
    for a given news article prediction based on linear model coefficients.
    """
    # Pack dense features
    X_dense = np.array([dense_feats_list], dtype=np.float64)
    
    # 2. Extract coefficients (supports LogisticRegression or SGDClassifier)
    if hasattr(model, "estimator"):
        base_model = None
        if hasattr(model.estimator, "estimators_"):
            for name, est in model.estimator.estimators_:
                if hasattr(est, "coef_"):
                    base_model = est
                    break
        elif hasattr(model, "calibrated_classifiers_"):
            for cal in model.calibrated_classifiers_:
                if hasattr(cal.base_estimator, "coef_"):
                    base_model = cal.base_estimator
                    break
        
        if base_model is not None:
            coefficients = base_model.coef_[0]
        else:
            return {"error": "Model does not support coefficient extraction for explainability."}
    elif hasattr(model, "coef_") and model.coef_ is not None:
        coefficients = model.coef_[0]
    else:
        return {"error": "Model does not support coefficient extraction for explainability."}
        
    # 3. Calculate word-level contributions
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    row = vectorized_tfidf.tocoo()
    word_contributions = []
    for col_idx, value in zip(row.col, row.data):
        if col_idx < len(feature_names):
            feat_name = feature_names[col_idx]
            coef = coefficients[col_idx]
            contrib = value * coef
            word_contributions.append((feat_name, float(contrib), float(value)))
        
    word_contributions = sorted(word_contributions, key=lambda x: x[1], reverse=True)
    
    top_real_words = [item for item in word_contributions if item[1] > 0][:8]
    top_fake_words = sorted([item for item in word_contributions if item[1] < 0], key=lambda x: x[1])[:8]
    
    # 4. Calculate dense metadata / stylometrics contributions
    dense_contributions = []
    dense_offset = len(feature_names)
    
    for idx, name in enumerate(DENSE_FEATURE_NAMES):
        val = dense_feats_list[idx]
        coef_idx = dense_offset + idx
        coef = coefficients[coef_idx] if coef_idx < len(coefficients) else 0.0
        contrib = val * coef
        dense_contributions.append({
            "feature": name,
            "value": float(val),
            "contribution": float(contrib)
        })
        
    dense_contributions = sorted(dense_contributions, key=lambda x: x["contribution"], reverse=True)
    
    # Summarize raw contributions
    style_contrib = sum(c["contribution"] for c in dense_contributions if c["feature"] in ["Average Sentence Length", "Lexical Diversity", "Entropy", "Flesch Reading Ease"])
    credibility_contrib = sum(c["contribution"] for c in dense_contributions if c["feature"] in ["Quotes Count", "Named Entities Est", "Credibility Citations"])
    speculation_contrib = sum(c["contribution"] for c in dense_contributions if c["feature"] == "Speculation Ratio")
    clickbait_contrib = sum(c["contribution"] for c in dense_contributions if c["feature"] == "Clickbait Score")
    emotion_contrib = sum(c["contribution"] for c in dense_contributions if c["feature"] in ["Polarity", "Subjectivity", "Emotional Intensity"])
    
    # Word-level total raw contributions
    word_contrib_sum = sum(item[1] for item in word_contributions)
    
    # Absolute values for normalization
    raw_contribs = {
        "Writing Style (Group B)": style_contrib,
        "Source Credibility (Group C)": credibility_contrib,
        "Speculation (Group D)": speculation_contrib,
        "Clickbait (Group E)": clickbait_contrib,
        "Emotion (Group F)": emotion_contrib,
        "Linguistic Vocabulary (Group A)": word_contrib_sum
    }
    
    total_abs = sum(abs(v) for v in raw_contribs.values())
    if total_abs == 0:
        total_abs = 1e-5
        
    # Calculate share percentages summing to 100%
    normalized_shares = {}
    for category, val in raw_contribs.items():
        normalized_shares[category] = {
            "share": (abs(val) / total_abs) * 100.0,
            "direction": "Positive (+)" if val >= 0 else "Negative (-)"
        }

    return {
        "top_real_words": top_real_words,
        "top_fake_words": top_fake_words,
        "dense_contributions": dense_contributions,
        "category_summary": normalized_shares
    }
