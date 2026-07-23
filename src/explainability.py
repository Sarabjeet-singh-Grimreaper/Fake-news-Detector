import numpy as np
import scipy.sparse as sp

DENSE_FEATURE_NAMES = [
    "Lexical Diversity", "Type-Token Ratio", "Hapax Legomena Ratio", "Avg Sentence Length", "Avg Word Length",
    "Punctuation Density", "Digit Ratio", "Uppercase Ratio", "Stopword Ratio",
    "Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog Index", "Coleman-Liau Index", "SMOG Index",
    "Sentiment Polarity", "Subjectivity Score", "Emotion Intensity",
    "Excessive Punctuation Indicator", "Clickbait Word Ratio", "All Caps Ratio", "Sensational Phrase Ratio", "Urgency Indicator Ratio",
    "Title Chars Count", "Title Words Count", "Article Chars Count",
    "Paragraph Count", "Quotation Count", "External Links Count", "Word Count", "Reading Time"
]

def explain_prediction(text_raw: str, clean_str: str, vectorized_input, dense_feats_list, model, vectorizer, selector) -> dict:
    """
    Computes local feature contribution explanations (linear SHAP-like approximations)
    for a given news article prediction.
    """
    # 1. Align/Transform input features
    # Transform TF-IDF
    X_tfidf_selected = selector.transform(vectorized_input)
    
    # Pack dense features
    X_dense = np.array([dense_feats_list], dtype=np.float64)
    
    # 2. Extract coefficients (supports LogisticRegression or SGDClassifier)
    if not hasattr(model, "coef_") or model.coef_ is None:
        return {"error": "Model does not support coefficient extraction for explainability."}
        
    coefficients = model.coef_[0]
    
    # 3. Calculate word-level contributions
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Selected feature indices mapping
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = feature_names[selected_indices]
    
    # Multiply local TF-IDF value by the coefficient
    row = X_tfidf_selected.tocoo()
    word_contributions = []
    for col_idx, value in zip(row.col, row.data):
        feat_name = selected_feature_names[col_idx]
        coef = coefficients[col_idx]
        contrib = value * coef
        word_contributions.append((feat_name, float(contrib), float(value)))
        
    # Sort contributions
    # Positive pushes to Real (1), Negative pushes to Fake (0)
    word_contributions = sorted(word_contributions, key=lambda x: x[1], reverse=True)
    
    top_real_words = [item for item in word_contributions if item[1] > 0][:8]
    top_fake_words = sorted([item for item in word_contributions if item[1] < 0], key=lambda x: x[1])[:8]
    
    # 4. Calculate dense metadata / stylometrics contributions
    dense_contributions = []
    dense_offset = len(selected_indices)
    
    for idx, name in enumerate(DENSE_FEATURE_NAMES):
        val = dense_feats_list[idx]
        coef = coefficients[dense_offset + idx]
        contrib = val * coef
        dense_contributions.append({
            "feature": name,
            "value": float(val),
            "contribution": float(contrib)
        })
        
    dense_contributions = sorted(dense_contributions, key=lambda x: x["contribution"], reverse=True)
    
    # Summarize contributions by categories
    # Readability (indices 9 to 13)
    readability_contrib = sum(dense_contributions[i]["contribution"] for i, name in enumerate(DENSE_FEATURE_NAMES) if "index" in name.lower() or "ease" in name.lower() or "grade" in name.lower() or "fog" in name.lower() or "liau" in name.lower() or "smog" in name.lower())
    # Sentiment / Clickbait (indices 14 to 21)
    emotion_clickbait_contrib = sum(dense_contributions[idx]["contribution"] for idx, name in enumerate(DENSE_FEATURE_NAMES) if any(kw in name.lower() for kw in ["sentiment", "polarity", "subjectivity", "clickbait", "urgency", "sensational", "emotion"]))
    # Stylometrics (indices 0 to 8)
    style_contrib = sum(dense_contributions[idx]["contribution"] for idx, name in enumerate(DENSE_FEATURE_NAMES) if any(kw in name.lower() for kw in ["diversity", "ratio", "length", "density"]))
    # Metadata (indices 22 to 29)
    metadata_contrib = sum(dense_contributions[idx]["contribution"] for idx, name in enumerate(DENSE_FEATURE_NAMES) if any(kw in name.lower() for kw in ["chars", "words", "count", "links", "time"]))

    return {
        "top_real_words": top_real_words,
        "top_fake_words": top_fake_words,
        "dense_contributions": dense_contributions,
        "category_summary": {
            "Readability": float(readability_contrib),
            "Style & Lexicon": float(style_contrib),
            "Emotion & Sensationalism": float(emotion_clickbait_contrib),
            "Article Metadata": float(metadata_contrib)
        }
    }
