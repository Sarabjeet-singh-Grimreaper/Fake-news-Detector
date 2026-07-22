import os
import json
import pickle
import re
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from src.preprocessing import clean_text, tokenize_and_filter

# Define paths
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/logreg_model.pkl"
DRIFT_REVIEW_PATH = "data/drift_review.json"

# Lexicon for lightweight, dependency-free Sentiment Bias calculation
POSITIVE_WORDS = {
    'great', 'excellent', 'good', 'verify', 'truth', 'true', 'confirmed', 'positive', 
    'success', 'support', 'credible', 'reliable', 'validated', 'factual', 'correct'
}
NEGATIVE_WORDS = {
    'fake', 'claim', 'alleged', 'worst', 'terrible', 'bad', 'false', 'hoax', 'lie', 
    'disaster', 'negative', 'unverified', 'suspicious', 'misleading', 'conspiracy', 'rumor'
}

def compute_sentiment_bias(text):
    """
    Computes a lightweight, dependency-free sentiment bias score based on word matching.
    Returns a float in the range [-1.0, 1.0].
    """
    text_lower = text.lower()
    words = text_lower.split()
    if not words:
        return 0.0
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (pos_count - neg_count) / len(words)

def extract_hybrid_features(raw_text, vectorizer):
    """
    Extracts TF-IDF features and appends custom dense engineering metadata features:
    1. Clickbait Capitalization Ratio
    2. Punctuation Density (exclamation and question marks)
    3. Average Word Length
    4. Sentiment Bias
    
    Returns a combined scipy.sparse.csr_matrix.
    """
    # 1. Compute text TF-IDF vector (sparse representation)
    clean_str = clean_text(raw_text)
    filtered_str = tokenize_and_filter(clean_str)
    
    # TF-IDF transform requires a list/iterable
    tfidf_sparse = vectorizer.transform([filtered_str])
    
    # 2. Compute custom engineering metadata features (dense statistical)
    if not raw_text:
        cap_ratio = 0.0
        punct_density = 0.0
        avg_word_len = 0.0
        sentiment_bias = 0.0
    else:
        # Clickbait Capitalization Ratio
        letters_only = [c for c in raw_text if c.isalpha()]
        cap_ratio = (sum(1 for c in letters_only if c.isupper()) / len(letters_only)) if letters_only else 0.0
        
        # Punctuation Density (tracking ! and ?)
        excl_q_count = raw_text.count('!') + raw_text.count('?')
        punct_density = excl_q_count / len(raw_text)
        
        # Average Word Length
        words = raw_text.split()
        avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0
        
        # Sentiment Bias
        sentiment_bias = compute_sentiment_bias(raw_text)
        
    dense_features = np.array([[cap_ratio, punct_density, avg_word_len, sentiment_bias]], dtype=np.float64)
    dense_sparse = sp.csr_matrix(dense_features)
    
    # Combine sparse TF-IDF and dense features
    combined_vector = sp.hstack([tfidf_sparse, dense_sparse], format="csr")
    return combined_vector

def is_git_lfs_pointer(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "rb") as f:
            header = f.read(100)
            return b"version https://git-lfs" in header
    except Exception:
        return False

def load_or_init_model_and_vectorizer():
    """
    Loads the global TF-IDF vectorizer and online model.
    Adapts weights dynamically to fit the combined 5004-feature space.
    """
    # Load Vectorizer
    if not os.path.exists(VECTORIZER_PATH) or is_git_lfs_pointer(VECTORIZER_PATH):
        raise FileNotFoundError(f"Global vectorizer file not found or is a Git LFS placeholder at {VECTORIZER_PATH}. Initialize the pipeline first.")
    
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    
    # Load or initialize model
    model = None
    if os.path.exists(MODEL_PATH) and not is_git_lfs_pointer(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            
            # Ensure it is an SGDClassifier supporting partial_fit and log_loss
            if not hasattr(model, "partial_fit") or getattr(model, "loss", None) != "log_loss":
                print("[Warning] Loaded model does not support incremental learning or probability estimation.")
                print("[Warning] Upgrading/converting model to SGDClassifier with log_loss...")
                
                coef = getattr(model, "coef_", None)
                intercept = getattr(model, "intercept_", None)
                
                model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
                model.classes_ = np.array([0, 1])
                
                if coef is not None and intercept is not None:
                    model.coef_ = np.array(coef, dtype=np.float64)
                    model.intercept_ = np.array(intercept, dtype=np.float64)
        except Exception as e:
            print(f"[Warning] Failed to load model due to unpickling/version error: {e}")
            print("[Warning] Initializing a clean SGDClassifier model fallback...")
            model = None

    if model is None:
        print("[Engine] Initializing new SGDClassifier model baseline...")
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        model.classes_ = np.array([0, 1])

        
    # Dynamically expand coefficients if shape mismatch exists (5000 TF-IDF features -> 5004 hybrid features)
    expected_features = len(vectorizer.get_feature_names_out()) + 4
    if hasattr(model, "coef_") and model.coef_ is not None:
        current_features = model.coef_.shape[1]
        if current_features < expected_features:
            diff = expected_features - current_features
            print(f"[Engine] Shape alignment: expanding model coefficient matrix from {current_features} to {expected_features} features.")
            model.coef_ = np.hstack([model.coef_, np.zeros((model.coef_.shape[0], diff))])
    else:
        # Initialize coefficients block to zero if model is not yet fit
        model.coef_ = np.zeros((1, expected_features))
        model.intercept_ = np.zeros(1)
        
    return model, vectorizer

def log_drift_article(article_dict, confidence, prediction):
    """
    Appends anomalous, low-confidence articles (<65% confidence) to data/drift_review.json.
    """
    os.makedirs(os.path.dirname(DRIFT_REVIEW_PATH), exist_ok=True)
    
    # Read existing review records
    records = []
    if os.path.exists(DRIFT_REVIEW_PATH):
        try:
            with open(DRIFT_REVIEW_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []
            
    # Append new anomaly metadata
    record = {
        "text": article_dict.get("text", ""),
        "title": article_dict.get("title", "Untitled Stream Article"),
        "confidence": float(confidence),
        "predicted_label": int(prediction),
        "timestamp": float(np.datetime64('now').astype(int))
    }
    records.append(record)
    
    # Write back to disk
    with open(DRIFT_REVIEW_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def process_unlabeled_stream(new_articles_list, elite_threshold=0.96, review_threshold=0.65):
    """
    Confidence-gated self-training processor for real-time incoming article streams.
    Updates weights dynamically and tracks telemetry stats.
    """
    model, vectorizer = load_or_init_model_and_vectorizer()
    
    # Extract baseline parameter states for tracking drift rate
    orig_coef = np.copy(model.coef_)
    orig_intercept = np.copy(model.intercept_)
    
    # Telemetry metrics
    telemetry = {
        "total_processed": 0,
        "pseudo_labeled": 0,
        "drift_reviewed": 0,
        "ignored": 0,
        "parameter_drift_rate": 0.0,
        "top_feature_shifts": []
    }
    
    unique_classes = np.array([0, 1])
    
    for item in new_articles_list:
        telemetry["total_processed"] += 1
        
        # Support both strings and dictionaries in the input stream
        if isinstance(item, dict):
            raw_text = item.get("text", "")
            if not raw_text:
                raw_text = item.get("title", "")
        else:
            raw_text = str(item)
            item = {"text": raw_text, "title": "Streamed raw text"}
            
        if not raw_text.strip():
            telemetry["ignored"] += 1
            continue
            
        # 1. Extract combined hybrid sparse vector
        x_sparse = extract_hybrid_features(raw_text, vectorizer)
        
        # 2. Compute probabilities
        probs = model.predict_proba(x_sparse)[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        # 3. Confidence-Gated Decision Rules
        if confidence >= elite_threshold:
            # Pseudo-Labeling autonomous update rule
            model.partial_fit(x_sparse, np.array([prediction]), classes=unique_classes)
            telemetry["pseudo_labeled"] += 1
            
        elif confidence < review_threshold:
            # Anomalous Flagging Rule
            log_drift_article(item, confidence, prediction)
            telemetry["drift_reviewed"] += 1
            
        else:
            # Safe zone: bypass update, no flag
            telemetry["ignored"] += 1
            
    # Serialize the updated parameters back to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
        
    # 4. Telemetry tracking computations
    # Calculate parameter drift rate (Euclidean distance)
    coef_drift = np.linalg.norm(model.coef_ - orig_coef)
    intercept_drift = np.linalg.norm(model.intercept_ - orig_intercept)
    telemetry["parameter_drift_rate"] = float(coef_drift + intercept_drift)
    
    # Calculate dynamic feature weight shifts
    diff = model.coef_[0] - orig_coef[0]
    feature_names = list(vectorizer.get_feature_names_out()) + [
        "meta_capitalization_ratio", 
        "meta_punctuation_density", 
        "meta_avg_word_length", 
        "meta_sentiment_bias"
    ]
    
    # Sort features by absolute coefficient shift magnitude
    shifts = []
    for idx, (name, d_val) in enumerate(zip(feature_names, diff)):
        shifts.append({
            "feature": name,
            "index": idx,
            "shift": float(d_val),
            "new_weight": float(model.coef_[0][idx])
        })
    shifts = sorted(shifts, key=lambda x: abs(x["shift"]), reverse=True)
    telemetry["top_feature_shifts"] = shifts[:10]
    
    return telemetry

if __name__ == "__main__":
    # Test/Demo execution setup
    print("[Engine] Initializing demonstration stream loop run...")
    dummy_articles = [
        {
            "title": "Unbelievable Breakthrough!!!",
            "text": "BREAKING NEWS: You won't believe this amazing discovery! A secret cure has been found! Is it real???"
        },
        {
            "title": "Federal Reserve Rate Update",
            "text": "The Federal Reserve indicated on Tuesday that interest rates would remain stable pending next quarter's inflation reports."
        },
        {
            "title": "Conspiracy Confirmed!",
            "text": "They are hiding the truth! The local government is controlled by alien shape-shifters! Confirming details soon!"
        }
    ]
    
    try:
        metrics = process_unlabeled_stream(dummy_articles)
        print("\n=== Live Stream Loop Telemetry Results ===")
        print(f"Total Articles Digested: {metrics['total_processed']}")
        print(f"Pseudo-labeled (Auto-Fit): {metrics['pseudo_labeled']}")
        print(f"Drift Reviewed (Flagged): {metrics['drift_reviewed']}")
        print(f"Ignored: {metrics['ignored']}")
        print(f"Parameter Drift Rate: {metrics['parameter_drift_rate']:.6f}")
        print("\nTop 5 Dynamic Feature Weight Shifts:")
        for idx, shift_info in enumerate(metrics['top_feature_shifts'][:5]):
            print(f"  {idx+1}. Feature: {shift_info['feature']} | Shift: {shift_info['shift']:.6f} | Current Weight: {shift_info['new_weight']:.6f}")
    except Exception as e:
        print(f"[Error] Stream simulation encountered an error: {e}")
