import os
import json
import hashlib
import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from src.preprocessing import full_preprocess_pipeline, compute_dense_features

FAILED_PREDS_DIR = "failed_predictions"

def save_failed_prediction(title, text, url, predicted_label, true_label, confidence, top_features, failure_reason):
    """
    Saves a misclassified prediction details to the failed_predictions/ directory.
    """
    os.makedirs(FAILED_PREDS_DIR, exist_ok=True)
    
    # Generate unique hash for the article
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    timestamp = int(time.time())
    filename = f"{FAILED_PREDS_DIR}/failure_{timestamp}_{text_hash}.json"
    
    data = {
        "title": title,
        "text": text,
        "url": url,
        "predicted_label": int(predicted_label),
        "true_label": int(true_label),
        "confidence": float(confidence),
        "top_contributing_features": top_features,
        "failure_reason": failure_reason,
        "timestamp": timestamp
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[Hard Example Mining] Saved failed prediction to {filename}")

def retrain_from_failures():
    """
    Loads all failed predictions, appends them to the combined dataset,
    and runs the generalization training pipeline again.
    """
    if not os.path.exists(FAILED_PREDS_DIR):
        print("No failed predictions directory found.")
        return False
        
    failures = []
    for f_name in os.listdir(FAILED_PREDS_DIR):
        if f_name.endswith(".json"):
            path = os.path.join(FAILED_PREDS_DIR, f_name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    failures.append(json.load(f))
            except Exception as e:
                print(f"Error reading {f_name}: {e}")
                
    if not failures:
        print("No failure records found to mine.")
        return False
        
    print(f"[Hard Example Mining] Found {len(failures)} failure records. Appending to training data...")
    
    combined_path = "data/processed/combined_news.csv"
    if not os.path.exists(combined_path):
        print(f"Error: {combined_path} not found.")
        return False
        
    df = pd.read_csv(combined_path)
    
    # Convert failures to DataFrame
    failures_data = []
    for fail in failures:
        failures_data.append({
            "title": fail["title"],
            "text": fail["text"],
            "subject": "worldnews" if fail["true_label"] == 1 else "politics",
            "label": fail["true_label"]
        })
        
    df_fails = pd.DataFrame(failures_data)
    df_updated = pd.concat([df, df_fails], ignore_index=True)
    df_updated.to_csv(combined_path, index=False)
    print(f"[Hard Example Mining] Dataset expanded to {len(df_updated)} rows.")
    
    # Trigger train_generalization
    from src.scripts.train_generalization import train_generalization
    train_generalization()
    return True
