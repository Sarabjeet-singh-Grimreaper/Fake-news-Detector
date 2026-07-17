import os
import json
import pickle
import numpy as np
import scipy.sparse as sp
from src.preprocessing import full_preprocess_pipeline
from src.scraper import scrape_article
from src.scripts.autonomous_engine import load_or_init_model_and_vectorizer, extract_hybrid_features

def run_strict_news_classification_test():
    print("="*60)
    print("SYSTEM DIAGNOSTICS: STRICT FAKE VS REAL VALIDATION")
    print("="*60)
    
    # 1. Load trained ensemble models
    models = {}
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        print("[Error] TF-IDF Vectorizer is missing. Please train the baseline models first.")
        return
        
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    model_paths = {
        "Online Logic Engine (SGD)": "models/logreg_model.pkl",
        "Random Forest Automata": "models/random_forest_model.pkl",
        "Neural Network Array (MLP)": "models/neuralnet_model.pkl",
        "K-Nearest Neighbors (KNN)": "models/knn_model.pkl"
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load {name}: {e}")
        else:
            print(f"[Warning] Model file {path} not found.")
            
    if not models:
        print("[Error] No trained models loaded. Cannot run validation.")
        return

    # 2. Formulate highly realistic fake news payload
    fake_news_payload = {
        "title": "SHOCKING SECRET CONSPIRACY EXPOSED!!!",
        "text": "BREAKING: You won't believe this leaked secret! Subterranean clones have infiltrated the central banking systems and are controlling global energy levers to enforce a new viral order. Share this urgent news before it is taken down!"
    }
    
    # 3. Formulate highly realistic real news payload (structured, objective journalism style)
    real_news_payload = {
        "title": "Federal Reserve Holds Benchmark Interest Rates Steady",
        "text": "The Federal Reserve announced on Tuesday that it would maintain the benchmark interest rate within the current range. Chairman Powell indicated that economic indicators register steady growth and inflation metrics remain aligned with long-term stabilization objectives."
    }

    # 4. Crawl targets for live URL testing
    test_urls = {
        "Real URL Target (Wikipedia Fake News)": "https://en.wikipedia.org/wiki/Fake_news",
        "Malformed Target (Non-existent page)": "https://en.wikipedia.org/wiki/non-existent-page-12345"
    }

    def evaluate_payload(payload_dict, label):
        print(f"\nEvaluating payload: [{label}] -> '{payload_dict['title']}'")
        raw_text = payload_dict['text']
        
        # Extract features
        x_combined = extract_hybrid_features(raw_text, vectorizer)
        
        predictions = []
        real_probs = []
        
        for name, model in models.items():
            if not hasattr(model, "predict_proba"):
                continue
                
            pred = model.predict(x_combined)[0]
            probs = model.predict_proba(x_combined)[0]
            lbl = "REAL" if pred == 1 else "FAKE"
            real_probs.append(probs[1])
            predictions.append(f"  - {name}: {lbl} ({probs[pred]*100:.2f}% Certainty)")
            
        consensus = np.mean(real_probs) * 100
        verdict = "VERIFIED AUTHENTIC CONTEXT" if consensus >= 50.0 else "HIGH RISK MISINFORMATION"
        
        for p in predictions:
            print(p)
        print(f" consensus score (Likelihood of being Real): {consensus:.2f}%")
        print(f" final system verdict: {verdict}")
        return consensus

    # Run Text Evaluators
    fake_score = evaluate_payload(fake_news_payload, "Realistic Fake News Payload")
    real_score = evaluate_payload(real_news_payload, "Realistic Real News Payload")

    # Verify separation
    print("\n" + "-"*50)
    print("SEPARATION ANALYSIS:")
    print(f"Fake Payload Real-Score: {fake_score:.2f}%")
    print(f"Real Payload Real-Score: {real_score:.2f}%")
    if real_score > fake_score:
        print("[SUCCESS] The ensemble successfully distinguishes fake and real news payloads.")
    else:
        print("[FAIL] Model separation is insufficient.")
    print("-"*50)

    # 5. Run live URL parsing and verification checks
    print("\nRUNNING LIVE URL PARSING & VERIFICATION TESTS:")
    for label, url in test_urls.items():
        print(f"\nCrawling URL: {label} -> {url}")
        res = scrape_article(url)
        
        if "error" in res:
            print(f"  [ERROR] Web Crawler status: Failed ({res['error']}) - Handled Gracefully")
        else:
            print(f"  [SUCCESS] Web Crawler status: Succeeded!")
            print(f"  Title Extracted: '{res['title']}'")
            print(f"  Text Excerpt: {res['text'][:150]}...")
            
            # Predict on crawled text
            x_combined = extract_hybrid_features(res['text'], vectorizer)
            real_probs = [model.predict_proba(x_combined)[0][1] for model in models.values() if hasattr(model, "predict_proba")]
            consensus = np.mean(real_probs) * 100
            verdict = "VERIFIED AUTHENTIC CONTEXT" if consensus >= 50.0 else "HIGH RISK MISINFORMATION"
            print(f"  Ensemble Consensus Score: {consensus:.2f}% -> Verdict: {verdict}")

if __name__ == "__main__":
    run_strict_news_classification_test()
