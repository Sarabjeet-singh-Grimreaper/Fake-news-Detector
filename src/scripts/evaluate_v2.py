import os
import pickle
import numpy as np
import scipy.sparse as sp
from src.pipeline import NewsCredibilityPipeline

def run_isolated_evaluation():
    print("\n[V2.0 Evaluation] Starting Out-of-Domain Benchmark Evaluation...")
    
    # 1. Load pipeline
    pipeline = NewsCredibilityPipeline()
    try:
        pipeline.load("models")
    except FileNotFoundError:
        print("[Error] Production pipeline assets not found in models/. Train models first.")
        return

    # 2. Load model
    model_name = "Voting Ensemble"
    model_file = "models/voting_ensemble_model.pkl"
    if not os.path.exists(model_file):
        model_name = "Logistic Regression"
        model_file = "models/logreg_model.pkl"
        
    print(f"[V2.0 Evaluation] Evaluating model: {model_name}...")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # 3. Read evaluation datasets
    categories = {"real": 1, "fake": 0}
    correct = 0
    total = 0
    
    results = []
    
    for category, label in categories.items():
        dir_path = os.path.join("evaluation", category)
        if not os.path.exists(dir_path):
            continue
            
        for filename in os.listdir(dir_path):
            if not filename.endswith(".txt"):
                continue
                
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            if not content:
                continue
                
            # Run transformation
            X_combined, clean_txt, dense_list = pipeline.transform([content], [filename])
            
            # Predict
            pred = model.predict(X_combined)[0]
            probs = model.predict_proba(X_combined)[0]
            confidence = probs[pred]
            
            is_correct = (pred == label)
            if is_correct:
                correct += 1
            total += 1
            
            verdict = "Real" if pred == 1 else "Fake"
            actual = "Real" if label == 1 else "Fake"
            status = "PASS" if is_correct else "FAIL"
            
            results.append({
                "source": f"{category}/{filename}",
                "actual": actual,
                "predicted": verdict,
                "confidence": confidence,
                "status": status
            })

    print("\n[V2.0 Evaluation] --- Results Table ---")
    print(f"{'Source':<30} | {'Actual':<8} | {'Predicted':<8} | {'Confidence':<10} | {'Status':<6}")
    print("-" * 72)
    for r in results:
        print(f"{r['source']:<30} | {r['actual']:<8} | {r['predicted']:<8} | {r['confidence']*100:6.2f}%    | {r['status']:<6}")
        
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n[V2.0 Evaluation] Benchmark Final Accuracy: {accuracy:.2f}% ({correct}/{total} passed)")

if __name__ == "__main__":
    run_isolated_evaluation()
