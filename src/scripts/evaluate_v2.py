import os
import sys
import pickle
import numpy as np
import scipy.sparse as sp

# Ensure project root is on sys.path for standalone script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
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
            
            # 1. Baseline ML Predict
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

    print("\n[V2.0 Evaluation] --- Baseline ML Results Table ---")
    print(f"{'Source':<30} | {'Actual':<8} | {'Predicted':<8} | {'Confidence':<10} | {'Status':<6}")
    print("-" * 72)
    for r in results:
        print(f"{r['source']:<30} | {r['actual']:<8} | {r['predicted']:<8} | {r['confidence']*100:6.2f}%    | {r['status']:<6}")
        
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n[V2.0 Evaluation] Baseline ML Final Accuracy: {accuracy:.2f}% ({correct}/{total} passed)")

    # 4. Multi-Signal Fusion Engine Evaluation (including real-time verification and satire detection)
    from src.credibility_engine import evaluate_comprehensive_credibility
    print("\n" + "=" * 72)
    print("[V2.0 Evaluation] --- Multi-Signal Real-Time Fusion Engine Evaluation ---")
    print("=" * 72)
    
    multi_correct = 0
    multi_total = 0
    multi_results = []
    
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
                
            res = evaluate_comprehensive_credibility(
                content, title="", url="", pipeline=pipeline, model=model, check_realtime=True
            )
            
            is_pred_real = 1 if "Real" in res["verdict"] else 0
            is_correct = (is_pred_real == label)
            if is_correct:
                multi_correct += 1
            multi_total += 1
            
            status = "PASS" if is_correct else "FAIL"
            multi_results.append({
                "source": f"{category}/{filename}",
                "actual": "Real" if label == 1 else "Fake",
                "verdict": res["verdict"],
                "score": res["composite_score"],
                "status": status,
                "reason": res["reason"][:45] + "..." if len(res["reason"]) > 45 else res["reason"]
            })
            
    print(f"{'Source':<26} | {'Actual':<6} | {'Multi-Signal Verdict':<22} | {'Score':<6} | {'Status':<6}")
    print("-" * 75)
    for r in multi_results:
        print(f"{r['source']:<26} | {r['actual']:<6} | {r['verdict']:<22} | {r['score']:5.1f}% | {r['status']:<6}")
        
    multi_accuracy = (multi_correct / multi_total) * 100 if multi_total > 0 else 0
    print(f"\n[V2.0 Evaluation] Multi-Signal Real-Time Engine Accuracy: {multi_accuracy:.2f}% ({multi_correct}/{multi_total} passed)")

    # 5. Write to reports/generalization_report.md
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/generalization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Real-World Out-of-Domain Generalization Report\n\n")
        f.write("This report benchmarks the Fake News Credibility Engine against raw out-of-domain articles from public news agencies, health authorities, and conspiracy feeds.\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Total Benchmark Articles**: {multi_total}\n")
        f.write(f"- **Baseline ML Accuracy**: {accuracy:.2f}% ({correct}/{total} passed)\n")
        f.write(f"- **Multi-Signal Real-Time Engine Accuracy**: {multi_accuracy:.2f}% ({multi_correct}/{multi_total} passed)\n")
        f.write(f"- **Status**: {'[PASS] High Robustness' if multi_accuracy >= 90 else '[WARN] Verification Recommended'}\n\n")
        f.write("## 2. Multi-Signal Fusion Evaluation Breakdown\n\n")
        f.write("| Source File | Ground Truth | Multi-Signal Verdict | Score | Status | Primary Decision Factor |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for mr in multi_results:
            icon = "✅ PASS" if mr["status"] == "PASS" else "❌ FAIL"
            f.write(f"| `{mr['source']}` | {mr['actual']} | {mr['verdict']} | {mr['score']:.1f}% | {icon} | {mr['reason']} |\n")
        f.write("\n## 3. Baseline ML Standalone Breakdown\n\n")
        f.write("| Source File | Ground Truth | Model Prediction | Confidence | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            icon = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL"
            f.write(f"| `{r['source']}` | {r['actual']} | {r['predicted']} | {r['confidence']*100:.1f}% | {icon} |\n")
    print(f"[V2.0 Evaluation] Generalization report successfully written to {report_path}")

if __name__ == "__main__":
    run_isolated_evaluation()
