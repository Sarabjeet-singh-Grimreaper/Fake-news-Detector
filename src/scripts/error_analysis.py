import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
from src.features import extract_features

def run_error_analysis():
    print("[Error Analysis] Loading data and preparing Logistic Regression model...")
    # Load original combined news to display raw text for analysis
    df_raw = pd.read_csv("data/processed/combined_news.csv")
    df_cleaned = pd.read_csv("data/processed/cleaned_news.csv")
    
    # Run the same feature extraction and split
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    # We want to keep track of the test indices
    test_indices = y_test.index
    
    # Train Logistic Regression with best parameter C=10.0, max_iter=5000
    model = LogisticRegression(C=10.0, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities and classes
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test_arr = np.array(y_test)
    
    # Ensure directories exist
    os.makedirs("reports/plots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    print("[Error Analysis] Generating Evaluation Plots...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_arr, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake (0)', 'Real (1)'], yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('Confusion Matrix - Error Analysis')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("reports/plots/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/plots/roc_curve.png")
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test_arr, y_prob)
    ap_score = average_precision_score(y_test_arr, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/plots/precision_recall_curve.png")
    plt.close()
    
    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test_arr, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/plots/calibration_curve.png")
    plt.close()
    
    # 5. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/plots/learning_curve.png")
    plt.close()
    
    # Find FP and FN indices
    fps = np.where((y_pred == 1) & (y_test_arr == 0))[0]
    fns = np.where((y_pred == 0) & (y_test_arr == 1))[0]
    
    # Determine "Hardest" articles (highest confidence errors)
    errors = np.where(y_pred != y_test_arr)[0]
    error_confidences = []
    for idx in errors:
        prob_real = y_prob[idx]
        conf = prob_real if y_pred[idx] == 1 else (1.0 - prob_real)
        error_confidences.append((idx, conf))
        
    hardest_errors = sorted(error_confidences, key=lambda x: x[1], reverse=True)[:5]
    
    # Compile markdown audit report
    print("[Error Analysis] Writing error_analysis_report.md...")
    report_path = "reports/error_analysis_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Robustness & Error Analysis Diagnostics Report\n\n")
        f.write("## 1. Classification Metrics Summary\n\n")
        f.write(f"- **Total Test Evaluation Samples**: {len(y_test)}\n")
        f.write(f"- **Misclassified Instances**: {len(errors)} / {len(y_test)} (Accuracy: {(1 - len(errors)/len(y_test))*100:.4f}%)\n")
        f.write(f"- **False Positives (Fake predicted as Real)**: {len(fps)}\n")
        f.write(f"- **False Negatives (Real predicted as Fake)**: {len(fns)}\n\n")
        
        f.write("### Detailed Classification Report\n")
        f.write("```\n")
        f.write(classification_report(y_test_arr, y_pred, target_names=["Fake (0)", "Real (1)"]))
        f.write("```\n\n")
        
        f.write("## 2. Hardest Articles (Highest Confidence Errors)\n\n")
        for i, (idx, conf) in enumerate(hardest_errors):
            orig_idx = test_indices[idx]
            raw_row = df_raw.iloc[orig_idx]
            true_lbl = "Real" if y_test_arr[idx] == 1 else "Fake"
            pred_lbl = "Real" if y_pred[idx] == 1 else "Fake"
            
            f.write(f"### {i+1}. Original Index: {orig_idx} (Error Confidence: {conf*100:.2f}%)\n")
            f.write(f"- **Title**: {raw_row.get('title', 'N/A')}\n")
            f.write(f"- **Actual Label**: {true_lbl} | **Model Prediction**: {pred_lbl}\n")
            f.write(f"- **Content Excerpt**:\n  > *\"{str(raw_row.get('text', ''))[:350]}...\"*\n\n")
            
        # Top Vocabulary coefficients
        f.write("## 3. Discriminative Vocabulary Coefficients\n\n")
        # Load the feature selector support
        with open("models/feature_selector.pkl", "rb") as sel_file:
            selector = pickle.load(sel_file)
            
        feature_names = np.array(vectorizer.get_feature_names_out())
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = feature_names[selected_indices]
        
        coefficients = model.coef_[0]
        # Only vocabulary features coefficients (indices 0 to 3999)
        vocab_coeffs = coefficients[:len(selected_indices)]
        sorted_idx = np.argsort(vocab_coeffs)
        
        f.write("### Top 15 Words Predicting FAKE News:\n")
        for idx in sorted_idx[:15]:
            f.write(f"- `{selected_feature_names[idx]}`: {vocab_coeffs[idx]:.4f}\n")
        f.write("\n")
        
        f.write("### Top 15 Words Predicting REAL News:\n")
        for idx in sorted_idx[-15:][::-1]:
            f.write(f"- `{selected_feature_names[idx]}`: {vocab_coeffs[idx]:.4f}\n")
            
    print(f"[Error Analysis] Report saved successfully to {report_path}")
    print("[Error Analysis] Diagnostic charts generated successfully in reports/plots/")

if __name__ == "__main__":
    run_error_analysis()
