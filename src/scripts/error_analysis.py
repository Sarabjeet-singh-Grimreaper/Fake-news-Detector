import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.features import extract_features

def run_error_analysis():
    print("[Error Analysis] Loading data and preparing Logistic Regression model...")
    # Load original combined news to display raw text for analysis
    df_raw = pd.read_csv("data/processed/combined_news.csv")
    df_cleaned = pd.read_csv("data/processed/cleaned_news.csv")
    
    # Run the same feature extraction and split
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    # We want to keep track of the test indices
    # train_test_split on pandas Series returns the original indexes in y_test.index
    test_indices = y_test.index
    
    # Train Logistic Regression with best parameter C=10.0
    model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_test)
    
    # Convert y_test and y_pred to numpy arrays for comparison
    y_test_arr = y_test.values
    
    # Find indices of misclassified instances
    misclassified_mask = y_pred != y_test_arr
    misclassified_test_indices = np.where(misclassified_mask)[0]
    
    print(f"\n[Error Analysis] Total Test Samples: {len(y_test)}")
    print(f"[Error Analysis] Total Misclassified Samples: {len(misclassified_test_indices)}")
    
    # Print details of the top misclassified articles
    print("\n" + "="*80)
    print("DETAILS OF MISCLASSIFIED ARTICLES")
    print("="*80)
    
    shown = 0
    for idx in misclassified_test_indices:
        if shown >= 5:
            break
        # Get original dataframe index
        orig_df_idx = test_indices[idx]
        
        # Get raw data values
        raw_row = df_raw.iloc[orig_df_idx]
        cleaned_row = df_cleaned.iloc[orig_df_idx]
        
        true_label = "Real (1)" if y_test_arr[idx] == 1 else "Fake (0)"
        pred_label = "Real (1)" if y_pred[idx] == 1 else "Fake (0)"
        
        print(f"\n[{shown+1}] Original Index: {orig_df_idx}")
        print(f"    -> Title: {raw_row.get('title', 'N/A')}")
        print(f"    -> True Label: {true_label}")
        print(f"    -> Predicted Label: {pred_label}")
        print(f"    -> Snippet (Raw): {str(raw_row.get('text', ''))[:300]}...")
        print(f"    -> Snippet (Cleaned): {str(cleaned_row.get('clean_text', ''))[:300]}...")
        print("-" * 50)
        shown += 1
        
    # Analyze Feature Leakage / Model Coefficients
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    
    # Sort coefficients
    sorted_coef_idx = np.argsort(coefficients)
    
    # Most predictive of Fake (Class 0, large negative coefficients)
    top_fake_words = feature_names[sorted_coef_idx[:15]]
    top_fake_weights = coefficients[sorted_coef_idx[:15]]
    
    # Most predictive of Real (Class 1, large positive coefficients)
    top_real_words = feature_names[sorted_coef_idx[-15:]][::-1]
    top_real_weights = coefficients[sorted_coef_idx[-15:]][::-1]
    
    print("\n" + "="*80)
    print("TOP LEAKAGE SIGNATURES / DISCRIMINATIVE WORDS")
    print("="*80)
    print("Top 15 words predicting FAKE news (Negative Coefficients):")
    for word, weight in zip(top_fake_words, top_fake_weights):
        print(f"   -> {word:<15}: {weight:.4f}")
        
    print("\nTop 15 words predicting REAL news (Positive Coefficients):")
    for word, weight in zip(top_real_words, top_real_weights):
        print(f"   -> {word:<15}: {weight:.4f}")

if __name__ == "__main__":
    run_error_analysis()
