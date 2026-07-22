import pickle
import pandas as pd
from src.preprocessing import full_preprocess_pipeline

def verify_detector():
    print("[Verification] Loading raw dataset to extract test samples...")
    df = pd.read_csv("data/processed/combined_news.csv")
    
    # Get a real news article sample (label 1)
    real_sample = df[df['label'] == 1].iloc[0]
    # Get a fake news article sample (label 0)
    fake_sample = df[df['label'] == 0].iloc[0]
    
    print("\n--- SAMPLE 1: ACTUAL REAL NEWS ---")
    print(f"Title: {real_sample['title']}")
    print(f"Snippet: {str(real_sample['text'])[:150]}...")
    
    print("\n--- SAMPLE 2: ACTUAL FAKE NEWS ---")
    print(f"Title: {fake_sample['title']}")
    print(f"Snippet: {str(fake_sample['text'])[:150]}...")
    
    # Load assets
    print("\n[Verification] Loading saved model assets...")
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    models = {
        "K-Nearest Neighbors": "knn",
        "Logistic Regression": "logreg",
        "Random Forest": "random_forest",
        "MLP Neural Network": "neuralnet",
        "Support Vector Machine": "svm",
        "Online SGD Classifier": "sgd_online"
    }
    
    import scipy.sparse as sp
    import numpy as np
    EMOTIONAL_WORDS = {'shocking', 'conspiracy', 'leaked', 'secret', 'urgent', 'viral', 'breaking', 'exposed', 'unbelievable', 'miracle', 'truth', 'warning', 'agenda', 'censored', 'anonymous', 'classified', 'insider', 'hiding', 'scandal', 'banned', 'shocked', 'chaos', 'destroys', 'slam', 'blasts', 'panic', 'terror', 'crisis', 'must-see', 'revealed', 'prophecy', 'secretly'}

    # Run predictions
    for sample_name, sample in [("Real News Sample", real_sample), ("Fake News Sample", fake_sample)]:
        print(f"\n==================== PREDICTIONS FOR: {sample_name} ====================")
        text_raw = str(sample['text'])
        cleaned = full_preprocess_pipeline(text_raw)
        vectorized = vectorizer.transform([cleaned])
        
        from src.preprocessing import compute_dense_features
        dense_feats_list = compute_dense_features(text_raw, cleaned)
        dense_feats = np.array([dense_feats_list], dtype=np.float64)
        final_vector = sp.hstack([vectorized, sp.csr_matrix(dense_feats)])
        
        for model_name, file_key in models.items():
            try:
                # Resolve sgd_online to sgd_online_model
                filename = f"models/sgd_online_model.pkl" if file_key == "sgd_online" else f"models/{file_key}_model.pkl"
                with open(filename, "rb") as f:
                    model = pickle.load(f)
                
                # Align dimensionality dynamically if needed
                current_vector = final_vector
                if hasattr(model, "n_features_in_"):
                    expected = model.n_features_in_
                    actual = current_vector.shape[1]
                    if actual > expected:
                        current_vector = current_vector[:, :expected]
                    elif actual < expected:
                        padding = sp.csr_matrix((current_vector.shape[0], expected - actual))
                        current_vector = sp.hstack([current_vector, padding], format="csr")

                pred = model.predict(current_vector)[0]
                probs = model.predict_proba(current_vector)[0]
                
                verdict = "Real (1)" if pred == 1 else "Fake (0)"
                confidence = probs[pred] * 100
                print(f"{model_name:<20} -> Prediction: {verdict:<10} | Confidence: {confidence:.2f}%")
            except Exception as e:
                print(f"{model_name:<20} -> Error loading model: {e}")

if __name__ == "__main__":
    verify_detector()
