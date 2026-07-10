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
        "Online SGD Classifier": "sgd_online"
    }
    
    # Run predictions
    for sample_name, sample in [("Real News Sample", real_sample), ("Fake News Sample", fake_sample)]:
        print(f"\n==================== PREDICTIONS FOR: {sample_name} ====================")
        cleaned = full_preprocess_pipeline(sample['text'])
        vectorized = vectorizer.transform([cleaned])
        
        for model_name, file_key in models.items():
            try:
                with open(f"models/{file_key}_model.pkl", "rb") as f:
                    model = pickle.load(f)
                pred = model.predict(vectorized)[0]
                probs = model.predict_proba(vectorized)[0]
                
                verdict = "Real (1)" if pred == 1 else "Fake (0)"
                confidence = probs[pred] * 100
                print(f"{model_name:<20} -> Prediction: {verdict:<10} | Confidence: {confidence:.2f}%")
            except Exception as e:
                print(f"{model_name:<20} -> Error loading model: {e}")

if __name__ == "__main__":
    verify_detector()
