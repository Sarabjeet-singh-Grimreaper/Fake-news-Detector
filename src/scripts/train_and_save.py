import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import NewsCredibilityPipeline
from src.models import get_models

def train_and_save_all():
    processed_path = "data/processed/combined_news.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Combined data missing at {processed_path}. Please run --preprocess first.")
        
    print("[V2.0 Train] Loading combined shuffled dataset...")
    df = pd.read_csv(processed_path)
    
    # Fill NA values
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['label'] = df['label'].fillna(0).astype(int)

    # Instantiate the unified pipeline
    pipeline = NewsCredibilityPipeline(max_features=4000) # Keep 4000 features like V1
    
    print("[V2.0 Train] Extracting features and fitting pipeline...")
    # To prevent memory issues with massive datasets in local testing, let's limit or train on the dataset
    # The dataset can be large (around 45k rows). Let's fit on it.
    X_combined = pipeline.fit(df['text'].tolist(), df['title'].tolist())
    y = df['label'].values
    
    # Save the pipeline components (vectorizer, dense scaler)
    pipeline.save("models")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_dict = get_models()
    
    model_files = {
        "LogReg": "logreg_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl",
        "Voting Ensemble": "voting_ensemble_model.pkl"
    }
    
    for name, model in models_dict.items():
        print(f"\n[V2.0 Train] Fitting model: {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"[V2.0 Train] {name} Generalization Test Accuracy: {score * 100:.2f}%")
        
        filename = f"models/{model_files[name]}"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"[V2.0 Train] Serialized {name} to {filename}")
        
    print("\n[V2.0 Train] All production models successfully trained and serialized.")

if __name__ == "__main__":
    train_and_save_all()
