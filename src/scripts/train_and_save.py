import os
import pickle
from src.features import extract_features
from src.models import get_models

def train_and_save_all():
    print("[Trainer] Starting full model training and serialization...")
    
    # Extract features using full clean dataset
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    # Train all models
    models = get_models()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the vectorizer first
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[Trainer] Saved TF-IDF Vectorizer to: {vectorizer_path}")
    
    for name, model in models.items():
        print(f"[Trainer] Training {name} on the complete training set...")
        model.fit(X_train, y_train)
        
        # Save model
        safe_name = name.lower().replace(" ", "_")
        model_path = f"models/{safe_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[Trainer] Saved {name} model to: {model_path}")
        
    print("[Trainer] All models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_all()
