import os
import pickle
import numpy as np
from src.features import extract_features
from src.models import get_models

def train_and_save_all():
    print("[Automata Engine] Commencing baseline feature extraction processing...")
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    models_dict = get_models()
    online_model = models_dict["Online_Logistic_Regression"]
    
    print("[Automata Engine] Fitting baseline models using SGD Online Learning paradigms...")
    # Initialize classes explicitly for subsequent partial_fit streams
    unique_classes = np.array([0, 1])
    
    # Baseline batch training phase
    online_model.fit(X_train, y_train)
    
    # Evaluate baseline accuracy on holdout test set split
    baseline_score = online_model.score(X_test, y_test)
    print(f"[Automata Engine] Initial Baseline Generalization Accuracy: {baseline_score * 100:.2f}%")
    
    # Save the iteration state to disk
    with open("models/logreg_model.pkl", "wb") as f:
        pickle.dump(online_model, f)
    print("[Automata Engine] Models successfully serialized to models/ directory.")

if __name__ == "__main__":
    train_and_save_all()
