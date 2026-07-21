import os
import pickle
import numpy as np
from src.features import extract_features
from src.models import get_models

def train_and_save_all():
    print("[Automata Engine] Commencing baseline feature extraction processing...")
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    models_dict = get_models()
    
    model_files = {
        "KNN": "knn_model.pkl",
        "LogReg": "logreg_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "NeuralNet": "neuralnet_model.pkl",
        "SVM": "svm_model.pkl",
        "Online_Logistic_Regression": "sgd_online_model.pkl"
    }
    
    for name, model in models_dict.items():
        print(f"\n[Automata Engine] Fitting model: {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"[Automata Engine] {name} Generalization Accuracy: {score * 100:.2f}%")
        
        filename = f"models/{model_files[name]}"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"[Automata Engine] Serialized {name} to {filename}")
        
    print("\n[Automata Engine] All models successfully trained and serialized.")

if __name__ == "__main__":
    train_and_save_all()

