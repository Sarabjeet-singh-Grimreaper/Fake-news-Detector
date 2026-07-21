import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.features import extract_features
from src.models import get_models

def tune_models(subset_size=5000):
    print(f"[Tuner] Loading features and splitting data (using subset of {subset_size} for speed)...")
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    # Slice the training data for faster hyperparameter search
    if subset_size and subset_size < X_train.shape[0]:
        X_train_sub = X_train[:subset_size]
        y_train_sub = y_train[:subset_size]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Define grids for each model
    param_grids = {
        "KNN": {
            "model_ref": "KNN",
            "grid": {
                "n_neighbors": [3, 5, 9],
                "weights": ["uniform", "distance"]
            },
            "search_type": "grid"
        },
        "LogReg": {
            "model_ref": "LogReg",
            "grid": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"]
            },
            "search_type": "grid"
        },
        "Random Forest": {
            "model_ref": "Random Forest",
            "grid": {
                "n_estimators": [50, 100],
                "max_depth": [10, 30, None]
            },
            "search_type": "random",
            "n_iter": 5
        },
        "NeuralNet": {
            "model_ref": "NeuralNet",
            "grid": {
                "alpha": [0.0001, 0.01],
                "hidden_layer_sizes": [(50,), (100,)]
            },
            "search_type": "grid"
        },
        "SVM": {
            "model_ref": "SVM",
            "grid": {
                "C": [0.1, 1.0, 10.0]
            },
            "search_type": "grid"
        }
    }

    base_models = get_models()
    best_params = {}

    for name, config in param_grids.items():
        model = base_models[name]
        grid = config["grid"]
        search_type = config["search_type"]
        
        print(f"\n[Tuner] Tuning {name} using {search_type.upper()} search...")
        start_time = time.time()
        
        if search_type == "grid":
            search = GridSearchCV(
                estimator=model,
                param_grid=grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=grid,
                n_iter=config.get("n_iter", 5),
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
        search.fit(X_train_sub, y_train_sub)
        elapsed = time.time() - start_time
        
        print(f"[Tuner] Completed tuning {name} in {elapsed:.2f} seconds.")
        print(f"        -> Best Params: {search.best_params_}")
        print(f"        -> Best CV Accuracy: {search.best_score_:.4f}")
        
        best_params[name] = search.best_params_

    # Print summary of results
    print("\n" + "="*50)
    print("[Tuner] SUMMARY OF BEST HYPERPARAMETERS")
    print("="*50)
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")
    print("="*50)

if __name__ == "__main__":
    tune_models(subset_size=5000)
