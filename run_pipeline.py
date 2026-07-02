import sys
import argparse
from src.features import extract_features
from src.models import get_models
from src.evaluate import evaluate_model

# Import utility scripts dynamically from the internal script package
from src.scripts.stage_data import stage_dataset
from src.scripts.preprocess_all import run_mass_preprocessing
from src.scripts.train_and_save import train_and_save_all
from src.scripts.tune_hyperparameters import tune_models
from src.scripts.error_analysis import run_error_analysis
from src.scripts.test_detector import verify_detector

def run_news_detection_pipeline():
    print("[Pipeline] Starting Fake News Detection Baseline Evaluation...")
    
    # 1. Extract TF-IDF features and split dataset
    X_train, X_test, y_train, y_test, vectorizer = extract_features()
    
    # 2. Get the dictionary of models
    models = get_models()
    
    # 3. Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n[Model] Training {model_name}...")
        model.fit(X_train, y_train)
        
        print(f"[Model] Predicting with {model_name}...")
        y_pred = model.predict(X_test)
        
        # 4. Generate evaluation reports & confusion matrices
        evaluate_model(y_test, y_pred, model_name)
        
    print("\n[Pipeline] Baseline evaluation execution completed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Veritas AI: Unified Command Line Interface for Fake News Detection Pipeline."
    )
    
    # Define flags
    parser.add_argument("--stage", action="store_true", help="Download raw files and shuffle dataset streams.")
    parser.add_argument("--preprocess", action="store_true", help="Run clean-up and tokenization on the raw corpus.")
    parser.add_argument("--train", action="store_true", help="Train and serialize models to models/ folder.")
    parser.add_argument("--tune", action="store_true", help="Execute hyperparameter sweeps across configurations.")
    parser.add_argument("--analyze", action="store_true", help="Run coefficient audit and leakage diagnostics.")
    parser.add_argument("--test", action="store_true", help="Verify predictions against real and fake news samples.")
    parser.add_argument("--run", action="store_true", help="Run standard pipeline evaluation on training and test split.")
    
    args = parser.parse_args()
    
    # Dispatcher
    if args.stage:
        stage_dataset()
    elif args.preprocess:
        run_mass_preprocessing()
    elif args.train:
        train_and_save_all()
    elif args.tune:
        tune_models(subset_size=5000)
    elif args.analyze:
        run_error_analysis()
    elif args.test:
        verify_detector()
    elif args.run or len(sys.argv) == 1:
        run_news_detection_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()