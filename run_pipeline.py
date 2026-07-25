import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.pipeline import NewsCredibilityPipeline
from src.models import get_models
from src.evaluate import evaluate_model

# Import utility scripts dynamically from the internal script package
from src.scripts.stage_data import stage_dataset
from src.scripts.preprocess_all import run_mass_preprocessing
from src.scripts.train_and_save import train_and_save_all
from src.scripts.test_detector import verify_detector
from src.scripts.evaluate_v2 import run_isolated_evaluation
 
def run_news_detection_pipeline():
    print("[Pipeline] Starting Fake News Detection Baseline Evaluation...")
    
    processed_path = "data/processed/combined_news.csv"
    df = pd.read_csv(processed_path)
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['label'] = df['label'].fillna(0).astype(int)

    pipeline = NewsCredibilityPipeline(max_features=4000)
    X_combined = pipeline.fit(df['text'].tolist(), df['title'].tolist())
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
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
        description="VerifiQ: Unified Command Line Interface for Fake News Detection Pipeline."
    )
    
    # Define flags
    parser.add_argument("--stage", action="store_true", help="Download raw files and shuffle dataset streams.")
    parser.add_argument("--preprocess", action="store_true", help="Run clean-up and tokenization on the raw corpus.")
    parser.add_argument("--train", action="store_true", help="Train and serialize models to models/ folder.")
    parser.add_argument("--test", action="store_true", help="Verify predictions against real and fake news samples.")
    parser.add_argument("--benchmark", action="store_true", help="Evaluate models against out-of-domain evaluation/ benchmarks.")
    parser.add_argument("--run", action="store_true", help="Run standard pipeline evaluation on training and test split.")
    
    args = parser.parse_args()
    
    # Dispatcher
    if args.stage:
        stage_dataset()
    elif args.preprocess:
        run_mass_preprocessing()
    elif args.train:
        train_and_save_all()
    elif args.test:
        verify_detector()
        run_isolated_evaluation()
    elif args.benchmark:
        run_isolated_evaluation()
    elif args.run or len(sys.argv) == 1:
        run_news_detection_pipeline()
    else:
        parser.print_help()
 
if __name__ == "__main__":
    main()