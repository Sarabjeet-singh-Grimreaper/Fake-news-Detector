import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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

def run_cli_verification(query_or_url: str):
    import os
    import pickle
    from src.scraper import scrape_article
    from src.credibility_engine import evaluate_comprehensive_credibility

    print(f"\n[VerifiQ CLI] Initiating real-time credibility audit for:")
    print(f"  Target: {query_or_url}")
    print("-" * 65)

    title, text, url = "", query_or_url, ""
    if query_or_url.startswith("http://") or query_or_url.startswith("https://"):
        url = query_or_url
        print("[VerifiQ CLI] Scraping web article...")
        article_data = scrape_article(url)
        if "error" in article_data:
            print(f"[Error] Failed to fetch article: {article_data['error']}")
            return
        title = article_data.get("title", "")
        text = article_data.get("text", "")
        print(f"[VerifiQ CLI] Extracted Title: {title}")
        print(f"[VerifiQ CLI] Extracted Body: {len(text.split())} words")
    else:
        title = query_or_url

    pipeline = NewsCredibilityPipeline()
    model = None
    try:
        pipeline.load("models")
        model_path = "models/voting_ensemble_model.pkl"
        if not os.path.exists(model_path):
            model_path = "models/logreg_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
    except Exception:
        print("[Warning] ML model assets not loaded. Running rule & real-time mode.")

    res = evaluate_comprehensive_credibility(
        text=text, title=title, url=url, pipeline=pipeline, model=model, check_realtime=True
    )

    print("\n" + "=" * 65)
    print(f"[VERDICT]    : {res['verdict'].upper()} (Credibility Score: {res['composite_score']:.1f}/100)")
    print(f"[RISK TIER]  : {res['risk_label']} | Confidence: {res['confidence_tier']}")
    print(f"[EXPLANATION]: {res['reason']}")
    print("CRITICAL SIGNAL BREAKDOWN:")
    signals = res['signals']
    print(f"  * Machine Learning Ensemble    : {signals['ml_score']:>5.1f}%  ({signals['ml_prediction']})")
    print(f"  * Live News Wire Corroboration : {signals['realtime_score']:>5.1f}%  ({signals['realtime_status']})")
    print(f"  * Source Domain Credibility    : {signals['domain_score']:>5.1f}%  ({signals['domain_badge']})")
    print(f"  * Stylometric / Integrity      : {signals['stylistic_score']:>5.1f}%")

    matches = res.get('realtime_details', {}).get('matching_articles', [])
    if matches:
        print(f"\nMATCHING LIVE NEWS WIRE STORIES ({len(matches)}):")
        for idx, art in enumerate(matches[:5], 1):
            print(f"  [{idx}] {art['title']}")
            print(f"      Source: {art['source']} | Date: {art['pub_date']}")
    print("-" * 65)
 
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
    parser.add_argument("--verify", type=str, metavar="CLAIM_OR_URL", help="Verify a news headline, claim, or URL live from the terminal.")
    parser.add_argument("--run", action="store_true", help="Run standard pipeline evaluation on training and test split.")
    
    args = parser.parse_args()
    
    # Dispatcher
    if args.verify:
        run_cli_verification(args.verify)
    elif args.stage:
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