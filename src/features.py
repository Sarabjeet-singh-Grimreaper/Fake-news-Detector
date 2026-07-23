import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Simple local emotional keywords for sentiment bias calculation
EMOTIONAL_WORDS = {
    'shocking', 'conspiracy', 'leaked', 'secret', 'urgent', 'viral', 'breaking', 
    'exposed', 'unbelievable', 'miracle', 'truth', 'warning', 'agenda', 'censored', 
    'anonymous', 'classified', 'insider', 'hiding', 'scandal', 'banned', 'shocked', 
    'chaos', 'destroys', 'slam', 'blasts', 'panic', 'terror', 'crisis', 'must-see', 
    'revealed', 'prophecy', 'secretly', 'unconfirmed', 'hoax', 'fraud', 'illegal',
    'conspire', 'collusion', 'deepstate', 'rigged', 'covert', 'plot', 'cover-up'
}

def extract_features():
    processed_path = "data/processed/combined_news.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Cleaned training stream missing at {processed_path}. Please run --preprocess first.")
        
    df = pd.read_csv(processed_path)
    df['clean_text'] = df['clean_text'].fillna('')
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    
    # Optimized TF-IDF setup
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.98
    )
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    
    # Feature Selection (SelectKBest Chi-Square)
    from sklearn.feature_selection import SelectKBest, chi2
    selector = SelectKBest(score_func=chi2, k=4000)
    X_tfidf_selected = selector.fit_transform(X_tfidf, df['label'])
    
    # Compute dense metadata features using upgraded stylometric, readability, clickbait, and sentiment indicators
    from src.preprocessing import compute_dense_features
    dense_features = []
    for raw_text, clean_str, title in zip(df['text'], df['clean_text'], df['title']):
        feats = compute_dense_features(raw_text, clean_str, title)
        dense_features.append(feats)
        
    # Scale dense features to [0, 1] range to match TF-IDF magnitude
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dense_scaled = scaler.fit_transform(dense_features)
    X_dense = sp.csr_matrix(dense_scaled)
    
    # Combine sparse selected TF-IDF and dense features
    X_combined = sp.hstack([X_tfidf_selected, X_dense], format="csr")
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    
    # Serialize assets to disk
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/feature_selector.pkl", "wb") as f:
        pickle.dump(selector, f)
    with open("models/dense_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    return X_train, X_test, y_train, y_test, vectorizer