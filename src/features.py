import os
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Simple local emotional keywords for sentiment bias calculation
EMOTIONAL_WORDS = {'shocking', 'conspiracy', 'leaked', 'secret', 'urgent', 'viral'}

def extract_features():
    processed_path = "data/processed/combined_news.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Cleaned training stream missing at {processed_path}. Please run --preprocess first.")
        
    df = pd.read_csv(processed_path)
    df['clean_text'] = df['clean_text'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # 5,000 feature constraint ensures quick online inference overhead
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    
    # Compute dense metadata features
    dense_features = []
    for raw_text, clean_str in zip(df['text'], df['clean_text']):
        raw_chars = max(1, len(raw_text))
        raw_words = max(1, len(raw_text.split()))
        
        cap_ratio = sum(1 for c in raw_text if c.isupper()) / raw_chars
        punc_density = sum(1 for c in raw_text if c in ['!', '?']) / raw_chars
        avg_word_len = np.mean([len(w) for w in raw_text.split()]) if raw_text.split() else 0.0
        sentiment_bias = sum(1 for w in clean_str.split() if w in EMOTIONAL_WORDS) / raw_words
        
        dense_features.append([cap_ratio, punc_density, avg_word_len, sentiment_bias])
        
    X_dense = sp.csr_matrix(dense_features)
    
    # Combine sparse TF-IDF and dense features
    X_combined = sp.hstack([X_tfidf, X_dense], format="csr")
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    
    # Serialize the global vectorizer mapping to disk
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    return X_train, X_test, y_train, y_test, vectorizer