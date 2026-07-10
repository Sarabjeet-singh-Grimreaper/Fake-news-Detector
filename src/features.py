import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_features():
    processed_path = "data/processed/combined_news.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Cleaned training stream missing at {processed_path}. Please run --preprocess first.")
        
    df = pd.read_csv(processed_path)
    # Handle any potential empty fields gracefully
    df['clean_text'] = df['clean_text'].fillna('')
    
    # 5,000 feature constraint ensures quick online inference overhead
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Serialize the global vectorizer mapping to disk
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    return X_train, X_test, y_train, y_test, vectorizer