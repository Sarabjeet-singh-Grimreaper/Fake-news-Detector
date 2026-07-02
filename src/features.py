import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(data_path="data/processed/cleaned_news.csv", max_features=5000):
    """
    Loads the cleaned text data, converts strings into numerical matrices 
    using TF-IDF, and splits the data into training and validation sets.
    """
    print("📖 Loading preprocessed text data...")
    df = pd.read_csv(data_path)
    
    # Fill any accidental null values in text with an empty string
    X = df['clean_text'].fillna("")
    y = df['label']
    
    print(f"🧮 Extracting top {max_features} TF-IDF features...")
    # Initialize the vectorizer to keep only the top 5000 most significant words
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Learn the vocabulary dictionary and return a document-term matrix
    X_vectorized = vectorizer.fit_transform(X)
    
    print("🔀 Splitting dataset into Train (80%) and Test (20%) sets...")
    # Stratify ensure classes are evenly distributed between train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Feature extraction complete.")
    print(f"   -> Train Shape: {X_train.shape}")
    print(f"   -> Test Shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, vectorizer