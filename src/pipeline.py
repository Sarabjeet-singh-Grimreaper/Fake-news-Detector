import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import full_preprocess_pipeline
from src.features import extract_dense_features

class NewsCredibilityPipeline:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.98
        )
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def fit(self, texts, titles=None):
        """
        Fits the TF-IDF vectorizer and the dense feature scaler.
        texts: list of raw text strings.
        titles: list of title strings (optional).
        """
        if titles is None:
            titles = [""] * len(texts)

        print("[Pipeline] Running preprocessing and cleaning...")
        clean_texts = [full_preprocess_pipeline(t) for t in texts]

        print("[Pipeline] Fitting TF-IDF Vectorizer...")
        X_tfidf = self.vectorizer.fit_transform(clean_texts)

        print("[Pipeline] Extracting and scaling dense features...")
        dense_feats_list = []
        for raw_txt, clean_txt, title in zip(texts, clean_texts, titles):
            _, feats = extract_dense_features(raw_txt, clean_txt, title)
            dense_feats_list.append(feats)

        dense_array = np.array(dense_feats_list, dtype=np.float64)
        dense_scaled = self.scaler.fit_transform(dense_array)

        self.is_fitted = True
        
        # Combine TF-IDF and dense features
        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(dense_scaled)], format="csr")
        return X_combined

    def transform(self, texts, titles=None):
        """
        Transforms texts and titles using the fitted TF-IDF and dense feature scaler.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit first.")
        if titles is None:
            titles = [""] * len(texts)

        clean_texts = [full_preprocess_pipeline(t) for t in texts]
        X_tfidf = self.vectorizer.transform(clean_texts)

        dense_feats_list = []
        for raw_txt, clean_txt, title in zip(texts, clean_texts, titles):
            _, feats = extract_dense_features(raw_txt, clean_txt, title)
            dense_feats_list.append(feats)

        dense_array = np.array(dense_feats_list, dtype=np.float64)
        dense_scaled = self.scaler.transform(dense_array)

        X_combined = sp.hstack([X_tfidf, sp.csr_matrix(dense_scaled)], format="csr")
        return X_combined, clean_texts, dense_feats_list

    def save(self, directory="models"):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(directory, "dense_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Pipeline] Serialized TF-IDF vectorizer and scaler to '{directory}/'")

    def load(self, directory="models"):
        with open(os.path.join(directory, "tfidf_vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(directory, "dense_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"[Pipeline] Loaded vectorizer and scaler from '{directory}/'")
