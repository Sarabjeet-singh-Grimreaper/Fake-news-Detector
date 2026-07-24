import unittest
import os
import pickle
import numpy as np
import scipy.sparse as sp
from src.preprocessing import full_preprocess_pipeline, compute_dense_features
from src.scraper import scrape_article
from src.domain_trust import get_domain_credibility
from src.explainability import explain_prediction

class TestFakeNewsDetector(unittest.TestCase):
    
    def test_preprocessing_pipeline(self):
        """Unit Test: Verifies preprocessing text normalization, cleaning, and filtering."""
        text = "Check out this BREAKING news details at http://example.com!!! It's shocking."
        cleaned = full_preprocess_pipeline(text)
        # Verify URL is removed
        self.assertNotIn("http", cleaned)
        # Verify lowercase
        self.assertEqual(cleaned, cleaned.lower())
        # Verify custom expanded stopwords are functioning (e.g. 'breaking' should be filtered out)
        self.assertNotIn("breaking", cleaned)

    def test_dense_features_extraction(self):
        """Unit Test: Verifies all 32 dense statistical features are calculated correctly."""
        text = "This is a standard test sentence. It contains some text. Let's see what happens!"
        feats = compute_dense_features(text, "test sentence text happens", "Test Title")
        
        self.assertEqual(len(feats), 32)
        # Assert specific feature indices ranges (e.g. all features are float/int)
        for val in feats:
            self.assertTrue(isinstance(val, (int, float, np.float64, np.int64)))
            
        # Title length features validation
        self.assertEqual(feats[24], len("Test Title")) # Title chars count
        self.assertEqual(feats[25], 2) # Title words count

    def test_domain_trust_engine(self):
        """Unit Test: Verifies domain trust categorization and scoring."""
        res_reuters = get_domain_credibility("https://www.reuters.com/article/some-news")
        self.assertEqual(res_reuters["score"], 100)
        self.assertEqual(res_reuters["badge"], "Trusted")
        
        res_gov = get_domain_credibility("https://whitehouse.gov/briefing-room")
        self.assertEqual(res_gov["score"], 98)
        self.assertEqual(res_gov["badge"], "Trusted")
        
        res_fake = get_domain_credibility("http://www.infowars.com/breaking-details")
        self.assertEqual(res_fake["score"], 10)
        self.assertEqual(res_fake["badge"], "Low Trust")

    def test_pipeline_asset_integrity(self):
        """Pipeline Test: Verifies models, vectorizer, selector, and scaler are loaded correctly."""
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        selector_path = "models/feature_selector.pkl"
        scaler_path = "models/dense_scaler.pkl"
        ensemble_path = "models/voting_ensemble_model.pkl"
        
        self.assertTrue(os.path.exists(vectorizer_path), "TF-IDF Vectorizer file missing.")
        self.assertTrue(os.path.exists(selector_path), "Feature Selector file missing.")
        self.assertTrue(os.path.exists(scaler_path), "Dense Scaler file missing.")
        self.assertTrue(os.path.exists(ensemble_path), "Voting Ensemble model file missing.")
        
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(selector_path, "rb") as f:
            selector = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
        # Verify shape sizes match expectations (4000 selected vocabulary + 32 dense = 4032)
        support_len = np.sum(selector.get_support())
        self.assertEqual(support_len, 4000)
        self.assertEqual(len(scaler.scale_), 32)

    def test_local_explainability_surrogate(self):
        """Model Explainability Test: Verifies that local linear contributions are calculated correctly."""
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/feature_selector.pkl", "rb") as f:
            selector = pickle.load(f)
        with open("models/dense_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("models/logreg_model.pkl", "rb") as f:
            logreg = pickle.load(f)
            
        text = "Government policy interest rates held steady by Federal Reserve."
        clean_str = full_preprocess_pipeline(text)
        
        vec_in = vectorizer.transform([clean_str])
        dense_feats = compute_dense_features(text, clean_str, "Fed Rates Update")
        
        expl = explain_prediction(text, clean_str, vec_in, dense_feats, logreg, vectorizer, selector)
        self.assertNotIn("error", expl)
        self.assertIn("category_summary", expl)
        self.assertIn("top_real_words", expl)
        self.assertIn("top_fake_words", expl)

if __name__ == "__main__":
    unittest.main()
