import unittest
import os
import pickle
import numpy as np
import scipy.sparse as sp
from src.preprocessing import full_preprocess_pipeline
from src.features import extract_dense_features
from src.domain_trust import get_domain_credibility
from src.explainability import explain_prediction
from src.pipeline import NewsCredibilityPipeline

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
        """Unit Test: Verifies all V2.0 Group B-F dense statistical features are calculated correctly."""
        text = "This is a standard test sentence. It contains some text. Let's see what happens!"
        clean_str = full_preprocess_pipeline(text)
        feats_dict, feats_list = extract_dense_features(text, clean_str, "Test Title")
        
        self.assertEqual(len(feats_list), 12)
        # Assert all features are float/int
        for val in feats_list:
            self.assertTrue(isinstance(val, (int, float, np.float64, np.int64)))
            
        # Verify specific feature outputs
        self.assertGreater(feats_dict["avg_sentence_len"], 0)
        self.assertTrue(0 <= feats_dict["lexical_diversity"] <= 1.0)
        self.assertTrue(0 <= feats_dict["flesch_reading_ease"] <= 100.0)

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
        """Pipeline Test: Verifies models, vectorizer, and scaler are loaded correctly."""
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        scaler_path = "models/dense_scaler.pkl"
        ensemble_path = "models/voting_ensemble_model.pkl"
        
        # We only assert check files if they have been trained/exist
        if os.path.exists(vectorizer_path):
            pipeline = NewsCredibilityPipeline()
            pipeline.load("models")
            self.assertTrue(pipeline.is_fitted)
            self.assertEqual(len(pipeline.scaler.scale_), 12)

    def test_local_explainability_surrogate(self):
        """Model Explainability Test: Verifies that local linear contributions are calculated correctly."""
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        logreg_path = "models/logreg_model.pkl"
        
        if os.path.exists(vectorizer_path) and os.path.exists(logreg_path):
            pipeline = NewsCredibilityPipeline()
            pipeline.load("models")
            
            with open(logreg_path, "rb") as f:
                logreg = pickle.load(f)
                
            text = "Government policy interest rates held steady by Federal Reserve."
            clean_str = full_preprocess_pipeline(text)
            
            vec_in = pipeline.vectorizer.transform([clean_str])
            _, dense_feats = extract_dense_features(text, clean_str, "Fed Rates Update")
            
            expl = explain_prediction(text, clean_str, vec_in, dense_feats, logreg, pipeline.vectorizer)
            self.assertNotIn("error", expl)
            self.assertIn("category_summary", expl)
            self.assertIn("top_real_words", expl)
            self.assertIn("top_fake_words", expl)

if __name__ == "__main__":
    unittest.main()
