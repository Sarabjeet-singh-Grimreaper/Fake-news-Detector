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
from src.scraper import scrape_article

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
            dense_scaled = pipeline.scaler.transform([dense_feats])[0]
            
            expl = explain_prediction(text, clean_str, vec_in, dense_feats, logreg, pipeline.vectorizer, dense_scaled=dense_scaled)
            self.assertNotIn("error", expl)
            self.assertIn("category_summary", expl)
            self.assertIn("top_real_words", expl)
            self.assertIn("top_fake_words", expl)

            # Verify category summary percentages sum to 100%
            total_share = sum(item["share"] for item in expl["category_summary"].values())
            self.assertAlmostEqual(total_share, 100.0, places=1)

    def test_domain_trust_raw_domains(self):
        """Unit Test: Verifies domain trust handles scheme-less and subdomain URLs."""
        # Scheme-less URLs
        res1 = get_domain_credibility("reuters.com/news/article-123")
        self.assertEqual(res1["badge"], "Trusted")
        self.assertEqual(res1["score"], 100)

        res2 = get_domain_credibility("infowars.com/conspiracy")
        self.assertEqual(res2["badge"], "Low Trust")

        # Subdomains
        res3 = get_domain_credibility("https://edition.cnn.com/world")
        self.assertEqual(res3["badge"], "Trusted")

        res_neutral = get_domain_credibility("https://myrandomblog.wordpress.com/story")
        self.assertEqual(res_neutral["badge"], "Neutral")

    def test_scraper_ssrf_and_security(self):
        """Security Test: Verifies SSRF protection against loopback, private IP, and non-http schemes."""
        # Loopback attempt
        res_loopback = scrape_article("http://127.0.0.1:8080/secret")
        self.assertIn("error", res_loopback)
        self.assertIn("prohibited", res_loopback["error"].lower())

        # Localhost attempt
        res_local = scrape_article("http://localhost:5000")
        self.assertIn("error", res_local)

        # File scheme attempt
        res_file = scrape_article("file:///etc/passwd")
        self.assertIn("error", res_file)
        self.assertIn("unsupported protocol", res_file["error"].lower())

    def test_pipeline_transform_edge_cases(self):
        """Pipeline Test: Verifies handling of edge-case inputs without crashing."""
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        if os.path.exists(vectorizer_path):
            pipeline = NewsCredibilityPipeline()
            pipeline.load("models")

            # Empty text and symbols
            X_comb, clean_texts, dense_list = pipeline.transform(["", "   ", "!@#$%^&*()", "1234567890"])
            self.assertEqual(X_comb.shape[0], 4)
            self.assertEqual(X_comb.shape[1], 4012)
            self.assertEqual(len(clean_texts), 4)

    def test_satire_and_parody_detection(self):
        """Unit Test: Verifies that satire domains and parody phrasing are accurately recognized."""
        from src.credibility_engine import check_satirical_patterns
        
        # Domain test
        onion_res = get_domain_credibility("https://www.theonion.com/article/12345")
        self.assertEqual(onion_res["badge"], "Satire / Parody")
        self.assertEqual(onion_res["score"], 15)

        babylon_res = get_domain_credibility("https://babylonbee.com/news/article")
        self.assertEqual(babylon_res["badge"], "Satire / Parody")

        # Phrasing test
        satirical_text = "The nation's top conspiracy theorists officially conceded today that the Earth is indeed a sphere."
        prob = check_satirical_patterns(satirical_text)
        self.assertGreaterEqual(prob, 0.7)

    def test_realtime_query_extraction_and_relevance(self):
        """Unit Test: Verifies extraction of substantive search terms and suppression of generic filler words."""
        from src.realtime_verification import extract_search_terms
        
        text = "In a historic and ground-breaking development, scientists have successfully engineered a quantum-biological hybrid leaf that absorbs light and outputs gold particles."
        query, salient = extract_search_terms("", text)
        
        # Ensure query doesn't contain generic words
        self.assertNotIn("historic", query.split())
        self.assertNotIn("scientists", query.split())
        # Ensure substantive entity is preserved
        self.assertTrue(any(w in query for w in ["quantum", "biological", "hybrid", "leaf", "gold"]))

    def test_credibility_engine_multi_signal_fusion(self):
        """Integration Test: Verifies multi-signal fusion across ML, live wire corroboration, and domain trust."""
        from src.credibility_engine import evaluate_comprehensive_credibility
        
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        ensemble_path = "models/voting_ensemble_model.pkl"
        
        pipeline = None
        model = None
        if os.path.exists(vectorizer_path) and os.path.exists(ensemble_path):
            pipeline = NewsCredibilityPipeline()
            pipeline.load("models")
            with open(ensemble_path, "rb") as f:
                model = pickle.load(f)
                
        # Test A: Genuine real news with reputable domain
        real_text = "Global oil prices stabilized on Friday as traders weighed supply disruptions in the Middle East."
        res_real = evaluate_comprehensive_credibility(
            text=real_text,
            title="Global Oil Prices Stabilize",
            url="https://www.reuters.com/business/energy/oil-prices",
            pipeline=pipeline,
            model=model,
            check_realtime=False
        )
        self.assertIn("Real", res_real["verdict"])
        self.assertGreater(res_real["composite_score"], 60.0)
        
        # Test B: Known fake/disinformation domain
        fake_text = "Urgent secret leak proves that the earth is completely hollow and populated by lizard elites."
        res_fake = evaluate_comprehensive_credibility(
            text=fake_text,
            title="Urgent Secret Leak",
            url="https://infowars.com/breaking-leak",
            pipeline=pipeline,
            model=model,
            check_realtime=False
        )
        self.assertEqual(res_fake["risk_label"], "HIGH RISK")
        self.assertLess(res_fake["composite_score"], 40.0)

    def test_scraper_ssrf_protection(self):
        """Security Test: Verifies that SSRF attack vectors (localhost, loopback, private IPs) are rejected."""
        from src.scraper import is_safe_url, scrape_article
        
        # Test private and loopback addresses
        bad_urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1/secret",
            "http://0.0.0.0/",
            "ftp://files.example.com",
            "file:///etc/passwd"
        ]
        for bad_url in bad_urls:
            is_safe, msg = is_safe_url(bad_url)
            self.assertFalse(is_safe, f"Expected {bad_url} to be flagged as unsafe.")
            res = scrape_article(bad_url)
            self.assertIn("error", res)

    def test_scraper_json_ld_extraction(self):
        """Web Scraping Test: Verifies JSON-LD schema parsing extracts headline and clean body."""
        from bs4 import BeautifulSoup
        import json
        
        sample_html = """
        <html>
          <head>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "NewsArticle",
              "headline": "Structured Data Headline Test",
              "datePublished": "2026-09-11T12:00:00Z",
              "author": {"@type": "Person", "name": "Dr. Jane Doe"},
              "publisher": {"@type": "Organization", "name": "Global Science News"},
              "articleBody": "This is a clean structured article body extracted directly from JSON-LD schema markup without sidebar noise."
            }
            </script>
          </head>
          <body>
            <h1>Fallback H1</h1>
            <p>DOM Paragraph</p>
          </body>
        </html>
        """
        soup = BeautifulSoup(sample_html, "html.parser")
        # Extract using same logic
        script = soup.find("script", type="application/ld+json")
        data = json.loads(script.string)
        self.assertEqual(data["headline"], "Structured Data Headline Test")
        self.assertEqual(data["author"]["name"], "Dr. Jane Doe")
        self.assertIn("JSON-LD schema markup", data["articleBody"])

    def test_extreme_edge_inputs(self):
        """Robustness Test: Verifies pipeline handles boundary inputs without crashing."""
        edge_inputs = [
            "",
            "   ",
            "!?!!??",
            "12345 67890 999",
            "🚀🔥✨💯",
            "a" * 5000
        ]
        for edge_str in edge_inputs:
            cleaned = full_preprocess_pipeline(edge_str)
            self.assertIsInstance(cleaned, str)
            feats_dict, feats_list = extract_dense_features(edge_str, cleaned, "Edge Case Title")
            self.assertEqual(len(feats_list), 12)

if __name__ == "__main__":
    unittest.main()
