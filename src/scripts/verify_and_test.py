import os
import json
import pickle
import unittest
import shutil
import time
import tracemalloc
import numpy as np
from unittest.mock import patch, MagicMock

# Import pipeline components
from src.preprocessing import clean_text, tokenize_and_filter, full_preprocess_pipeline
from src.scraper import scrape_article
from src.scripts.autonomous_engine import (
    process_unlabeled_stream, 
    load_or_init_model_and_vectorizer, 
    extract_hybrid_features,
    MODEL_PATH,
    DRIFT_REVIEW_PATH
)

class TestLeakageAndPreprocessing(unittest.TestCase):
    """
    PHASE 1: PREPROCESSING & ADVERSARIAL LEAKAGE AUDIT
    """
    def test_leakage_stripping(self):
        # Explicit publisher datelines and social footer attributions
        inputs = [
            "WASHINGTON (Reuters) - The president signed the bill today.",
            "LONDON (Reuters) — Officials declared an emergency.",
            "A suspicious incident occurred. Read more via Facebook loops",
            "Breaking reports on the event via @twitterhandle now."
        ]
        
        for text in inputs:
            cleaned = clean_text(text)
            # Verify publisher datelines and social media handles/attribution signatures are stripped
            self.assertNotIn("Reuters", cleaned)
            self.assertNotIn("reuters", cleaned)
            self.assertNotIn("Facebook", cleaned)
            self.assertNotIn("facebook", cleaned)
            self.assertNotIn("twitter", cleaned)
            self.assertNotIn("washington", cleaned)
            self.assertNotIn("london", cleaned)
            
    def test_boundary_edge_cases(self):
        # Edge inputs that could cause crash
        edge_inputs = [
            "",
            "   ",
            "!!!???!!!",
            "1234567890",
            "the and of in to a it is that", # academic stopwords only
            None
        ]
        
        for text in edge_inputs:
            try:
                result = full_preprocess_pipeline(text)
                self.assertIsInstance(result, str)
            except Exception as e:
                self.fail(f"Pipeline crashed on edge case '{text}' with exception: {e}")

class TestCrawlerAndInference(unittest.TestCase):
    """
    PHASE 2: REAL-TIME INFERENCE & WEB CRAWLER STABILITY
    """
    @patch('requests.get')
    def test_scraper_payload_verification(self, mock_get):
        # 1. Clean HTML Stream
        mock_response_clean = MagicMock()
        mock_response_clean.status_code = 200
        mock_response_clean.content = b"<html><head><title>Test Title</title></head><body><h1>Main Heading</h1><p>This is a test paragraph with more than five words in it.</p></body></html>"
        
        mock_get.return_value = mock_response_clean
        result_clean = scrape_article("http://example.com/clean")
        
        self.assertNotIn("error", result_clean)
        self.assertEqual(result_clean["title"], "Main Heading")
        self.assertIn("test paragraph", result_clean["text"])
        
        # 2. Malformed/Broken DOM
        mock_response_broken = MagicMock()
        mock_response_broken.status_code = 200
        mock_response_broken.content = b"<html><head><title>Broken Title</title><body><p>Unclosed paragraph tags and missing tags"
        
        mock_get.return_value = mock_response_broken
        result_broken = scrape_article("http://example.com/broken")
        self.assertNotIn("error", result_broken)
        
        # 3. HTTP Error Response
        mock_get.side_effect = Exception("HTTP 404 Not Found")
        result_error = scrape_article("http://example.com/error")
        self.assertIn("error", result_error)

    def test_prediction_alignment(self):
        model, vectorizer = load_or_init_model_and_vectorizer()
        test_text = "This is a random neutral political news update regarding current infrastructure planning initiatives."
        
        x_sparse = extract_hybrid_features(test_text, vectorizer)
        probs = model.predict_proba(x_sparse)[0]
        
        # Double-precision float ranges verification
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

class TestSelfLearningAndDrift(unittest.TestCase):
    """
    PHASE 3: AUTONOMOUS SELF-LEARNING & PARAMETER DRIFT VERIFICATION
    """
    def setUp(self):
        # Safely backup the real model file to avoid test pollution
        self.backup_model_path = MODEL_PATH + ".bak"
        if os.path.exists(MODEL_PATH):
            shutil.copyfile(MODEL_PATH, self.backup_model_path)
            
        self.backup_drift_path = DRIFT_REVIEW_PATH + ".bak"
        if os.path.exists(DRIFT_REVIEW_PATH):
            shutil.copyfile(DRIFT_REVIEW_PATH, self.backup_drift_path)
            # Remove live file for pristine testing
            os.remove(DRIFT_REVIEW_PATH)

    def tearDown(self):
        # Restore original models
        if os.path.exists(self.backup_model_path):
            shutil.copyfile(self.backup_model_path, MODEL_PATH)
            os.remove(self.backup_model_path)
            
        if os.path.exists(self.backup_drift_path):
            shutil.copyfile(self.backup_drift_path, DRIFT_REVIEW_PATH)
            os.remove(self.backup_drift_path)
        elif os.path.exists(DRIFT_REVIEW_PATH):
            os.remove(DRIFT_REVIEW_PATH)

    def test_confidence_gated_pseudo_labeling(self):
        # Load baseline weights and pre-fit to ensure fitted state
        model_before, vectorizer = load_or_init_model_and_vectorizer()
        
        # Pre-fit on dummy examples of both classes to make it non-trivial
        x_dummy_0 = extract_hybrid_features("The Federal Reserve indicated that interest rates would remain stable.", vectorizer)
        x_dummy_1 = extract_hybrid_features("ALERT BREAKING! Alien spaceship landed with fake news details!", vectorizer)
        model_before.partial_fit(x_dummy_0, np.array([0]), classes=np.array([0, 1]))
        model_before.partial_fit(x_dummy_1, np.array([1]), classes=np.array([0, 1]))
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_before, f)
            
        coef_before = np.copy(model_before.coef_)
        
        # Article with high-confidence signature (strong indicators)
        elite_article = [{
            "title": "Severe Financial Crisis Imminent",
            "text": "The Federal Reserve indicated on Tuesday that interest rates would remain stable pending next quarter's inflation reports."
        }]
        
        # Run through active learning loop with strict elite threshold
        process_unlabeled_stream(elite_article, elite_threshold=0.51, review_threshold=0.50)
        
        # Load weights after update
        model_after, _ = load_or_init_model_and_vectorizer()
        coef_after = np.copy(model_after.coef_)
        
        # Weight changes verify the training happened autonomously
        self.assertFalse(np.array_equal(coef_before, coef_after), "Model weights did not shift after processing high-confidence stream.")

    def test_anomaly_isolation(self):
        # Ensure model is initialized
        model_before, vectorizer = load_or_init_model_and_vectorizer()
        x_dummy_0 = extract_hybrid_features("The Federal Reserve indicated that interest rates would remain stable.", vectorizer)
        x_dummy_1 = extract_hybrid_features("ALERT BREAKING! Alien spaceship landed with fake news details!", vectorizer)
        model_before.partial_fit(x_dummy_0, np.array([0]), classes=np.array([0, 1]))
        model_before.partial_fit(x_dummy_1, np.array([1]), classes=np.array([0, 1]))
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_before, f)
            
        # Ambiguous article designed to output low confidence
        ambiguous_article = [{
            "title": "Neutral Update",
            "text": "Yesterday something happened, or maybe it didn't. Some details were reported but they are totally unconfirmed."
        }]
        
        # Ensure it hits the review path (threshold set high to force low confidence anomaly logging)
        process_unlabeled_stream(ambiguous_article, elite_threshold=1.1, review_threshold=1.05)



        
        self.assertTrue(os.path.exists(DRIFT_REVIEW_PATH), "Drift review output file was not created.")
        with open(DRIFT_REVIEW_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        self.assertTrue(len(records) > 0, "No records found in drift review file.")
        self.assertIn("Yesterday something happened", records[-1]["text"])

def run_performance_stress_benchmark():
    """
    PHASE 4: PERFORMANCE TELEMETRY & STRESS AUDIT
    """
    print("\n" + "="*50)
    print("PHASE 4: PERFORMANCE TELEMETRY & STRESS AUDIT")
    print("="*50)
    
    # Generate 50 multi-domain article data objects sequentially
    mock_articles = []
    for i in range(50):
        domain = "finance" if i % 2 == 0 else "conspiracy"
        text_content = (
            f"The central bank announced financial guidelines for fiscal year {2026 + i} in standard market briefing."
            if domain == "finance" else
            f"ALERT!!! Alien spaceships landed near area {50 + i} warning the local population about secret plans!"
        )
        mock_articles.append({
            "title": f"Sequential Test Document {i}",
            "text": text_content,
            "domain": domain
        })
        
    # Start tracking memory
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()
    
    # Run latency benchmarks
    start_time = time.perf_counter()
    scraping_times = []
    inference_times = []
    
    model, vectorizer = load_or_init_model_and_vectorizer()
    
    for doc in mock_articles:
        # Simulate scraping overhead (very small random delay to represent fast local mock parser processing)
        scrap_start = time.perf_counter()
        # Dummy HTML parse representation
        _ = scrape_article("http://mock-live-source.org")
        scraping_times.append((time.perf_counter() - scrap_start) * 1000) # ms
        
        # Predict / Inference overhead
        inf_start = time.perf_counter()
        x_sparse = extract_hybrid_features(doc["text"], vectorizer)
        _ = model.predict_proba(x_sparse)
        inference_times.append((time.perf_counter() - inf_start) * 1000) # ms
        
    total_duration = (time.perf_counter() - start_time) * 1000
    mem_peak, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    avg_scraping = np.mean(scraping_times)
    avg_inference = np.mean(inference_times)
    peak_mem_mb = mem_peak / (1024 * 1024)
    
    print(f"Benchmark run completed for {len(mock_articles)} articles.")
    print(f"Total time elapsed: {total_duration:.2f} ms")
    print(f"Average Inference Latency: {avg_inference:.4f} ms")
    print(f"Average Scraping Overhead: {avg_scraping:.4f} ms")
    print(f"Peak Memory Footprint Shift: {peak_mem_mb:.6f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    import sys
    # Run Phases 1-3 via unittest runner
    print("Executing Pipeline Automation Unit & Integration Tests...")
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run Phase 4 Performance Telemetry Benchmark
    if result.wasSuccessful():
        run_performance_stress_benchmark()
    else:
        print("[Warning] Unit tests failed. Skipping performance benchmark.")
        sys.exit(1)

