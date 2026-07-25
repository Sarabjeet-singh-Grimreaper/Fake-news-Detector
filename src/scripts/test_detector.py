import os
import pickle
import scipy.sparse as sp
import numpy as np
from src.pipeline import NewsCredibilityPipeline

def verify_detector():
    print("[V2.0 Verification] Loading saved model assets...")
    
    pipeline = NewsCredibilityPipeline()
    try:
        pipeline.load("models")
    except FileNotFoundError:
        print("[Error] Pipeline assets not found. Run training first.")
        return
        
    models = {
        "Logistic Regression": "logreg_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl",
        "Voting Ensemble": "voting_ensemble_model.pkl"
    }
    
    loaded_models = {}
    for name, filename in models.items():
        filepath = os.path.join("models", filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                loaded_models[name] = pickle.load(f)
        else:
            print(f"[Warning] Model file {filename} not found.")

    real_world_tests = [
        ("BBC News (Real)", "The UK economy grew by 0.2% in the latest quarter, official statistics show. The Office for National Statistics said growth was driven by the service sector, particularly retail and hospitality, offsetting a slight decline in manufacturing."),
        ("CNN News (Real)", "NASA's James Webb Space Telescope has captured a stunning new image of the Cartwheel Galaxy, revealing new details about star formation and the galaxy's central black hole. The observatory used its Near-Infrared Camera to peer through thick dust clouds."),
        ("Reuters News (Real)", "Global oil prices stabilized on Friday as traders weighed supply disruptions in the Middle East against rising production in North America. Brent crude futures settled at $78.50 a barrel, while West Texas Intermediate rose slightly."),
        ("AP News (Real)", "Severe storms swept across the Midwest on Wednesday, leaving thousands without power and causing significant structural damage to homes and businesses. Local authorities have declared a state of emergency in affected counties."),
        ("The Guardian (Real)", "Researchers at Oxford University have developed a new malaria vaccine that has shown up to 80% efficacy in clinical trials. The vaccine, which targets the parasite's life cycle, could save hundreds of thousands of lives annually."),
        ("Partisan Fake News (Fake)", "URGENT BREAKING: Leaked secret documents confirm that the space launch was entirely staged in a desert warehouse. Global elites are using CGI holographic projection models to fake satellite pictures and control the populations' minds!")
    ]

    for sample_name, text_raw in real_world_tests:
        print(f"\n==================== PREDICTIONS FOR: {sample_name} ====================")
        # Transform using the pipeline
        X_combined, clean_txt, dense_list = pipeline.transform([text_raw])
        
        for name, model in loaded_models.items():
            try:
                pred = model.predict(X_combined)[0]
                probs = model.predict_proba(X_combined)[0]
                
                verdict = "Real" if pred == 1 else "Fake"
                confidence = probs[pred] * 100
                print(f"{name:<20} -> Prediction: {verdict:<10} | Confidence: {confidence:.2f}%")
            except Exception as e:
                print(f"{name:<20} -> Prediction error: {e}")

if __name__ == "__main__":
    verify_detector()
