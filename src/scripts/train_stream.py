import os
import pickle
import numpy as np
import pandas as pd
from src.preprocessing import full_preprocess_pipeline

def run_stream_training(epochs=5, chunk_size=5000):
    processed_path = "data/processed/combined_news.csv"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model_path = "models/logreg_model.pkl"
    
    if not os.path.exists(processed_path):
        print("[Error] No processed data matrix located. Run preprocessing routines first.")
        return

    # 1. Load the pre-configured Global Vocabulary Vectorizer
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("[Stream Engine] Global feature structure loaded successfully.")
    else:
        print("[Error] tfidf_vectorizer.pkl not found. Please initialize the pipeline mapping.")
        return

    # 2. Retrieve existing model parameters or build an adaptive baseline layout
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("[Stream Engine] Resuming training from historical serialized weights.")
    else:
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        print("[Stream Engine] Initializing brand new online learning weights layout.")

    # Unique classification alignments for online partial_fit registration
    unique_classes = np.array([0, 1])

    print(f"[Stream Engine] Beginning {epochs} progressive incremental optimization passes...")
    
    for epoch in range(epochs):
        print(f"\n--- Processing Pass Epoch {epoch + 1}/{epochs} ---")
        
        # Read the file in clean, memory-friendly micro-batches
        data_iterator = pd.read_csv(processed_path, chunksize=chunk_size)
        
        for i, chunk in enumerate(data_iterator):
            # Fill empty blocks safely
            chunk['clean_text'] = chunk['clean_text'].fillna('')
            
            # Map clean strings into numerical spatial fields
            X_batch = vectorizer.transform(chunk['clean_text'])
            y_batch = chunk['label'].values
            
            # Update the model parameters via stochastic gradient steps
            model.partial_fit(X_batch, y_batch, classes=unique_classes)
            
            if i % 2 == 0:
                print(f" Chunk {i}: Digested {len(y_batch)} articles. Weights iteratively adjusted.")

    # 3. Serialize optimized states safely back to your workspace configuration
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print("\n[Stream Engine] Iterative optimization cycle completed. All structural states saved to disk.")

if __name__ == "__main__":
    run_stream_training()
