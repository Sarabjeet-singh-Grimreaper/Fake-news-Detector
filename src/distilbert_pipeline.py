import os
import re
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# Configuration Constants
MODEL_NAME = "distilbert-base-uncased"
SAVED_MODEL_DIR = "./saved_model"

# =====================================================================
# 1. PIPELINE TRAINING & EVALUATION (Google Colab / GPU Context)
# =====================================================================
def run_training_pipeline(dataset_path="train.csv"):
    print("[1/3] Starting Training Pipeline...")
    
    # Load dataset (assumes columns: 'id', 'title', 'author', 'text', 'label')
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please provide data to train.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Fill missing values and concatenate title + body into unified text feature
    df['title'] = df['title'].fillna("")
    df['text'] = df['text'].fillna("")
    df['label'] = df['label'].fillna(0).astype(int)
    
    df['combined_text'] = df['title'] + " " + df['text']
    df = df.drop_duplicates(subset=['combined_text'])
    
    # Remove excessive whitespace datelines and headers
    def clean_text(t):
        return " ".join(t.split()).lower()
    df['combined_text'] = df['combined_text'].apply(clean_text)
    
    # Stratified split (80% train, 20% test)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Convert DataFrames to Hugging Face Dataset format
    train_ds = Dataset.from_pandas(train_df[['combined_text', 'label']].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[['combined_text', 'label']].reset_index(drop=True))
    
    def tokenize_function(examples):
        return tokenizer(examples["combined_text"], truncation=True, padding="max_length", max_length=512)
        
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)
    
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Setup Sequence Classifier
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )
    
    # Train and Save model weights & tokenizer configurations
    trainer.train()
    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)
    print(f"[Done] Saved model weights & configuration configurations to: {SAVED_MODEL_DIR}")

# =====================================================================
# 2. REAL-TIME FLASK WEB APP DEPLOYMENT (Unified Backend)
# =====================================================================
app = Flask(__name__)

# Device allocation (GPU if available, otherwise fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for prediction engine
predictor_model = None
predictor_tokenizer = None

def init_prediction_engine():
    global predictor_model, predictor_tokenizer
    if os.path.exists(SAVED_MODEL_DIR):
        print(f"[Engine] Loading serialized weights from: {SAVED_MODEL_DIR}")
        predictor_tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_DIR)
        predictor_model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)
        predictor_model.to(device)
        predictor_model.eval()
    else:
        print("[Warning] Trained weights directory not found. Please train the pipeline first or configure path.")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict API Endpoint returning classification label and confidence."""
    global predictor_model, predictor_tokenizer
    if predictor_model is None or predictor_tokenizer is None:
        return jsonify({"status": "ERROR", "message": "Model weights are not loaded."}), 500
        
    data = request.get_json(force=True)
    text_input = data.get("text", "").strip()
    
    if not text_input:
        return jsonify({"status": "ERROR", "message": "Missing article text payload."}), 400
        
    # Standardize input casing
    processed_text = " ".join(text_input.split()).lower()
    
    # Tokenization and local inference pass
    inputs = predictor_tokenizer(
        processed_text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = predictor_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
    pred_class = np.argmax(probs)
    # Mapping label: 1 = Unreliable (Fake), 0 = Reliable (Real)
    verdict = "UNRELIABLE" if pred_class == 1 else "RELIABLE"
    confidence = probs[pred_class]
    
    return jsonify({
        "status": "SUCCESS",
        "prediction": verdict,
        "confidence_score": float(confidence)
    })

# =====================================================================
# 3. FRONTEND INLINE ROUTING
# =====================================================================
HTML_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Veritas AI | DistilBERT news verification</title>
    <style>
        body {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            width: 100%;
            max-width: 650px;
            background: #111827;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.05);
        }
        h2 {
            margin-top: 0;
            background: linear-gradient(135deg, #a5b4fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            background: #1f2937;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #f1f5f9;
            padding: 12px;
            box-sizing: border-box;
            font-size: 0.95rem;
            resize: vertical;
        }
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        .result-box {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
            text-align: center;
            font-weight: 600;
        }
        .unreliable { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
        .reliable { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    </style>
</head>
<body>
    <div class="container">
        <h2>Veritas AI news verification</h2>
        <p style="color: #94a3b8; text-align: center; font-size: 0.9rem;">DistilBERT Engine Classifier Interface</p>
        <textarea id="newsText" placeholder="Paste full news headline or article body here..."></textarea>
        <button onclick="runAudit()">Audit Content</button>
        <div id="resultBox" class="result-box"></div>
    </div>

    <script>
        async function runAudit() {
            const text = document.getElementById("newsText").value.trim();
            const resultBox = document.getElementById("resultBox");
            if (!text) {
                alert("Please enter news content first.");
                return;
            }
            
            resultBox.style.display = "block";
            resultBox.className = "result-box";
            resultBox.innerText = "Analyzing language vectors...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                const data = await response.json();
                
                if (data.status === "SUCCESS") {
                    const cleanClass = data.prediction === "UNRELIABLE" ? "unreliable" : "reliable";
                    resultBox.className = `result-box ${cleanClass}`;
                    resultBox.innerHTML = `Verdict: ${data.prediction} <br> <span style="font-size: 0.85rem; font-weight: normal;">Confidence score: ${(data.confidence_score * 100).toFixed(2)}%</span>`;
                } else {
                    resultBox.innerText = "Prediction Error: " + data.message;
                }
            } catch (err) {
                resultBox.innerText = "Connection error. Ensure Flask backend is running.";
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def dashboard():
    """Renders the HTML workspace."""
    return render_template_string(HTML_DASHBOARD_TEMPLATE)

# =====================================================================
# PIPELINE INITIATOR
# =====================================================================
if __name__ == "__main__":
    init_prediction_engine()
    # Run the web server
    app.run(host="0.0.0.0", port=5000, debug=False)
