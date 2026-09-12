import os
import requests
import numpy as np
import torch
import xml.etree.ElementTree as ET
from flask import Flask, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
MODEL_DIR = "./saved_model"
FALLBACK_MODEL = "distilbert-base-uncased"
RSS_FEED_URL = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

# =====================================================================
# 1. COMPACT PIPELINE & SINGLETON MODEL LOADING
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Initialization] Using hardware acceleration device: {device}")

# Determine path: use saved model if available, else load base pre-trained model as fallback
model_source = MODEL_DIR if os.path.exists(MODEL_DIR) else FALLBACK_MODEL
print(f"[Initialization] Loading tokenizer and model from: {model_source}")

tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2)
model.to(device)
model.eval()

def predict_batch(text_list):
    """
    Executes a single-batch parallel inference pass for maximum efficiency.
    Returns predictions (RELIABLE/UNRELIABLE) and winning class confidence.
    """
    if not text_list:
        return []
        
    # Standardize spaces and casing
    processed_texts = [" ".join(text.split()).lower() for text in text_list]
    
    # Tokenize input list in parallel
    inputs = tokenizer(
        processed_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
    results = []
    for row in probs:
        pred_class = np.argmax(row)
        # Mapping: 1 = Unreliable (Fake), 0 = Reliable (Real)
        prediction = "UNRELIABLE" if pred_class == 1 else "RELIABLE"
        results.append({
            "prediction": prediction,
            "confidence": float(row[pred_class])
        })
    return results

# =====================================================================
# 2. LIVE INGESTION ENGINE WITH RSS FEED
# =====================================================================
def fetch_realtime_feed():
    """
    Ingests live news feeds using Google News RSS, falling back to a curated local
    array of Indian & Global headlines if connection fails.
    """
    try:
        response = requests.get(RSS_FEED_URL, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            articles = []
            for item in root.findall(".//item")[:9]:
                title = item.find("title").text if item.find("title") is not None else ""
                source_elem = item.find("source")
                source = source_elem.text if source_elem is not None else "Google News"
                
                # Clean up title if source name is appended
                if " - " in title:
                    title = " - ".join(title.split(" - ")[:-1])
                
                if title:
                    articles.append({"title": title, "source": source})
            if articles:
                return articles
    except Exception as e:
        print(f"[Ingestion] RSS feed fetch failed: {e}. Serving fallback context.")

    # Ingestion Fallback Datastore (3 Global, 3 Indian)
    return [
        {"title": "Federal Reserve maintains interest rates target range amid steady inflation data.", "source": "Reuters"},
        {"title": "Global summit reaches new cooperative agreement on carbon emission targets.", "source": "Associated Press"},
        {"title": "Scientists capture high-resolution imagery of black hole magnetic fields.", "source": "NASA Spaceflight"},
        {"title": "CONFIRMED: Government to disable all mobile networks starting tomorrow night.", "source": "Social Media Rumor"},
        {"title": "Reserve Bank of India announces digital rupee integration guidelines for retail banking.", "source": "Economic Times"},
        {"title": "BREAKING: NASA claims a giant asteroid will collide with Mumbai next Tuesday.", "source": "Whatsapp Broadcast"}
    ]

# =====================================================================
# 3. WEB STREAMING BACKEND (Flask)
# =====================================================================
app = Flask(__name__)

@app.route("/api/stream", methods=["GET"])
def stream_api():
    """Ingests, batches predictions, and streams JSON news payloads."""
    articles = fetch_realtime_feed()
    titles = [a["title"] for a in articles]
    
    # Run batch inference pass
    predictions = predict_batch(titles)
    
    stream_payload = []
    for art, pred in zip(articles, predictions):
        stream_payload.append({
            "title": art["title"],
            "source": art["source"],
            "prediction": pred["prediction"],
            "confidence": pred["confidence"]
        })
        
    return jsonify(stream_payload)

# =====================================================================
# 4. AUTO-REFRESHING VISUAL DASHBOARD
# =====================================================================
HTML_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Veritas AI | Real-Time News Monitoring Stream</title>
    <!-- Tailwind CSS CDN for visual interface styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #080b11; }
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col">

    <!-- Header Section -->
    <header class="border-b border-slate-800/60 bg-slate-950/60 backdrop-blur-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">🛡️</span>
                <div>
                    <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">VERITAS AI</h1>
                    <p class="text-xs text-slate-500 font-mono">DISTILBERT REAL-TIME CLASSIFIER STREAM</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <span class="relative flex h-3.5 w-3.5">
                    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span class="relative inline-flex rounded-full h-3.5 w-3.5 bg-emerald-500"></span>
                </span>
                <span id="syncStatus" class="text-xs text-slate-400 font-mono">Syncing...</span>
            </div>
        </div>
    </header>

    <!-- Dashboard Content -->
    <main class="flex-grow max-w-7xl mx-auto px-6 py-8 w-full">
        <div class="mb-6">
            <h2 class="text-lg font-semibold text-slate-200">Global & Regional Ingestion Monitor</h2>
            <p class="text-sm text-slate-400">Continuous Google News RSS feed classification. Dashboard auto-refreshes every 10 seconds.</p>
        </div>

        <div id="gridContainer" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <!-- Dynamic news card layouts will render here -->
        </div>
    </main>

    <!-- JavaScript Stream Orchestrator -->
    <script>
        const gridContainer = document.getElementById("gridContainer");
        const syncStatus = document.getElementById("syncStatus");

        async function refreshStream() {
            syncStatus.innerText = "Syncing feed...";
            try {
                const res = await fetch("/api/stream");
                const data = await res.json();
                
                // Clear grid layout
                gridContainer.innerHTML = "";
                
                data.forEach(item => {
                    const isUnreliable = item.prediction === "UNRELIABLE";
                    const borderAccent = isUnreliable ? "border-red-500/30 hover:border-red-500/60" : "border-emerald-500/30 hover:border-emerald-500/60";
                    const badgeBg = isUnreliable ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
                    
                    const cardHtml = `
                        <div class="glass-card rounded-2xl p-6 flex flex-col justify-between border ${borderAccent} transition-all duration-300 transform hover:-translate-y-1 shadow-lg shadow-black/30">
                            <div>
                                <div class="flex justify-between items-start mb-4">
                                    <span class="text-xs font-mono text-slate-500">${item.source}</span>
                                    <span class="px-2.5 py-0.5 text-xs font-bold rounded-full border ${badgeBg}">
                                        ${item.prediction}
                                    </span>
                                </div>
                                <h3 class="text-slate-100 font-medium text-base leading-snug mb-4">${item.title}</h3>
                            </div>
                            <div class="pt-4 border-t border-slate-800/40 flex justify-between items-center">
                                <span class="text-xs text-slate-500">Confidence Score</span>
                                <span class="text-sm font-semibold text-slate-300">${(item.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `;
                    gridContainer.innerHTML += cardHtml;
                });
                
                const now = new Date();
                syncStatus.innerText = `Synced at ${now.toTimeString().split(' ')[0]}`;
            } catch (err) {
                syncStatus.innerText = "Connection Error";
                console.error("Stream Synchronization Error: ", err);
            }
        }

        // Initialize polling intervals
        refreshStream();
        setInterval(refreshStream, 10000);
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """Renders the auto-refreshing dashboard."""
    return render_template_string(HTML_DASHBOARD_TEMPLATE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
