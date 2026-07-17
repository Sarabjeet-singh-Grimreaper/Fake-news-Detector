# VerifiQ: AI-Powered Fake News Detection Using Text Classification

Welcome to the documentation for the project **"AI-Powered Fake News Detection Using Text Classification"** developed for the **Summer Internship Program in AI & ML 2026**.

This repository implements a complete end-to-end Machine Learning pipeline to classify news articles as **Real** or **Fake**, featuring a live ML classifier portal and a high-performance interactive simulation.

---

## 🛡️ Core Academic Constraints

1. **Custom NLP Preprocessing (From Scratch):** High-level NLP packages (like NLTK or SpaCy wrappers) are **not** used for tokenization or preprocessing. Instead, the cleaning, URL stripping, punctuation stripping, and stopwords removal are written from scratch using core regular expressions and basic Python string mechanics.
2. **Machine Learning Framework:** Features are extracted using scikit-learn's `TfidfVectorizer` (constrained to the top 5,000 features), and modeled using KNN, Logistic Regression, Random Forest, and MLP Neural Networks.
3. **IEEE Compliance:** The architectural design, pipeline verification, and model evaluations are built to support final drafting under strict academic IEEE report formats.

---

## 📂 Project Architecture

```text
fake-news-detector/
│
├── .github/workflows/
│   └── deploy.yml          # GitHub Pages deployment automation script
│
├── data/
│   ├── raw/                # Untouched True.csv and Fake.csv archives
│   └── processed/          # Combined (shuffled) and clean_text datasets
│
├── models/                 # Serialized pickle assets (vectorizer & models)
│
├── reports/
│   └── plots/              # Saved confusion matrix heatmaps
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Regex cleaning automata & manual stopwords list
│   ├── features.py        # TF-IDF conversion (max_features=5000) & split (80/20)
│   ├── models.py          # Configuration definitions for KNN, LogReg, RF, and MLP
│   ├── evaluate.py        # Heatmap plot generator & classification report printer
│   └── scripts/            # Modular pipeline utility files
│       ├── stage_data.py   # Downloader and shuffler
│       ├── preprocess_all.py # Mass cleaner orchestrator
│       ├── train_stream.py # Memory-efficient online/incremental training script
│       ├── tune_hyperparameters.py # Hyperparameter optimizer
│       ├── error_analysis.py # Leakage and coefficient examiner
│       └── test_detector.py # Verification testing script
│
├── app.py                  # State-of-the-art interactive Streamlit frontend portal
├── index.html              # Mobile-optimized interactive simulation landing page
├── style.css               # Glassmorphic responsive styling
├── app.js                  # Retina-scaled canvas physics simulation engine
├── run_pipeline.py         # Unified pipeline CLI orchestrator
└── requirements.txt        # Python package dependencies
```

---

## 🚀 Executive Summary & Core Mission

Modern digital landscapes suffer from information asymmetry, where fabricated clickbait spreads faster than verified journalism. Traditional natural language processing models fail here because they rely on batch offline training (which becomes obsolete instantly) or pick up shallow contextual artifacts (feature leakage) like publisher names.

**VerifIQ** solves this by establishing a decoupled, leakage-resilient, and continuously evolving detection framework. Its core mission is to provide an unbiased, real-time auditing system that parses custom text inputs or live URL scrapes, processes them through a custom regex-based cleaning engine, maps them to a hybrid 5,004-dimensional sparse-dense feature space, and outputs a consensus verification score.

---

## 🛠️ The Modern Tech Stack & Architectural Purpose

Every layer of VerifIQ has been chosen to optimize performance, minimize latency, and run efficiently on both desktop and mobile devices:

* **🐍 Python (Core Backend & ML Operations):** The fundamental programming language powering the entire scientific data science pipeline. It orchestrates the ingestion of news articles, executes the preprocess regular expression loops, runs evaluations, computes metrics, and serializes the resulting models.
* **📊 Streamlit (Diagnostic & AI Portal UI):** A lightweight Python framework designed to build high-performance data portals. It acts as the primary interface for users to enter custom articles, crawl URLs, trigger the dynamic models, and view consensus classification metrics. Injected CSS overrides are applied to style the portal as a premium dark glassmorphic interface.
* **🌐 HTML5 Canvas, CSS3, & Vanilla JavaScript (Visual Information Flow Simulator):** High-performance, client-side rendering engine deployed on GitHub Pages to simulate particle physics and information propagation. It handles high-DPI (Retina) displays, adjusts viewport bounds dynamically, maps desktop pointer controls to mobile touches, and uses `touch-action: none` to isolate gesture events.
* **🧠 Scikit-Learn (Vectorization & Modeling Core):**
  * *TfidfVectorizer:* Translates clean text strings into **5,000 spatial language fields** (TF-IDF sparse matrices) based on inverse document frequency weights.
  * *SGDClassifier:* Deployed with a log-loss objective function. It serves as the incremental active learning engine, using the `partial_fit` API to adjust gradients and write updated model weights to disk dynamically without needing a full batch training rebuild.
  * *Classifiers (Random Forest, MLP, KNN, Logistic Regression):* Run in parallel to perform ensemble predictions and output consensus verdicts.
* **⚡ SciPy Sparse Modules (Matrix Operations):** Integrates the 5,000-dimensional text vectors with the 4-dimensional dense metadata parameters using `scipy.sparse.hstack`. This prevents memory crashes when working with high-dimensional matrices.
* **🕸️ BeautifulSoup4 & Requests (Asynchronous Web Scraping):** Sends HTTP requests to any article URL, fetches the raw HTML page, filters structural markup, and extracts the core textual elements for validation.
* **🚀 GitHub Actions & GitHub Pages (DevOps & CI/CD Deployment):** Automatically tests, packages, and deploys the static Canvas simulation assets to the web on every git push to the main branch via a secure workflow.

---

## 🔄 Step-by-Step Runtime Project Flow

1. **Ingestion Layer:** The portal captures data via the Manual Text Stream input box or runs the real-time target domain URL web crawler using `BeautifulSoup4` and `Requests`.
2. **Leakage-Resilient Cleaning Automata:** The custom regex preprocessor sanitizes the text, completely stripping out publisher datelines (e.g., "Reuters", "AP") and platform attributions to block shallow vocabulary leaks.
3. **Hybrid Feature Layer Engineering:** The clean text maps into a 5,000-dimensional TF-IDF spatial field. Simultaneously, the framework measures 4 dense style metadata metrics: Clickbait Capitalization Ratio, Punctuation Density ('!' and '?'), Average Word Length, and Sentiment Intensity Bias.
4. **The 5,004 Matrix Transformation:** The system stacks these features together using `scipy.sparse.hstack`, passing a unified matrix of exactly 5,004 parameters to the classifiers.
5. **Ensemble Assessment:** The core models run calculations, outputting actual prediction scores via a balanced mathematical consensus.
6. **Consumer Telemetry Output:** The UI renders a clean, public-friendly horizontal "Real vs. Fake" percentage distribution bar along with system telemetry details.

---

## 🧠 Incremental Online Training Methodology

Traditional Machine Learning pipelines suffer from **Model Drift**—as news topics shift over time, static models trained months ago lose accuracy. Re-training traditional models requires **Batch Learning**, where the developer must rebuild the entire pipeline from scratch, passing the entire historical dataset back through vectorization and model fitting. This is slow, expensive, and impractical for real-time applications.

### The Online Learning Solution
VerifIQ solves this using **Incremental Online Learning** backed by scikit-learn's `SGDClassifier` and `partial_fit` API:
* **Live Adaptability:** When a user flags a classification error or provides correct label feedback, VerifIQ does not rebuild itself.
* **Gradient Adjustments:** The model computes loss gradient adjustments solely on the new input, updating its stored weights file directly on disk (`models/sgd_online_model.pkl`).
* **Active Feedback Loop:** This feedback loop allows the model to learn new misinformation signatures and vocabulary shifts on-the-fly, mimicking the active learning strategies of state-of-the-art AI systems.

---

## ⚙️ Setup and Installation

### 🚀 Simplified Quick Start (One-Click Launch)
We have provided a unified launcher script to handle virtual environment creation, package installation, and application launching automatically.

Simply run the batch script from your terminal:
```powershell
.\run.bat
```
*(Or simply double-click `run.bat` in Windows File Explorer).*

---

## 🔄 Pipeline Workflow Execution & Retraining

If you ever modify the training datasets or want to retrain the models from scratch, all stages are unified inside the single CLI runner `run_pipeline.py`. Use the following command flags:

### Step 1: Programmatic Data Fetching & Staging
Downloads and shuffles the raw Kaggle dataset:
```powershell
python run_pipeline.py --stage
```

### Step 2: Custom Text Preprocessing
Processes all 44,898 raw articles using our custom string engine:
```powershell
python run_pipeline.py --preprocess
```

### Step 3: Progressive Stream Model Training
Fits the vectorizer and progressively trains the model in memory-efficient increments:
```powershell
python run_pipeline.py --train
```

### Step 4: Run Baseline Pipeline Evaluator
Evaluates the baseline models and saves metrics and confusion matrix heatmaps to `reports/plots/`:
```powershell
python run_pipeline.py --run
# Or run with no arguments:
python run_pipeline.py
```

### Step 5: (Optional) Hyperparameter Grid Sweep
Runs a randomized grid search to find the optimal hyperparameters:
```powershell
python run_pipeline.py --tune
```

### Step 6: (Optional) Error & Leakage Analysis
Inspects misclassified samples and outputs coefficient weights:
```powershell
python run_pipeline.py --analyze
```

### Step 7: (Optional) Quick Integration Verification
Verifies predictions against sample real and fake news articles:
```powershell
python run_pipeline.py --test
```

---

## 🧪 Interactive Live Interfaces

### 🎨 1. Interactive Information Flow Simulator (GitHub Pages)
A mobile-optimized, high-performance HTML5 Canvas simulation designed to model information propagation, particle physics collisions, and news classification.
* **Live Site:** [sarabjeet-singh-grimreaper.github.io/Fake-news-Detector](https://Sarabjeet-singh-Grimreaper.github.io/Fake-news-Detector/)
* **Key Enhancements:**
  * **Retina/High-DPI Sharpness:** Accounts for `window.devicePixelRatio` to prevent blurriness on iPhones, tablets, and high-DPI desktop screens.
  * **Cross-Platform Responsiveness:** Auto-scales viewport and adjusts coordinates dynamically on window resize or rotation.
  * **Touch Optimizations:** Combines unified modern `PointerEvents` and applies `touch-action: none;` to suppress system gesture collisions (pinch-to-zoom, page-bounce).
  * **Action Workflow Deployment:** Auto-deploys static elements recursively through [.github/workflows/deploy.yml](.github/workflows/deploy.yml).

### 🛡️ 2. Custom Classification Portal (Streamlit)
To interact with our trained machine learning models, enter custom article texts, or parse live URLs:
* **Live App Portal:** [fake-news-detector-sarabjeet2448060.streamlit.app](https://fake-news-detector-sarabjeet2448060.streamlit.app/)
* **To Run Locally:**
  ```powershell
  streamlit run app.py
  ```
  *(Or execute `.\run.bat` which handles setup, virtual environment, and launches the app automatically).*

### 📰 3. Auto-Refreshing RSS Feed Classifier (Flask)
To start the live Google News RSS feed monitoring dashboard (evaluating new global headlines every 10 seconds) on your machine:
```powershell
python src/realtime_dashboard.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser to view the live ingestion feed.

---

## 🔍 Critical Senior ML Insights: Feature Leakage
During **Error Analysis** (`python error_analysis.py`), we uncovered a significant data artifact known as **Feature Leakage**:
1. **The Reuters Dateline Stamp:** In the dataset, authentic articles contain a publisher stamp (e.g. `WASHINGTON (Reuters) - ...`). Logistic Regression weights indicate that the word **`reuters`** is the single strongest indicator of Real news (coefficient weight **`+58.68`**).
2. **Hyperlinks and Social Attributions:** Fake news articles contain footer tags like *"read more via [website]"* or image attributions. The word **`via`** became the single strongest indicator of Fake news (coefficient weight **`-26.64`**).

*Note: For an academic paper, discussing this leakage and model vulnerability to publisher tags adds substantial technical depth and demonstrates rigorous machine learning practices.*
