# Veritas AI: AI-Powered Fake News Detection Using Text Classification

Welcome to the documentation for the project **"AI-Powered Fake News Detection Using Text Classification"** developed for the **Summer Internship Program in AI & ML 2026**.

This repository implements a complete end-to-end Machine Learning pipeline to classify news articles as **Real** or **Fake**.

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
├── run_pipeline.py         # Unified pipeline CLI orchestrator
└── requirements.txt        # Python package dependencies
```

---

## ⚙️ Setup and Installation

### 1. Initialize Virtual Environment
Open your terminal (PowerShell/Command Prompt) in the project directory:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```

### 2. Install Required Dependencies
Install the required scientific packages:
```powershell
pip install -r requirements.txt
```

### 3. 🚀 Quick Start (Detect Fake News Now)
The machine learning models are **already trained** and saved in your `models/` folder. You do not need to train them again to start using the system.

To launch the interactive website immediately and begin verifying news articles, run:
```powershell
streamlit run app.py
```

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

## 🧪 Real-time Interfaces (Frontend & Ingestion Stream)

### 1. Interactive Custom Portal (Streamlit)
To launch the modern dark-themed interactive SaaS verification portal in your browser:
```powershell
streamlit run app.py
```
This application allows you to paste any news article, scrape URL contents, and cross-evaluate authenticity verdicts dynamically with optimized visual controls.

### 2. Auto-Refreshing RSS Feed Classifier (Flask)
To start the live Google News RSS feed monitoring dashboard (evaluating new global headlines every 10 seconds):
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
