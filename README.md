# 🛡️ Fake News Detection & AI Verification System

A multi-layered, explainable Fake News Detection and Verification web application that combines classical Machine Learning (TF-IDF + LinearSVC), Model Explainability, AI-Powered Cross-Source Search, URL Web Article Extraction, Established Fact-Checking API Integration, and an evidence-based **Decision Engine** (`REAL`, `FAKE`, `UNCERTAIN`).

---

## 🌟 Key Features

- **⚖️ Evidence-Based Final Assessment Engine**: Synthesizes classical ML prediction with AI cross-source verification, official domain checks (`.gov`/`.org`), established news outlet coverage (Indian & International), and published fact-checks into a unified verdict (`REAL`, `FAKE`, `UNCERTAIN`).
- **🎯 Deterministic Truth Matrix & Strict Claim Verification**:
  - **Truth Table Synthesis**:
    - ML REAL + AI REAL → **REAL**
    - ML REAL + AI FAKE → **FAKE**
    - ML FAKE + AI REAL → **REAL**
    - ML FAKE + AI FAKE → **FAKE**
  - **Strict Claim Priority**: If *any* major extracted claim is verified **FALSE** or remains **UNVERIFIED**, the AI overall prediction strictly defaults to **FAKE**. AI prediction requires **ALL** major claims to be verified **TRUE** by evidence to return **REAL**.
- **🤖 Classical ML Classification**: Uses a trained `LinearSVC` model with 50,000 TF-IDF unigram & bigram features to evaluate statistical vocabulary and writing style patterns.
- **💡 Model Explainability & Pandas Linguistic Table**: Highlights top words influencing the ML model (with TF-IDF scores, model weights, and directional contribution) rendered cleanly via Pandas DataFrames (`st.table`), alongside automated clickbait pattern detection.
- **📰 Reputable News Organization Verification**: Checks claim reporting across established Indian national news outlets (*Aaj Tak*, *India Today*, *Zee News*, *NDTV*, *News18*, *The Hindu*, *The Indian Express*, *Hindustan Times*, *Times of India*, *ANI*, *PTI*) and international outlets (*Reuters*, *AP*, *BBC*, *AFP*, *Al Jazeera*, *CNN*, *The Guardian*).
- **🔎 Published Fact-Check Integration**: Integrates Google Fact Check Tools API and direct searches across Snopes, PolitiFact, Reuters Fact Check, AP Fact Check, and FactCheck.org with normalized rating classifications.
- **🔗 Safe New-Tab Link Handling & Web Extractor**: Automatically extracts main article text from URLs using `trafilatura`. All external source links open safely in a **new browser tab** (`target="_blank" rel="noopener noreferrer"`) with strict URL format validation to prevent broken or fake links.
- **🎨 Polished Streamlit UI**: 100% clean UI rendering with zero raw HTML tag leakage, custom glassmorphism cards, status badges, expanders, and metric counters.

---

## 🏗️ Project Architecture

```text
fake-news-detection/
│
├── app.py                      # Step 10 Streamlit Web Application (Frontend & System Pipeline)
│
├── notebooks/                  # Step-by-Step Task Notebooks (Task 1 - Task 9)
│   ├── Task1_Dataset_Collection_Analysis.ipynb
│   ├── Task2_NLP_Preprocessing.ipynb
│   ├── Task3_TF_IDF_Feature_Extraction.ipynb
│   ├── Task4_ML_Model_Training.ipynb
│   ├── Step5_Prediction_Pipeline.ipynb
│   ├── Task6_Explainability.ipynb
│   ├── Task7_AI_News_Verification.ipynb
│   ├── Task8_URL_Article_Extraction.ipynb
│   └── Task9_Fact_Checking_Integration.ipynb
│
├── models/                     # Saved Machine Learning Artifacts
│   ├── best_model.pkl          # Trained LinearSVC Classifier
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer (50,000 features)
│
├── src/                        # Reusable Python Pipeline Modules
│   ├── prediction.py           # Step 5: Prediction & Decision Margin Confidence
│   ├── explainability.py       # Step 6: Feature Contribution & Clickbait Scanner
│   ├── ai_verification.py      # Step 7: AI Claim Extraction, Web Search & Strict Claim Rules
│   ├── web_search.py           # Step 7: DuckDuckGo Live Search & Social Handle Check
│   ├── source_analysis.py      # Step 7: Source Tiering (Tier 1/2/3) & Outlet Classification
│   ├── article_extractor.py    # Step 8: Trafilatura URL Extraction & Content Cleaning
│   ├── fact_checker.py         # Step 9: Google Fact Check API & Rating Normalizer
│   └── decision_engine.py      # Step 10: Multi-Dimensional Evidence Decision Engine
│
├── data/                       # Datasets
│   ├── Fake.csv / True.csv
│   └── news_dataset.csv
├── .env.example                # Template for API keys
├── .gitignore                  # Git exclusions (excludes .env)
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Frontend UI**: Streamlit 1.30+
- **Machine Learning**: Scikit-Learn (`LinearSVC`), NumPy, SciPy, Joblib
- **NLP & Extraction**: NLTK, BeautifulSoup4, Trafilatura
- **AI & Live Search**: Google Gemini API (`google-genai`), DuckDuckGo Search (`duckduckgo_search`), Google Fact Check Tools API
- **Decision System**: Evidence Rule-Synthesis & Deterministic Decision Engine (`src/decision_engine.py`)

---

## 🧪 AI & Decision Engine Rules

### 1. Overall Decision Truth Table Matrix

| ML Model Prediction | AI Verification Prediction | Final Overall Result | Notes / Reasoning |
| :---: | :---: | :---: | :--- |
| **REAL** | **REAL** | **REAL** | Both statistical ML and external AI evidence confirm legitimacy. |
| **REAL** | **FAKE** | **FAKE** | External evidence or false/unverified claims override statistical style model. |
| **FAKE** | **REAL** | **REAL** | External cross-source evidence overrides vocabulary shift / false ML alarm. |
| **FAKE** | **FAKE** | **FAKE** | Both statistical ML and external evidence indicate fake news. |

### 2. Strict AI Claim Verification Priority Order

After extracting and checking major factual claims within an article:

1. **If ANY major claim is verified FALSE** → AI Prediction = **FAKE**
2. **Else if ANY major claim is UNVERIFIED** → AI Prediction = **FAKE**
3. **Else if ALL major claims are verified TRUE** → AI Prediction = **REAL**

*Example:* `TRUE + TRUE + TRUE + UNVERIFIED` → **FAKE**

---

## ⚡ Quick Start & Setup Instructions

### 1. Clone & Set Up Directory

```bash
cd fake-news-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your API keys in `.env` (optional, fallback web search logic activates automatically if keys are omitted):

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key_here
```

### 4. Launch Streamlit Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ⚠️ Disclaimer & Ethical Guidelines

This system provides an automated multi-dimensional news assessment based on machine learning vocabulary patterns, publicly available web news reporting, AI cross-source reasoning, and published fact-checks. It does **not** claim 100% infallibility and never displays `100% REAL` or `100% FAKE`. Users should always consult primary government documents and official sources when verifying critical real-world news.
