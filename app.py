"""
=============================================================================
  FAKE NEWS DETECTOR — STREAMLIT FRONTEND & COMPLETE SYSTEM INTEGRATION
=============================================================================

This is the main web application for the Fake News Detection project.
It integrates all previous pipeline components (Steps 1–9) with Step 10 Decision Engine:

  - Step 5: ML Model Prediction & Confidence Score
  - Step 6: Model Explainability & Clickbait Language Scanner
  - Step 7: AI-Powered Cross-Source Verification & Claim Analysis
  - Step 8: URL / Web Article Extraction & Content Cleaning
  - Step 9: Fact-Checking API Integration & Rating Normalization
  - Step 10: Multi-Dimensional Evidence Decision Engine (FINAL ASSESSMENT)

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import re
import sys
import time
import warnings

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Ensure current project directory and src/ are in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, 'src')
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Load environment variables
load_dotenv(os.path.join(_THIS_DIR, '.env'))

# Import pipeline backend modules
import prediction as pred
import explainability as exp
import ai_verification as av
import article_extractor as ae
import fact_checker as fc
import decision_engine as de


# ---------------------------------------------------------------------------
# Frontend Rendering Helpers (Safe Badges & New-Tab Links - Requirements 1,3,4,5,12,15)
# ---------------------------------------------------------------------------

def is_valid_url(url):
    """
    Validates URL string to prevent broken, fake, or missing links.
    Filters out text source names (e.g. 'Independent Reporting Index', 'Reuters'),
    placeholders ('None', 'null', 'undefined', '#', 'N/A'), and invalid strings with spaces.
    """
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if u in ('', '#', 'None', 'null', 'undefined', 'N/A', 'example.com'):
        return False
    
    # Fake URL check: domain part cannot contain spaces or %20
    host_part = u.split('://')[-1].split('/')[0]
    if ' ' in host_part or '%20' in host_part.lower():
        return False
        
    if ' ' in u:
        return False
        
    if u.startswith(('http://', 'https://')):
        rest = u.split('://', 1)[1].strip()
        domain = rest.split('/')[0]
        return bool(domain and '.' in domain and not domain.startswith('.'))
        
    if '.' in u and not u.startswith('.'):
        domain = u.split('/')[0]
        return bool(domain and '.' in domain)
        
    return False


def render_external_link(label, url):
    """
    Renders an external link that opens in a NEW browser tab (target="_blank").
    Validates URL before rendering to prevent broken or fake links.
    """
    if not is_valid_url(url):
        st.caption("ℹ️ Source link unavailable")
        return

    clean_url = url.strip()
    if not clean_url.startswith(('http://', 'https://')):
        clean_url = 'https://' + clean_url

    clean_label = label.strip() if (label and isinstance(label, str) and label.strip()) else "View Source"

    html = f"""<a href="{clean_url}" target="_blank" rel="noopener noreferrer" style="display: inline-flex; align-items: center; gap: 4px; background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); padding: 4px 12px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 500; margin-top: 4px; margin-bottom: 4px;">🔗 {clean_label} ↗</a>"""
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(status):
    """
    Renders a visual status badge without leaking raw HTML tags into markdown.
    """
    if not status or not isinstance(status, str):
        st.markdown("<span style='background: #334155; color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: inline-block;'>UNKNOWN STATUS</span>", unsafe_allow_html=True)
        return

    st_clean = status.strip().upper()

    if st_clean in ('SUPPORTED', 'REAL', 'HIGHLY_SUPPORTED', 'CONFIRMED', 'TRUE'):
        bg_color = "rgba(16, 185, 129, 0.2)"
        border_color = "#10b981"
        text_color = "#34d399"
        icon = "✓"
    elif st_clean in ('CONTRADICTED', 'FAKE', 'FALSE', 'DEBUNKED', 'REFUTED', 'DISPROVEN'):
        bg_color = "rgba(239, 68, 68, 0.2)"
        border_color = "#ef4444"
        text_color = "#f87171"
        icon = "✕"
    elif st_clean in ('PARTIALLY_SUPPORTED', 'PARTIALLY_TRUE', 'MIXED'):
        bg_color = "rgba(245, 158, 11, 0.2)"
        border_color = "#f59e0b"
        text_color = "#fbbf24"
        icon = "⚠"
    else:  # UNVERIFIED, UNCERTAIN, UNKNOWN
        bg_color = "rgba(148, 163, 184, 0.2)"
        border_color = "#64748b"
        text_color = "#cbd5e1"
        icon = "❓"

    html = f"""<span style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; display: inline-block;">{icon} {st_clean}</span>"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector & AI Verification System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, dark theme, violet accents, and glass cards
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background & headers */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #311b92 50%, #4c1d95 100%);
        border-radius: 16px;
        padding: 32px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.25);
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    .header-title {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .header-subtitle {
        font-size: 1.05rem;
        color: #ddd6fe;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Card Containers */
    .glass-card {
        background: #1e293b;
        border-radius: 14px;
        padding: 22px 26px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* FINAL ASSESSMENT HERO CARDS */
    .hero-card-real {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
        margin-bottom: 20px;
    }

    .hero-card-fake {
        background: linear-gradient(135deg, #881337 0%, #4c0519 100%);
        border: 2px solid #f43f5e;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(244, 63, 94, 0.2);
        margin-bottom: 20px;
    }

    .hero-card-uncertain {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.2);
        margin-bottom: 20px;
    }

    .hero-title-real {
        font-size: 3.2rem;
        font-weight: 800;
        color: #34d399;
        text-transform: uppercase;
        margin: 8px 0;
        letter-spacing: 0.05em;
    }

    .hero-title-fake {
        font-size: 3.2rem;
        font-weight: 800;
        color: #fb7185;
        text-transform: uppercase;
        margin: 8px 0;
        letter-spacing: 0.05em;
    }

    .hero-title-uncertain {
        font-size: 3.2rem;
        font-weight: 800;
        color: #fbbf24;
        text-transform: uppercase;
        margin: 8px 0;
        letter-spacing: 0.05em;
    }

    .hero-substatus {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .badge-supported { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-contradicted { background-color: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid #f43f5e; }
    .badge-partially { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-unverified { background-color: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid #64748b; }

    .breaking-news-box {
        background-color: rgba(59, 130, 246, 0.12);
        border-left: 5px solid #3b82f6;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #dbeafe;
    }

    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }

    .styled-table th {
        background-color: #334155;
        color: #f8fafc;
        text-align: left;
        padding: 12px 16px;
        font-size: 0.9rem;
    }

    .styled-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #334155;
        font-size: 0.92rem;
    }

    /* Streamlit UI elements override */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        color: #ffffff;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        transform: translateY(-1px);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 1: Session State Initialization & Pipeline Execution
# ---------------------------------------------------------------------------

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'last_input' not in st.session_state:
    st.session_state.last_input = None


def run_full_pipeline(input_text_or_url, is_url=False, article_title=""):
    """
    Executes the complete end-to-end detection pipeline (Steps 5 -> 10)
    with real progress feedback.
    """
    status_container = st.status("Analyzing news article & evaluating evidence...", expanded=True)

    try:
        # Step 1: Article Extraction if URL
        if is_url:
            status_container.write("🔍 **Step 1/6:** Extracting webpage content & metadata...")
            extracted = ae.extract_article(input_text_or_url)
            if extracted.get('extraction_status') == 'FAILED' or 'error' in extracted:
                status_container.update(label="❌ Article extraction failed.", state="error")
                return {
                    'error': extracted.get('error', 'Article extraction failed.'),
                    'error_type': extracted.get('error_type', 'EXTRACTION_FAILED'),
                    'fallback_suggested': extracted.get('fallback_suggested', True),
                    'extraction_status': 'FAILED'
                }

            raw_text = extracted['text']
            title = extracted.get('title') or "Web News Article"
            extractor_used = extracted.get('extractor_used', 'trafilatura')
            word_count = extracted.get('word_count', len(raw_text.split()))

            status_container.write(f"✓ **Step 1/6:** Article extracted successfully ({word_count} words detected using {extractor_used}).")
            art_metadata = {
                'title': title,
                'author': extracted.get('author') or 'Not specified',
                'publication_date': extracted.get('publication_date') or 'Not specified',
                'source': extracted.get('source') or 'Web Domain',
                'url': extracted.get('url') or input_text_or_url,
                'word_count': word_count,
                'input_mode': f'URL Extraction ({extractor_used})',
                'extraction_status': 'SUCCESS',
                'extractor_used': extractor_used
            }
        else:
            status_container.write("📝 **Step 1/6:** Validating article text input...")
            raw_text = input_text_or_url.strip()
            word_count = len(raw_text.split())

            if word_count < 25:
                status_container.update(label="❌ Insufficient article text.", state="error")
                return {'error': 'The article contains insufficient text for reliable analysis (minimum 25 words required).'}

            title = article_title.strip() if article_title else "Pasted News Article"
            art_metadata = {
                'title': title,
                'author': 'Direct User Input',
                'publication_date': 'Not specified',
                'source': 'Pasted Article',
                'url': None,
                'word_count': word_count,
                'input_mode': 'Pasted Article',
                'extraction_status': 'N/A'
            }

        # Step 2: ML Model Prediction (Step 5)
        status_container.write("🤖 **Step 2/6:** Running statistical feature analysis...")
        ml_res = pred.predict_news(raw_text, title=title)
        if 'error' in ml_res:
            status_container.update(label="❌ ML Model failure.", state="error")
            return {'error': f"ML Model error: {ml_res['error']}"}

        # Step 3: Explainability (Step 6)
        status_container.write("💡 **Step 3/6:** Computing linguistic feature patterns & clickbait scanner...")
        exp_res = exp.get_explanation(raw_text, top_n=10)

        # Step 4: AI Cross-Source Verification (Step 7)
        status_container.write("🌐 **Step 4/6:** Extracting claims & searching external news/official sources...")
        ai_res = av.verify_article(raw_text)

        # Step 5: Fact-Checking Integration (Step 9)
        status_container.write("🔎 **Step 5/6:** Querying fact-checking databases & evaluating ratings...")
        step7_claims = ai_res.get('claim_verifications', []) if isinstance(ai_res, dict) else []
        try:
            fact_res = fc.fact_check_article(raw_text, claims=step7_claims)
        except Exception as exc:
            logger.warning(f"Fact checking module exception: {exc}")
            fact_res = {
                'status': 'UNAVAILABLE',
                'error': 'Fact-checking service temporarily unavailable.',
                'results': [],
                'fact_checks_found_total': 0,
                'overall_evidence_status': 'UNAVAILABLE'
            }

        # Step 6: Multi-Dimensional Evidence Decision Engine (Step 10)
        status_container.write("⚖️ **Step 6/6:** Synthesizing evidence into Final Assessment...")
        final_dec = de.make_final_decision(ml_res, ai_res, fact_res, article_metadata=art_metadata)

        status_container.update(label="✅ News Analysis & Verification Complete!", state="complete", expanded=False)

        ai_dict = ai_res if isinstance(ai_res, dict) else {}
        ml_dict = ml_res if isinstance(ml_res, dict) else {}
        exp_dict = exp_res if isinstance(exp_res, dict) else {}
        v_summary = ai_dict.get('verification_summary') or {}
        if not isinstance(v_summary, dict):
            v_summary = {}

        master_report = {
            'article': art_metadata or {},
            'prediction': {
                'label': ml_dict.get('prediction', 'UNKNOWN'),
                'confidence': ml_dict.get('confidence', 0.0),
                'confidence_type': ml_dict.get('confidence_type', ''),
                'model_used': ml_dict.get('model_used', '')
            },
            'final_decision': final_dec or {},
            'explainability': {
                'influential_features': (exp_dict.get('influential_features') or [])[:6],
                'features_for_real': exp_dict.get('features_for_real') or [],
                'features_for_fake': exp_dict.get('features_for_fake') or [],
                'suspicious_language': exp_dict.get('suspicious_language') or []
            },
            'ai_verification': {
                'overall_status': v_summary.get('overall_status', 'UNVERIFIED'),
                'ai_assessment': ai_dict.get('ai_assessment', ''),
                'ai_prediction': ai_dict.get('ai_prediction', 'UNCERTAIN'),
                'claims': ai_dict.get('claim_verifications') or [],
                'sources_analysis': ai_dict.get('sources_analysis') or {},
                'official_sources': ai_dict.get('official_sources') or [],
                'social_sources': ai_dict.get('social_sources') or [],
                'reputable_news_sources': ai_dict.get('reputable_news_sources') or [],
                'reputable_news_confirmations_count': v_summary.get('reputable_news_confirmations_count', 0)
            },
            'fact_checking': fact_res or {}
        }

        return master_report

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        status_container.update(label="❌ Pipeline execution error.", state="error")
        return {'error': f"An unexpected error occurred during processing: {str(e)}"}


# ---------------------------------------------------------------------------
# Section 2: Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🛡️ FAKE NEWS DETECTOR")
st.sidebar.markdown("Integrated News Verification Engine")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🔍 Analyze News", "ℹ️ How It Works", "📌 About System"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
st.sidebar.markdown(f"• **Verification Engine**: `Active`")
st.sidebar.markdown(f"• **Decision System**: `Integrated Evidence Engine`")
st.sidebar.markdown(f"• **AI Cross-Source Search**: `{'Online (Gemini)' if os.environ.get('GEMINI_API_KEY') else 'Active (Web Fallback)'}`")
st.sidebar.markdown(f"• **Fact Check Database**: `{'Configured' if os.environ.get('GOOGLE_FACT_CHECK_API_KEY') else 'Active (Web Search Fallback)'}`")


# ---------------------------------------------------------------------------
# Section 3: Navigation Page 1 — Analyze News
# ---------------------------------------------------------------------------

if nav_choice == "🔍 Analyze News":

    # Header Banner
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">
            <span>🛡️ Fake News Detector</span>
        </div>
        <p class="header-subtitle">
            Evidence-Based News Assessment combining Cross-Source Search, Fact-Checking Databases, and Statistical Feature Analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Input Section with Tabs
    tab_paste, tab_url = st.tabs(["📰 Paste Article Text", "🔗 Analyze News URL"])

    input_text = ""
    input_url = ""
    article_title_input = ""
    trigger_analyze = False
    is_url_mode = False

    with tab_paste:
        st.markdown("##### Option A: Paste Raw Article Text")
        article_title_input = st.text_input("Article Title (Optional)", placeholder="e.g., Federal Reserve Raises Interest Rates")
        input_text = st.text_area("Article Text", height=200, placeholder="Paste full news article content here...")
        if st.button("🚀 Analyze News Article", key="btn_paste"):
            if not input_text.strip():
                st.error("Please enter a news article to analyze.")
            else:
                trigger_analyze = True
                is_url_mode = False

    with tab_url:
        st.markdown("##### Option B: Enter News Article URL")
        input_url = st.text_input("News Article URL", placeholder="https://www.reuters.com/business/finance/example-article")
        if st.button("🌐 Analyze URL", key="btn_url"):
            val_res = ae.validate_url(input_url)
            if not val_res['valid']:
                st.error(val_res['error'])
            else:
                trigger_analyze = True
                is_url_mode = True

    # Execute Analysis if triggered
    if trigger_analyze:
        target_input = input_url if is_url_mode else input_text
        res = run_full_pipeline(target_input, is_url=is_url_mode, article_title=article_title_input)
        st.session_state.analysis_result = res
        st.session_state.last_input = target_input

    # -----------------------------------------------------------------------
    # Render Results if available in Session State
    # -----------------------------------------------------------------------
    res = st.session_state.analysis_result

    if res:
        if 'error' in res or res.get('extraction_status') == 'FAILED':
            st.error(f"⚠️ {res.get('error', 'Extraction failed.')}")
            if res.get('fallback_suggested'):
                st.markdown("---")
                st.warning("⚠️ **Automatic Extraction Unsuccessful**\n\nWe couldn't automatically retrieve enough article content from this URL. Please paste the article text manually below to analyze:")
                fallback_paste = st.text_area("Paste Article Text Manually", height=200, key="manual_fallback_text_area", placeholder="Paste the news article text here...")
                fallback_title = st.text_input("Article Title (Optional)", key="manual_fallback_title_input", placeholder="Article Title...")
                if st.button("🚀 Analyze Pasted Article", key="btn_manual_fallback"):
                    if not fallback_paste.strip() or len(fallback_paste.strip().split()) < 25:
                        st.error("Please paste at least 25 words of article text to perform analysis.")
                    else:
                        fb_res = run_full_pipeline(fallback_paste.strip(), is_url=False, article_title=fallback_title)
                        st.session_state.analysis_result = fb_res
                        st.rerun()
        else:
            art = res['article']
            exp_data = res['explainability']
            ai_data = res['ai_verification']
            fact_data = res['fact_checking']
            final_dec = res['final_decision']

            st.markdown("---")

            # ---------------------------------------------------------------
            # 1. FINAL ASSESSMENT HERO CARD (Clean Single Verdict - Section 15 Prompt)
            # ---------------------------------------------------------------
            f_label = final_dec.get('final_label') or final_dec.get('final_decision') or 'UNCERTAIN'
            f_level = final_dec.get('evidence_level') or final_dec.get('sub_status') or 'LIMITED EVIDENCE'
            
            if f_label == 'REAL':
                hero_card_cls = "hero-card-real"
                hero_title_cls = "hero-title-real"
                hero_icon = "✅"
            elif f_label == 'FAKE':
                hero_card_cls = "hero-card-fake"
                hero_title_cls = "hero-title-fake"
                hero_icon = "🚨"
            else:
                hero_card_cls = "hero-card-uncertain"
                hero_title_cls = "hero-title-uncertain"
                hero_icon = "⚠️"

            st.markdown(f"""
            <div class="{hero_card_cls}">
                <p style="font-size: 1.15rem; text-transform: uppercase; letter-spacing: 0.15em; color: #cbd5e1; margin-bottom: 2px;">
                    FINAL ASSESSMENT
                </p>
                <h1 class="{hero_title_cls}">{hero_icon} {f_label}</h1>
                <p class="hero-substatus">{f_level}</p>
            </div>
            """, unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # 2. WHY? (FINAL REASON) (Section 13 & 16 Prompt Layout)
            # ---------------------------------------------------------------
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #a78bfa; margin-top: 0; margin-bottom: 8px;">❓ Reason</h4>
                <p style="font-size: 1.05rem; line-height: 1.6; color: #f8fafc; margin: 0;">
                    {final_dec['reason']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # 3. BREAKING NEWS / DEVELOPING EVENT ALERT (Section 10 Prompt Layout)
            # ---------------------------------------------------------------
            if final_dec.get('is_breaking_news'):
                st.markdown("""
                <div class="breaking-news-box">
                    <strong>⚡ LIMITED EVIDENCE (DEVELOPING EVENT):</strong><br>
                    This appears to be a newly developing event with limited initial external reporting. There is currently not enough independent evidence for a confident conclusion.
                </div>
                """, unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # 4. ARTICLE INFORMATION (Section 14 Prompt Layout)
            # ---------------------------------------------------------------
            st.markdown("### 📄 ARTICLE INFORMATION")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.write(f"**Title:** {art.get('title', 'N/A')}")
                st.write(f"**Source:** `{art.get('source', 'N/A')}`")
                st.write(f"**Input Type:** {art.get('input_mode', 'Text')}")
            with col_a2:
                st.write(f"**Publication Date:** {art.get('publication_date', 'N/A')}")
                st.write(f"**Word Count:** {art.get('word_count', 0)} words")
                if art.get('url') and art['url'] != 'N/A':
                    st.write("**Original URL:**")
                    render_external_link("View Original Article", art['url'])

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # 5. EVIDENCE SUMMARY OVERVIEW (Section 16 Prompt Layout)
            # ---------------------------------------------------------------
            es = final_dec.get('evidence_summary', {})
            st.markdown("### 📊 EVIDENCE SUMMARY")
            col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
            with col_e1:
                st.metric("Official Sources", es.get('official_sources', 0))
            with col_e2:
                st.metric("Independent Outlets", es.get('independent_sources', 0))
            with col_e3:
                st.metric("Fact Checks Found", es.get('fact_checks', 0))
            with col_e4:
                st.metric("Supporting Evidence", es.get('supporting_sources', 0))
            with col_e5:
                st.metric("Contradicting Evidence", es.get('contradicting_sources', 0))

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # 6. DETAILED VERIFICATION SECTIONS & TABS (Sections 15-18 Prompt Layout)
            # ---------------------------------------------------------------
            r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
                "🌐 AI Verification & Claims",
                "🔎 Fact Checking",
                "🏛️ Sources & Evidence",
                "💡 Linguistic Insights",
                "📌 System Disclaimer"
            ])

            # ── Tab 1: AI Verification & Claim Analysis ──
            with r_tab1:
                st.markdown("### 🌐 AI Cross-Source Verification")
                st.write("**Cross-Source Search Status:**")
                render_status_badge(ai_data.get('overall_status', 'UNVERIFIED'))
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"**Evidence Summary:** {ai_data.get('ai_assessment', 'No assessment summary.')}")

                st.markdown("#### Factual Claim Analysis")
                claims_list = ai_data.get('claims', []) or ai_data.get('claim_verifications', [])
                if not claims_list:
                    st.write("No verifiable factual claims extracted for this article.")
                else:
                    for i, c in enumerate(claims_list, 1):
                        c_status = c.get('status', 'UNVERIFIED') if isinstance(c, dict) else 'UNVERIFIED'
                        c_text = c.get('claim', 'Factual Claim') if isinstance(c, dict) else str(c)
                        c_sum = c.get('summary', 'Claim evaluation details.') if isinstance(c, dict) else ''

                        with st.expander(f"CLAIM {i}: \"{c_text[:80]}...\"", expanded=(i == 1)):
                            st.markdown(f"**Full Claim:** {c_text}")
                            st.write("**Status:**")
                            render_status_badge(c_status)
                            st.markdown(f"**Evidence Summary:** {c_sum}")

                            if isinstance(c, dict) and c.get('supporting_sources'):
                                st.markdown("**Supporting Sources:**")
                                for s in c['supporting_sources']:
                                    render_external_link("View Supporting Source", s)
                            if isinstance(c, dict) and c.get('contradicting_sources'):
                                st.markdown("**Contradicting Sources:**")
                                for s in c['contradicting_sources']:
                                    render_external_link("View Contradicting Source", s)

            # ── Tab 2: Fact Checking Section ──
            with r_tab2:
                st.markdown("### 🔎 Fact-Checking Integration")
                if fact_data.get('status') == 'UNAVAILABLE':
                    st.warning("⚠️ **Fact-checking service temporarily unavailable.**\n\nFinal assessment was generated using remaining available evidence.")
                else:
                    fc_cnt = fact_data.get('fact_checks_found_total', 0)
                    st.write(f"**Claims Checked:** `{fact_data.get('claims_checked', 0)}` | **Fact Checks Found:** `{fc_cnt}`")
                    st.write(f"**Overall Fact-Check Status:** `{fact_data.get('overall_evidence_status', 'NO_FACT_CHECK_FOUND')}`")
                    st.write(fact_data.get('summary', 'No matching published fact-checks found.'))

                    fc_results = fact_data.get('results', [])
                    if not fc_results or fc_cnt == 0:
                        st.info("ℹ️ **No published fact-check reviews found in established databases.**")
                    else:
                        for i, fr in enumerate(fc_results, 1):
                            with st.expander(f"Fact Check Analysis {i}: \"{fr.get('claim', '')[:70]}...\"", expanded=True):
                                st.markdown(f"**Claim Evaluated:** \"{fr.get('claim', '')}\"")
                                st.markdown(f"**Status:** `{fr.get('status', 'UNVERIFIED')}`")
                                st.markdown(f"**Reason:** {fr.get('reason', '')}")
                                
                                if fr.get('fact_checks'):
                                    st.markdown("##### Published Reviews Found:")
                                    for item in fr['fact_checks']:
                                        st.markdown(f"""
                                        - **Publisher:** {item.get('publisher', 'Fact Checker')} (`{item.get('source_quality', 'HIGH')}` Quality)
                                        - **Original Rating:** `{item.get('original_rating', 'FALSE')}` -> **Normalized:** `{item.get('normalized_rating', 'FALSE')}`
                                        - **Review Title:** {item.get('title', 'Fact Check Review')}
                                        """)
                                        render_external_link(f"View Fact Check on {item.get('publisher', 'Fact Checker')}", item.get('url'))

            # ── Tab 3: Official & External Sources ──
            with r_tab3:
                st.markdown("### 🏛️ OFFICIAL / PRIMARY SOURCES")
                off_sources = final_dec.get('official_sources_list', [])
                if off_sources:
                    st.success("✅ **Primary / Official Domain (.gov / .org) Confirmation Found:**")
                    for off in off_sources:
                        st.markdown(f"✓ **{off.get('source', 'Official Source')}** — {off.get('title', 'Announcement')}")
                        render_external_link(f"Open Official Source: {off.get('source', 'Gov Source')}", off.get('url'))
                else:
                    st.info("ℹ️ No direct primary government website (.gov / .org) confirmation identified.")

                st.markdown("---")

                # ── Reputable News Confirmations panel ──
                st.markdown("### 📰 REPUTABLE NEWS CONFIRMATIONS")
                rep_sources = ai_data.get('reputable_news_sources', []) or ai_data.get('reputable_sources', [])
                rep_count = ai_data.get('reputable_news_confirmations_count', 0) or len(rep_sources)
                if rep_sources:
                    st.success(
                        f"✅ **{rep_count} established news organization(s) independently reported on the same event** "
                        f"(Indian national or international outlets):"
                    )
                    for rep in rep_sources:
                        src_label = rep.get('publisher') or rep.get('source') or rep.get('domain') or 'Established Outlet'
                        title_label = rep.get('headline') or rep.get('title') or 'News Report'
                        url_label = rep.get('url', '')
                        snippet = rep.get('snippet', '')
                        st.markdown(f"- 📰 **{src_label}**: {title_label}")
                        if snippet:
                            st.caption(f"  › {snippet[:160]}...")
                        render_external_link(f"Read Source on {src_label}", url_label)
                else:
                    st.info(
                        "ℹ️ No established reputable news outlet coverage was found in search results for this article."
                    )

                st.markdown("---")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("### 🟢 SUPPORTING EVIDENCE")
                    sup_list = final_dec.get('supporting_evidence', [])
                    if sup_list:
                        for s_item in sup_list:
                            st.write(f"- **Claim:** \"{s_item.get('claim', '')[:60]}...\"")
                            st.write(f"  **Detail:** {s_item.get('detail', '')}")
                            src_val = s_item.get('url') or s_item.get('source') or ''
                            if is_valid_url(src_val):
                                render_external_link("Read Supporting Evidence", src_val)
                            elif src_val:
                                st.caption(f"  ℹ️ **Source Outlet:** {src_val}")
                    else:
                        st.write("No explicit supporting external web links recorded.")

                with col_s2:
                    st.markdown("### 🔴 CONTRADICTING EVIDENCE")
                    con_list = final_dec.get('contradicting_evidence', [])
                    if con_list:
                        for c_item in con_list:
                            st.write(f"- **Claim:** \"{c_item.get('claim', '')[:60]}...\"")
                            st.write(f"  **Detail:** {c_item.get('detail', '')}")
                            src_val = c_item.get('url') or c_item.get('source') or ''
                            if is_valid_url(src_val):
                                render_external_link("Read Contradicting Evidence", src_val)
                            elif src_val:
                                st.caption(f"  ℹ️ **Source Outlet:** {src_val}")
                    else:
                        st.success("✓ No contradictory evidence was detected.")

            # ── Tab 4: Linguistic Insights (Section 6,7,8,9,10 Prompt Requirements) ──
            with r_tab4:
                st.markdown("### 💡 LINGUISTIC INSIGHTS")
                st.write("Key vocabulary patterns influencing automated statistical analysis:")

                feats = exp_data.get('influential_features', [])
                if not feats or not isinstance(feats, list) or len(feats) == 0:
                    st.info("No linguistic insights available for this article.")
                else:
                    data_for_df = []
                    for f in feats[:10]:
                        if isinstance(f, dict):
                            data_for_df.append({
                                "Vocabulary Feature": f.get('word', f.get('feature', '')),
                                "TF-IDF Score": round(float(f.get('tfidf_score', f.get('tfidf', 0.0))), 4),
                                "Model Weight": round(float(f.get('coefficient', f.get('weight', 0.0))), 4),
                                "Direction / Class": f.get('direction', 'UNKNOWN')
                            })

                    if data_for_df:
                        df_ling = pd.DataFrame(data_for_df)
                        st.table(df_ling)
                    else:
                        st.info("No linguistic insights available for this article.")

                susp = exp_data.get('suspicious_language', [])
                if susp and isinstance(susp, list):
                    st.markdown("<br>##### ⚠️ Potentially Suspicious Language Patterns Detected:", unsafe_allow_html=True)
                    for s in susp:
                        if isinstance(s, dict):
                            cat = str(s.get('category', 'LANGUAGE')).upper()
                            desc = s.get('pattern_desc', 'Pattern detected')
                            txt = s.get('matched_text', '')
                            st.warning(f"**[{cat}]** {desc} — *matched: \"{txt}\"*")
                else:
                    st.info("No suspicious language patterns detected.")

            # ── Tab 5: System Disclaimer ──
            with r_tab5:
                st.markdown("### 📌 System Disclaimer & Limitations")
                st.markdown("""
                - **No Claim of Absolute Truth**: This system does NOT display `100% REAL` or `100% FAKE`. It explicitly acknowledges evidence boundaries and search indexing limits.
                - **Integrated Evidence Approach**: The **FINAL ASSESSMENT** is rendered by synthesizing cross-source web search reporting, published fact-checking databases, primary domain verification, and linguistic style patterns.
                - **User Guidance**: Users should always verify critical news using primary government and official documentation.
                """)


# ---------------------------------------------------------------------------
# Section 4: Navigation Page 2 — How It Works
# ---------------------------------------------------------------------------

elif nav_choice == "ℹ️ How It Works":
    st.markdown("# ℹ️ How The Integrated Engine Works")
    st.markdown("---")

    st.markdown("""
    ### 🏗️ Integrated Pipeline Architecture

    ```
    Raw Text / URL Input
            ↓
    URL Article Extraction & Cleaning (Trafilatura / BeautifulSoup)
            ↓
    Statistical Vocabulary & Writing Style Analysis
            ↓
    Linguistic Feature Weight & Clickbait Pattern Scanner
            ↓
    Factual Claim Extraction + Live Web Search + Official Domain Check
            ↓
    Fact-Checking API & Database Search (Google Fact Check / Snopes / PolitiFact)
            ↓
    Multi-Dimensional Evidence Synthesis Engine
            ↓
    FINAL ASSESSMENT (REAL / FAKE / UNCERTAIN)
    ```

    ---

    ### 🔬 Final Assessment Hierarchy

    1. **Strong Credible & Official Evidence**: Primary `.gov`/`.org` domain sources and multiple independent established news outlets.
    2. **Published Fact-Check Reviews**: Verified ratings from Snopes, PolitiFact, Reuters Fact Check, AP, etc.
    3. **AI Cross-Source Claim Analysis**: Verification of extracted factual claims against search indexes.
    4. **Statistical Linguistic Analysis**: Learned vocabulary classification patterns.
    """)


# ---------------------------------------------------------------------------
# Section 5: Navigation Page 3 — About System
# ---------------------------------------------------------------------------

elif nav_choice == "📌 About System":
    st.markdown("# 📌 About The Fake News Detection Project")
    st.markdown("---")

    st.markdown("""
    ### 🎯 Project Overview
    This project provides an **explainable, integrated fake news detection system**.
    It combines statistical Machine Learning with live AI evidence verification, established fact-checking database searches, and a multi-dimensional Decision Engine.

    ### 🛠️ Technology Stack
    - **Language**: Python 3.10+
    - **Frontend UI**: Streamlit 1.57+
    - **ML Framework**: Scikit-Learn (`LinearSVC`), NumPy, SciPy
    - **NLP**: NLTK, BeautifulSoup4, Trafilatura
    - **AI & Web Integration**: Google Gemini API (`google-genai`), DuckDuckGo Search (`duckduckgo_search`), Google Fact Check Tools API
    - **Decision Logic**: Integrated Evidence Engine (`src/decision_engine.py`)

    ---
    ### 🛡️ Ethical Principles & Limitations
    - **Unified Verdict**: Presents a clean, single final assessment based on multi-source evidence.
    - **API Security**: All API keys are loaded via `.env` and strictly excluded from Git repositories.
    - **Transparency**: Clearly presents verifiable external source evidence and published fact-checks.
    """)
