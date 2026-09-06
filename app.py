"""
=============================================================================
  FAKE NEWS DETECTOR — STREAMLIT SAAS FRONTEND & SYSTEM INTEGRATION
=============================================================================

Main web application frontend for the Fake News Detection SaaS product.
Integrates all backend pipeline components (Steps 5–10):
  - Step 5: ML Model Prediction & Confidence Score
  - Step 6: Model Explainability & Clickbait Language Scanner
  - Step 7: AI-Powered Cross-Source Verification & Claim Analysis
  - Step 8: URL / Web Article Extraction & Content Cleaning
  - Step 9: Fact-Checking API Integration & Rating Normalization
  - Step 10: Multi-Dimensional Evidence Decision Engine (FINAL ASSESSMENT)
"""

import logging
import os
import sys
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

# Bridge Streamlit Cloud secrets into os.environ
try:
    if hasattr(st, "secrets") and st.secrets:
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# Import pipeline backend modules
import prediction as pred
import explainability as exp
import ai_verification as av
import article_extractor as ae
import fact_checker as fc
import decision_engine as de

# Import SVG icon system and UI rendering components
from svg_icons import get_svg
from ui_components import (
    is_valid_url,
    render_external_link,
    render_status_badge,
    render_hero_assessment_card,
    render_metric_card,
    render_empty_state
)


# ---------------------------------------------------------------------------
# Page Configuration & Modern Dark SaaS Theme CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector — AI News Verification SaaS",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/shield-check.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global reset and font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main App Background */
    .stApp {
        background-color: #0b0f17;
        color: #f8fafc;
    }

    /* Content Area Width & Spacing */
    .main .block-container {
        max-width: 1120px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0f17;
    }
    ::-webkit-scrollbar-thumb {
        background: #263143;
        border-radius: 4px;
    }

    /* Top Bar Banner */
    .top-header-card {
        background: linear-gradient(135deg, #151c28 0%, #1a2332 100%);
        border: 1px solid #263143;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }

    .top-header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.02em;
    }

    .top-header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* SaaS Dashboard Cards */
    .saas-card {
        background: #151c28;
        border: 1px solid #263143;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .saas-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.15);
    }

    /* Breaking News Box */
    .breaking-news-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #fef3c7;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }

    .sidebar-brand-box {
        padding: 16px 12px 20px 12px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
    }

    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
    }

    .sidebar-brand-sub {
        font-size: 0.78rem;
        font-weight: 500;
        color: #8b5cf6;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }

    /* System Status Panel in Sidebar */
    .status-panel-card {
        background: #151c28;
        border: 1px solid #263143;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 24px;
    }

    .status-panel-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(38, 49, 67, 0.5);
    }
    .status-row:last-child {
        border-bottom: none;
    }

    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        margin-right: 6px;
    }

    /* Streamlit Controls Styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 0.92rem;
        border-radius: 10px;
        border: 1px solid rgba(139, 92, 246, 0.4);
        padding: 10px 24px;
        transition: all 0.2s ease;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
        border-color: #8b5cf6;
        transform: translateY(-1px);
    }

    /* Textarea & Inputs Override */
    .stTextArea textarea, .stTextInput input {
        background-color: #151c28 !important;
        color: #f8fafc !important;
        border: 1px solid #263143 !important;
        border-radius: 10px !important;
        font-size: 0.92rem !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.25) !important;
    }

    /* Segmented Selector & Radio Styling */
    div[data-testid="stForm"] {
        border-color: #263143;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #151c28;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #263143;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0px 16px;
        background-color: transparent;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Expanders Styling */
    .streamlit-expanderHeader {
        background-color: #151c28 !important;
        border: 1px solid #263143 !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background-color: #101622 !important;
        border: 1px solid #263143 !important;
        border-top: none !important;
        border-bottom-left-radius: 10px !important;
        border-bottom-right-radius: 10px !important;
    }

    /* Sidebar Radio Buttons Styling */
    div[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        padding-bottom: 6px;
        display: block;
    }

    div[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem;
        color: #cbd5e1;
    }

    div[data-testid="stSidebar"] .stRadio div[role="radio"] {
        padding: 6px 4px;
        border-radius: 8px;
        margin-bottom: 2px;
    }

    /* Streamlit Metrics Override */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* Dataframe table dark styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #263143;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Info/Warning/Success Override for dark theme */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Custom Data Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 0.88rem;
        background: #151c28;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #263143;
    }
    .styled-table th {
        background-color: #1e293b;
        color: #94a3b8;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        padding: 12px 16px;
        text-align: left;
    }
    .styled-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #263143;
        color: #f8fafc;
    }
    .styled-table tr:last-child td {
        border-bottom: none;
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
    with real progress feedback using SVG status indicators.
    """
    status_container = st.status("Analyzing news article & evaluating evidence...", expanded=True)

    try:
        # Step 1: Article Extraction if URL
        if is_url:
            status_container.write(f"Step 1/6: Extracting webpage content & metadata...")
            extracted = ae.extract_article(input_text_or_url)
            if extracted.get('extraction_status') == 'FAILED' or 'error' in extracted:
                status_container.update(label="Article extraction failed.", state="error")
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

            status_container.write(f"✓ Step 1/6: Article extracted ({word_count} words via {extractor_used}).")
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
            status_container.write("Step 1/6: Validating article text input...")
            raw_text = input_text_or_url.strip()
            word_count = len(raw_text.split())

            if word_count < 25:
                status_container.update(label="Insufficient article text.", state="error")
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
        status_container.write("Step 2/6: Running statistical feature analysis...")
        ml_res = pred.predict_news(raw_text, title=title)
        if 'error' in ml_res:
            status_container.update(label="ML Model failure.", state="error")
            return {'error': f"ML Model error: {ml_res['error']}"}

        # Step 3: Explainability (Step 6)
        status_container.write("Step 3/6: Computing linguistic feature patterns & clickbait scanner...")
        exp_res = exp.get_explanation(raw_text, top_n=10)

        # Step 4: AI Cross-Source Verification (Step 7)
        status_container.write("Step 4/6: Extracting claims & searching external news/official sources...")
        ai_res = av.verify_article(raw_text)

        # Step 5: Fact-Checking Integration (Step 9)
        status_container.write("Step 5/6: Querying fact-checking databases & evaluating ratings...")
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
        status_container.write("Step 6/6: Synthesizing evidence into Final Assessment...")
        final_dec = de.make_final_decision(ml_res, ai_res, fact_res, article_metadata=art_metadata)

        status_container.update(label="News Verification Completed", state="complete", expanded=False)

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
                'influential_features': (exp_dict.get('influential_features') or [])[:10],
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
        status_container.update(label="Pipeline execution error.", state="error")
        return {'error': f"An unexpected error occurred during processing: {str(e)}"}


# ---------------------------------------------------------------------------
# Section 2: Sidebar Navigation & Live Monitoring System Status
# ---------------------------------------------------------------------------

with st.sidebar:
    # SVG Brand Header
    brand_shield = get_svg('shield', size=26, color='#7c3aed')
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div class="sidebar-brand-title">
            {brand_shield}
            <span>Fake News Detector</span>
        </div>
        <div class="sidebar-brand-sub">AI News Verification SaaS</div>
    </div>
    """, unsafe_allow_html=True)

    # Workspace Navigation
    nav_choice = st.radio(
        "WORKSPACE",
        ["Analyze News", "How It Works", "About System"],
        index=0,
        label_visibility="visible"
    )

    # System Status Monitoring Panel
    activity_icon = get_svg('activity', size=16, color='#94a3b8')
    st.markdown(f"""
    <div class="status-panel-card">
        <div class="status-panel-header">
            {activity_icon} System Status
        </div>
        <div class="status-row">
            <span style="color: #cbd5e1;">Verification Engine</span>
            <span><span class="status-dot"></span><strong style="color: #34d399;">ONLINE</strong></span>
        </div>
        <div class="status-row">
            <span style="color: #cbd5e1;">AI Search</span>
            <span><span class="status-dot"></span><strong style="color: #34d399;">ONLINE</strong></span>
        </div>
        <div class="status-row">
            <span style="color: #cbd5e1;">Fact Check API</span>
            <span><span class="status-dot"></span><strong style="color: #fbbf24;">READY</strong></span>
        </div>
        <div class="status-row">
            <span style="color: #cbd5e1;">Decision System</span>
            <span><span class="status-dot"></span><strong style="color: #34d399;">ACTIVE</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 3: Navigation Page 1 — Analyze News
# ---------------------------------------------------------------------------

if nav_choice == "Analyze News":

    # Top Header Banner
    top_shield = get_svg('shield', size=28, color='#7c3aed')
    st.markdown(f"""
    <div class="top-header-card">
        <div class="top-header-title">
            {top_shield}
            <span>AI News Verification Workspace</span>
        </div>
        <p class="top-header-subtitle">
            Evidence-based news assessment combining multi-source web verification, live fact-checking databases, and statistical machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Input Mode Selector: Document vs URL
    mode_choice = st.radio(
        "Select Input Mode",
        ["Document Text", "News URL"],
        horizontal=True,
        label_visibility="collapsed"
    )

    input_text = ""
    input_url = ""
    article_title_input = ""
    trigger_analyze = False
    is_url_mode = (mode_choice == "News URL")

    # Document Input Mode Card
    if not is_url_mode:
        st.markdown(f"""
        <div style="background: #151c28; border: 1px solid #263143; border-radius: 12px; padding: 20px 22px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1rem; color: #f8fafc; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
                {get_svg('file-text', size=18, color='#8b5cf6')} Article Content & Metadata
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 16px;">
                Paste the full text of the article you wish to verify.
            </div>
        </div>
        """, unsafe_allow_html=True)

        article_title_input = st.text_input(
            "Article Title (Optional)",
            placeholder="Enter article title or headline...",
            key="input_title_doc"
        )
        input_text = st.text_area(
            "Article Text",
            height=210,
            placeholder="Paste news article content here...",
            key="input_text_doc"
        )

        word_count_calc = len(input_text.strip().split()) if input_text.strip() else 0
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 10px; display: flex; align-items: center; gap: 6px;">
                {get_svg('hash', size=14, color='#94a3b8')} <strong>{word_count_calc}</strong> words detected
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            if st.button("Analyze Article", key="btn_paste_doc"):
                if not input_text.strip():
                    st.error("Please enter an article text to analyze.")
                else:
                    trigger_analyze = True
                    is_url_mode = False

    # URL Input Mode Card
    else:
        st.markdown(f"""
        <div style="background: #151c28; border: 1px solid #263143; border-radius: 12px; padding: 20px 22px; margin-bottom: 20px;">
            <div style="font-weight: 600; font-size: 1rem; color: #f8fafc; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
                {get_svg('globe', size=18, color='#8b5cf6')} News Article URL
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 16px;">
                Provide a web link to automatically extract and analyze article content.
            </div>
        </div>
        """, unsafe_allow_html=True)

        input_url = st.text_input(
            "News Article Link",
            placeholder="https://example.com/news/article-path",
            key="input_url_field"
        )
        if st.button("Analyze URL", key="btn_url_mode"):
            val_res = ae.validate_url(input_url)
            if not val_res['valid']:
                st.error(val_res['error'])
            else:
                trigger_analyze = True
                is_url_mode = True

    # Execute Analysis
    if trigger_analyze:
        target_input = input_url if is_url_mode else input_text
        res = run_full_pipeline(target_input, is_url=is_url_mode, article_title=article_title_input)
        st.session_state.analysis_result = res
        st.session_state.last_input = target_input

    # Render Results if available in Session State
    res = st.session_state.analysis_result

    if res:
        if 'error' in res or res.get('extraction_status') == 'FAILED':
            st.error(f"Unable to complete analysis: {res.get('error', 'Extraction failed.')}")
            if res.get('fallback_suggested'):
                st.markdown("---")
                st.warning("Automatic content extraction was unsuccessful for this URL. Please paste the article text manually:")
                fallback_paste = st.text_area("Paste Article Text Manually", height=200, key="manual_fallback_text_area")
                fallback_title = st.text_input("Article Title (Optional)", key="manual_fallback_title_input")
                if st.button("Analyze Pasted Text", key="btn_manual_fallback"):
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

            # 1. FINAL ASSESSMENT HERO CARD
            render_hero_assessment_card(final_dec)

            # 2. BREAKING NEWS ALERT BOX
            if final_dec.get('is_breaking_news'):
                alert_icon = get_svg('alert-triangle', size=18, color='#f59e0b')
                st.markdown(f"""
                <div class="breaking-news-box">
                    <div style="font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        {alert_icon} Developing Event / Limited Reporting Detected
                    </div>
                    <div>This article discusses a recent or developing event. Initial external web reporting is limited, requiring cautious verification.</div>
                </div>
                """, unsafe_allow_html=True)

            # 3. EVIDENCE SUMMARY METRICS DASHBOARD
            es = final_dec.get('evidence_summary', {})
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                render_metric_card("Official Sources", es.get('official_sources', 0), icon_name="shield")
            with m_col2:
                render_metric_card("Independent Outlets", es.get('independent_sources', 0), icon_name="newspaper")
            with m_col3:
                render_metric_card("Fact Checks Found", es.get('fact_checks', 0), icon_name="database")
            with m_col4:
                render_metric_card("Supporting Evidence", es.get('supporting_sources', 0), icon_name="check-circle")
            with m_col5:
                render_metric_card("Contradicting Evidence", es.get('contradicting_sources', 0), icon_name="x-circle")

            st.markdown("<br>", unsafe_allow_html=True)

            # 4. ARTICLE INFORMATION CARD
            info_icon = get_svg('file-text', size=18, color='#8b5cf6')
            st.markdown(f"""
            <div style="background: #151c28; border: 1px solid #263143; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;">
                <div style="font-weight: 700; font-size: 0.95rem; color: #f8fafc; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                    {info_icon} ARTICLE INFORMATION
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.88rem; color: #cbd5e1;">
                    <div>
                        <div style="color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;">Title</div>
                        <div style="font-weight: 600; color: #f8fafc; margin-top: 2px;">{art.get('title', 'N/A')}</div>
                    </div>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;">Publication Date</div>
                        <div style="font-weight: 600; color: #f8fafc; margin-top: 2px;">{art.get('publication_date', 'Not specified')}</div>
                    </div>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;">Source Outlet</div>
                        <div style="font-weight: 600; color: #f8fafc; margin-top: 2px;">{art.get('source', 'N/A')}</div>
                    </div>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;">Word Count / Mode</div>
                        <div style="font-weight: 600; color: #f8fafc; margin-top: 2px;">{art.get('word_count', 0)} words ({art.get('input_mode', 'Text')})</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if art.get('url') and is_valid_url(art['url']):
                render_external_link("View Original Source Webpage", art['url'])

            st.markdown("<br>", unsafe_allow_html=True)

            # 5. DETAILED VERIFICATION SECTIONS & TABS
            r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
                "AI Verification & Claims",
                "Fact Checking",
                "Sources & Evidence",
                "Linguistic Insights",
                "System Disclaimer"
            ])

            # Tab 1: AI Verification & Claim Analysis
            with r_tab1:
                st.markdown("##### AI Cross-Source Verification Summary")
                render_status_badge(ai_data.get('overall_status', 'UNVERIFIED'))
                st.markdown("<br>", unsafe_allow_html=True)

                if ai_data.get('ai_assessment'):
                    st.markdown(f"""
                    <div style="background: #101622; border: 1px solid #263143; border-radius: 10px; padding: 16px; font-size: 0.9rem; line-height: 1.5; color: #e2e8f0; margin-bottom: 20px;">
                        <strong>Synthesis Overview:</strong> {ai_data.get('ai_assessment')}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("##### Factual Claim Analysis")
                claims_list = ai_data.get('claims', []) or ai_data.get('claim_verifications', [])
                if not claims_list:
                    render_empty_state("No claims extracted", "No factual claims were identified for cross-source verification.")
                else:
                    for i, c in enumerate(claims_list, 1):
                        c_status = c.get('status', 'UNVERIFIED') if isinstance(c, dict) else 'UNVERIFIED'
                        c_text = c.get('claim', 'Factual Claim') if isinstance(c, dict) else str(c)
                        c_sum = c.get('summary', 'Claim evaluation details.') if isinstance(c, dict) else ''

                        with st.expander(f"Claim {i:02d}: \"{c_text[:75]}...\"", expanded=(i == 1)):
                            st.markdown(f"**Full Claim:** {c_text}")
                            st.write("**Verification Status:**")
                            render_status_badge(c_status)
                            st.markdown(f"**Evidence Summary:** {c_sum}")

                            if isinstance(c, dict) and c.get('supporting_sources'):
                                st.markdown("**Supporting Reports:**")
                                for s in c['supporting_sources']:
                                    render_external_link("Read Supporting Source", s)
                            if isinstance(c, dict) and c.get('contradicting_sources'):
                                st.markdown("**Contradicting Reports:**")
                                for s in c['contradicting_sources']:
                                    render_external_link("Read Contradicting Source", s)

            # Tab 2: Fact-Checking Section
            with r_tab2:
                st.markdown("##### Fact-Checking Database Query Results")
                if fact_data.get('status') == 'UNAVAILABLE':
                    st.warning("Fact-checking API service temporarily unavailable.")
                else:
                    fc_cnt = fact_data.get('fact_checks_found_total', 0)
                    st.write(f"**Claims Checked:** `{fact_data.get('claims_checked', 0)}` | **Fact Checks Identified:** `{fc_cnt}`")

                    fc_results = fact_data.get('results', [])
                    if not fc_results or fc_cnt == 0:
                        render_empty_state("No Published Fact Checks Found", "No matching reviews identified in Google Fact Check API or major fact-checking outlets.")
                    else:
                        for i, fr in enumerate(fc_results, 1):
                            with st.expander(f"Fact Check Review {i:02d}: \"{fr.get('claim', '')[:70]}...\"", expanded=True):
                                st.markdown(f"**Evaluated Claim:** \"{fr.get('claim', '')}\"")
                                st.write("**Status:**")
                                render_status_badge(fr.get('status', 'UNVERIFIED'))
                                st.markdown(f"**Rationale:** {fr.get('reason', '')}")

                                if fr.get('fact_checks'):
                                    st.markdown("###### Reviews Found:")
                                    for item in fr['fact_checks']:
                                        pub = item.get('publisher', 'Fact Checker')
                                        orig_rating = item.get('original_rating', 'FALSE')
                                        norm_rating = item.get('normalized_rating', 'FALSE')
                                        title = item.get('title', 'Fact Check Review')

                                        st.markdown(f"""
                                        <div style="background: #151c28; border: 1px solid #263143; border-radius: 8px; padding: 12px; margin-top: 8px;">
                                            <div style="font-weight: 600; color: #f8fafc; font-size: 0.9rem;">{title}</div>
                                            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                                                Publisher: <strong>{pub}</strong> | Original Rating: <code style="color:#fb7185;">{orig_rating}</code> → Normalized: <code style="color:#f87171;">{norm_rating}</code>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        render_external_link(f"View Review on {pub}", item.get('url'))

            # Tab 3: Official & External Sources
            with r_tab3:
                st.markdown("##### Official & Primary Domain Confirmations")
                off_sources = final_dec.get('official_sources_list', [])
                if off_sources:
                    for off in off_sources:
                        st.markdown(f"✓ **{off.get('source', 'Official Source')}** — {off.get('title', 'Official Announcement')}")
                        render_external_link(f"Open Official Source ({off.get('source', 'Gov Domain')})", off.get('url'))
                else:
                    st.info("No direct primary domain (.gov / .org) confirmation identified.")

                st.markdown("---")
                st.markdown("##### Independent Reputable News Outlets")
                rep_sources = ai_data.get('reputable_news_sources', []) or ai_data.get('reputable_sources', [])
                if rep_sources:
                    for rep in rep_sources:
                        src_label = rep.get('publisher') or rep.get('source') or rep.get('domain') or 'Established Outlet'
                        title_label = rep.get('headline') or rep.get('title') or 'News Report'
                        url_label = rep.get('url', '')
                        snippet = rep.get('snippet', '')

                        st.markdown(f"📰 **{src_label}**: {title_label}")
                        if snippet:
                            st.caption(f"  › {snippet[:150]}...")
                        render_external_link(f"Read Source on {src_label}", url_label)
                else:
                    render_empty_state("No Established News Coverage Found", "No independent reporting from recognized national or international outlets was identified.")

                st.markdown("---")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown("##### Supporting Evidence")
                    sup_list = final_dec.get('supporting_evidence', [])
                    if sup_list:
                        for s_item in sup_list:
                            st.write(f"- **Claim:** \"{s_item.get('claim', '')[:50]}...\"")
                            st.caption(f"  {s_item.get('detail', '')}")
                            src_val = s_item.get('url') or s_item.get('source') or ''
                            if is_valid_url(src_val):
                                render_external_link("View Supporting Evidence", src_val)
                    else:
                        st.write("No explicit supporting links recorded.")

                with col_s2:
                    st.markdown("##### Contradicting Evidence")
                    con_list = final_dec.get('contradicting_evidence', [])
                    if con_list:
                        for c_item in con_list:
                            st.write(f"- **Claim:** \"{c_item.get('claim', '')[:50]}...\"")
                            st.caption(f"  {c_item.get('detail', '')}")
                            src_val = c_item.get('url') or c_item.get('source') or ''
                            if is_valid_url(src_val):
                                render_external_link("View Contradicting Evidence", src_val)
                    else:
                        st.write("No contradictory evidence detected.")

            # Tab 4: Linguistic Insights
            with r_tab4:
                st.markdown("##### Model Explainability & Vocabulary Feature Weights")
                st.write("Key vocabulary terms influencing statistical model predictions:")

                feats = exp_data.get('influential_features', [])
                if not feats or not isinstance(feats, list) or len(feats) == 0:
                    render_empty_state("No Linguistic Insights", "Linguistic feature weights could not be computed for this input.")
                else:
                    data_for_df = []
                    for f in feats[:10]:
                        if isinstance(f, dict):
                            data_for_df.append({
                                "Feature Word": f.get('word', f.get('feature', '')),
                                "TF-IDF Score": round(float(f.get('tfidf_score', f.get('tfidf', 0.0))), 4),
                                "Model Weight": round(float(f.get('coefficient', f.get('weight', 0.0))), 4),
                                "Classification Direction": str(f.get('direction', 'UNKNOWN')).upper()
                            })

                    if data_for_df:
                        df_ling = pd.DataFrame(data_for_df)
                        st.dataframe(df_ling, use_container_width=True, hide_index=True)

                susp = exp_data.get('suspicious_language', [])
                if susp and isinstance(susp, list):
                    st.markdown("<br>##### Suspicious Language Patterns Detected", unsafe_allow_html=True)
                    for s in susp:
                        if isinstance(s, dict):
                            cat = str(s.get('category', 'LANGUAGE')).upper()
                            desc = s.get('pattern_desc', 'Pattern detected')
                            txt = s.get('matched_text', '')
                            st.markdown(f"""
                            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; margin-top: 6px; font-size: 0.85rem; color: #fef3c7;">
                                <strong>[{cat}]</strong> {desc} — <em>matched: "{txt}"</em>
                            </div>
                            """, unsafe_allow_html=True)

            # Tab 5: System Disclaimer
            with r_tab5:
                st.markdown("##### System Principles & Verification Boundaries")
                st.markdown("""
                - **Evidence-Based Assessment**: The Final Assessment is generated by evaluating primary government domain reporting, independent news indexing, published fact checks, and statistical writing style.
                - **Dynamic Web Indexing**: Articles covering recent or unindexed events may yield `UNCERTAIN` or `LIMITED EVIDENCE` statuses.
                - **User Guidance**: Always verify sensitive or critical news stories using official primary domain announcements.
                """)


# ---------------------------------------------------------------------------
# Section 4: Navigation Page 2 — How It Works
# ---------------------------------------------------------------------------

elif nav_choice == "How It Works":
    layers_icon = get_svg('layers', size=26, color='#7c3aed')
    st.markdown(f"""
    <div class="top-header-card">
        <div class="top-header-title">
            {layers_icon}
            <span>How The Verification Engine Works</span>
        </div>
        <p class="top-header-subtitle">
            An end-to-end multi-dimensional pipeline synthesizing live web verification, fact-checking databases, and statistical NLP.
        </p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Article Input", "User submits raw article text or news link.", "file-text"),
        ("02", "Article Extraction", "Clean text & metadata extraction using Trafilatura & BeautifulSoup.", "globe"),
        ("03", "ML Feature Analysis", "TF-IDF vocabulary classification using trained LinearSVC model.", "brain"),
        ("04", "Cross-Source Search", "Live web search across official domains (.gov/.org) & reputable news outlets.", "search"),
        ("05", "Fact Check Query", "Queries Google Fact Check API & database ratings (Snopes, Reuters, etc.).", "database"),
        ("06", "Evidence Synthesis", "Multi-dimensional rules evaluate supporting vs. contradicting evidence.", "scale"),
        ("07", "Final Assessment", "Produces clean verdict: REAL, FAKE, or UNCERTAIN with confidence levels.", "shield")
    ]

    for step_num, step_title, step_desc, icon_name in steps:
        step_svg = get_svg(icon_name, size=22, color='#8b5cf6')
        st.markdown(f"""
        <div style="background: #151c28; border: 1px solid #263143; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; display: flex; align-items: center; gap: 16px;">
            <div style="background: rgba(124, 58, 237, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); color: #c4b5fd; font-weight: 800; font-size: 1rem; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                {step_num}
            </div>
            <div style="flex-grow: 1;">
                <div style="font-weight: 700; font-size: 1rem; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                    {step_svg} {step_title}
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">
                    {step_desc}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 5: Navigation Page 3 — About System
# ---------------------------------------------------------------------------

elif nav_choice == "About System":
    info_icon = get_svg('info', size=26, color='#7c3aed')
    st.markdown(f"""
    <div class="top-header-card">
        <div class="top-header-title">
            {info_icon}
            <span>About The AI News Verification System</span>
        </div>
        <p class="top-header-subtitle">
            An explainable SaaS solution for automated news credibility assessment and evidence synthesis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="saas-card">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                {get_svg('shield', size=20, color='#8b5cf6')} Project Overview
            </div>
            <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                This system provides automated fake news assessment by combining machine learning style classification with real-time web search verification and fact-checking database queries.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="saas-card">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                {get_svg('cpu', size=20, color='#8b5cf6')} Technology Stack
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;">
                • <strong>Frontend UI:</strong> Streamlit 1.57+<br>
                • <strong>ML Framework:</strong> Scikit-Learn (LinearSVC & TF-IDF)<br>
                • <strong>AI Verification:</strong> Google Gemini API & DuckDuckGo Search<br>
                • <strong>Fact Check Engine:</strong> Google Fact Check Tools API<br>
                • <strong>Article Extractor:</strong> Trafilatura & BeautifulSoup4
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="saas-card">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                {get_svg('check-circle', size=20, color='#10b981')} Capabilities
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;">
                • Multi-source cross-verification<br>
                • Automatic claim extraction & rating normalization<br>
                • Model explainability via feature weight inspection<br>
                • Support for raw text and URL input formats
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="saas-card">
            <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                {get_svg('alert-triangle', size=20, color='#f59e0b')} Boundaries & Principles
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.8;">
                • Explicit evidence boundaries for developing events<br>
                • Secure API key handling via standard environment configs<br>
                • Direct clickable external links to verified sources
            </div>
        </div>
        """, unsafe_allow_html=True)
