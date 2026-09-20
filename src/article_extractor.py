"""
Fake News Detection — Step 8: URL / News Article Extraction Module
=====================================================================

This module accepts a news article URL, validates it, fetches the webpage content,
extracts the main article text and metadata (title, author, date, domain), cleans the
text, and passes it into the complete Fake News Detection pipeline:

    URL → Article Extraction → Clean Text → Step 5 ML → Step 6 Explainability → Step 7 AI Verification → Step 9 Fact Checking → Step 10 Decision Engine

Key Features:
  - Multi-stage robust article extraction:
      Method 1: Trafilatura (primary article body & metadata)
      Method 2: JSON-LD (Schema.org NewsArticle parsing)
      Method 3: BeautifulSoup Tag & Heuristic Article Extraction
  - Security validation against invalid protocols, non-HTTP schemas, and SSRF targets.
  - Redirect handling: Safe follow of HTTP/HTTPS redirects and extraction of final domain.
  - Quality assurance: Detects boilerplate/navigation/cookie notices to prevent false "insufficient text" errors on long articles.
  - Detailed internal debugging: Logs URL, HTTP status, final URL, extractor used, word count, and content preview.
  - Unified `analyze_url(url)` orchestrator function ready for Streamlit UI integration.

Usage:
    from article_extractor import extract_article, analyze_url, display_url_analysis
    result = analyze_url("https://www.bbc.com/news/article-12345678")
    display_url_analysis(result)
"""

import json
import logging
import os
import re
import sys
import urllib.parse
import warnings

import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try importing trafilatura for high-quality main content & metadata extraction
_TRAFILATURA_AVAILABLE = False
try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

# Import project pipeline modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import prediction as pred
import explainability as exp
import ai_verification as av
import fact_checker as fc
import decision_engine as de


# ---------------------------------------------------------------------------
# Section 1: URL Validation & Domain Extraction
# ---------------------------------------------------------------------------

_BLOCKED_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '169.254.169.254', '::1'}

def validate_url(url):
    """
    Validate that an input string is a valid, safe public HTTP/HTTPS URL.

    Args:
        url (str): Input URL candidate string.

    Returns:
        dict:
            {"valid": True, "clean_url": str, "domain": str}
            or
            {"valid": False, "error": str}
    """
    if not url or not isinstance(url, str) or not url.strip():
        return {
            'valid': False,
            'error': 'Invalid URL. Please enter a valid news article URL (e.g. https://example.com/article).'
        }

    url_str = url.strip()

    # Check scheme
    parsed = urllib.parse.urlparse(url_str)
    if parsed.scheme.lower() not in ('http', 'https'):
        return {
            'valid': False,
            'error': 'Unsupported protocol. Only HTTP and HTTPS URLs are supported.'
        }

    # Security check: SSRF guard against local/internal addresses
    hostname = (parsed.hostname or '').lower()
    if not hostname or hostname in _BLOCKED_HOSTS or hostname.startswith('192.168.') or hostname.startswith('10.') or hostname.startswith('172.16.') or hostname.startswith('172.31.'):
        return {
            'valid': False,
            'error': 'Invalid URL. Private, local, or internal addresses cannot be accessed.'
        }

    # Domain extraction
    domain = hostname
    if domain.startswith('www.'):
        domain = domain[4:]

    return {
        'valid': True,
        'clean_url': url_str,
        'domain': domain
    }


# ---------------------------------------------------------------------------
# Section 2: Core Article Extraction Engine & Fallbacks
# ---------------------------------------------------------------------------

_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, Gecko) Chrome/128.0.0.0 Safari/537.36'
)

_MIN_ARTICLE_WORDS = 25  # Minimum words required for valid article content

_BOILERPLATE_PATTERNS = [
    r'privacy policy', r'terms of service', r'all rights reserved', r'cookie policy',
    r'subscribe to', r'follow us on', r'sign in', r'log in', r'create account',
    r'skip to main content', r'advertisement'
]


def _is_quality_article_text(text, min_words=_MIN_ARTICLE_WORDS):
    """
    Verify if extracted text represents genuine article content rather than navigation,
    cookie banners, or site boilerplate.
    """
    if not text or not isinstance(text, str):
        return False
    words = text.split()
    if len(words) < min_words:
        return False

    text_lower = text.lower()
    bp_matches = sum(1 for pat in _BOILERPLATE_PATTERNS if re.search(pat, text_lower))
    if len(words) < 50 and bp_matches >= 3:
        return False

    return True


def _json_ld_extract(html):
    """
    Method 2 Fallback: Extract article body and metadata from application/ld+json scripts.
    """
    if not html:
        return None, None, None, None
    try:
        soup = BeautifulSoup(html, 'html.parser')
        body_text, title, author, pub_date = None, None, None, None

        for script in soup.find_all('script', type='application/ld+json'):
            content = script.string or script.get_text()
            if not content:
                continue
            try:
                data = json.loads(content)
            except Exception:
                continue

            items = data if isinstance(data, list) else [data]
            if isinstance(data, dict) and '@graph' in data and isinstance(data['@graph'], list):
                items = data['@graph']

            for item in items:
                if not isinstance(item, dict):
                    continue
                type_ = str(item.get('@type', ''))
                if any(t in type_ for t in ['NewsArticle', 'Article', 'ReportageNewsArticle', 'BlogPosting', 'WebPage']):
                    if not title:
                        title = item.get('headline') or item.get('name')
                    if not body_text and 'articleBody' in item and isinstance(item['articleBody'], str):
                        body_text = item['articleBody']
                    if not author and 'author' in item:
                        auth = item['author']
                        if isinstance(auth, dict):
                            author = auth.get('name')
                        elif isinstance(auth, list) and auth and isinstance(auth[0], dict):
                            author = auth[0].get('name')
                        elif isinstance(auth, str):
                            author = auth
                    if not pub_date:
                        pub_date = item.get('datePublished') or item.get('dateCreated')
        return body_text, title, author, pub_date
    except Exception:
        return None, None, None, None


def _bs4_heuristic_extract(html):
    """
    Method 3 Fallback: Article text and metadata extraction using BeautifulSoup tag analysis.
    """
    if not html:
        return None, None, None, None
    try:
        soup = BeautifulSoup(html, 'html.parser')
        title, author, pub_date = None, None, None

        # Meta title / author / date
        og_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'twitter:title'})
        if og_title and og_title.get('content'):
            title = og_title['content'].strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        og_author = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', property='article:author')
        if og_author and og_author.get('content'):
            author = og_author['content'].strip()

        og_date = soup.find('meta', property='article:published_time') or soup.find('meta', attrs={'name': 'pubdate'}) or soup.find('meta', attrs={'name': 'date'})
        if og_date and og_date.get('content'):
            pub_date = og_date['content'].strip()

        # Clean non-content tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript', 'svg', 'button']):
            tag.decompose()

        # Remove boilerplate containers
        for el in soup.find_all(class_=re.compile(r'nav|menu|cookie|footer|header|sidebar|ad-|banner|comment|related|popular|trending|promo', re.I)):
            el.decompose()

        # Locate article container
        article_tag = (
            soup.find('article') or
            soup.find('main') or
            soup.find(attrs={'itemprop': 'articleBody'}) or
            soup.find('div', class_=re.compile(r'article|content|story|post|entry|main', re.I))
        )

        target = article_tag if article_tag else soup
        paragraphs = []
        for p in target.find_all(['p', 'h2', 'h3']):
            txt = p.get_text(strip=True)
            if len(txt.split()) >= 5:
                if not re.search(r'privacy policy|terms of service|all rights reserved|cookie policy', txt, re.I):
                    paragraphs.append(txt)

        text = '\n\n'.join(paragraphs) if paragraphs else None
        return text, title, author, pub_date
    except Exception as e:
        logger.warning(f"BS4 extraction exception: {e}")
        return None, None, None, None


def extract_article(url):
    """
    Fetch and extract main article text and metadata from a news URL with multi-stage fallbacks.

    Args:
        url (str): News article URL.

    Returns:
        dict on success:
            {
                "status": "SUCCESS",
                "extraction_status": "SUCCESS",
                "extractor_used": str ("trafilatura", "json_ld", or "bs4_heuristic"),
                "title": str,
                "author": str,
                "publication_date": str,
                "source": str (domain),
                "url": str (final redirected URL),
                "original_url": str,
                "text": str (raw main article body),
                "word_count": int,
                "http_status": int
            }
        dict on error:
            {
                "status": "FAILED",
                "extraction_status": "FAILED",
                "error": str (user-friendly error message),
                "error_type": str ("RESTRICTED", "NOT_FOUND", "INSUFFICIENT_CONTENT", "FETCH_ERROR", "INVALID_URL"),
                "fallback_suggested": True,
                "url": str,
                "source": str,
                "http_status": int or None
            }
    """
    # 1. Validate URL
    val = validate_url(url)
    if not val['valid']:
        return {
            'status': 'FAILED',
            'extraction_status': 'FAILED',
            'error': val['error'],
            'error_type': 'INVALID_URL',
            'fallback_suggested': True,
            'url': url,
            'source': 'Unknown'
        }

    target_url = val['clean_url']
    original_domain = val['domain']

    headers = {
        'User-Agent': _USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Upgrade-Insecure-Requests': '1',
    }

    session = requests.Session()
    response = None
    http_status = None
    final_url = target_url
    html_content = None

    # 2. Fetch Webpage Content (Handling Redirects & Headers)
    try:
        response = session.get(target_url, headers=headers, timeout=12, allow_redirects=True)
        http_status = response.status_code
        final_url = response.url

        # Security check on final redirected URL
        val_final = validate_url(final_url)
        if not val_final['valid']:
            return {
                'status': 'FAILED',
                'extraction_status': 'FAILED',
                'error': 'Redirected to invalid or restricted address.',
                'error_type': 'INVALID_REDIRECT',
                'fallback_suggested': True,
                'url': final_url,
                'source': original_domain
            }
        domain = val_final['domain']

        if http_status in (401, 403):
            return {
                'status': 'FAILED',
                'extraction_status': 'FAILED',
                'error': 'This article requires restricted access and could not be read automatically. Please paste the article text manually.',
                'error_type': 'RESTRICTED',
                'fallback_suggested': True,
                'url': final_url,
                'source': domain,
                'http_status': http_status
            }
        elif http_status == 404:
            return {
                'status': 'FAILED',
                'extraction_status': 'FAILED',
                'error': 'Article page not found (404 Error). Please check the URL or paste the article text manually.',
                'error_type': 'NOT_FOUND',
                'fallback_suggested': True,
                'url': final_url,
                'source': domain,
                'http_status': http_status
            }
        elif http_status >= 400:
            return {
                'status': 'FAILED',
                'extraction_status': 'FAILED',
                'error': f'We couldn\'t extract the article from this website (HTTP Error {http_status}). Please paste the article text manually.',
                'error_type': 'HTTP_ERROR',
                'fallback_suggested': True,
                'url': final_url,
                'source': domain,
                'http_status': http_status
            }

        html_content = response.text

    except requests.exceptions.Timeout:
        return {
            'status': 'FAILED',
            'extraction_status': 'FAILED',
            'error': 'Webpage request timed out. Please check your network connection or paste the article text manually.',
            'error_type': 'TIMEOUT',
            'fallback_suggested': True,
            'url': target_url,
            'source': original_domain
        }
    except Exception as e:
        logger.warning(f"Fetch failed for {target_url}: {e}")
        # Fallback fetcher via trafilatura
        if _TRAFILATURA_AVAILABLE:
            try:
                html_content = trafilatura.fetch_url(target_url)
            except Exception:
                pass
        if not html_content:
            return {
                'status': 'FAILED',
                'extraction_status': 'FAILED',
                'error': 'We couldn\'t extract the article from this website. Please paste the article text manually.',
                'error_type': 'FETCH_ERROR',
                'fallback_suggested': True,
                'url': target_url,
                'source': original_domain
            }

    domain = validate_url(final_url)['domain']

    # 3. Multi-Stage Content & Metadata Extraction
    extracted_text = None
    title = None
    author = None
    pub_date = None
    extractor_used = None

    # Method 1: Trafilatura Primary Extractor
    if _TRAFILATURA_AVAILABLE and html_content:
        try:
            t_text = trafilatura.extract(
                html_content,
                favor_recall=True,
                include_comments=False,
                include_tables=True,
                include_links=False,
                output_format='txt'
            )
            t_meta = trafilatura.extract_metadata(html_content)
            if t_meta:
                title = t_meta.title
                author = t_meta.author
                pub_date = t_meta.date

            if t_text and _is_quality_article_text(t_text):
                extracted_text = t_text
                extractor_used = 'trafilatura'
        except Exception as ex:
            logger.warning(f"Trafilatura extraction exception: {ex}")

    # Method 2: JSON-LD Schema.org Extraction Fallback
    if not extracted_text and html_content:
        j_text, j_title, j_author, j_date = _json_ld_extract(html_content)
        if j_text and _is_quality_article_text(j_text):
            extracted_text = j_text
            extractor_used = 'json_ld'
        if not title and j_title: title = j_title
        if not author and j_author: author = j_author
        if not pub_date and j_date: pub_date = j_date

    # Method 3: BeautifulSoup Tag & Heuristic Fallback
    if not extracted_text and html_content:
        b_text, b_title, b_author, b_date = _bs4_heuristic_extract(html_content)
        if b_text and _is_quality_article_text(b_text):
            extracted_text = b_text
            extractor_used = 'bs4_heuristic'
        if not title and b_title: title = b_title
        if not author and b_author: author = b_author
        if not pub_date and b_date: pub_date = b_date

    # Clean text formatting
    if extracted_text:
        extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text).strip()

    word_count = len(extracted_text.split()) if extracted_text else 0

    # 4. Final Quality & Content Length Validation
    if not extracted_text or word_count < _MIN_ARTICLE_WORDS:
        logger.info(f"Extraction failed quality check for {final_url}. Word count: {word_count}")
        return {
            'status': 'FAILED',
            'extraction_status': 'FAILED',
            'error': 'We couldn\'t extract enough article content from this URL. Please paste the article text manually.',
            'error_type': 'INSUFFICIENT_CONTENT',
            'fallback_suggested': True,
            'url': final_url,
            'source': domain,
            'word_count': word_count,
            'http_status': http_status or 200
        }

    if not title:
        title = f"News Article from {domain}"

    # 5. Internal Debug Logging (Prompt Requirement 6)
    logger.info("========================================")
    logger.info("EXTRACTION DEBUG")
    logger.info(f"URL: {target_url}")
    logger.info(f"HTTP Status: {http_status or 200}")
    logger.info(f"Final URL: {final_url}")
    logger.info(f"Extractor: {extractor_used}")
    logger.info(f"Title: {title}")
    logger.info(f"Title Length: {len(title)}")
    logger.info(f"Article Text Length: {len(extracted_text)}")
    logger.info(f"Word Count: {word_count}")
    logger.info(f"Preview: {repr(extracted_text[:300])}")
    logger.info("========================================")

    return {
        'status': 'SUCCESS',
        'extraction_status': 'SUCCESS',
        'extractor_used': extractor_used,
        'title': title,
        'author': author if author else 'N/A',
        'publication_date': pub_date if pub_date else 'N/A',
        'source': domain,
        'url': final_url,
        'original_url': target_url,
        'text': extracted_text,
        'word_count': word_count,
        'http_status': http_status or 200
    }


# ---------------------------------------------------------------------------
# Section 3: Master Orchestrator — analyze_url()
# ---------------------------------------------------------------------------

def analyze_url(url):
    """
    Master orchestrator function for URL-based Fake News Detection.

    Flow:
        URL
         ↓
        Article Extraction (trafilatura / json_ld / bs4_heuristic)
         ↓
        Validation (Content length & HTTP checks)
         ↓
        Step 5 ML Prediction (Prediction + Confidence %)
         ↓
        Step 6 Explainability (Influential features & Clickbait language)
         ↓
        Step 7 AI Verification (Claims + Live Web Search + Source Tiers)
         ↓
        Step 9 Fact Checking (Dedicated API query)
         ↓
        Step 10 Decision Engine (Multi-dimensional Evidence Synthesis)

    Args:
        url (str): Input news URL.

    Returns:
        dict: Complete unified analysis result ready for Streamlit UI or notebook.
    """
    # 1. Extract Article
    extracted = extract_article(url)
    if extracted.get('extraction_status') == 'FAILED' or 'error' in extracted:
        return {
            'error': extracted.get('error', 'Article extraction failed.'),
            'error_type': extracted.get('error_type', 'EXTRACTION_FAILED'),
            'fallback_suggested': extracted.get('fallback_suggested', True),
            'extraction_status': 'FAILED',
            'article': {
                'title': 'N/A',
                'source': extracted.get('source', 'Web Domain'),
                'url': extracted.get('url', url),
                'extraction_status': 'FAILED'
            }
        }

    article_text = extracted['text']
    article_title = extracted['title']

    # 2. Step 5 ML Model Prediction
    ml_result = pred.predict_news(article_text, title=article_title)
    if 'error' in ml_result:
        return {
            'error': f"ML Model Analysis Error: {ml_result['error']}",
            'extraction_status': 'SUCCESS',
            'article': extracted
        }

    # 3. Step 6 Model Explainability
    exp_result = exp.get_explanation(article_text, top_n=10)

    # 4. Step 7 AI News Verification & Cross-Source Analysis
    ai_result = av.verify_article(article_text)

    # 5. Step 9 Fact-Checking API Integration
    step7_claims = ai_result.get('claim_verifications', []) if isinstance(ai_result, dict) else []
    try:
        fact_check_result = fc.fact_check_article(article_text, claims=step7_claims)
    except Exception as exc:
        logger.warning(f"Fact checking module exception: {exc}")
        fact_check_result = {
            'status': 'UNAVAILABLE',
            'error': 'Fact-checking service temporarily unavailable.',
            'results': [],
            'fact_checks_found_total': 0,
            'overall_evidence_status': 'UNAVAILABLE'
        }

    # 6. Step 10 Multi-Dimensional Evidence Decision Engine
    art_meta = {
        'title': extracted['title'],
        'author': extracted['author'],
        'publication_date': extracted['publication_date'],
        'source': extracted['source'],
        'url': extracted['url'],
        'word_count': extracted['word_count'],
        'extractor_used': extracted['extractor_used']
    }
    final_dec_report = de.make_final_decision(ml_result, ai_result, fact_check_result, article_metadata=art_meta)

    ai_dict = ai_result if isinstance(ai_result, dict) else {}
    v_summary = ai_dict.get('verification_summary') or {}
    if not isinstance(v_summary, dict):
        v_summary = {}

    # Combine into unified master report
    master_report = {
        'article': {
            'title': extracted['title'],
            'author': extracted['author'],
            'publication_date': extracted['publication_date'],
            'source': extracted['source'],
            'url': extracted['url'],
            'word_count': extracted['word_count'],
            'extraction_status': 'SUCCESS',
            'extractor_used': extracted['extractor_used']
        },
        'prediction': {
            'label': ml_result.get('prediction', 'UNKNOWN') if isinstance(ml_result, dict) else 'UNKNOWN',
            'confidence': ml_result.get('confidence', 0.0) if isinstance(ml_result, dict) else 0.0,
            'confidence_type': ml_result.get('confidence_type', '') if isinstance(ml_result, dict) else '',
            'model_used': ml_result.get('model_used', '') if isinstance(ml_result, dict) else ''
        },
        'final_decision': final_dec_report or {},
        'explainability': {
            'influential_features': (exp_result.get('influential_features') or []) if isinstance(exp_result, dict) else [],
            'suspicious_language': (exp_result.get('suspicious_language') or []) if isinstance(exp_result, dict) else []
        },
        'ai_verification': {
            'overall_status': v_summary.get('overall_status', 'UNVERIFIED'),
            'disagreement_detected': v_summary.get('disagreement_detected', False),
            'disagreement_warning': v_summary.get('disagreement_warning', ''),
            'ai_assessment': ai_dict.get('ai_assessment', ''),
            'claims': ai_dict.get('claim_verifications') or [],
            'sources_analysis': ai_dict.get('sources_analysis') or {},
            'official_sources': ai_dict.get('official_sources') or [],
            'social_sources': ai_dict.get('social_sources') or []
        },
        'fact_checking': fact_check_result or {}
    }

    return master_report


# ---------------------------------------------------------------------------
# Section 4: Formatting & Debug Display Helper
# ---------------------------------------------------------------------------

def display_url_analysis(result):
    """
    Pretty-print the full URL Analysis Result for notebook/console output
    including Prompt Requirement 13 EXTRACTION DEBUG header.
    """
    SEP = '=' * 60

    print(SEP)
    print("  EXTRACTION DEBUG & URL ARTICLE ANALYSIS REPORT")
    print(SEP)

    if 'error' in result or result.get('extraction_status') == 'FAILED':
        art = result.get('article', {})
        print("  EXTRACTION STATUS: FAILED")
        print(f"  URL              : {art.get('url', result.get('url', 'N/A'))}")
        print(f"  Source Domain    : {art.get('source', result.get('source', 'N/A'))}")
        print(f"  Error            : {result.get('error')}")
        if result.get('fallback_suggested'):
            print()
            print("  FALLBACK INSTRUCTION:")
            print("  Automatic extraction failed. You can still analyze this news by")
            print("  pasting the article text manually.")
        print(SEP)
        return

    art = result['article']
    pred_data = result['prediction']
    exp_data = result['explainability']
    ai_data = result['ai_verification']

    print("EXTRACTION DEBUG METADATA:")
    print(f"  Title            : {art['title']}")
    print(f"  Source (Domain)  : {art['source']}")
    print(f"  Publication Date : {art['publication_date']}")
    print(f"  Author           : {art['author']}")
    print(f"  URL              : {art['url']}")
    print(f"  Word Count       : {art['word_count']} words")
    print(f"  Extractor Used   : {art.get('extractor_used', 'trafilatura')}")
    print(f"  Extraction Status: {art['extraction_status']}")
    print()

    print("-" * 60)
    print(f"ML PREDICTION  : {pred_data['label']}")
    print(f"Confidence     : {pred_data['confidence']:.2f}% ({pred_data['confidence_type']})")
    print(f"Model          : {pred_data['model_used']}")
    print("-" * 60)
    print()

    print("EXPLAINABILITY (Top Influential Features):")
    for i, feat in enumerate(exp_data['influential_features'][:5], 1):
        arrow = "-> REAL" if feat['direction'] == 'REAL' else "-> FAKE"
        print(f"  {i}. {feat['word']:25s} {arrow} (Contribution: {feat['contribution']:+.4f})")
    print()

    if exp_data['suspicious_language']:
        print("POTENTIALLY SUSPICIOUS LANGUAGE DETECTED:")
        for s in exp_data['suspicious_language']:
            print(f"  - [{s['category']}] {s['pattern_desc']}")
        print()

    print("-" * 60)
    print(f"AI VERIFICATION STATUS : {ai_data['overall_status']}")
    print(f"Claims Evaluated       : {len(ai_data['claims'])}")
    print(f"AI Assessment Summary  :")
    print(f"  {ai_data['ai_assessment']}")
    print("-" * 60)
    print()

    if ai_data['disagreement_detected']:
        print(f"[!] WARNING: {ai_data['disagreement_warning']}")
        print()

    print("Extraction & Full Pipeline Analysis: SUCCESS")
    print(SEP)
