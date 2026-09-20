"""
Fake News Detection — Step 9: Fact-Checking API Integration & Evidence Validation
===================================================================================

This module provides a dedicated fact-checking layer that searches established
fact-checking organizations (via Google Fact Check Tools API or targeted web search)
to determine whether specific claims in a news article have already been investigated
and debunked/verified.

Key Features:
  1. Google Fact Check Tools API Integration (via GOOGLE_FACT_CHECK_API_KEY in .env).
  2. Targeted Fact-Check Web Search Fallback across Snopes, PolitiFact, Reuters Fact Check,
     AP Fact Check, Full Fact, FactCheck.org, and Lead Stories.
  3. Rating Normalization into standardized categories (TRUE, MOSTLY_TRUE, MIXED,
     MISLEADING, MOSTLY_FALSE, FALSE, UNVERIFIED).
  4. Claim Relevance Filtering & Semantic Overlap Guard (prevents false matches).
  5. Multi-Fact-Check Aggregation & Conflicting Fact Check Detection.
  6. Reuses claims extracted in Step 7 (`ai_verification`) to prevent duplicate processing.
  7. In-memory query caching & graceful error fallback (never crashes the pipeline).

Usage:
    from fact_checker import fact_check_article, display_fact_check_report
    report = fact_check_article(article_text, claims=existing_claims)
    display_fact_check_report(report)
"""

import json
import logging
import os
import re
import sys
import urllib.parse
import warnings

import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Ensure current src/ directory is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Load environment variables
_env_path = os.path.join(_PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

import web_search as ws

# Google Fact Check Tools API Key
_GOOGLE_FACTCHECK_KEY = os.environ.get('GOOGLE_FACT_CHECK_API_KEY', '').strip()

# Simple in-memory search query cache
_FACT_CHECK_CACHE = {}


# ---------------------------------------------------------------------------
# Section 1: Established Fact-Check Publishers & Quality Mapping
# ---------------------------------------------------------------------------

ESTABLISHED_FACT_CHECKERS = {
    'snopes.com': ('Snopes', 'HIGH'),
    'politifact.com': ('PolitiFact', 'HIGH'),
    'reuters.com': ('Reuters Fact Check', 'HIGH'),
    'apnews.com': ('AP Fact Check', 'HIGH'),
    'fullfact.org': ('Full Fact', 'HIGH'),
    'factcheck.org': ('FactCheck.org', 'HIGH'),
    'afp.com': ('AFP Fact Check', 'HIGH'),
    'leadstories.com': ('Lead Stories', 'HIGH'),
    'altnews.in': ('Alt News', 'HIGH'),
    'boomlive.in': ('BOOM Live', 'HIGH'),
    'washingtonpost.com': ('Washington Post Fact Checker', 'HIGH'),
}


def _get_publisher_quality(publisher_name, url=''):
    """Identify publisher name and quality tier (HIGH, MEDIUM, UNKNOWN)."""
    pub_lower = (publisher_name or '').lower()
    url_lower = (url or '').lower()

    for domain, (name, quality) in ESTABLISHED_FACT_CHECKERS.items():
        if domain in url_lower or domain in pub_lower:
            return name, quality

    if any(kw in pub_lower for kw in ['fact check', 'fact-check', 'factcheck', 'verifier']):
        return publisher_name or 'Independent Fact-Checker', 'MEDIUM'

    return publisher_name or 'Web Fact-Checker', 'UNKNOWN'


# ---------------------------------------------------------------------------
# Section 2: Fact-Check Rating Normalization
# ---------------------------------------------------------------------------

def normalize_rating(rating_str):
    """
    Normalize arbitrary publisher rating strings into standardized internal categories:
    TRUE, MOSTLY_TRUE, MIXED, MISLEADING, MOSTLY_FALSE, FALSE, UNVERIFIED.

    Args:
        rating_str (str): Original textual rating from fact-checker.

    Returns:
        dict:
            {
                "original_rating": str,
                "normalized_rating": str
            }
    """
    if not rating_str or not isinstance(rating_str, str):
        return {'original_rating': 'N/A', 'normalized_rating': 'UNVERIFIED'}

    r_lower = rating_str.strip().lower()

    # FALSE / DEBUNKED / HOAX
    if any(kw in r_lower for kw in ['false', 'pants on fire', 'fake', 'hoax', 'incorrect', 'debunked', 'refuted', 'wrong', 'untrue', 'fabricated', 'baseless']):
        if 'mostly false' in r_lower:
            norm = 'MOSTLY_FALSE'
        else:
            norm = 'FALSE'

    # MOSTLY FALSE / BARELY TRUE
    elif any(kw in r_lower for kw in ['mostly false', 'barely true', 'partially false']):
        norm = 'MOSTLY_FALSE'

    # MISLEADING / OUT OF CONTEXT
    elif any(kw in r_lower for kw in ['misleading', 'context', 'altered', 'exaggerated', 'distorted', 'deceptive', 'spin', 'cherry-picked']):
        norm = 'MISLEADING'

    # MIXED / HALF TRUE
    elif any(kw in r_lower for kw in ['half true', 'mixed', 'half-true', 'partly true', 'inconclusive', 'disputed']):
        norm = 'MIXED'

    # MOSTLY TRUE
    elif any(kw in r_lower for kw in ['mostly true', 'largely true', 'substantially true']):
        norm = 'MOSTLY_TRUE'

    # TRUE / CORRECT / VERIFIED
    elif any(kw in r_lower for kw in ['true', 'correct', 'accurate', 'confirmed', 'verified']):
        norm = 'TRUE'

    else:
        norm = 'UNVERIFIED'

    return {
        'original_rating': rating_str.strip(),
        'normalized_rating': norm
    }


# ---------------------------------------------------------------------------
# Section 3: Search Query Generation & Fact-Check Search
# ---------------------------------------------------------------------------

def _clean_claim_for_search(claim_text):
    """Generate concise search query terms from claim text."""
    clean = re.sub(r'[^\w\s]', '', claim_text).lower()
    words = [w for w in clean.split() if len(w) > 3 and w not in ('this', 'that', 'with', 'from', 'they', 'have', 'were', 'been', 'said', 'according')]
    return ' '.join(words[:6])


def search_fact_checks(claim_text, max_results=4):
    """
    Search established fact-checking sources for investigations matching a claim.

    Uses Google Fact Check Tools API if GOOGLE_FACT_CHECK_API_KEY is configured,
    with targeted web search fallback across top fact-checking domains.

    Args:
        claim_text (str): Factual claim statement.
        max_results (int): Max results to retrieve.

    Returns:
        list[dict]: Unified fact-check records.
    """
    if not claim_text or not isinstance(claim_text, str) or not claim_text.strip():
        return []

    cache_key = claim_text.strip().lower()
    if cache_key in _FACT_CHECK_CACHE:
        return _FACT_CHECK_CACHE[cache_key]

    results = []

    # 1. Strategy A: Google Fact Check Tools API
    if _GOOGLE_FACTCHECK_KEY:
        try:
            query = _clean_claim_for_search(claim_text)
            api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'query': query,
                'key': _GOOGLE_FACTCHECK_KEY,
                'languageCode': 'en',
                'pageSize': max_results
            }
            resp = requests.get(api_url, params=params, timeout=8)

            if resp.status_code == 200:
                data = resp.json()
                claims_data = data.get('claims', [])

                for item in claims_data:
                    c_reviewed = item.get('text', '')
                    reviews = item.get('claimReview', [])

                    for rev in reviews:
                        if not isinstance(rev, dict):
                            continue
                        pub_info = rev.get('publisher') or {}
                        pub_name = pub_info.get('name', 'Fact-Checker') if isinstance(pub_info, dict) else 'Fact-Checker'
                        pub_site = pub_info.get('site', '') if isinstance(pub_info, dict) else ''
                        url = rev.get('url', '')
                        title = rev.get('title', c_reviewed or 'Fact Check Review')
                        raw_rating = rev.get('textualRating', 'Unverified')
                        review_date = rev.get('reviewDate', 'N/A')

                        norm = normalize_rating(raw_rating)
                        resolved_pub, quality = _get_publisher_quality(pub_name, pub_site or url)

                        results.append({
                            'publisher': resolved_pub,
                            'publisher_url': pub_site or ws._extract_domain(url),
                            'url': url,
                            'title': title,
                            'original_rating': norm['original_rating'],
                            'normalized_rating': norm['normalized_rating'],
                            'source_quality': quality,
                            'date': review_date[:10] if len(review_date) >= 10 else review_date,
                            'claim_reviewed': c_reviewed or claim_text
                        })

        except Exception as e:
            logger.warning(f"Google Fact Check API query failed: {e}. Falling back to web search.")

    # 2. Strategy B: Targeted Fact-Check Web Search Fallback
    if not results:
        results = _fallback_fact_check_search(claim_text, max_results=max_results)

    _FACT_CHECK_CACHE[cache_key] = results[:max_results]
    return results[:max_results]


def _fallback_fact_check_search(claim_text, max_results=4):
    """Search DuckDuckGo specifically targeting top fact-checking domains."""
    query = f"\"{_clean_claim_for_search(claim_text)}\" site:snopes.com OR site:politifact.com OR site:reuters.com/fact-check OR site:apnews.com OR site:fullfact.org OR site:factcheck.org OR site:leadstories.com"
    web_res = ws.search_claim(query, max_results=max_results)

    fact_checks = []
    for item in web_res:
        url = item.get('url', '')
        title = item.get('title', '')
        snippet = item.get('snippet', '')
        domain = item.get('domain', '')

        # Filter to ensure URL/domain is from a fact-checking site
        if any(fc_dom in url or fc_dom in domain for fc_dom in ESTABLISHED_FACT_CHECKERS):
            resolved_pub, quality = _get_publisher_quality(item.get('source', ''), url)
            
            # Infer rating from snippet/title keywords
            rating_match = re.search(r'\b(false|mostly false|misleading|true|mostly true|mixture|pants on fire|half true|debunked)\b', (title + " " + snippet), re.I)
            raw_rating = rating_match.group(1).title() if rating_match else 'Fact Checked'
            norm = normalize_rating(raw_rating)

            fact_checks.append({
                'publisher': resolved_pub,
                'publisher_url': domain,
                'url': url,
                'title': title,
                'original_rating': norm['original_rating'],
                'normalized_rating': norm['normalized_rating'],
                'source_quality': quality,
                'date': item.get('published_date', 'N/A'),
                'claim_reviewed': snippet[:150]
            })

    return fact_checks


# ---------------------------------------------------------------------------
# Section 4: Claim-Level Fact Check Evaluation
# ---------------------------------------------------------------------------

def fact_check_claim(claim_obj):
    """
    Perform fact-checking evaluation for a single factual claim.

    Args:
        claim_obj (dict or str): Claim dictionary or text string.

    Returns:
        dict:
            {
                "claim": str,
                "status": str (normalized status or NO_FACT_CHECK_FOUND),
                "confidence": float,
                "fact_checks_found_count": int,
                "fact_checks": list[dict],
                "reason": str,
                "independence": str
            }
    """
    claim_text = claim_obj.get('claim', '') if isinstance(claim_obj, dict) else str(claim_obj)
    
    if not claim_text or not claim_text.strip():
        return {
            'claim': '',
            'status': 'NO_FACT_CHECK_FOUND',
            'confidence': 0.0,
            'fact_checks_found_count': 0,
            'fact_checks': [],
            'reason': 'Empty claim provided.',
            'independence': 'UNKNOWN'
        }

    raw_checks = search_fact_checks(claim_text, max_results=4)

    if not raw_checks:
        return {
            'claim': claim_text,
            'status': 'NO_FACT_CHECK_FOUND',
            'confidence': 0.0,
            'fact_checks_found_count': 0,
            'fact_checks': [],
            'reason': 'No relevant fact-check was found in established fact-checking sources for this claim.',
            'independence': 'UNKNOWN'
        }

    # Filter out weak/irrelevant matches (relevance check)
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
    relevant_checks = []

    for fc in raw_checks:
        fc_text = (fc.get('title', '') + " " + fc.get('claim_reviewed', '')).lower()
        fc_words = set(re.findall(r'\b\w{4,}\b', fc_text))
        overlap = len(claim_words.intersection(fc_words))
        
        # Require at least 2 significant word overlaps
        if overlap >= 2 or len(raw_checks) == 1:
            relevant_checks.append(fc)

    if not relevant_checks:
        return {
            'claim': claim_text,
            'status': 'NO_FACT_CHECK_FOUND',
            'confidence': 0.0,
            'fact_checks_found_count': 0,
            'fact_checks': [],
            'reason': 'Retrieved fact-checks were not sufficiently relevant to this specific claim.',
            'independence': 'UNKNOWN'
        }

    # Analyze ratings across fact-checkers
    ratings = [fc['normalized_rating'] for fc in relevant_checks]
    publishers = [fc['publisher'] for fc in relevant_checks]

    # Check for conflicting ratings
    has_false = any(r in ('FALSE', 'MOSTLY_FALSE', 'MISLEADING') for r in ratings)
    has_true = any(r in ('TRUE', 'MOSTLY_TRUE') for r in ratings)

    if has_false and has_true:
        status = 'CONFLICTING_FACT_CHECKS'
        reason = f"Conflicting ratings found across fact-checkers ({', '.join(set(publishers))})."
        confidence = 0.50
    else:
        # Most frequent rating
        dominant_rating = max(set(ratings), key=ratings.count)
        if dominant_rating == 'FALSE':
            status = 'FACT_CHECKED_FALSE'
            reason = f"Investigated and rated FALSE by {publishers[0]}."
        elif dominant_rating == 'MOSTLY_FALSE':
            status = 'FACT_CHECKED_MOSTLY_FALSE'
            reason = f"Investigated and rated MOSTLY FALSE by {publishers[0]}."
        elif dominant_rating == 'MISLEADING':
            status = 'FACT_CHECKED_MISLEADING'
            reason = f"Investigated and rated MISLEADING by {publishers[0]}."
        elif dominant_rating == 'MIXED':
            status = 'FACT_CHECKED_MIXED'
            reason = f"Investigated and rated MIXED evidence by {publishers[0]}."
        elif dominant_rating == 'MOSTLY_TRUE':
            status = 'FACT_CHECKED_MOSTLY_TRUE'
            reason = f"Investigated and rated MOSTLY TRUE by {publishers[0]}."
        elif dominant_rating == 'TRUE':
            status = 'FACT_CHECKED_TRUE'
            reason = f"Investigated and rated TRUE by {publishers[0]}."
        else:
            status = 'FACT_CHECK_FOUND'
            reason = f"Existing fact-check found from {publishers[0]}."

        confidence = 0.90 if relevant_checks[0]['source_quality'] == 'HIGH' else 0.75

    return {
        'claim': claim_text,
        'status': status,
        'confidence': confidence,
        'fact_checks_found_count': len(relevant_checks),
        'fact_checks': relevant_checks,
        'reason': reason,
        'independence': 'appears_to_reference_same_report' if len(relevant_checks) > 1 else 'UNKNOWN'
    }


# ---------------------------------------------------------------------------
# Section 5: Article-Level Fact Check Engine — fact_check_article()
# ---------------------------------------------------------------------------

def fact_check_article(article_text, claims=None):
    """
    Main entry point for Step 9 Fact-Checking Integration.

    Args:
        article_text (str): Raw article text.
        claims (list[dict] or list[str], optional): Pre-extracted claims from Step 7
            to avoid duplicate processing.

    Returns:
        dict:
            {
                "status": "COMPLETED" | "UNAVAILABLE" | "NO_FACT_CHECKS_FOUND",
                "claims_checked": int,
                "fact_checks_found_total": int,
                "overall_evidence_status": str,
                "results": list[dict],
                "summary": str
            }
    """
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        return {'status': 'UNAVAILABLE', 'error': 'Empty article text provided.', 'results': []}

    # 1. Use Step 7 claims if supplied, else extract claims via Step 7 AI Verification
    if not claims:
        try:
            import ai_verification as av
            extracted_objs = av.extract_claims(article_text, max_claims=5)
        except Exception as e:
            logger.warning(f"Could not extract claims via ai_verification: {e}")
            extracted_objs = [{'claim': article_text[:200], 'importance': 'high'}]
    else:
        extracted_objs = claims

    if not extracted_objs:
        return {
            'status': 'NO_FACT_CHECKS_FOUND',
            'claims_checked': 0,
            'fact_checks_found_total': 0,
            'overall_evidence_status': 'NO_FACT_CHECK_FOUND',
            'results': [],
            'summary': 'No verifiable factual claims extracted for fact-checking.'
        }

    # 2. Fact check each claim (limit 3-5 claims for efficiency)
    results = []
    total_checks_found = 0

    for c_obj in extracted_objs[:5]:
        c_res = fact_check_claim(c_obj)
        results.append(c_res)
        total_checks_found += c_res['fact_checks_found_count']

    # 3. Aggregate overall evidence status
    statuses = [r['status'] for r in results]
    
    if any(s in ('FACT_CHECKED_FALSE', 'FACT_CHECKED_MOSTLY_FALSE') for s in statuses):
        overall_status = 'FACT_CHECKED_FALSE'
        summary = f"One or more factual claims in this article have been debunked or rated FALSE by established fact-checking organizations."
    elif any(s == 'FACT_CHECKED_MISLEADING' for s in statuses):
        overall_status = 'FACT_CHECKED_MISLEADING'
        summary = f"One or more factual claims have been rated MISLEADING by established fact-checking organizations."
    elif any(s in ('FACT_CHECKED_TRUE', 'FACT_CHECKED_MOSTLY_TRUE') for s in statuses) and not any('FALSE' in s for s in statuses):
        overall_status = 'FACT_CHECKED_TRUE'
        summary = f"Key factual claims in this article have been verified as TRUE by established fact-checking organizations."
    elif 'CONFLICTING_FACT_CHECKS' in statuses or ('FACT_CHECKED_TRUE' in statuses and 'FACT_CHECKED_FALSE' in statuses):
        overall_status = 'MIXED_EVIDENCE'
        summary = f"Fact-checking organizations returned mixed or conflicting ratings across claims."
    elif total_checks_found > 0:
        overall_status = 'FACT_CHECK_FOUND'
        summary = f"Found {total_checks_found} relevant fact-check review(s) for the claims in this article."
    else:
        overall_status = 'NO_FACT_CHECK_FOUND'
        summary = f"No previous fact-checks were found for the specific claims in this article in configured fact-checking databases."

    return {
        'status': 'COMPLETED',
        'claims_checked': len(results),
        'fact_checks_found_total': total_checks_found,
        'overall_evidence_status': overall_status,
        'results': results,
        'summary': summary
    }


# ---------------------------------------------------------------------------
# Section 6: Pretty-Print Display Helper
# ---------------------------------------------------------------------------

def display_fact_check_report(report):
    """Pretty-print fact-checking report for notebook/console output."""
    SEP = '=' * 60

    print(SEP)
    print("  STEP 9: FACT-CHECKING INTEGRATION & EVIDENCE REPORT")
    print(SEP)
    print()

    if report.get('status') == 'UNAVAILABLE':
        print("  STATUS: UNAVAILABLE")
        print(f"  Note: {report.get('error', 'Fact-checking service unavailable.')}")
        print()
        print(SEP)
        return

    print(f"  Overall Fact-Check Status : {report['overall_evidence_status']}")
    print(f"  Claims Checked            : {report['claims_checked']}")
    print(f"  Fact-Checks Found         : {report['fact_checks_found_total']}")
    print(f"  Summary                   : {report['summary']}")
    print()
    print("-" * 60)

    for i, item in enumerate(report.get('results', []), 1):
        print(f"  CLAIM {i}: \"{item['claim']}\"")
        print(f"    Status: {item['status']} (Confidence: {int(item['confidence']*100)}%)")
        print(f"    Reason: {item['reason']}")
        
        if item['fact_checks']:
            print("    Fact-Checks Found:")
            for fc in item['fact_checks']:
                print(f"      - [{fc['publisher']}] Rating: {fc['original_rating']} ({fc['normalized_rating']})")
                print(f"        Title: {fc['title']}")
                print(f"        URL  : {fc['url']}")
        else:
            print("    Fact-Checks Found: None")
        print()

    print(SEP)
