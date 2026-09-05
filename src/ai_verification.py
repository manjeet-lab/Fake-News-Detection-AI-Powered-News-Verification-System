"""
Fake News Detection — Step 7: AI-Powered Verification & Cross-Source Analysis Engine
======================================================================================

This module builds an intelligent verification layer on top of Step 5 (Prediction)
and Step 6 (Explainability).

Key Workflow:
  1. Receives News Article text.
  2. Runs Step 5 ML Model (Prediction + Confidence).
  3. Runs Step 6 Explainability (Influential Words + Suspicious Language).
  4. Extracts 3-7 verifiable factual claims from the article text.
  5. Searches web, official (.gov/.org), and social sources for each claim.
  6. Classifies sources into Tiers (Tier 1: Official, Tier 2: Established, Tier 3: Other).
  7. Evaluates source independence and detects syndicated reporting.
  8. Uses Google Gemini LLM reasoning (with local NLP fallback) to evaluate evidence.
  9. Produces a structured verification report comparing ML prediction vs AI evidence analysis.

Environment Configuration:
  - Requires GEMINI_API_KEY in `.env` (loaded automatically).
  - Add `.env` to `.gitignore`. Never print or hard-code secrets.

Usage:
    from ai_verification import verify_article, display_verification_report
    report = verify_article("News text...")
    display_verification_report(report)
"""

import json
import logging
import os
import re
import sys
import warnings

from dotenv import load_dotenv

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Ensure current src/ directory is in sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Load environment variables from .env in project root
_env_path = os.path.join(_PROJECT_ROOT, '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

# Import project modules
import prediction as pred
import explainability as exp
import web_search as ws
import source_analysis as sa


# Determine API Provider & Key
_GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
_AI_PROVIDER = os.environ.get('AI_PROVIDER', 'gemini').lower()

# Check Gemini SDK availability
_GENAI_CLIENT = None
_GENAI_TYPE = None

if _GEMINI_KEY:
    try:
        from google import genai
        _GENAI_CLIENT = genai.Client(api_key=_GEMINI_KEY)
        _GENAI_TYPE = 'genai'
    except ImportError:
        try:
            import google.generativeai as gai
            gai.configure(api_key=_GEMINI_KEY)
            _GENAI_CLIENT = gai
            _GENAI_TYPE = 'generativeai'
        except Exception as e:
            logger.warning(f"Could not initialize Google Gemini SDK: {e}")


def _call_llm_json(prompt, system_instruction="You are an expert news verification and fact-checking AI."):
    """
    Call Google Gemini API with fallback to local JSON logic if API key is missing or fails.
    """
    if not _GEMINI_KEY or not _GENAI_CLIENT:
        return None

    try:
        full_prompt = f"{system_instruction}\n\n{prompt}\n\nIMPORTANT: Return ONLY raw, valid JSON with no markdown formatting or commentary."

        if _GENAI_TYPE == 'genai':
            response = _GENAI_CLIENT.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt,
            )
            raw_text = response.text
        elif _GENAI_TYPE == 'generativeai':
            model = _GENAI_CLIENT.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            raw_text = response.text
        else:
            return None

        # Clean JSON markdown blocks if present
        clean_text = raw_text.strip()
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        if clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        return json.loads(clean_text)

    except Exception as e:
        logger.warning(f"LLM API call failed: {e}. Degrading to local NLP fallback.")
        return None


# ---------------------------------------------------------------------------
# Section 1: Claim Extraction
# ---------------------------------------------------------------------------

def extract_claims(article_text, max_claims=5):
    """
    Extract 3-7 verifiable factual claims from the article text.

    Args:
        article_text (str): Raw article text.
        max_claims (int): Maximum claims to extract.

    Returns:
        list[dict]:
            [
                {
                    "claim": "...",
                    "importance": "high" | "medium",
                    "verifiable": True
                }
            ]
    """
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        return []

    # Attempt LLM Claim Extraction first if API available
    prompt = f"""
Analyze the following news article and extract 3 to {max_claims} key VERIFIABLE FACTUAL CLAIMS.
Exclude opinions, predictions, jokes, or vague statements.

Article Text:
{article_text[:2000]}

Return JSON array in exact format:
[
  {{"claim": "exact factual assertion", "importance": "high", "verifiable": true}}
]
"""
    llm_result = _call_llm_json(prompt, "You are a professional fact-checker extracting verifiable factual assertions.")
    if isinstance(llm_result, list) and len(llm_result) > 0:
        cleaned_claims = []
        for c in llm_result[:max_claims]:
            if isinstance(c, dict) and 'claim' in c:
                cleaned_claims.append({
                    'claim': str(c['claim']).strip(),
                    'importance': str(c.get('importance', 'high')).lower(),
                    'verifiable': bool(c.get('verifiable', True))
                })
        if cleaned_claims:
            return cleaned_claims

    # Local Heuristic NLP Extraction Fallback
    return _local_extract_claims(article_text, max_claims)


def _local_extract_claims(text, max_claims=5):
    """Local rule-based fallback for claim extraction."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []

    # Sentences containing numbers, dates, organizations, or action verbs are good candidates
    claim_indicators = ['said', 'announced', 'raised', 'reported', 'found', 'according', 'decided', 'approved', 'signed', 'passed']

    for sent in sentences:
        s_clean = sent.strip()
        if len(s_clean.split()) >= 6:
            has_number = bool(re.search(r'\d+', s_clean))
            has_indicator = any(ind in s_clean.lower() for ind in claim_indicators)
            
            if has_number or has_indicator:
                claims.append({
                    'claim': s_clean,
                    'importance': 'high' if has_number and has_indicator else 'medium',
                    'verifiable': True
                })
                if len(claims) >= max_claims:
                    break

    if not claims and sentences:
        # Fallback to first few substantial sentences
        for s in sentences[:max_claims]:
            if len(s.split()) >= 4:
                claims.append({'claim': s.strip(), 'importance': 'medium', 'verifiable': True})

    return claims[:max_claims]


# ---------------------------------------------------------------------------
# Section 2: Evidence Comparison per Claim
# ---------------------------------------------------------------------------

def compare_evidence(claim_obj, sources):
    """
    Compare a single claim against retrieved external sources.

    Args:
        claim_obj (dict): Claim dictionary containing 'claim'.
        sources (list[dict]): Retrieved search result objects.

    Returns:
        dict:
            {
                "claim": str,
                "status": "SUPPORTED" | "CONTRADICTED" | "PARTIALLY_SUPPORTED" | "UNVERIFIED",
                "supporting_sources": list[str],
                "contradicting_sources": list[str],
                "summary": str
            }
    """
    claim_text = claim_obj.get('claim', '') if isinstance(claim_obj, dict) else str(claim_obj)
    
    if not sources:
        return {
            'claim': claim_text,
            'status': 'UNVERIFIED',
            'supporting_sources': [],
            'contradicting_sources': [],
            'summary': 'No reliable external reporting found to confirm or refute this claim.'
        }

    # Attempt LLM evidence comparison
    prompt = f"""
Compare the following Factual Claim against the retrieved Search Evidence snippets.

Claim: "{claim_text}"

Evidence Snippets:
{json.dumps([{ 'source': s.get('source'), 'snippet': s.get('snippet') } for s in sources], indent=2)}

Determine if the evidence supports, contradicts, partially supports, or cannot verify the claim.

Return JSON object in format:
{{
  "status": "SUPPORTED" | "CONTRADICTED" | "PARTIALLY_SUPPORTED" | "UNVERIFIED",
  "supporting_sources": ["Source Name 1"],
  "contradicting_sources": [],
  "summary": "Concise 1-sentence evaluation"
}}
"""
    llm_eval = _call_llm_json(prompt, "You are a factual evidence verification expert.")
    if isinstance(llm_eval, dict) and 'status' in llm_eval:
        return {
            'claim': claim_text,
            'status': llm_eval.get('status', 'UNVERIFIED').upper(),
            'supporting_sources': llm_eval.get('supporting_sources', []),
            'contradicting_sources': llm_eval.get('contradicting_sources', []),
            'summary': llm_eval.get('summary', 'Evidence evaluated against sources.')
        }

    # Local Rule-based Evidence Comparison Fallback
    return _local_compare_evidence(claim_text, sources)


def _local_compare_evidence(claim_text, sources):
    """Local text matching for claim verification fallback."""
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
    supporting = []
    contradicting = []

    negation_words = ['false', 'denies', 'refutes', 'debunked', 'fake', 'incorrect', 'hoax', 'no evidence', 'untrue']

    for src in sources:
        snip = src.get('snippet', '').lower()
        src_name = src.get('source', 'Unknown')
        snip_words = set(re.findall(r'\b\w{4,}\b', snip))

        overlap = len(claim_words.intersection(snip_words))
        if overlap >= 2:
            is_contradiction = any(neg in snip for neg in negation_words)
            if is_contradiction:
                contradicting.append(src_name)
            else:
                supporting.append(src_name)

    if supporting and contradicting:
        status = 'PARTIALLY_SUPPORTED'
        summary = f"Evidence is mixed: supported by {', '.join(supporting[:2])} but contradicted by {', '.join(contradicting[:2])}."
    elif supporting:
        status = 'SUPPORTED'
        summary = f"Claim supported by {len(supporting)} source(s) including {supporting[0]}."
    elif contradicting:
        status = 'CONTRADICTED'
        summary = f"Claim contradicted by reporting from {', '.join(contradicting[:2])}."
    else:
        status = 'UNVERIFIED'
        summary = "No direct confirmation found in retrieved source snippets."

    return {
        'claim': claim_text,
        'status': status,
        'supporting_sources': list(set(supporting)),
        'contradicting_sources': list(set(contradicting)),
        'summary': summary
    }


# ---------------------------------------------------------------------------
# Section 3: Comprehensive Article Verification Engine
# ---------------------------------------------------------------------------

def verify_article(article_text):
    """
    Main entry point for AI-Powered News Verification & Cross-Source Analysis.

    Integrates:
      - Step 5 ML Model Prediction & Confidence
      - Step 6 Model Explainability & Suspicious Language Detection
      - Step 7 Claim Extraction, Live Web Search, Official Source & Social Checks
      - Source Tier Classification & Independence Analysis
      - ML vs AI Comparison & Disagreement Handling

    Args:
        article_text (str): Raw article text.

    Returns:
        dict: Complete structured verification report for Streamlit or notebook display.
    """
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        return {'error': 'Please provide a valid news article text to verify.'}

    # 1. Step 5 ML Prediction
    ml_res = pred.predict_news(article_text)
    if 'error' in ml_res:
        return ml_res

    # 2. Step 6 Explainability
    exp_res = exp.get_explanation(article_text, top_n=10)

    # 3. Step 7 Factual Claim Extraction
    extracted_claims = extract_claims(article_text, max_claims=5)

    # 4. Search & Evidence Collection per Claim
    all_retrieved_sources = []
    claim_verifications = []
    official_confirmations = []
    social_confirmations = []

    for c_item in extracted_claims:
        claim_str = c_item['claim']

        # Web search for general reporting
        web_sources = ws.search_claim(claim_str, max_results=4)
        all_retrieved_sources.extend(web_sources)

        # Reputable news search — Indian & international Tier 2 outlets
        reputable_sources = ws.search_reputable_sources(claim_str, max_results=6)
        all_retrieved_sources.extend(reputable_sources)

        # Official / Government source search
        off_sources = ws.find_official_sources(claim_str, max_results=3)
        all_retrieved_sources.extend(off_sources)
        if off_sources:
            official_confirmations.extend(off_sources)

        # Social media account check
        soc_res = ws.find_social_confirmation(claim_str, max_results=2)
        if soc_res['status'] == 'CONFIRMED':
            social_confirmations.extend(soc_res['findings'])

        # Compare claim against retrieved sources (reputable + web + official)
        c_sources = reputable_sources + web_sources + off_sources
        eval_res = compare_evidence(c_item, c_sources)
        claim_verifications.append(eval_res)

    # Deduplicate sources by URL
    unique_sources_dict = {}
    for s in all_retrieved_sources:
        url = s.get('url')
        if url and url not in unique_sources_dict:
            unique_sources_dict[url] = s
    unique_sources = list(unique_sources_dict.values())

    # 5. Source Analysis & Tier Classification
    source_analysis_res = sa.analyze_sources(unique_sources)

    # 5a. Reputable Source Confirmation — count how many Tier 2 established news
_STOPWORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'is', 'was', 'are', 'were',
    'been', 'be', 'by', 'that', 'with', 'from', 'as', 'has', 'have', 'had', 'this', 'it',
    'and', 'or', 'but', 'says', 'said', 'according', 'report', 'reports', 'news'
}

def extract_key_concepts(text):
    """Extract non-stopword content terms for relevance matching."""
    if not text or not isinstance(text, str):
        return []
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
    return [w for w in words if w not in _STOPWORDS]


def is_relevant_reputable_match(claim_text, search_item, input_url=""):
    """
    Checks if a search result item is from a configured reputable news outlet (Tier 1/Tier 2)
    AND is relevant to the claim event (not just a generic mention of an entity).
    """
    if not isinstance(search_item, dict):
        return False

    url = search_item.get('url', '')
    source = search_item.get('source', '')
    title = search_item.get('title', '')
    snippet = search_item.get('snippet', '')

    # 1. Ignore input article itself or duplicate URLs
    if input_url and url and input_url.strip().lower() == url.strip().lower():
        return False

    # 2. Source Tier Check: Must be Tier 1 (Official) or Tier 2 (Established News)
    tier_info = sa.classify_source_tier(url, source)
    if tier_info['tier'] not in ('TIER_1_OFFICIAL', 'TIER_2_ESTABLISHED_NEWS'):
        return False

    # 3. Content Relevance Check
    claim_terms = extract_key_concepts(claim_text)
    if not claim_terms:
        return False

    combined_text = (title + " " + snippet).lower()
    search_terms = set(extract_key_concepts(combined_text))

    matched = [t for t in claim_terms if t in search_terms]

    if len(claim_terms) <= 3:
        return len(matched) >= 2
    else:
        ratio = len(matched) / len(claim_terms)
        return ratio >= 0.40 or len(matched) >= 3


def normalize_claim_status(status_str):
    """
    Normalizes claim verification status into standard categories:
    TRUE, FALSE, UNVERIFIED.
    """
    if not status_str or not isinstance(status_str, str):
        return 'UNVERIFIED'
    st = status_str.upper().strip()
    if st in ('SUPPORTED', 'TRUE', 'VERIFIED_TRUE', 'CONFIRMED', 'HIGHLY_SUPPORTED'):
        return 'TRUE'
    elif st in ('CONTRADICTED', 'FALSE', 'VERIFIED_FALSE', 'REFUTED', 'DEBUNKED', 'MISLEADING', 'DISPROVEN'):
        return 'FALSE'
    else:
        return 'UNVERIFIED'


def determine_ai_prediction(claims=None):
    """
    STRICT CLAIM RULE (Requirements 1, 2, 6, 16):
      1. If ANY major claim = FALSE -> FAKE
      2. Else if ANY major claim = UNVERIFIED -> FAKE
      3. Else if ALL major claims = TRUE -> REAL
    REAL only if EVERY major claim is verified TRUE.
    Otherwise -> FAKE.
    """
    if not claims or not isinstance(claims, list):
        return 'FAKE'

    normalized_statuses = []
    for claim in claims:
        st = claim.get('status') if isinstance(claim, dict) else str(claim)
        normalized_statuses.append(normalize_claim_status(st))

    if 'FALSE' in normalized_statuses:
        return 'FAKE'

    if 'UNVERIFIED' in normalized_statuses:
        return 'FAKE'

    if len(normalized_statuses) > 0 and all(status == 'TRUE' for status in normalized_statuses):
        return 'REAL'

    return 'FAKE'


# ---------------------------------------------------------------------------
# Section 3: Comprehensive Article Verification Engine
# ---------------------------------------------------------------------------

def verify_article(article_text, article_url=""):
    """
    Main entry point for AI Web-Search Verification & Cross-Source Analysis.

    Strict Claim-Level Rule:
      - Extracts all major claims
      - Verifies every claim against web & reputable sources
      - ALL claims TRUE -> REAL
      - ANY claim FALSE or UNVERIFIED -> FAKE
    """
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        return {'error': 'Please provide a valid news article text to verify.'}

    # 1. Step 5 ML Prediction
    ml_res = pred.predict_news(article_text)
    if 'error' in ml_res:
        return ml_res

    # 2. Step 6 Explainability
    exp_res = exp.get_explanation(article_text, top_n=10)

    # 3. Step 7 Factual Claim Extraction
    extracted_claims = extract_claims(article_text, max_claims=5)

    # 4. Search & Evidence Collection per Claim
    all_retrieved_sources = []
    claim_verifications = []
    official_confirmations = []
    social_confirmations = []

    for c_item in extracted_claims:
        claim_str = c_item['claim']

        # Search established reputable news outlets
        reputable_sources = ws.search_reputable_sources(claim_str, max_results=6)
        all_retrieved_sources.extend(reputable_sources)

        # General web search fallback
        web_sources = ws.search_claim(claim_str, max_results=4)
        all_retrieved_sources.extend(web_sources)

        # Official / Government source search
        off_sources = ws.find_official_sources(claim_str, max_results=3)
        all_retrieved_sources.extend(off_sources)
        if off_sources:
            official_confirmations.extend(off_sources)

        # Social media account check
        soc_res = ws.find_social_confirmation(claim_str, max_results=2)
        if soc_res['status'] == 'CONFIRMED':
            social_confirmations.extend(soc_res['findings'])

        c_sources = reputable_sources + web_sources + off_sources
        eval_res = compare_evidence(c_item, c_sources)
        claim_verifications.append(eval_res)

    # 5. Filter & Deduplicate Relevant Reputable News Confirmations
    reputable_matching_sources = []
    seen_urls = set()

    for s in all_retrieved_sources:
        url = s.get('url', '')
        if not url or url in seen_urls:
            continue

        # Check relevance against extracted article claims
        is_match = False
        for c_item in extracted_claims:
            claim_str = c_item.get('claim', '') if isinstance(c_item, dict) else str(c_item)
            if is_relevant_reputable_match(claim_str, s, input_url=article_url):
                is_match = True
                break

        if is_match:
            seen_urls.add(url)
            reputable_matching_sources.append({
                'publisher': s.get('source', 'Established News'),
                'headline': s.get('title', 'News Report'),
                'url': url,
                'snippet': s.get('snippet', ''),
                'source': s.get('source', 'Established News'),
                'title': s.get('title', 'News Report')
            })

    reputable_confirmations_count = len(reputable_matching_sources)

    # Calculate claim-level breakdown
    true_claims_cnt = 0
    false_claims_cnt = 0
    unverified_cnt = 0

    for cv in claim_verifications:
        st = cv.get('status', 'UNVERIFIED') if isinstance(cv, dict) else 'UNVERIFIED'
        norm = normalize_claim_status(st)
        if norm == 'TRUE':
            true_claims_cnt += 1
        elif norm == 'FALSE':
            false_claims_cnt += 1
        else:
            unverified_cnt += 1

    # 6. Apply Strict Claim-Level Core Rule:
    # ALL TRUE -> REAL | ANY FALSE or UNVERIFIED -> FAKE
    ai_prediction = determine_ai_prediction(claim_verifications)

    if ai_prediction == "REAL":
        overall_status = "SUPPORTED"
        ai_assessment = "All major factual claims were verified by external sources."
    else:
        overall_status = "UNVERIFIED"
        ai_assessment = "One or more major claims could not be sufficiently verified."

    # Debug Output
    print("\n--- DEBUG OUTPUT ---")
    print("AI Prediction :", ai_prediction)
    print("Claims Checked:", len(extracted_claims))
    print("True Claims   :", true_claims_cnt)
    print("False Claims  :", false_claims_cnt)
    print("Unverified    :", unverified_cnt)
    print("--------------------\n")

    unique_sources = list({s.get('url'): s for s in all_retrieved_sources if s.get('url')}.values())
    source_analysis_res = sa.analyze_sources(unique_sources)

    ml_pred = ml_res['prediction']
    ml_conf = ml_res['confidence']

    disagreement = False
    disagreement_warning = ""
    if ml_pred == 'FAKE' and ai_prediction == 'REAL':
        disagreement = True
        disagreement_warning = (
            "[!] ML prediction and external evidence do not fully agree: "
            "The ML model predicted FAKE based on vocabulary style, but "
            "all major factual claims were verified by external news search."
        )
    elif ml_pred == 'REAL' and ai_prediction == 'FAKE':
        disagreement = True
        disagreement_warning = (
            "[!] ML prediction and external evidence do not fully agree: "
            "The ML model predicted REAL based on formal writing style, but "
            "one or more major claims could not be sufficiently verified."
        )

    # Assemble Final Report
    report = {
        'ml_classification': {
            'prediction': ml_pred,
            'confidence': ml_conf,
            'confidence_type': ml_res.get('confidence_type', ''),
            'model_used': ml_res.get('model_used', ''),
        },
        'explainability_summary': {
            'influential_features': (exp_res.get('influential_features') or [])[:6],
            'suspicious_language': exp_res.get('suspicious_language') or [],
        },
        'verification_summary': {
            'overall_status': overall_status,
            'ai_prediction': ai_prediction,
            'disagreement_detected': disagreement,
            'disagreement_warning': disagreement_warning,
            'claims_count': len(extracted_claims),
            'claims_checked': len(extracted_claims),
            'true_claims': true_claims_cnt,
            'false_claims': false_claims_cnt,
            'unverified_claims': unverified_cnt,
            'reputable_confirmations_count': reputable_confirmations_count,
            'reputable_news_confirmations_count': reputable_confirmations_count,
            'sources_checked_count': len(unique_sources),
            'official_confirmations_count': len(official_confirmations),
            'social_confirmations_count': len(social_confirmations),
            'tier1_official_count': source_analysis_res['tier1_count'],
            'tier2_established_count': source_analysis_res['tier2_count'],
            'independence_status': source_analysis_res['independence_status'],
        },
        'claim_verifications': claim_verifications,
        'claims': claim_verifications,
        'claims_checked': len(extracted_claims),
        'true_claims': true_claims_cnt,
        'false_claims': false_claims_cnt,
        'unverified_claims': unverified_cnt,
        'sources_analysis': source_analysis_res,
        'official_sources': official_confirmations[:3],
        'social_sources': social_confirmations[:3],
        'reputable_sources': reputable_matching_sources,
        'reputable_news_sources': reputable_matching_sources,
        'reputable_confirmations_count': reputable_confirmations_count,
        'reputable_news_confirmations_count': reputable_confirmations_count,
        'ai_assessment': ai_assessment,
        'ai_prediction': ai_prediction,
        'overall_status': overall_status,
        'limitations': [
            "AI verification strictly requires all major factual claims to be verified TRUE.",
            "If any major claim is false or unverified, overall AI prediction becomes FAKE."
        ]
    }

    return report

    return report


def _synthesize_ai_assessment(article_text, ml_prediction, overall_status, claim_verifications, source_analysis, reputable_confirmation_count=0):
    """Generate concise reasoning synthesis using LLM or local template."""
    prompt = f"""
Synthesize a concise 3-sentence verification summary based on this evidence:

ML Model Prediction: {ml_prediction}
External Verification Status: {overall_status}
Total Claims Evaluated: {len(claim_verifications)}
Source Tiers: {source_analysis['tier1_count']} Official, {source_analysis['tier2_count']} Established News.
Reputable News Confirmations: {reputable_confirmation_count} established Indian/international outlets independently reported this.
Source Independence: {source_analysis['independence_status']}

Claims & Statuses:
{json.dumps([{ 'claim': c['claim'], 'status': c['status'], 'summary': c['summary'] } for c in claim_verifications], indent=2)}

Provide a factual, objective assessment explaining whether reporting supports, contradicts, or leaves claims unverified.
"""
    llm_synth = _call_llm_json(prompt, "You are a news verification analyst writing an objective executive summary.")
    if isinstance(llm_synth, str) and len(llm_synth.strip()) > 20:
        return llm_synth.strip()

    # Local template synthesis fallback
    tier1_cnt = source_analysis['tier1_count']
    tier2_cnt = source_analysis['tier2_count']
    total_src = source_analysis['total_sources']

    if reputable_confirmation_count >= 2:
        return (
            f"The article's key factual claims are corroborated by {reputable_confirmation_count} reputable established news "
            f"organization(s) (including Indian and/or international outlets), providing strong independent confirmation. "
            f"Overall verification status: {overall_status}."
        )
    elif overall_status == 'HIGHLY_SUPPORTED':
        return (
            f"The article's key factual claims are strongly corroborated by external reporting across "
            f"{total_src} source(s), including {tier1_cnt} official primary source(s) and {tier2_cnt} established news outlet(s). "
            f"Independent reporting aligns with the core assertions."
        )
    elif overall_status == 'CONFLICTING_EVIDENCE':
        return (
            f"External verification retrieved conflicting or contradictory evidence regarding the main claims. "
            f"While some sources mention the topic, key assertions are refuted by established news or official statements."
        )
    elif overall_status == 'PARTIALLY_SUPPORTED':
        return (
            f"Verification found partial support for the claims in external news reports ({total_src} source(s) checked), "
            f"though official confirmation from primary governing bodies remains limited or unverified."
        )
    else:
        return (
            f"No direct external news reporting or official press announcements were found to verify the specific claims. "
            f"The claims remain unverified by independent Tier 1 or Tier 2 reporting."
        )


# ---------------------------------------------------------------------------
# Section 4: Display Helper
# ---------------------------------------------------------------------------

def display_verification_report(report):
    """
    Pretty-print the full AI-Powered News Verification Report.
    """
    SEP = '=' * 60

    print(SEP)
    print("  STEP 7: AI-POWERED NEWS VERIFICATION & CROSS-SOURCE REPORT")
    print(SEP)
    print()

    if 'error' in report:
        print("  ERROR:", report['error'])
        print(SEP)
        return

    ml = report['ml_classification']
    vs = report['verification_summary']

    print("+----------------------------------------------------------+")
    print(f"| ML MODEL CLASSIFICATION                                  |")
    print(f"|   Prediction : {ml['prediction']:<41} |")
    print(f"|   Confidence : {ml['confidence']:.2f}% ({ml['confidence_type']:<28}) |")
    print("+----------------------------------------------------------+")
    print(f"| AI EXTERNAL VERIFICATION STATUS                          |")
    print(f"|   Status     : {vs['overall_status']:<41} |")
    print(f"|   Claims Evaluated : {vs['claims_count']:<37} |")
    print(f"|   Sources Checked  : {vs['sources_checked_count']:<37} |")
    print(f"|   Tier 1 Official  : {vs['tier1_official_count']:<37} |")
    print(f"|   Tier 2 Established News : {vs['tier2_established_count']:<30} |")
    print("+----------------------------------------------------------+")
    print()

    if vs['disagreement_detected']:
        print(f"[!] WARNING: {vs['disagreement_warning']}")
        print()

    print("AI VERIFICATION ASSESSMENT:")
    print(f"  {report['ai_assessment']}")
    print()

    print("CLAIMS & EVIDENCE BREAKDOWN:")
    for i, c in enumerate(report['claim_verifications'], 1):
        status_icon = "[OK]" if c['status'] == 'SUPPORTED' else ("[-] " if c['status'] == 'CONTRADICTED' else "[?]")
        print(f"  {i}. [{c['status']}] {status_icon} Claim: \"{c['claim']}\"")
        print(f"     Summary: {c['summary']}")
        if c['supporting_sources']:
            print(f"     Supported by: {', '.join(c['supporting_sources'])}")
        if c['contradicting_sources']:
            print(f"     Contradicted by: {', '.join(c['contradicting_sources'])}")
        print()

    print("SOURCE TIER & INDEPENDENCE ANALYSIS:")
    sa_res = report['sources_analysis']
    print(f"  Total Sources : {sa_res['total_sources']}")
    print(f"  Independence  : {sa_res['independence_status']}")
    print(f"  Analysis      : {sa_res['independence_analysis']}")
    print()

    if report['official_sources']:
        print("PRIMARY / OFFICIAL SOURCES FOUND:")
        for off in report['official_sources']:
            print(f"  - {off['source']} -- {off['title']}")
            print(f"    URL: {off['url']}")
        print()

    if report['social_sources']:
        print("PUBLIC OFFICIAL SOCIAL MEDIA POSTS / HANDLES:")
        for soc in report['social_sources']:
            print(f"  - {soc['source']} -- {soc['title']}")
            print(f"    URL: {soc['url']}")
        print()

    print(SEP)
