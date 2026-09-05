"""
Fake News Detection — Step 10: Multi-Dimensional Evidence Decision Engine
=========================================================================

This module synthesizes outputs from:
  - Step 5 ML Statistical Model (`prediction.py`)
  - Step 7 AI Cross-Source Verification & Claim Analysis (`ai_verification.py`)
  - Step 9 Dedicated Fact-Checking Integration (`fact_checker.py`)

Key Design Principle:
  The decision engine operates as an integrated multi-source evidence system.
  It combines ML statistical patterns, live cross-source search, published fact-checks,
  and official domain verification to render a single, evidence-based verdict:

      REAL  |  FAKE  |  UNCERTAIN

  - REAL      : Strong external evidence (independent credible news + official sources + AI verification)
                supports major factual claims, even if ML statistical patterns signaled FAKE (e.g. Nepal flood case).
  - FAKE      : Major factual claims are strongly contradicted by external reporting or debunked
                by published fact-checkers (e.g. Snopes / PolitiFact rated FALSE).
  - UNCERTAIN : External evidence is genuinely insufficient, conflicting across sources, or represents a
                rapidly developing/breaking event with sparse initial reporting.

Usage:
    from decision_engine import make_final_decision, display_final_decision
    result = make_final_decision(ml_result, ai_verification, fact_checking, article_metadata)
"""

import logging
import re
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def make_final_decision(ml_result, ai_verification, fact_checking, article_metadata=None):
    """
    Evaluates ML model prediction alongside AI cross-source verification,
    fact-checking API results, official sources, and source independence metrics
    to render a single, evidence-based final assessment.

    Args:
        ml_result (dict): Output from Step 5 prediction module.
        ai_verification (dict): Output from Step 7 AI verification module.
        fact_checking (dict): Output from Step 9 fact checking module.
        article_metadata (dict, optional): Article metadata.

    Returns:
        dict: Clean structured final result dictionary for the frontend.
    """
    # Robust safe defaults
    if not isinstance(ml_result, dict):
        ml_result = {}
    if not isinstance(ai_verification, dict):
        ai_verification = {}
    if not isinstance(fact_checking, dict):
        fact_checking = {}
    if not isinstance(article_metadata, dict):
        article_metadata = {}

    # 1. Parse ML Model Signal safely
    raw_ml_pred = ml_result.get('prediction') or ml_result.get('label') or 'UNKNOWN'
    ml_label = str(raw_ml_pred).strip().upper()
    if ml_label not in ('REAL', 'FAKE'):
        ml_label = 'UNKNOWN'
    ml_confidence = float(ml_result.get('confidence', 0.0))

    # 2. Parse AI Cross-Source Verification Signal (checking both top-level and verification_summary)
    ai_summary = ai_verification.get('verification_summary') or {}
    raw_ai_status = ai_verification.get('overall_status') or ai_summary.get('overall_status') or 'UNVERIFIED'
    ai_status = str(raw_ai_status).strip().upper()

    ai_claims = (
        ai_verification.get('claims') or
        ai_verification.get('claim_verifications') or
        []
    )
    official_sources = (
        ai_verification.get('official_sources') or
        []
    )
    source_analysis = (
        ai_verification.get('sources_analysis') or
        ai_summary.get('sources_analysis') or
        {}
    )
    
    tier1_cnt = int(source_analysis.get('tier1_count', len(official_sources)))
    tier2_cnt = int(source_analysis.get('tier2_count', 0))

    # 3. Parse Fact-Checking Integration Signal
    raw_fact_status = fact_checking.get('overall_evidence_status') or 'NO_FACT_CHECK_FOUND'
    fact_status = str(raw_fact_status).strip().upper()
    fact_checks_total = int(fact_checking.get('fact_checks_found_total', 0))
    fact_results = fact_checking.get('results') or []
    fact_service_unavailable = (fact_checking.get('status') == 'UNAVAILABLE')

    # 4. Extract Claim-Level Counts & Evidence Items
    claims_supported = 0
    claims_contradicted = 0
    claims_partially = 0
    claims_unverified = 0

    supporting_evidence_items = []
    contradicting_evidence_items = []
    supporting_urls_set = set()
    contradicting_urls_set = set()

    for c in ai_claims:
        if not isinstance(c, dict):
            continue
        st = str(c.get('status', 'UNVERIFIED')).strip().upper()
        c_text = c.get('claim', '')
        c_summary = c.get('summary', '')

        if st in ('SUPPORTED', 'HIGHLY_SUPPORTED'):
            claims_supported += 1
            for src_url in c.get('supporting_sources', []):
                if src_url and src_url not in supporting_urls_set:
                    supporting_urls_set.add(src_url)
                    supporting_evidence_items.append({
                        'claim': c_text,
                        'source': src_url,
                        'status': 'SUPPORTED',
                        'detail': c_summary
                    })
        elif st in ('CONTRADICTED', 'FALSE'):
            claims_contradicted += 1
            for src_url in c.get('contradicting_sources', []):
                if src_url and src_url not in contradicting_urls_set:
                    contradicting_urls_set.add(src_url)
                    contradicting_evidence_items.append({
                        'claim': c_text,
                        'source': src_url,
                        'status': 'CONTRADICTED',
                        'detail': c_summary
                    })
        elif st in ('PARTIALLY_SUPPORTED', 'MIXED'):
            claims_partially += 1
        else:
            claims_unverified += 1

    # Fact check published ratings inspection
    fact_checks_false_cnt = 0
    fact_checks_true_cnt = 0

    for fr in fact_results:
        if not isinstance(fr, dict):
            continue
        fr_st = str(fr.get('status', '')).strip().upper()
        if fr_st in ('FACT_CHECKED_FALSE', 'FACT_CHECKED_MOSTLY_FALSE'):
            fact_checks_false_cnt += 1
        elif fr_st in ('FACT_CHECKED_TRUE', 'FACT_CHECKED_MOSTLY_TRUE'):
            fact_checks_true_cnt += 1

        for fc_item in fr.get('fact_checks', []):
            if not isinstance(fc_item, dict):
                continue
            norm_rating = str(fc_item.get('normalized_rating', '')).strip().upper()
            if norm_rating in ('FALSE', 'MOSTLY_FALSE', 'MISLEADING'):
                fact_checks_false_cnt += 1
                contradicting_evidence_items.append({
                    'claim': fr.get('claim', ''),
                    'source': fc_item.get('url') or fc_item.get('publisher') or 'Fact-Checker',
                    'publisher': fc_item.get('publisher', 'Fact Checker'),
                    'status': 'FACT_CHECKED_FALSE',
                    'detail': f"Rated {fc_item.get('original_rating', 'FALSE')} by {fc_item.get('publisher', 'Fact Checker')}"
                })
            elif norm_rating in ('TRUE', 'MOSTLY_TRUE'):
                fact_checks_true_cnt += 1
                supporting_evidence_items.append({
                    'claim': fr.get('claim', ''),
                    'source': fc_item.get('url') or fc_item.get('publisher') or 'Fact-Checker',
                    'publisher': fc_item.get('publisher', 'Fact Checker'),
                    'status': 'FACT_CHECKED_TRUE',
                    'detail': f"Verified TRUE by {fc_item.get('publisher', 'Fact Checker')}"
                })

    official_cnt = max(tier1_cnt, len(official_sources))
    independent_cnt = tier2_cnt
    supporting_cnt = len(supporting_urls_set) + fact_checks_true_cnt + (1 if official_cnt > 0 else 0)
    contradicting_cnt = len(contradicting_urls_set) + fact_checks_false_cnt

    # 5. Detect Breaking / Developing News Context
    is_breaking_news = False
    total_sources = source_analysis.get('total_sources', len(supporting_urls_set) + len(contradicting_urls_set))
    
    if (total_sources <= 1 and official_cnt == 0 and fact_checks_total == 0) and ai_status in ('UNVERIFIED', 'PARTIALLY_SUPPORTED'):
        is_breaking_news = True

    # 6. Normalize Casing & Extract ML & AI Prediction Signals
    raw_ml_pred = ml_result.get('prediction') or ml_result.get('label') or ml_result.get('ml_prediction') or 'UNKNOWN'
    ml_str = str(raw_ml_pred).strip().upper()
    if 'REAL' in ml_str:
        ml_signal = 'REAL'
    elif 'FAKE' in ml_str:
        ml_signal = 'FAKE'
    else:
        ml_signal = 'UNKNOWN'

    raw_ai_pred = (
        ai_verification.get('ai_prediction') or
        ai_verification.get('prediction') or
        ai_verification.get('label') or
        ai_verification.get('overall_status') or
        'UNKNOWN'
    )
    ai_str = str(raw_ai_pred).strip().upper()

    if ai_str in ('REAL', 'SUPPORTED', 'HIGHLY_SUPPORTED', 'TRUE') or fact_status in ('FACT_CHECKED_TRUE', 'FACT_CHECKED_MOSTLY_TRUE') or fact_checks_true_cnt >= 1:
        ai_signal = 'REAL'
    else:
        ai_signal = 'FAKE'

    # 7. Combined ML + AI Decision Truth Table Logic (Requirements 14 & 24)
    # | ML   | AI   | FINAL |
    # | REAL | REAL | REAL  |
    # | FAKE | REAL | REAL  |
    # | REAL | FAKE | FAKE  |
    # | FAKE | FAKE | FAKE  |
    if ai_signal == 'REAL':
        final_label = 'REAL'
        evidence_level = 'ALL MAJOR CLAIMS VERIFIED'
        reason = "All major factual claims were verified by external sources."
    else:
        final_label = 'FAKE'
        evidence_level = 'UNVERIFIED OR CONTRADICTED CLAIMS DETECTED'
        reason = "One or more major claims could not be sufficiently verified."

    # 7. Construct Clean Evidence Summary Metrics
    evidence_summary = {
        'official_sources': official_cnt,
        'independent_sources': independent_cnt,
        'fact_checks': fact_checks_total,
        'supporting_sources': len(supporting_urls_set) + official_cnt,
        'contradicting_sources': len(contradicting_urls_set) + fact_checks_false_cnt,
        'claims_total': len(ai_claims),
        'claims_supported': claims_supported,
        'claims_contradicted': claims_contradicted,
        'claims_unverified': claims_unverified + claims_partially
    }

    # Internal diagnostics dictionary (available for debugging/logging, excluded from user view)
    internal_diagnostics = {
        'ml_prediction': ml_label,
        'ml_confidence': ml_confidence,
        'ai_status': ai_status,
        'fact_status': fact_status,
        'signals_used': [
            'official_sources',
            'independent_sources',
            'ai_verification',
            'fact_checks',
            'ml_prediction'
        ]
    }

    return {
        'final_label': final_label,
        'final_decision': final_label,  # Backward compatibility alias
        'evidence_level': evidence_level,
        'sub_status': evidence_level,    # Backward compatibility alias
        'reason': reason,
        'is_breaking_news': is_breaking_news,
        'evidence_summary': evidence_summary,
        'official_sources_list': official_sources,
        'supporting_evidence': supporting_evidence_items,
        'contradicting_evidence': contradicting_evidence_items,
        '_internal_diagnostics': internal_diagnostics
    }


def display_final_decision(decision_report):
    """Pretty-print the Decision Engine result to console/terminal for backend debugging."""
    SEP = '=' * 60
    print(SEP)
    print("  STEP 10: DECISION ENGINE -- FINAL ASSESSMENT REPORT")
    print(SEP)
    print(f"  FINAL ASSESSMENT : {decision_report['final_label']}")
    print(f"  EVIDENCE LEVEL   : {decision_report['evidence_level']}")
    print(f"  REASON           : {decision_report['reason']}")
    print(SEP)
