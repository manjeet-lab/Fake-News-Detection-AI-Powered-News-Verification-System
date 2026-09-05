"""
Fake News Detection — Step 7: Source Analysis & Credibility Classification
===========================================================================

This module implements:
  1. Source Tier Classification (Tier 1: Official/Primary, Tier 2: Established News, Tier 3: Other).
  2. Source Independence Analysis — detects if multiple news sources are simply syndicating
     or referencing the same original wire service / primary report.
  3. Source Reputation & Credibility Analysis.

Usage:
    from source_analysis import classify_source_tier, detect_source_independence, analyze_sources
"""

import re
import urllib.parse

# ---------------------------------------------------------------------------
# Source Credibility Knowledge Base
# ---------------------------------------------------------------------------

TIER1_PRIMARY_DOMAINS = {
    'gov', 'gov.in', 'gov.uk', 'gov.au', 'gov.ca', 'mil', 'edu',
    'federalreserve.gov', 'whitehouse.gov', 'who.int', 'un.org', 'cdc.gov',
    'nasa.gov', 'nih.gov', 'pib.gov.in', 'sec.gov', 'ftc.gov', 'ecb.europa.eu'
}

TIER2_ESTABLISHED_NEWS = {
    # ── International Wire Services & Broadcasters ──────────────────────────
    'reuters.com': 'Reuters',
    'apnews.com': 'Associated Press',
    'afp.com': 'Agence France-Presse (AFP)',
    'bbc.com': 'BBC News',
    'bbc.co.uk': 'BBC News',
    'aljazeera.com': 'Al Jazeera',
    'bloomberg.com': 'Bloomberg',
    'wsj.com': 'The Wall Street Journal',
    'nytimes.com': 'The New York Times',
    'washingtonpost.com': 'The Washington Post',
    'guardian.com': 'The Guardian',
    'theguardian.com': 'The Guardian',
    'cnn.com': 'CNN',
    'nbcnews.com': 'NBC News',
    'cbsnews.com': 'CBS News',
    'abcnews.go.com': 'ABC News',
    'economist.com': 'The Economist',
    'ft.com': 'Financial Times',
    'npr.org': 'NPR',
    'time.com': 'TIME',
    'newsweek.com': 'Newsweek',

    # ── Indian National News Organizations ──────────────────────────────────
    'ndtv.com': 'NDTV',
    'thehindu.com': 'The Hindu',
    'indianexpress.com': 'The Indian Express',
    'hindustantimes.com': 'Hindustan Times',
    'timesofindia.com': 'Times of India',
    'timesofindia.indiatimes.com': 'Times of India',
    'indiatoday.in': 'India Today',
    'aajtak.in': 'Aaj Tak',
    'zeenews.india.com': 'Zee News',
    'news18.com': 'News18',
    'ani.in': 'ANI (Asian News International)',
    'aninews.in': 'ANI (Asian News International)',
    'ptinews.com': 'PTI (Press Trust of India)',
    'theprint.in': 'The Print',
    'thewire.in': 'The Wire',
    'scroll.in': 'Scroll.in',
    'livemint.com': 'Mint',
    'financialexpress.com': 'Financial Express',
    'businessstandard.com': 'Business Standard',
    'business-standard.com': 'Business Standard',
    'deccanherald.com': 'Deccan Herald',
    'telegraphindia.com': 'The Telegraph India',
    'wionews.com': 'WION',
    'indiatvnews.com': 'India TV News',
    'abplive.com': 'ABP Live',
    'abpnews.com': 'ABP News',
    'tv9bharatvarsh.com': 'TV9 Bharatvarsh',
    'republicworld.com': 'Republic World',
    'firstpost.com': 'Firstpost',
    'moneycontrol.com': 'Moneycontrol',
}


def _clean_domain(url_or_domain):
    """Normalize a domain string from URL or raw domain input."""
    if not url_or_domain:
        return 'unknown'
    if '://' in url_or_domain:
        try:
            parsed = urllib.parse.urlparse(url_or_domain)
            domain = parsed.netloc.lower()
        except Exception:
            domain = url_or_domain.lower()
    else:
        domain = url_or_domain.lower()

    if domain.startswith('www.'):
        domain = domain[4:]
    return domain.split('/')[0]


def classify_source_tier(url_or_domain, source_name=''):
    """
    Classify a source into Tier 1 (Official), Tier 2 (Established News), or Tier 3 (Other).

    Args:
        url_or_domain (str): URL or domain name.
        source_name (str): Friendly name of the source if available.

    Returns:
        dict:
            {
                "tier": "TIER_1_OFFICIAL" | "TIER_2_ESTABLISHED_NEWS" | "TIER_3_OTHER",
                "tier_name": str,
                "description": str,
                "credibility_level": "HIGH" | "MEDIUM-HIGH" | "VARIABLE/UNVERIFIED"
            }
    """
    domain = _clean_domain(url_or_domain)

    # Tier 1 Check: Government, International Agency, Official Domain
    is_tier1 = any(domain.endswith('.' + t) or domain == t for t in TIER1_PRIMARY_DOMAINS)
    if is_tier1 or 'official' in source_name.lower() or 'press release' in source_name.lower():
        return {
            'tier': 'TIER_1_OFFICIAL',
            'tier_name': 'Tier 1 — Official / Primary Source',
            'description': 'Direct primary source, government agency, official organization, or official statement.',
            'credibility_level': 'HIGH'
        }

    # Tier 2 Check: Established Reputable News Agencies
    is_tier2 = any(est in domain for est in TIER2_ESTABLISHED_NEWS)
    if is_tier2:
        matched_name = next((v for k, v in TIER2_ESTABLISHED_NEWS.items() if k in domain), source_name or domain)
        return {
            'tier': 'TIER_2_ESTABLISHED_NEWS',
            'tier_name': 'Tier 2 — Established News Organization',
            'description': f'Recognized major news organization ({matched_name}) with professional editorial standards.',
            'credibility_level': 'MEDIUM-HIGH'
        }

    # Tier 3 Check: General Web / Blogs / Unverified
    return {
        'tier': 'TIER_3_OTHER',
        'tier_name': 'Tier 3 — Other / Secondary Source',
        'description': 'General website, independent blog, forum, or unverified secondary outlet.',
        'credibility_level': 'VARIABLE/UNVERIFIED'
    }


def detect_source_independence(sources_list):
    """
    Analyze if multiple sources are independent or simply referencing/syndicating
    the same original wire report or primary source.

    Args:
        sources_list (list[dict]): List of search result dictionaries containing 'snippet', 'source', 'url'.

    Returns:
        dict:
            {
                "independence": "likely_independent" | "appears_to_reference_same_source" | "insufficient_sources",
                "primary_wire_detected": str or None,
                "analysis": str
            }
    """
    if not sources_list or len(sources_list) <= 1:
        return {
            'independence': 'insufficient_sources',
            'primary_wire_detected': None,
            'analysis': 'Fewer than 2 sources retrieved; source independence cannot be evaluated.'
        }

    snippets = [s.get('snippet', '').lower() for s in sources_list]
    sources = [s.get('source', '').lower() for s in sources_list]

    # Look for wire service attributions ("according to reuters", "reported by ap", etc.)
    wire_keywords = ['reuters', 'associated press', 'ap', 'afp', 'bloomberg', 'official statement']
    detected_wires = set()

    for snip in snippets:
        for wire in wire_keywords:
            if f'according to {wire}' in snip or f'reported by {wire}' in snip or f'cited {wire}' in snip:
                detected_wires.add(wire.upper())

    if len(detected_wires) > 0:
        wire_str = ', '.join(detected_wires)
        return {
            'independence': 'appears_to_reference_same_source',
            'primary_wire_detected': wire_str,
            'analysis': f'Multiple outlets appear to reference the same original report/wire service: {wire_str}.'
        }

    # Check snippet similarity (jaccard overlap on 4-grams or key terms)
    word_sets = [set(snip.split()) for snip in snippets if len(snip.split()) > 5]
    if len(word_sets) >= 2:
        overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i].intersection(word_sets[j]))
                union = len(word_sets[i].union(word_sets[j]))
                if union > 0:
                    overlaps.append(intersection / union)

        if overlaps and max(overlaps) > 0.45:
            return {
                'independence': 'appears_to_reference_same_source',
                'primary_wire_detected': 'Shared Wording / Syndication',
                'analysis': 'High textual overlap detected across source snippets; reporting likely derives from a single shared origin.'
            }

    return {
        'independence': 'likely_independent',
        'primary_wire_detected': None,
        'analysis': 'Sources present distinct reporting and phrasing; likely independent observations or reporting.'
    }


def analyze_sources(sources_list):
    """
    Perform complete source credibility and distribution analysis.

    Args:
        sources_list (list[dict]): List of retrieved sources.

    Returns:
        dict: Source analysis summary.
    """
    if not sources_list:
        return {
            'total_sources': 0,
            'tier1_count': 0,
            'tier2_count': 0,
            'tier3_count': 0,
            'independence_status': 'insufficient_sources',
            'independence_analysis': 'No sources retrieved; independence cannot be evaluated.',
            'classified_sources': []
        }


    classified = []
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0

    for src in sources_list:
        tier_info = classify_source_tier(src.get('url', ''), src.get('source', ''))
        item = dict(src)
        item.update(tier_info)
        classified.append(item)

        if tier_info['tier'] == 'TIER_1_OFFICIAL':
            tier1_count += 1
        elif tier_info['tier'] == 'TIER_2_ESTABLISHED_NEWS':
            tier2_count += 1
        else:
            tier3_count += 1

    indep_info = detect_source_independence(sources_list)

    return {
        'total_sources': len(classified),
        'tier1_count': tier1_count,
        'tier2_count': tier2_count,
        'tier3_count': tier3_count,
        'independence_status': indep_info['independence'],
        'independence_analysis': indep_info['analysis'],
        'classified_sources': classified
    }
