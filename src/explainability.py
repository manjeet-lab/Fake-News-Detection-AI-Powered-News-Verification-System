"""
Fake News Detection — Step 6: Explainability Module
=====================================================

This module adds explainability to the prediction pipeline built in Step 5.
It does NOT retrain the model or re-fit the vectorizer. It loads the SAME
artifacts that Step 5 uses and analyses their internals to explain decisions.

Three levels of explanation are provided:

  1. Global feature importance
       Which words the model *generally* associates with FAKE vs REAL news,
       derived from the model's coefficient vector.

  2. Article-specific explanation
       Which words *in this specific article* pushed the model toward its
       prediction, computed as:
           contribution[i] = tfidf_score[i] × coef[i]
       Positive contribution → pushes toward REAL.
       Negative contribution → pushes toward FAKE.

  3. Suspicious / clickbait language detection
       A small, manually-curated list of common clickbait/sensational language
       patterns searched for in the raw (unprocessed) text.
       Clearly labeled as "potentially suspicious language" — NOT as proof of
       fake news. The model-based explanation and this heuristic list are
       always kept separate in the output.

Compatible model types:
  - LogisticRegression  (uses model.coef_[0])
  - LinearSVC           (uses model.coef_[0])
  Both expose a coef_ attribute with the same directional semantics for a
  binary 0/1 classifier.

Usage (from notebook or Streamlit app):
    import sys; sys.path.insert(0, '../src')   # or add src/ to PYTHONPATH
    from explainability import get_explanation, get_feature_importance

    result = get_explanation("Some raw news article text ...")
    # {
    #   'prediction'          : 'FAKE',
    #   'confidence'          : 91.25,
    #   'influential_features': [...],
    #   'suspicious_language' : [...]
    # }
"""

import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Import the Step 5 prediction module (do NOT duplicate its logic)
# ---------------------------------------------------------------------------
# This file lives in src/. prediction.py is also in src/, so inserting the
# directory of this file guarantees the import works regardless of how or from
# where this module is imported (notebook, Streamlit, CLI).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import prediction as _pred          # noqa: E402  (import after sys.path tweak)

# Pull the shared artifacts from prediction.py — we do NOT reload them.
_model       = _pred.model
_vectorizer  = _pred.vectorizer
_LABEL_NAMES = _pred.LABEL_NAMES    # {0: 'FAKE', 1: 'REAL'}
_preprocess  = _pred.preprocess_text
_validate    = _pred._validate_raw_input
_MIN_CHARS   = _pred._MIN_CHARS_CLEAN


# ===========================================================================
# Section 1 — Model introspection helpers
# ===========================================================================

def _get_coef():
    """
    Return the 1-D coefficient array (shape: [n_features]) for the model.

    For a binary 0/1 classifier with model.coef_ of shape (1, n_features):
        positive value → word pushes prediction toward classes_[-1] = 1 = REAL
        negative value → word pushes prediction toward classes_[0]  = 0 = FAKE

    Returns None when the model does not expose a coef_ attribute
    (e.g. tree-based models), so callers can degrade gracefully.
    """
    if not hasattr(_model, 'coef_'):
        return None
    return np.asarray(_model.coef_[0])


def _get_feature_names():
    """Return the TF-IDF vocabulary feature-name array."""
    return _vectorizer.get_feature_names_out()


def model_info():
    """
    Return a diagnostic summary of the loaded model and its explainability.

    Returns:
        dict with keys: model_type, has_coef, coef_shape, n_features,
                        classes, label_mapping, explainability_method.
    """
    coef = _get_coef()
    return {
        'model_type'            : type(_model).__name__,
        'has_coef'              : coef is not None,
        'coef_shape'            : tuple(coef.shape) if coef is not None else None,
        'n_features'            : len(_get_feature_names()),
        'classes'               : (list(_model.classes_)
                                   if hasattr(_model, 'classes_') else None),
        'label_mapping'         : _LABEL_NAMES,
        'explainability_method' : ('coefficient × tfidf_score'
                                   if coef is not None else 'unavailable'),
    }


# ===========================================================================
# Section 2 — Global feature importance
# ===========================================================================

def get_feature_importance(top_n=20):
    """
    Return the top model-level influential features for FAKE and REAL.

    This reflects the model's *global* learned associations — which words the
    model consistently links with REAL or FAKE news across the whole training
    corpus — without considering any specific article.

    Args:
        top_n (int): Number of top features to return per class. Default 20.

    Returns:
        dict:
          {
            'model_type'       : str,
            'method'           : str,
            'top_real_features': [{'word': str, 'coefficient': float}, ...],
            'top_fake_features': [{'word': str, 'coefficient': float}, ...],
            'note'             : str
          }
        Returns {'error': str} if the model does not expose coefficients.
    """
    coef = _get_coef()
    if coef is None:
        return {
            'error': (
                'The loaded model ({}) does not expose a coef_ attribute. '
                'Global feature importance via coefficients is unavailable '
                'for this model type.'.format(type(_model).__name__)
            )
        }

    feature_names = _get_feature_names()

    # Most positive coefficients → strongest push toward REAL
    top_real_idx = np.argsort(coef)[::-1][:top_n]
    top_real = [
        {'word': feature_names[i], 'coefficient': round(float(coef[i]), 6)}
        for i in top_real_idx
    ]

    # Most negative coefficients → strongest push toward FAKE
    top_fake_idx = np.argsort(coef)[:top_n]
    top_fake = [
        {'word': feature_names[i], 'coefficient': round(float(coef[i]), 6)}
        for i in top_fake_idx
    ]

    return {
        'model_type'        : type(_model).__name__,
        'method'            : 'model.coef_[0] mapped to TF-IDF vocabulary',
        'top_real_features' : top_real,
        'top_fake_features' : top_fake,
        'note': (
            'These are the words with the largest absolute model coefficients. '
            'They reflect the model\'s learned statistical patterns — NOT '
            'factual evidence that any word is inherently "fake" or "real".'
        ),
    }


# ===========================================================================
# Section 3 — Article-specific explanation
# ===========================================================================

def explain_prediction(text, top_n=10):
    """
    Explain the model's prediction for one specific article.

    Method:
        contribution[i]  =  tfidf_score[i]  ×  coef[i]

        tfidf_score[i] — TF-IDF weight of the i-th feature in *this* article.
        coef[i]        — model's global association of that feature with
                         REAL (positive) or FAKE (negative).

    Only features that actually appear in the article (non-zero TF-IDF) are
    considered; words absent from the article have zero contribution.

    Args:
        text  (str): Raw news article text (title + body, or combined).
        top_n (int): Number of top influential features to return. Default 10.

    Returns:
        dict on success:
          {
            'prediction'          : 'FAKE' | 'REAL',
            'confidence'          : float (0-100),
            'confidence_type'     : str,
            'model_used'          : str,
            'influential_features': [
                {
                  'word'        : str,
                  'tfidf_score' : float,   # how strongly word appears here
                  'coefficient' : float,   # model's global association
                  'contribution': float,   # tfidf_score × coefficient
                  'direction'   : 'REAL' | 'FAKE'
                }, ...
            ],
            'features_for_real'   : [...],   # subset pushing toward REAL
            'features_for_fake'   : [...],   # subset pushing toward FAKE
            'note'                : str
          }
        dict with 'error' key on invalid input or unsupported model.
    """
    # 1. Validate ─────────────────────────────────────────────────────────
    err = _validate(text)
    if err:
        return {'error': err}

    coef = _get_coef()
    if coef is None:
        return {
            'error': (
                'The loaded model ({}) does not expose model.coef_, so '
                'article-level explanation via coefficients is unavailable.'
                .format(type(_model).__name__)
            )
        }

    # 2. Preprocess — identical to Step 5 ─────────────────────────────────
    cleaned = _preprocess(text)
    if len(cleaned.strip()) < _MIN_CHARS:
        return {
            'error': ('After preprocessing, no meaningful text remained. '
                      'Please provide a longer, more substantive article.')
        }

    # 3. Vectorize (transform only — NEVER re-fit) ─────────────────────────
    X_vec = _vectorizer.transform([cleaned])

    # 4. Get the Step 5 prediction + confidence ───────────────────────────
    base_result = _pred.predict_news(text)
    if 'error' in base_result:
        return base_result

    # 5. Compute per-feature contributions ───────────────────────────────
    feature_names = _get_feature_names()
    cx            = X_vec.tocsr()
    # Use .indices and .data from the CSR internals — they are already flat
    # float arrays and avoid the sub-matrix return of fancy indexing.
    nonzero_cols  = cx.indices   # column (feature) indices of non-zero entries
    nonzero_vals  = cx.data      # corresponding TF-IDF float values

    contributions = []
    for idx, tfidf_score in zip(nonzero_cols, nonzero_vals):
        c            = float(coef[idx])
        contribution = float(tfidf_score) * c
        contributions.append({
            'word'        : feature_names[idx],
            'tfidf_score' : round(float(tfidf_score), 6),
            'coefficient' : round(c, 6),
            'contribution': round(contribution, 6),
            'direction'   : 'REAL' if contribution >= 0 else 'FAKE',
        })

    # 6. Rank by absolute contribution magnitude ──────────────────────────
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    top_features = contributions[:top_n]

    # 7. Split by direction ───────────────────────────────────────────────
    features_for_real = [f for f in top_features if f['direction'] == 'REAL']
    features_for_fake = [f for f in top_features if f['direction'] == 'FAKE']

    return {
        'prediction'          : base_result['prediction'],
        'confidence'          : base_result['confidence'],
        'confidence_type'     : base_result['confidence_type'],
        'model_used'          : base_result['model_used'],
        'influential_features': top_features,
        'features_for_real'   : features_for_real,
        'features_for_fake'   : features_for_fake,
        'note': (
            'These features had the largest influence on the model\'s prediction '
            'for this specific article. They reflect learned statistical patterns '
            '— NOT factual evidence about whether the article\'s content is true.'
        ),
    }


# ===========================================================================
# Section 4 — Suspicious / clickbait language detection
# ===========================================================================

# Small, transparent curated list of common clickbait / sensational patterns.
# Each entry: (regex_pattern,  human_readable_description,  category)
#
# • Operates on the RAW (unprocessed) text to preserve capitalisation signals.
# • These are HEURISTIC indicators only — they do NOT prove an article is fake.
# • The regex flags=re.IGNORECASE is applied so partial-caps forms also match.
_SUSPICIOUS_PATTERNS = [
    # ── Sensational urgency ────────────────────────────────────────────────
    (r'\bBREAKING\b',
     '"BREAKING" urgency framing',            'sensational urgency'),
    (r'\bURGENT\b',
     '"URGENT" urgency framing',              'sensational urgency'),
    (r'\bBOMBSHELL\b',
     '"BOMBSHELL" sensational language',      'sensational urgency'),
    (r'\bEXPLOSIVE\b',
     '"EXPLOSIVE" sensational language',      'sensational urgency'),
    # ── Clickbait calls to action ─────────────────────────────────────────
    (r'\bSHARE\s+THIS\s+NOW\b',
     '"SHARE THIS NOW" viral bait',           'clickbait call-to-action'),
    (r'\bMAKE\s+THIS\s+VIRAL\b',
     '"MAKE THIS VIRAL" viral bait',          'clickbait call-to-action'),
    (r'\byou\s+won.?t\s+believe\b',
     '"you won\'t believe" clickbait phrase', 'clickbait framing'),
    (r'\bbefore\s+it.?s\s+(too\s+late|deleted|banned|removed)\b',
     'urgency / censorship framing',          'clickbait framing'),
    (r'\bscientists\s+hate\b',
     '"scientists hate" clickbait trope',     'clickbait framing'),
    # ── Conspiracy language ───────────────────────────────────────────────
    (r'\bthey\s+don.?t\s+want\s+you\s+to\s+know\b',
     'conspiracy framing ("they don\'t want you to know")', 'conspiracy language'),
    (r'\bthe\s+government\s+(is\s+hiding|doesn.?t\s+want)\b',
     'conspiracy framing (government hiding)', 'conspiracy language'),
    (r'\bBig\s+Pharma\b',
     'conspiracy-coded term ("Big Pharma")',   'conspiracy language'),
    (r'\bdeep\s+state\b',
     'conspiracy-coded term ("deep state")',   'conspiracy language'),
    (r'\bnew\s+world\s+order\b',
     'conspiracy-coded term ("new world order")', 'conspiracy language'),
    (r'\bsecret\s+(cure|remedy|they|that|the\s+government)\b',
     'conspiracy framing ("secret ...")',      'conspiracy language'),
    # ── Exaggerated certainty ─────────────────────────────────────────────
    (r'\b100\s*%\s*(proven|guaranteed|confirmed|effective)\b',
     'exaggerated certainty claim ("100% proven/guaranteed")', 'exaggerated claim'),
    (r'\bGUARANTEED\b',
     '"GUARANTEED" exaggerated certainty',     'exaggerated claim'),
    (r'\bmiracle\s+(cure|treatment|remedy)\b',
     'exaggerated health claim ("miracle cure")', 'exaggerated claim'),
    # ── Emotional manipulation ────────────────────────────────────────────
    (r'\bSHOCKING\b',
     '"SHOCKING" emotional trigger word',      'emotional manipulation'),
    (r'\bDEVASTATING\b',
     '"DEVASTATING" emotional trigger word',   'emotional manipulation'),
    (r'\bOUTRAGEOUS\b',
     '"OUTRAGEOUS" emotional trigger word',    'emotional manipulation'),
]


def _detect_suspicious_language(raw_text):
    """
    Scan raw text for clickbait / suspicious linguistic patterns.

    Args:
        raw_text (str): The original (unprocessed) article text.

    Returns:
        list[dict]: One entry per matched pattern:
            {'pattern_desc': str, 'category': str, 'matched_text': str}
        Empty list if nothing matched.
    """
    if not isinstance(raw_text, str):
        return []

    findings   = []
    seen_descs = set()   # prevent the same description appearing twice

    for pattern, description, category in _SUSPICIOUS_PATTERNS:
        matches = re.findall(pattern, raw_text, flags=re.IGNORECASE)
        if matches and description not in seen_descs:
            seen_descs.add(description)
            raw_match = matches[0]
            if isinstance(raw_match, tuple):   # group captures in the pattern
                raw_match = ' '.join(m for m in raw_match if m)
            findings.append({
                'pattern_desc': description,
                'category'    : category,
                'matched_text': raw_match.strip(),
            })

    return findings


# ===========================================================================
# Section 5 — Combined get_explanation() — main public entry point
# ===========================================================================

def get_explanation(text, top_n=10):
    """
    Main explainability entry point combining all three explanation layers.

    Args:
        text  (str): Raw news article text (can include title + body).
        top_n (int): Number of top influential features to include. Default 10.

    Returns:
        dict on success:
          {
            'prediction'          : 'FAKE' | 'REAL',
            'confidence'          : float (0-100),
            'confidence_type'     : str,
            'model_used'          : str,
            'influential_features': [  # sorted by abs(contribution)
                {'word', 'tfidf_score', 'coefficient', 'contribution', 'direction'},
                ...
            ],
            'features_for_real'   : [...],   # subset with direction == 'REAL'
            'features_for_fake'   : [...],   # subset with direction == 'FAKE'
            'suspicious_language' : [        # heuristic linguistic patterns
                {'pattern_desc', 'category', 'matched_text'},
                ...
            ],
            'note'                : str
          }
        dict with 'error' key on invalid input.
    """
    # Article-specific explanation already includes prediction + confidence.
    result = explain_prediction(text, top_n=top_n)
    if 'error' in result:
        return result

    # Suspicious language runs on the raw text — append to the result.
    result['suspicious_language'] = _detect_suspicious_language(text)

    return result


# ===========================================================================
# Section 6 — Display helper (used by notebook and Streamlit app)
# ===========================================================================

def display_explanation(result, show_direction_split=True):
    """
    Pretty-print an explanation result dict from get_explanation().

    Args:
        result              (dict): Output from get_explanation().
        show_direction_split (bool): Also print features split by direction.
    """
    SEP = '=' * 44

    print(SEP)
    print('  EXPLAINABILITY RESULT')
    print(SEP)
    print()

    if 'error' in result:
        print('  ERROR:', result['error'])
        print()
        print(SEP)
        return

    prediction = result['prediction']
    confidence = result.get('confidence')
    conf_type  = result.get('confidence_type', '')
    model_used = result.get('model_used', 'unknown')

    print(f'  Prediction : {prediction}')
    if confidence is not None:
        print(f'  Confidence : {confidence:.2f}%')
    print(f'  Model      : {model_used}')
    if 'decision_margin' in conf_type:
        print('  (confidence is a sigmoid-squashed decision margin,')
        print('   not a calibrated probability)')
    print()

    # ── Top influential features ─────────────────────────────────────────
    features = result.get('influential_features', [])
    print('  Most influential features:')
    if features:
        for i, feat in enumerate(features, 1):
            arrow = '→ REAL' if feat['direction'] == 'REAL' else '→ FAKE'
            print(f'    {i:2d}. {feat["word"]:34s}  {arrow}  '
                  f'({feat["contribution"]:+.4f})')
    else:
        print('    No features extracted for this article.')
    print()

    # ── Directional breakdown ────────────────────────────────────────────
    if show_direction_split:
        real_feats = result.get('features_for_real', [])
        fake_feats = result.get('features_for_fake', [])
        supporting = real_feats if prediction == 'REAL' else fake_feats
        opposing   = fake_feats if prediction == 'REAL' else real_feats

        print(f'  Features supporting {prediction}:')
        if supporting:
            for f in supporting:
                print(f'    + {f["word"]}  ({f["contribution"]:+.4f})')
        else:
            print('    (none in top features)')
        print()

        print(f'  Features opposing {prediction}:')
        if opposing:
            for f in opposing:
                print(f'    − {f["word"]}  ({f["contribution"]:+.4f})')
        else:
            print('    (none in top features)')
        print()

    # ── Suspicious language ──────────────────────────────────────────────
    suspicious = result.get('suspicious_language', [])
    print('  Potentially suspicious language patterns:')
    if suspicious:
        for item in suspicious:
            print(f'    • [{item["category"]}]  {item["pattern_desc"]}')
            print(f'        matched: "{item["matched_text"]}"')
    else:
        print('    No obvious suspicious language detected.')
    print()

    # ── Disclaimer ───────────────────────────────────────────────────────
    print('  DISCLAIMER: The features listed above are the ones with the')
    print('  strongest influence on the MODEL\'s prediction for this article.')
    print('  They reflect learned statistical patterns — NOT factual evidence')
    print('  about whether the article\'s content is true or false.')
    print()
    print(SEP)
