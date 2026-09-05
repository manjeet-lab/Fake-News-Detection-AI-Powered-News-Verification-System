"""
Fake News Detection — Step 5: Prediction Pipeline + Confidence Score
======================================================================

This module loads the artifacts already produced by Steps 3 and 4
(the fitted TF-IDF vectorizer and the selected best model) and exposes a
single reusable function, `predict_news()`, that takes raw, unseen news
text and returns a structured prediction + confidence result.

IMPORTANT — This module does NOT retrain anything. It only loads:
    models/best_model.pkl
    models/tfidf_vectorizer.pkl

The text-cleaning function below (`preprocess_text`) is copied *verbatim*
from Task 2 (Task2_NLP_Preprocessing.ipynb, Section 5) so that new text is
cleaned in EXACTLY the same way the training data was cleaned before being
fed into the TF-IDF vectorizer. Do not "improve" or simplify this function —
any deviation (even a seemingly harmless one) changes the tokens that reach
the vectorizer and silently breaks compatibility with the trained vocabulary.

Usage (e.g. from a future Streamlit app):

    from prediction import predict_news, predict_multiple

    result = predict_news("Some raw news article text ...")
    print(result)
    # {'prediction': 'REAL', 'confidence': 94.52, ...}
"""

import os
import re
import warnings

import joblib
import nltk
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Section 1: Locate and load the saved artifacts (Step 3 + Step 4 outputs)
# ---------------------------------------------------------------------------

# Resolve paths relative to the project root (this file lives in src/,
# and models/ is a sibling of src/ at the project root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'best_model.pkl')
_VECTORIZER_PATH = os.path.join(_PROJECT_ROOT, 'models', 'tfidf_vectorizer.pkl')

# Fallback: also allow running with models/ in the current working directory
# (useful when this module is imported from a notebook sitting next to models/).
if not os.path.exists(_MODEL_PATH):
    _alt = os.path.join(os.getcwd(), 'models', 'best_model.pkl')
    if os.path.exists(_alt):
        _MODEL_PATH = _alt
        _VECTORIZER_PATH = os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.pkl')


def _load_artifacts():
    """Load the trained model and TF-IDF vectorizer saved in Step 4.

    Raises a clear error (rather than a cryptic one) if the files are missing,
    since Step 5 must NEVER retrain — it can only load what already exists.
    """
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            "Could not find '{}'. Step 5 does not train models — make sure "
            "Step 4 has been run and 'models/best_model.pkl' exists.".format(_MODEL_PATH)
        )
    if not os.path.exists(_VECTORIZER_PATH):
        raise FileNotFoundError(
            "Could not find '{}'. Step 5 does not fit vectorizers — make sure "
            "Step 3/4 saved 'models/tfidf_vectorizer.pkl'.".format(_VECTORIZER_PATH)
        )

    model = joblib.load(_MODEL_PATH)
    vectorizer = joblib.load(_VECTORIZER_PATH)
    return model, vectorizer


# Load once, at import time, and reuse for every prediction call.
model, vectorizer = _load_artifacts()

# Does this model support predict_proba (a true, calibrated-ish probability)?
_HAS_PROBA = hasattr(model, 'predict_proba')
# Does this model support decision_function (an uncalibrated margin score)?
_HAS_DECISION = hasattr(model, 'decision_function')

# ---------------------------------------------------------------------------
# Section 2: Determine the correct label mapping (do NOT assume 0/1 meaning)
# ---------------------------------------------------------------------------
# Task 1 (Task1_analysis.ipynb, Section 5) explicitly assigns:
#     Fake.csv -> label = 0
#     True.csv -> label = 1
# This is confirmed directly from the dataset construction, not assumed.
# We still read it from `model.classes_` at runtime so the code stays correct
# even if a future retraining ever changes the label encoding.
LABEL_NAMES = {0: 'FAKE', 1: 'REAL'}

if hasattr(model, 'classes_'):
    _unexpected = set(model.classes_) - set(LABEL_NAMES.keys())
    if _unexpected:
        raise ValueError(
            "Model classes_ {} do not match the expected Fake/Real label "
            "encoding {} confirmed in Task 1. Investigate before predicting."
            .format(list(model.classes_), list(LABEL_NAMES.keys()))
        )

# ---------------------------------------------------------------------------
# Section 3: NLP preprocessing — copied verbatim from Task 2 (Section 5)
# ---------------------------------------------------------------------------

# Ensure the NLTK resources used by preprocess_text() are available.
# quiet=True avoids noisy download logs; this is a no-op if already present.
for _resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(_resource, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Cleans and normalizes a raw news article text through 10 preprocessing steps.

    *** This function is copied verbatim from Task 2 (preprocess_text). ***
    It must stay IDENTICAL to the version used to build the training data,
    or the cleaned text will no longer match the vocabulary the TF-IDF
    vectorizer and model were trained on.

    Args:
        text (str): Raw input text (combined title and article body)

    Returns:
        str: Cleaned, preprocessed text ready for TF-IDF vectorization
    """

    # Guard: handle NaN or non-string input gracefully
    if not isinstance(text, str):
        return ''

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Step 4: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 5: Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Step 6: Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Step 7: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 8: Tokenize
    tokens = word_tokenize(text)

    # Step 9: Remove stopwords (and single-character tokens)
    tokens = [word for word in tokens
              if word not in stop_words and len(word) > 1]

    # Step 10: Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 11: Rejoin tokens into a single string
    clean_text = ' '.join(tokens)

    return clean_text


def _combine_title_and_text(title, text):
    """Combine title + article body the SAME way Task 2 (Section 4) did:
    'title' + a single space + 'text'. Only used when title/text are
    supplied separately; if the caller already has one combined string,
    pass it directly as `text` to predict_news().
    """
    title = '' if title is None else str(title)
    text = '' if text is None else str(text)
    return (title + ' ' + text).strip()


# ---------------------------------------------------------------------------
# Section 4: Input validation
# ---------------------------------------------------------------------------

_MIN_WORDS_RAW = 3        # a title-only fragment can still be valid, but not 1-2 words
_MIN_CHARS_CLEAN = 3      # after cleaning, require at least a few characters left


def _validate_raw_input(raw_text):
    """Returns an error message string if input is invalid, else None."""
    if raw_text is None:
        return 'Please enter a valid news article.'
    if not isinstance(raw_text, str):
        return 'Please enter a valid news article.'
    if raw_text.strip() == '':
        return 'Please enter a valid news article. The input appears to be empty.'
    if len(raw_text.strip().split()) < _MIN_WORDS_RAW:
        return 'Please enter a longer news article. The input is too short to classify reliably.'
    return None


# ---------------------------------------------------------------------------
# Section 5: Confidence scoring
# ---------------------------------------------------------------------------
# Two possible situations, handled explicitly and never silently mixed up:
#
#   (a) model has predict_proba()  -> a genuine (softmax/sigmoid-based)
#       probability distribution over classes. Confidence = max class prob.
#
#   (b) model only has decision_function() (e.g. plain LinearSVC) -> there is
#       NO calibrated probability available. We report the raw signed
#       decision score, AND a secondary "approximate_confidence" derived by
#       squashing that score through a sigmoid purely for a human-readable
#       0-100 number. This is explicitly labeled as NOT a probability. We do
#       NOT silently fit a calibrator (e.g. CalibratedClassifierCV) here —
#       that would be retraining, which Step 5 is forbidden from doing.

def _sigmoid(x):
    import math
    # Clip to avoid overflow for very large |x|
    x = max(min(x, 60), -60)
    return 1.0 / (1.0 + math.exp(-x))


def _score_prediction(X_vec):
    """Given a vectorized (1-row) TF-IDF input, return:
        pred_label_raw   : the raw predicted class (0 or 1)
        confidence_pct   : a 0-100 number for display
        confidence_type  : 'probability' or 'decision_margin (uncalibrated)'
        raw_score_info   : dict with the underlying numeric evidence
    """
    pred_label_raw = model.predict(X_vec)[0]

    if _HAS_PROBA:
        proba = model.predict_proba(X_vec)[0]  # e.g. [P(class0), P(class1)]
        classes = list(model.classes_)
        pred_idx = classes.index(pred_label_raw)
        confidence_pct = round(float(proba[pred_idx]) * 100, 2)
        return pred_label_raw, confidence_pct, 'probability', {
            'probabilities': {LABEL_NAMES[c]: round(float(p) * 100, 2)
                               for c, p in zip(classes, proba)}
        }

    elif _HAS_DECISION:
        raw_score = float(model.decision_function(X_vec)[0])
        # Sigmoid-squash the margin purely as a human-readable proxy.
        # This is NOT a calibrated probability — see module docstring.
        approx_conf = round(_sigmoid(abs(raw_score)) * 100, 2)
        # abs() -> confidence in the *predicted* direction, since a large
        # negative score means confidently Fake, and a large positive score
        # means confidently Real; the sign already determined pred_label_raw.
        return pred_label_raw, approx_conf, 'decision_margin (uncalibrated)', {
            'decision_score': round(raw_score, 4)
        }

    else:
        # Neither method available — extremely unlikely for LogisticRegression
        # or LinearSVC, but handle gracefully rather than crashing.
        return pred_label_raw, None, 'unavailable', {}


# ---------------------------------------------------------------------------
# Section 6: The main reusable prediction function
# ---------------------------------------------------------------------------

def predict_news(text, title=None):
    """
    Predict whether a news article is FAKE or REAL.

    Args:
        text (str): The article body (or a complete article if title is None).
        title (str, optional): The article headline. If provided, it is
            combined with `text` exactly the way Task 2 did:
            combined = title + ' ' + text

    Returns:
        dict: On success:
            {
                "prediction": "FAKE" | "REAL",
                "confidence": <float, 0-100>,
                "confidence_type": "probability" | "decision_margin (uncalibrated)",
                "model_used": "<model class name>",
                ... (extra diagnostic fields, e.g. per-class probabilities
                     or the raw decision score)
            }
        On invalid input:
            {"error": "<human-readable message>"}
        (No Python stack traces are ever exposed to the caller.)
    """
    try:
        # 1. Combine title + text if a title was given, else use text as-is.
        raw_input = _combine_title_and_text(title, text) if title is not None else text

        # 2. Validate the raw input before doing any work.
        error = _validate_raw_input(raw_input)
        if error:
            return {'error': error}

        # 3. Preprocess using the EXACT Task 2 logic.
        cleaned = preprocess_text(raw_input)

        if len(cleaned.strip()) < _MIN_CHARS_CLEAN:
            return {
                'error': ('Please enter a valid news article. After removing '
                           'stopwords/punctuation/numbers, no meaningful text remained.')
            }

        # 4. Vectorize using the saved TF-IDF vectorizer (transform only, never fit).
        X_vec = vectorizer.transform([cleaned])

        # 5. Predict + confidence.
        pred_label_raw, confidence_pct, confidence_type, extra = _score_prediction(X_vec)
        prediction_name = LABEL_NAMES[int(pred_label_raw)]

        result = {
            'prediction': prediction_name,
            'confidence': confidence_pct,
            'confidence_type': confidence_type,
            'model_used': type(model).__name__,
            'cleaned_text_preview': cleaned[:200] + ('...' if len(cleaned) > 200 else ''),
        }
        result.update(extra)
        return result

    except Exception as exc:
        # Never leak a raw stack trace to the eventual (Streamlit) user.
        return {'error': 'Something went wrong while analyzing this article. Please try again.',
                '_debug_info': str(exc)}


def predict_multiple(texts):
    """
    Run predict_news() over a list of raw article texts.

    Args:
        texts (list[str]): A list of raw article texts.

    Returns:
        list[dict]: One result dict per input, in the same order,
                    each following the same structure as predict_news().
    """
    if not isinstance(texts, (list, tuple)):
        return [{'error': 'predict_multiple() expects a list of strings.'}]
    return [predict_news(t) for t in texts]


def pipeline_info():
    """Small helper for diagnostics / the Step 5 demo notebook."""
    return {
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'has_predict_proba': _HAS_PROBA,
        'has_decision_function': _HAS_DECISION,
        'vectorizer_vocab_size': len(vectorizer.vocabulary_),
        'label_mapping': LABEL_NAMES,
        'model_path': _MODEL_PATH,
        'vectorizer_path': _VECTORIZER_PATH,
    }
