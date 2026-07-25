import re
import math
import numpy as np
import scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Word lists for different feature groups
SPECULATION_WORDS = [
    'anonymous', 'allegedly', 'reportedly', 'rumored', 'might', 'could', 
    'expected', 'according to sources', 'speculated', 'unconfirmed', 'claims', 'rumors'
]

CLICKBAIT_PHRASES = [
    'shocking', 'breaking', 'unbelievable', "you won't believe", 'finally exposed',
    'must-see', 'revealed', 'secret they', 'shocked the world', 'what happened next'
]

EMOTIONAL_WORDS = {
    'shocking', 'conspiracy', 'leaked', 'secret', 'urgent', 'viral', 'breaking', 
    'exposed', 'unbelievable', 'miracle', 'truth', 'warning', 'agenda', 'censored', 
    'anonymous', 'classified', 'insider', 'hiding', 'scandal', 'banned', 'shocked', 
    'chaos', 'destroys', 'slam', 'blasts', 'panic', 'terror', 'crisis', 'must-see', 
    'revealed', 'prophecy', 'secretly', 'unconfirmed', 'hoax', 'fraud', 'illegal',
    'conspire', 'collusion', 'deepstate', 'rigged', 'covert', 'plot', 'cover-up'
}

POSITIVE_WORDS = {"great", "excellent", "good", "verify", "truth", "true", "positive", "credible", "reliable", "validated", "factual", "correct", "success"}
NEGATIVE_WORDS = {"fake", "worst", "terrible", "bad", "false", "hoax", "lie", "disaster", "negative", "unverified", "suspicious", "misleading", "conspiracy", "rumor"}

# Pre-compile patterns
SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]+')
CAPITAL_WORD_PATTERN = re.compile(r'\b[A-Z][a-z]+\b')

def calculate_entropy(text):
    """Computes Shannon entropy of word frequencies in the text."""
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return entropy

def extract_dense_features(raw_text: str, clean_str: str, title: str = "") -> dict:
    """
    Extracts structured Group B-F features from raw and cleaned text.
    Returns a dictionary of raw features, and an ordered list for model ingestion.
    """
    if not isinstance(raw_text, str):
        raw_text = ""
    if not isinstance(clean_str, str):
        clean_str = ""
    if not isinstance(title, str):
        title = ""

    # Tokenizing
    words = raw_text.split()
    word_count = max(1, len(words))
    clean_words = clean_str.split()
    clean_word_count = max(1, len(clean_words))
    raw_chars = max(1, len(raw_text))

    # --- Group B: Writing Style ---
    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(raw_text) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_len = len(words) / sentence_count
    
    unique_clean_words = set(clean_words)
    lexical_diversity = len(unique_clean_words) / clean_word_count
    entropy = calculate_entropy(raw_text)
    
    # Syllable & Readability estimation
    text_lower = raw_text.lower()
    vowel_count = sum(text_lower.count(v) for v in 'aeiou')
    syllables = max(word_count, vowel_count)
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * (syllables / word_count)
    flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))

    # --- Group C: Credibility ---
    # Quotes count
    quotes_count = raw_text.count('"') + raw_text.count("'") + raw_text.count('“') + raw_text.count('”')
    
    # Capitalized words not at start of sentence (approximation of named entities)
    cap_words = CAPITAL_WORD_PATTERN.findall(raw_text)
    named_entities_est = sum(1 for w in cap_words if w.lower() not in clean_words) # simple heuristic
    
    # References & Citations: counting URLs, domain patterns, and reporting attribution keywords
    attribution_keywords = ['said', 'told', 'reported', 'according to', 'stated', 'announced', 'spokesman']
    attribution_count = sum(text_lower.count(kw) for kw in attribution_keywords)
    urls_count = raw_text.count('http://') + raw_text.count('https://') + raw_text.count('www.')
    credibility_citations = urls_count + attribution_count

    # --- Group D: Speculation ---
    speculation_count = sum(text_lower.count(word) for word in SPECULATION_WORDS)
    speculation_ratio = speculation_count / word_count

    # --- Group E: Clickbait ---
    clickbait_count = sum(text_lower.count(phrase) for phrase in CLICKBAIT_PHRASES)
    # Check for excessive capitalization/punctuation
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    excessive_punct = 1.0 if ("!!!" in raw_text or "???" in raw_text or "!?" in raw_text) else 0.0
    clickbait_score = (clickbait_count / sentence_count) + (all_caps_words / word_count) + excessive_punct

    # --- Group F: Emotion ---
    pos_count = sum(clean_words.count(w) for w in POSITIVE_WORDS)
    neg_count = sum(clean_words.count(w) for w in NEGATIVE_WORDS)
    polarity = (pos_count - neg_count) / (pos_count + neg_count + 1e-5)
    subjectivity = (pos_count + neg_count) / clean_word_count
    
    emotional_vocab_count = sum(clean_words.count(w) for w in EMOTIONAL_WORDS)
    emotional_intensity = emotional_vocab_count / clean_word_count

    # Compile structured features dict
    features_dict = {
        # Group B
        "avg_sentence_len": avg_sentence_len,
        "lexical_diversity": lexical_diversity,
        "entropy": entropy,
        "flesch_reading_ease": flesch_reading_ease,
        # Group C
        "quotes_count": quotes_count,
        "named_entities_est": named_entities_est,
        "credibility_citations": credibility_citations,
        # Group D
        "speculation_ratio": speculation_ratio,
        # Group E
        "clickbait_score": clickbait_score,
        # Group F
        "polarity": polarity,
        "subjectivity": subjectivity,
        "emotional_intensity": emotional_intensity
    }

    # Order of features for ingestion
    features_list = [
        avg_sentence_len, lexical_diversity, entropy, flesch_reading_ease, # Group B
        quotes_count, named_entities_est, credibility_citations,         # Group C
        speculation_ratio,                                                # Group D
        clickbait_score,                                                  # Group E
        polarity, subjectivity, emotional_intensity                      # Group F
    ]

    return features_dict, features_list

DENSE_FEATURE_NAMES = [
    "avg_sentence_len", "lexical_diversity", "entropy", "flesch_reading_ease",
    "quotes_count", "named_entities_est", "credibility_citations",
    "speculation_ratio",
    "clickbait_score",
    "polarity", "subjectivity", "emotional_intensity"
]