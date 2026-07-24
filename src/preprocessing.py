import re

# A manual list of common words (stopwords) to ignore, since we aren't using pre-built library shortcuts
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
    'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now',
    'reuters', 'via'  # Expanded to prevent feature leakage
}

# Pre-compile regular expressions at the module level for maximum execution performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
# Refined regex for publisher datelines (e.g. "WASHINGTON (Reuters) -", "LONDON (Reuters) -", "MOSCOW —", etc.)
DATELINE_PATTERN = re.compile(r'^\s*[A-Z\s]+(?:\s*\([^)]+\))?\s*[-—]\s*|^\s*\([^)]*(?:reuters|ap|afp|bbc|bloomberg|xinhua)[^)]*\)\s*[-—]\s*', re.IGNORECASE)
# Refined regex for standard social attribution signatures (e.g. "via @username", "via Facebook", etc.)
ATTRIBUTION_PATTERN = re.compile(r'\bvia\s+[@#]?[A-Za-z0-9_]+\b|\bvia\s+twitter\b|\bvia\s+facebook\b', re.IGNORECASE)
PUNCT_PATTERN = re.compile(r'\W')
NUMBER_PATTERN = re.compile(r'\b\d+\b')
SPACES_PATTERN = re.compile(r'\s+')

def clean_text(text):
    """
    Cleans raw text by removing punctuation, links, datelines, social attributions,
    and numbers, and converting everything to lowercase.
    """
    if not isinstance(text, str):
        return ""
    
    # Strip publisher datelines at the beginning of the text stream
    text = DATELINE_PATTERN.sub(' ', text)
    
    # Strip social attribution signatures
    text = ATTRIBUTION_PATTERN.sub(' ', text)
    
    # Lowercase everything
    text = text.lower()
    
    # Strip web links (URLs)
    text = URL_PATTERN.sub(' ', text)
    
    # Replace punctuation symbols with a space so words don't mash together
    text = PUNCT_PATTERN.sub(' ', text)
    
    # Remove solo numbers/digits
    text = NUMBER_PATTERN.sub(' ', text)
    
    # Fix extra accidental spaces
    text = SPACES_PATTERN.sub(' ', text).strip()
    
    return text

def tokenize_and_filter(text):
    """
    Splits sentences into single words and drops the stopwords.
    """
    if not text:
        return ""

    # Split into words, then filter out any empty strings and stopwords
    words = text.split(' ')
    return " ".join([word for word in words if word and word not in STOPWORDS])

def expand_contractions(text):
    contractions_dict = {
        "won't": "will not", "can't": "cannot", "don't": "do not", "shouldn't": "should not",
        "needn't": "need not", "hasn't": "has not", "haven't": "have not", "weren't": "were not",
        "wasn't": "was not", "didn't": "did not", "doesn't": "does not", "isn't": "is not",
        "aren't": "are not", "ain't": "is not", "it's": "it is", "i'm": "i am", "he's": "he is",
        "she's": "she is", "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would", "i'll": "i will", "you'll": "you will",
        "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will"
    }
    # Pattern to match contractions
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group(0)], text)

def full_preprocess_pipeline(text):
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Unicode Normalization (curly quotes, dashes, smart punctuation)
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'").replace('—', '-').replace('–', '-')
    
    # 2. Lowercase target string stream
    text = text.lower()
    
    # 3. Contraction Expansion
    text = expand_contractions(text)
    
    # 4. Repeated character normalization (e.g., 'sooooo' -> 'soo')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 5. Emoji & Non-ASCII removal (keep simple punctuation/alphanumeric)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 6. Strip Global Publisher Leakage & Datelines
    # Removes patterns like "london (reuters) - ", "tokyo (ap) - ", etc.
    text = re.sub(r'^[a-z\s\.,\-_\|]+(\(reuters\)[^\-]*\-|\(ap\)[^\-]*\-)', '', text)
    text = re.sub(r'\b(reuters|ap|bbc|cnn|foxnews|truthunleashed)\b', '', text)
    text = re.sub(r'\bread more via\b.*$', '', text)
    
    # 7. Strip URL patterns and email entities
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 8. Extract word tokens between 3 and 15 alphabetic characters
    words = re.findall(r'\b[a-z]{3,15}\b', text)
    
    # 9. Native Optimized Academic Stopwords List & Domain/Publisher Leakage Filters
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", 
        "in", "for", "on", "with", "at", "by", "from", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "which", "who", "whom", "here", "there",
        # Days of the week (to prevent day-of-week publisher artifacts)
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        # Months (to prevent monthly bias artifacts)
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        # Political/Figure entities to prevent topical name-memorization leakage
        "trump", "obama", "hillary", "clinton", "biden", "bush", "sanders", "donald", "barack",
        "gop", "republican", "republicans", "democrat", "democrats", "democratic", "sen", "rep",
        # Temporal markers (to prevent time bias artifacts)
        "today", "yesterday", "tomorrow", "tonight", "election",
        # Reporting verbs and publishing boilerplate
        "said", "told", "reporters", "spokesman", "spokeswoman", "added", "stated", "reporting", "statement", "comment",
        "read", "watch", "video", "image", "featured", "photo", "pic", "twitter", "com", "getty", "images",
        "breaking", "cops", "reuters", "ap", "news", "via", "est", "edt"
    }
    
    cleaned_tokens = [w for w in words if w not in stopwords]
    return " ".join(cleaned_tokens)

def compute_dense_features(raw_text, clean_str, title=""):
    import numpy as np
    from collections import Counter
    
    # Standard fallback values if text is empty
    raw_chars = max(1, len(raw_text))
    words = raw_text.split()
    raw_words = max(1, len(words))
    clean_words = clean_str.split()
    clean_words_count = max(1, len(clean_words))
    
    # --- Writing Style (Lexical & Stylometric) ---
    unique_words = set(clean_words)
    lexical_diversity = len(unique_words) / clean_words_count
    ttr = len(unique_words) / clean_words_count
    
    # Hapax Legomena Ratio (words occurring exactly once)
    clean_word_counts = Counter(clean_words)
    hapax_count = sum(1 for w, count in clean_word_counts.items() if count == 1)
    hapax_ratio = hapax_count / clean_words_count
    
    # Fast sentence count estimation
    sentences_count = max(1, raw_text.count('.') + raw_text.count('!') + raw_text.count('?'))
    avg_sentence_len = len(words) / sentences_count
    avg_word_len = sum(len(w) for w in words) / raw_words
    
    # Character ratios
    punct_density = sum(1 for c in raw_text if c in '.,!?;:"\'()[]{}') / raw_chars
    digit_ratio = sum(1 for c in raw_text if c.isdigit()) / raw_chars
    uppercase_ratio = len(re.findall(r'[A-Z]', raw_text)) / raw_chars
    
    # Stopword Ratio
    from src.preprocessing import STOPWORDS
    stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
    stopword_ratio = stopword_count / raw_words
    
    # --- Readability Metrics ---
    # Syllables estimation
    text_lower = raw_text.lower()
    total_vowels = sum(text_lower.count(v) for v in 'aeiou')
    syllables = max(raw_words, total_vowels)
    
    # Flesch Reading Ease (FRE)
    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * (syllables / raw_words)
    flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))
    
    # Flesch-Kincaid Grade Level
    flesch_kincaid_grade = 0.39 * avg_sentence_len + 11.8 * (syllables / raw_words) - 15.59
    flesch_kincaid_grade = max(0.0, min(20.0, flesch_kincaid_grade))
    
    # Complex words (syllables >= 3)
    complex_words = sum(1 for w in words if sum(1 for c in w.lower() if c in 'aeiou') >= 3)
    complex_word_ratio = complex_words / raw_words
    
    # Gunning Fog Index
    gunning_fog = 0.4 * (avg_sentence_len + 100 * complex_word_ratio)
    gunning_fog = max(0.0, min(20.0, gunning_fog))
    
    # Coleman-Liau Index
    letters_only = len(re.findall(r'[a-zA-Z0-9]', raw_text))
    L = (letters_only / raw_words) * 100
    S = (sentences_count / raw_words) * 100
    coleman_liau = 0.0588 * L - 0.296 * S - 15.8
    coleman_liau = max(0.0, min(20.0, coleman_liau))
    
    # SMOG Index
    smog_index = 1.0430 * np.sqrt(complex_words * (30 / sentences_count)) + 3.1291
    smog_index = max(0.0, min(20.0, smog_index))
    
    # --- Emotion Metrics ---
    pos_words = {"great", "excellent", "good", "verify", "truth", "true", "positive", "credible", "reliable", "validated", "factual", "correct", "success"}
    neg_words = {"fake", "worst", "terrible", "bad", "false", "hoax", "lie", "disaster", "negative", "unverified", "suspicious", "misleading", "conspiracy", "rumor"}
    pos_count = sum(clean_words.count(w) for w in pos_words)
    neg_count = sum(clean_words.count(w) for w in neg_words)
    
    polarity = (pos_count - neg_count) / (pos_count + neg_count + 1e-5)
    subjectivity = (pos_count + neg_count) / clean_words_count
    
    # Emotion Intensity (Sensationalized/Emotional Vocabulary density)
    emotional_vocabulary = {"shocking", "leaked", "secret", "urgent", "viral", "exposed", "unbelievable", "miracle", "warning", "banned", "chaos", "destroys", "slam", "blasts", "panic", "terror", "crisis"}
    emotion_intensity = sum(clean_words.count(w) for w in emotional_vocabulary) / clean_words_count
    
    # --- Clickbait Features ---
    excessive_punct = 1.0 if ("!!!" in raw_text or "???" in raw_text or "!?" in raw_text) else 0.0
    
    clickbait_words = {"breaking", "shocking", "must-see", "unbelievable", "secret", "exposed", "urgent", "viral", "insider", "banned", "conspiracy"}
    clickbait_ratio = sum(clean_words.count(w) for w in clickbait_words) / clean_words_count
    
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    all_caps_ratio = all_caps_words / raw_words
    
    sensational_phrases = {"you won't believe", "what happened next", "secret they don't want you to know", "shocked the world", "the truth about"}
    sensational_count = sum(1 for phrase in sensational_phrases if phrase in text_lower)
    sensational_ratio = sensational_count / sentences_count
    
    urgency_indicators = {"now", "urgent", "immediately", "hurry", "last chance", "breaking news"}
    urgency_count = sum(1 for indicator in urgency_indicators if indicator in text_lower)
    urgency_ratio = urgency_count / clean_words_count
    
    # --- Hedging and Certainty Language ---
    hedging_words = {"maybe", "perhaps", "possibly", "allegedly", "reportedly", "could", "might", "suggests", "seems", "apparently", "presumably", "suspected", "rumored"}
    hedging_count = sum(clean_words.count(w) for w in hedging_words)
    hedging_ratio = hedging_count / clean_words_count
    
    certainty_words = {"definitely", "absolutely", "always", "never", "proven", "fact", "undeniable", "unquestionably", "certainly", "truth", "obvious", "obviously", "clear", "clearly"}
    certainty_count = sum(clean_words.count(w) for w in certainty_words)
    certainty_ratio = certainty_count / clean_words_count

    # --- Metadata Features ---
    title_str = str(title) if title else ""
    title_length_chars = len(title_str)
    title_length_words = len(title_str.split())
    
    article_length_chars = len(raw_text)
    paragraph_count = max(1, raw_text.count('\n\n') + 1)
    quotation_count = raw_text.count('"') + raw_text.count("'")
    external_links = raw_text.count('http://') + raw_text.count('https://') + raw_text.count('www.')
    
    word_count = len(words)
    reading_time = word_count / 200.0  # Standard average reading speed
    
    return [
        lexical_diversity, ttr, hapax_ratio, avg_sentence_len, avg_word_len,
        punct_density, digit_ratio, uppercase_ratio, stopword_ratio,
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog, coleman_liau, smog_index,
        polarity, subjectivity, emotion_intensity,
        excessive_punct, clickbait_ratio, all_caps_ratio, sensational_ratio, urgency_ratio,
        hedging_ratio, certainty_ratio,
        title_length_chars, title_length_words, article_length_chars,
        paragraph_count, quotation_count, external_links, word_count, reading_time
    ]