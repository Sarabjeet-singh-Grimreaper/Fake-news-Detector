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
    """
    Backward-compatible bridge forwarding to canonical feature extraction in src.features.
    Returns the ordered list of 12 dense features matching the V2.0 pipeline and dense scaler.
    """
    from src.features import extract_dense_features
    _, feats_list = extract_dense_features(raw_text, clean_str, title)
    return feats_list