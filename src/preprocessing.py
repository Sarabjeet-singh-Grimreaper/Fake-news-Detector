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

def full_preprocess_pipeline(text):
    """
    The master function that runs both steps back-to-back.
    """
    cleaned_string = clean_text(text)
    final_output = tokenize_and_filter(cleaned_string)
    return final_output