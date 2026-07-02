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
    'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now'
}

def clean_text(text):
    """
    Cleans raw text by removing punctuation, links, and numbers,
    and converting everything to lowercase.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase everything
    text = text.lower()
    
    # Strip web links (URLs)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Replace punctuation symbols with a space so words don't mash together
    text = re.sub(r'\W', ' ', text)
    
    # Remove solo numbers/digits
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Fix extra accidental spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_filter(text):
    """
    Splits sentences into single words and drops the stopwords.
    """
    # Break sentence into a list of words using spaces
    raw_words = text.split(' ')
    
    # Keep the word only if it isn't blank and isn't a stopword
    filtered_words = [word for word in raw_words if word and word not in STOPWORDS]
    
    # Glue the cleaned words back together with a single space
    return " ".join(filtered_words)

def full_preprocess_pipeline(text):
    """
    The master function that runs both steps back-to-back.
    """
    cleaned_string = clean_text(text)
    final_output = tokenize_and_filter(cleaned_string)
    return final_output