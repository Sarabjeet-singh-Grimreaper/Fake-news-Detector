import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
from typing import Dict, List, Any, Optional

# In-memory LRU/TTL cache for live news queries: query -> (timestamp, result_dict)
_CACHE: Dict[str, tuple] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes cache

DEBUNK_KEYWORDS = [
    "fact check", "fact-check", "false claim", "debunked", "debunk", 
    "hoax", "untrue", "misleading", "fake news", "not true", 
    "snopes", "politifact", "ap fact check", "reuters fact check"
]

HIGH_AUTHORITY_SOURCES = {
    "reuters", "associated press", "ap news", "ap", "bbc", "bbc news",
    "bloomberg", "the wall street journal", "wsj", "the new york times",
    "the guardian", "the washington post", "financial times", "ft",
    "npr", "pbs", "afp", "agence france-presse", "deutsche welle", "dw",
    "france 24", "al jazeera", "cnbc", "forbes", "time", "the hindu",
    "world health organization", "who", "cdc", "nasa", "nature", "science"
}

RSS_TOPICS = {
    "all": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    "business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    "technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "health": "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-US&gl=US&ceid=US:en"
}

STOP_AND_GENERIC_WORDS = {
    'in', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should',
    'now', 'says', 'said', 'tells', 'told', 'reported', 'reports', 'per', 'according',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our',
    'breaking', 'urgent', 'shocking', 'truth', 'exposed', 'secret', 'leaked', 'historic',
    'development', 'scientists', 'researchers', 'experts', 'officials', 'confirmed',
    'revealed', 'exclusive', 'finally', 'unbelievable', 'watch', 'video', 'today',
    'yesterday', 'tomorrow', 'new', 'could', 'would', 'have', 'has', 'had', 'been',
    'were', 'announced', 'issued', 'expand', 'state', 'scheduled', 'tonight', 'inside',
    'reveal', 'warning', 'watch', 'video', 'before', 'gets', 'banned', 'rich', 'hours',
    'trick', 'absolutely', 'shocked', 'closed', 'doors', 'these', 'what', 'development'
}

def extract_search_terms(headline: str, body_text: str = "", max_words: int = 5) -> tuple:
    """
    Extracts the most salient keywords and entities from headline or body text
    to form an optimal news wire search query, and returns both the query string
    and the list of distinctive keywords for relevance scoring.
    """
    text = headline.strip() if headline and len(headline.strip()) > 5 else body_text.strip()
    if not headline.strip() and text:
        first_sent = re.split(r'[.!?\n]', text)[0]
        text = first_sent if len(first_sent.split()) >= 4 else text[:150]
        
    text = re.sub(r'^[A-Za-z\s]+[-—]\s*', '', text)
    text = re.sub(r'\(reuters\)|\(ap\)|\(afp\)', '', text, flags=re.IGNORECASE)
    
    # Clean words
    words = [w.lower() for w in re.sub(r'[^a-zA-Z0-9\$\-\s]', ' ', text).split() if len(w) > 2]
    salient_words = [w for w in words if w not in STOP_AND_GENERIC_WORDS]
    
    if not salient_words:
        salient_words = words[:max_words]
        
    query_str = " ".join(salient_words[:max_words])
    return query_str, salient_words[:8]

def search_live_news_rss(query: str, timeout: int = 5) -> List[Dict[str, str]]:
    """
    Searches Google News RSS for live news articles matching the query.
    Returns list of dicts: [{'title': ..., 'source': ..., 'pub_date': ..., 'link': ...}]
    """
    if not query or not query.strip():
        return []
        
    encoded_query = urllib.parse.quote(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            articles = []
            for item in root.findall(".//item")[:10]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                source_el = item.find("source")
                source = source_el.text if source_el is not None and source_el.text else "News Agency"
                
                # Clean title if it contains " - Publisher" at end
                clean_title = title
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    clean_title = parts[0].strip()
                    if not source or source == "News Agency":
                        source = parts[1].strip()
                        
                articles.append({
                    "title": clean_title,
                    "source": source,
                    "pub_date": pub_date,
                    "link": link
                })
            return articles
    except Exception:
        return []

def verify_realtime_news(headline: str, body_text: str = "", domain: str = "") -> Dict[str, Any]:
    """
    Performs real-time live verification of a news story or claim.
    1. Extracts search terms.
    2. Queries live global news feeds.
    3. Checks for live corroboration across major news wires.
    4. Scans for explicit fact-checking debunks / alerts.
    5. Filters articles by substantive entity overlap to eliminate spurious matches.
    """
    query, salient_terms = extract_search_terms(headline, body_text)
    if not query:
        return {
            "query": "",
            "corroboration_score": 50,
            "corroboration_status": "No search terms could be extracted",
            "badge": "Unverified",
            "articles_found": 0,
            "trusted_matches": 0,
            "fact_check_alert": False,
            "fact_check_details": "",
            "matching_articles": []
        }
        
    # Check cache
    cache_key = query.lower()
    now = time.time()
    if cache_key in _CACHE:
        ts, cached_res = _CACHE[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            return cached_res
            
    raw_articles = search_live_news_rss(query)
    
    # Filter by substantive relevance: article title must contain substantive query keywords
    relevant_articles = []
    salient_set = set(salient_terms)
    
    trusted_matches = 0
    fact_check_detected = False
    fact_check_details = ""
    
    for art in raw_articles:
        src_lower = art["source"].lower()
        title_lower = art["title"].lower()
        
        # Check debunk keywords across ALL returned items
        for kw in DEBUNK_KEYWORDS:
            if kw in title_lower or kw in src_lower:
                fact_check_detected = True
                fact_check_details = f"Fact-check / debunk detected from {art['source']}: '{art['title']}'"
                break
                
        # Substantive entity overlap check
        overlap = sum(1 for term in salient_set if term in title_lower)
        # Require at least 2 distinctive terms or single rare multi-part term
        if overlap >= 2 or (len(salient_set) <= 2 and overlap >= 1):
            is_trusted = any(h_src in src_lower for h_src in HIGH_AUTHORITY_SOURCES)
            if is_trusted:
                trusted_matches += 1
            relevant_articles.append({
                **art,
                "overlap_score": overlap,
                "is_trusted": is_trusted
            })
            
    total_relevant = len(relevant_articles)
    
    # Calculate Corroboration Rating (0 to 100)
    if fact_check_detected:
        corroboration_score = 12
        corroboration_status = "Debunked by Fact-Checking Media"
        badge = "Debunked"
    elif trusted_matches >= 2:
        corroboration_score = 96
        corroboration_status = f"Strongly Corroborated by {trusted_matches} Verified International Outlets"
        badge = "Corroborated"
    elif trusted_matches == 1:
        corroboration_score = 88
        corroboration_status = "Corroborated by Major News Wire"
        badge = "Corroborated"
    elif total_relevant >= 3:
        corroboration_score = 75
        corroboration_status = f"Corroborated by {total_relevant} General Media Reports"
        badge = "Corroborated"
    elif total_relevant >= 1:
        corroboration_score = 55
        corroboration_status = f"Limited Media Coverage ({total_relevant} source)"
        badge = "Unverified"
    else:
        corroboration_score = 25
        corroboration_status = "Uncorroborated on Live Global News Wires"
        badge = "Uncorroborated"
        
    result = {
        "query": query,
        "corroboration_score": corroboration_score,
        "corroboration_status": corroboration_status,
        "badge": badge,
        "articles_found": total_relevant,
        "trusted_matches": trusted_matches,
        "fact_check_alert": fact_check_detected,
        "fact_check_details": fact_check_details,
        "matching_articles": relevant_articles[:5]
    }
    
    # Store in cache
    _CACHE[cache_key] = (now, result)
    return result

def fetch_live_breaking_news(category: str = "all", limit: int = 15) -> List[Dict[str, str]]:
    """
    Fetches real-time breaking news items from top global live feeds.
    Supported categories: 'all', 'world', 'business', 'technology', 'science', 'health'.
    """
    feed_url = RSS_TOPICS.get(category.lower(), RSS_TOPICS["all"])
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(feed_url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            items = []
            for item in root.findall(".//item")[:limit]:
                title = item.find("title").text if item.find("title") is not None else "Untitled"
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                source_el = item.find("source")
                source = source_el.text if source_el is not None and source_el.text else "Global News Agency"
                
                clean_title = title
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    clean_title = parts[0].strip()
                    if not source or source == "Global News Agency":
                        source = parts[1].strip()
                        
                items.append({
                    "title": clean_title,
                    "source": source,
                    "pub_date": pub_date,
                    "link": link
                })
            return items
    except Exception:
        return []
