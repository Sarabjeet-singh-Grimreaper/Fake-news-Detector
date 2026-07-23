import requests
from bs4 import BeautifulSoup
import re

def scrape_article(url):
    """
    Fetches the web page at the given URL, extracts the article title
    and main body text, and removes boilerplate (header, footer, nav, script, style).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}
    
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove unwanted tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
            
        # Try to find the title
        title = ""
        title_element = soup.find("h1")
        if title_element:
            title = title_element.get_text().strip()
        else:
            title = soup.title.string.strip() if soup.title else "Untitled Article"
            
        # Try to find the main article container to filter out boilerplate sidebars/footers
        article_container = soup.find(["article", "main"]) or soup.find(id=re.compile(r'article|content|main', re.I)) or soup.find(class_=re.compile(r'article|content|main', re.I))
        
        # Try to extract paragraphs, headers, and list items
        if article_container:
            elements = article_container.find_all(["p", "h1", "h2", "h3", "h4", "li"])
        else:
            elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
            
        text_content = []
        for el in elements:
            el_text = el.get_text().strip()
            # Ignore short snippets/nav links
            if len(el_text.split()) > 4:
                text_content.append(el_text)
                
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
            # Fallback to general text extraction if no paragraphs found
            full_text = soup.get_text(separator="\n")
            # Clean up empty lines
            full_text = re.sub(r'\n+', '\n\n', full_text).strip()
            
        # Try to find author, date, publisher metadata
        author = "Unknown Author"
        date = "Unknown Date"
        publisher = "Unknown Publisher"
        category = "General"
        canonical_url = url
        language = "en"
        
        # Meta tag searches
        meta_author = soup.find("meta", {"name": re.compile(r'author|creator', re.I)}) or soup.find("meta", {"property": re.compile(r'author', re.I)})
        if meta_author:
            author = meta_author.get("content", "").strip() or author
            
        meta_date = soup.find("meta", {"name": re.compile(r'date|publish|time', re.I)}) or soup.find("meta", {"property": re.compile(r'publish_time|published_time', re.I)})
        if meta_date:
            date = meta_date.get("content", "").strip() or date
            
        meta_pub = soup.find("meta", {"name": re.compile(r'publisher|source', re.I)}) or soup.find("meta", {"property": re.compile(r'og:site_name', re.I)})
        if meta_pub:
            publisher = meta_pub.get("content", "").strip() or publisher
            
        meta_cat = soup.find("meta", {"name": re.compile(r'category|section|topic', re.I)}) or soup.find("meta", {"property": re.compile(r'article:section', re.I)})
        if meta_cat:
            category = meta_cat.get("content", "").strip() or category
            
        meta_canon = soup.find("link", {"rel": "canonical"}) or soup.find("meta", {"property": "og:url"})
        if meta_canon:
            canonical_url = meta_canon.get("href", "").strip() or meta_canon.get("content", "").strip() or canonical_url
            
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            language = html_tag.get("lang").strip().split('-')[0].lower()
            
        # Calculate statistics
        word_count = len(full_text.split())
        article_length = len(full_text)
        reading_time = max(0.5, round(word_count / 200.0, 1))
        
        # Quality check
        sentences_count = max(1, full_text.count('.') + full_text.count('!') + full_text.count('?'))
        poor_quality = word_count < 50 or sentences_count < 3
        
        return {
            "title": title,
            "text": full_text,
            "url": url,
            "author": author,
            "date": date,
            "publisher": publisher,
            "category": category,
            "canonical_url": canonical_url,
            "language": language,
            "word_count": word_count,
            "article_length": article_length,
            "reading_time": reading_time,
            "poor_quality": poor_quality
        }
    except Exception as e:
        return {"error": f"Failed to parse content: {str(e)}"}
