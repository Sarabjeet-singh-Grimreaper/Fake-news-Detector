import socket
import ipaddress
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
import json

MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2 MB max to prevent memory exhaustion

def is_safe_url(url: str):
    """
    Validates scheme and guards against SSRF by rejecting loopback, link-local,
    private, and non-HTTP(S) addresses.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "Unsupported protocol: only http and https are permitted."
        
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: missing hostname."
            
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False, "Access to localhost and loopback addresses is prohibited."
            
        # Resolve address to check IP range
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Access to private or local network address ({ip_str}) is prohibited."
                
        return True, ""
    except Exception as e:
        return False, f"URL resolution error: {str(e)}"

def scrape_article(url):
    """
    Fetches the web page at the given URL with SSRF guards and size limits,
    extracts the article title and main body text, and removes boilerplate.
    """
    if not url or not isinstance(url, str):
        return {"error": "Invalid URL provided."}
        
    url = url.strip()
    parsed_early = urlparse(url)
    if parsed_early.scheme and parsed_early.scheme not in ("http", "https"):
        return {"error": f"Unsupported protocol '{parsed_early.scheme}': only http and https are permitted."}

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    safe, msg = is_safe_url(url)
    if not safe:
        return {"error": msg}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=12, stream=True)
        response.raise_for_status()
        
        raw_ct = response.headers.get("Content-Type", "")
        content_type = str(raw_ct).lower() if isinstance(raw_ct, str) else ""
        if content_type and not any(ct in content_type for ct in ("text/html", "text/plain", "application/xhtml")):
            return {"error": f"Invalid content type ({content_type}). Only web text/HTML articles are supported."}
            
        # Read with size cap
        chunks = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > MAX_PAYLOAD_BYTES:
                break
        raw_html = b"".join(chunks)
        if not raw_html and hasattr(response, "content") and response.content:
            raw_html = response.content[:MAX_PAYLOAD_BYTES]
    except Exception as e:
        return {"error": f"Failed to fetch URL: {str(e)}"}
    
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # 1. Extract high-fidelity structured JSON-LD data before script decomposition
        json_ld_headline = ""
        json_ld_body = ""
        json_ld_author = ""
        json_ld_date = ""
        json_ld_publisher = ""
        
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                if not script.string:
                    continue
                data = json.loads(script.string.strip())
                items = data if isinstance(data, list) else (data.get("@graph", [data]) if isinstance(data, dict) else [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("@type", "")).lower()
                    if any(t in item_type for t in ("article", "newsarticle", "reportage", "blogposting", "webpage")):
                        if not json_ld_headline and item.get("headline"):
                            json_ld_headline = str(item.get("headline")).strip()
                        if not json_ld_body and item.get("articleBody"):
                            json_ld_body = str(item.get("articleBody")).strip()
                        if not json_ld_date and (item.get("datePublished") or item.get("dateModified")):
                            json_ld_date = str(item.get("datePublished") or item.get("dateModified")).strip()
                        if not json_ld_author and item.get("author"):
                            auth = item["author"]
                            if isinstance(auth, dict) and auth.get("name"):
                                json_ld_author = str(auth["name"]).strip()
                            elif isinstance(auth, list) and len(auth) > 0 and isinstance(auth[0], dict):
                                json_ld_author = str(auth[0].get("name", "")).strip()
                        if not json_ld_publisher and item.get("publisher"):
                            pub = item["publisher"]
                            if isinstance(pub, dict) and pub.get("name"):
                                json_ld_publisher = str(pub["name"]).strip()
            except Exception:
                continue

        # Remove unwanted tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()
            
        # Try to find the title
        title = json_ld_headline
        if not title:
            title_element = soup.find("h1")
            if title_element:
                title = title_element.get_text().strip()
            else:
                title = soup.title.string.strip() if soup.title else "Untitled Article"
            
        # Try to find the main article container to filter out boilerplate sidebars/footers
        if json_ld_body and len(json_ld_body.split()) >= 40:
            full_text = json_ld_body
        else:
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
        author = json_ld_author or "Unknown Author"
        date = json_ld_date or "Unknown Date"
        publisher = json_ld_publisher or "Unknown Publisher"
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
