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
            
        # Try to extract paragraphs, headers, and list items
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
            
        return {
            "title": title,
            "text": full_text,
            "url": url
        }
    except Exception as e:
        return {"error": f"Failed to parse content: {str(e)}"}
