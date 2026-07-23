import re
from urllib.parse import urlparse

# Trust database for domain evaluation
TRUSTED_DOMAINS = {
    "reuters.com": (100, "Highly Trusted International News Agency"),
    "apnews.com": (99, "Highly Trusted Cooperative Press Agency"),
    "bbc.com": (98, "Highly Trusted Public Broadcaster"),
    "bbc.co.uk": (98, "Highly Trusted Public Broadcaster"),
    "who.int": (100, "Official Global Health Authority Organization"),
    "nasa.gov": (100, "Official Government Space Agency"),
    "cdc.gov": (100, "Official National Public Health Agency"),
    "nih.gov": (100, "Official National Medical Research Agency"),
    "nytimes.com": (97, "Reputable Mainstream Journalism Publisher"),
    "bloomberg.com": (98, "Reputable Financial News Publisher"),
    "theguardian.com": (96, "Reputable Mainstream Journalism Publisher"),
    "npr.org": (96, "Reputable Public Radio Broadcaster"),
}

LOW_TRUST_DOMAINS = {
    "breitbart.com": (25, "Partisan News Publisher - High Fact-Check Redirection"),
    "infowars.com": (10, "Conspiracy Focus Outlet - Low Factual Accuracy History"),
    "naturalnews.com": (15, "Pseudo-Science Focus Outlet - Unverified Health Claims"),
    "activistpost.com": (30, "Alternative Focus Blog - Unverified Editorial Content"),
}

def get_domain_credibility(url: str) -> dict:
    """
    Parses a URL, extracts the domain name, and returns a credibility rating dictionary.
    """
    if not url or not isinstance(url, str):
        return {"domain": "N/A", "score": 50, "status": "No URL provided", "badge": "Unverified"}
        
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
            
        # 1. Exact match checks
        if domain in TRUSTED_DOMAINS:
            score, desc = TRUSTED_DOMAINS[domain]
            return {"domain": domain, "score": score, "status": desc, "badge": "Trusted"}
        if domain in LOW_TRUST_DOMAINS:
            score, desc = LOW_TRUST_DOMAINS[domain]
            return {"domain": domain, "score": score, "status": desc, "badge": "Low Trust"}
            
        # 2. Pattern and TLD matches (e.g. *.gov, *.mil, *.edu)
        if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".gov.in"):
            return {"domain": domain, "score": 98, "status": "Official Public Sector Domain", "badge": "Trusted"}
        if domain.endswith(".mil"):
            return {"domain": domain, "score": 98, "status": "Official Defense / Military Sector Domain", "badge": "Trusted"}
        if domain.endswith(".edu") or domain.endswith(".ac.uk"):
            return {"domain": domain, "score": 92, "status": "Academic Research & Educational Institution Domain", "badge": "Trusted"}
            
        # 3. Default fallback for unlisted domains
        return {
            "domain": domain,
            "score": 50,
            "status": "Independent or Generic Publisher (Standard Verification Required)",
            "badge": "Neutral"
        }
    except Exception as e:
        return {"domain": "Error", "score": 50, "status": f"Evaluation error: {str(e)}", "badge": "Unverified"}
