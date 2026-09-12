import re
from urllib.parse import urlparse

# Trust database for domain evaluation: (score, description, badge)
TRUSTED_DOMAINS = {
    # Global Press & Wire Services
    "reuters.com": (100, "Highly Trusted International News Agency", "Trusted"),
    "apnews.com": (99, "Highly Trusted Cooperative Press Agency", "Trusted"),
    "afp.com": (99, "Highly Trusted Global Wire Service", "Trusted"),
    "bloomberg.com": (98, "Reputable Financial & Global News Publisher", "Trusted"),
    
    # Public Broadcasters & Established Journalism
    "bbc.com": (98, "Highly Trusted Public Broadcaster", "Trusted"),
    "bbc.co.uk": (98, "Highly Trusted Public Broadcaster", "Trusted"),
    "npr.org": (96, "Reputable Public Radio Broadcaster", "Trusted"),
    "pbs.org": (96, "Reputable Public Broadcasting Service", "Trusted"),
    "dw.com": (96, "Reputable German Public International Broadcaster", "Trusted"),
    "france24.com": (95, "Reputable French International News Network", "Trusted"),
    "aljazeera.com": (92, "Reputable Global News Network", "Trusted"),
    "cbc.ca": (95, "Reputable Canadian Public Broadcaster", "Trusted"),
    "abc.net.au": (95, "Reputable Australian Public Broadcaster", "Trusted"),
    "nhk.or.jp": (96, "Reputable Japanese Public Broadcaster", "Trusted"),
    
    # Tier-1 Print & Digital Journalism
    "nytimes.com": (97, "Reputable Mainstream Journalism Publisher", "Trusted"),
    "wsj.com": (97, "Reputable Financial & Global News Publisher", "Trusted"),
    "theguardian.com": (96, "Reputable Mainstream Journalism Publisher", "Trusted"),
    "washingtonpost.com": (95, "Reputable Investigative Journalism Publisher", "Trusted"),
    "ft.com": (97, "Reputable Financial Times Publishing House", "Trusted"),
    "economist.com": (97, "Reputable Global Economics & Policy Publication", "Trusted"),
    "time.com": (93, "Reputable Weekly News Magazine", "Trusted"),
    "theatlantic.com": (94, "Reputable Cultural & Political Journalism Magazine", "Trusted"),
    "propublica.org": (98, "Pulitzer-Winning Non-profit Investigative Newsroom", "Trusted"),
    "reutersagency.com": (100, "Official Reuters News Agency Portal", "Trusted"),
    
    # Major National & International Outlets
    "cnn.com": (90, "Major Cable & Digital News Organization", "Trusted"),
    "nbcnews.com": (92, "Major National Broadcast News Network", "Trusted"),
    "cbsnews.com": (92, "Major National Broadcast News Network", "Trusted"),
    "abcnews.go.com": (92, "Major National Broadcast News Network", "Trusted"),
    "usatoday.com": (91, "National Daily Newspaper Publisher", "Trusted"),
    "thehindu.com": (94, "Reputable National Indian Daily Broadcaster", "Trusted"),
    "indianexpress.com": (93, "Reputable Indian Journalism Publication", "Trusted"),
    "timesofindia.indiatimes.com": (88, "Major Indian Commercial News Daily", "Trusted"),
    "smh.com.au": (93, "Reputable Australian Metropolitan Broadcaster", "Trusted"),
    
    # Scientific & Medical Journals / Authorities
    "who.int": (100, "Official Global Health Authority Organization", "Trusted"),
    "cdc.gov": (100, "Official National Public Health Agency", "Trusted"),
    "nih.gov": (100, "Official National Medical Research Agency", "Trusted"),
    "nasa.gov": (100, "Official Government Space Agency", "Trusted"),
    "nature.com": (99, "Peer-Reviewed High-Impact Scientific Journal", "Trusted"),
    "science.org": (99, "Leading International Scientific Journal", "Trusted"),
    "thelancet.com": (99, "Premier International Medical Journal", "Trusted"),
    "nejm.org": (99, "New England Journal of Medicine", "Trusted"),
    "scientificamerican.com": (95, "Authoritative Popular Science Publication", "Trusted"),
    
    # Fact-Checking Organizations
    "snopes.com": (98, "Independent Fact-Checking Organization", "Trusted"),
    "politifact.com": (98, "Pulitzer-Winning Fact-Checking Project", "Trusted"),
    "factcheck.org": (98, "Nonpartisan Educational Fact-Checking Project", "Trusted"),
    "fullfact.org": (98, "Independent UK Fact-Checking Charity", "Trusted"),
    "leadstories.com": (96, "IFCN Certified Fact-Checking Organization", "Trusted"),
    "boomlive.in": (95, "IFCN Certified Fact-Checking Newsroom", "Trusted"),
    "altnews.in": (94, "IFCN Certified Investigative Fact-Checking Portal", "Trusted"),
}

# Satire & Parody Publications (Recognized humorous or satirical content)
SATIRE_DOMAINS = {
    "theonion.com": (15, "Recognized Satirical News Publication", "Satire / Parody"),
    "babylonbee.com": (15, "Recognized Conservative Satirical News Outlet", "Satire / Parody"),
    "clickhole.com": (15, "Satirical Clickbait Parody Site", "Satire / Parody"),
    "thebeaverton.com": (15, "Canadian Satirical News Media Publication", "Satire / Parody"),
    "waterfordwhispersnews.com": (15, "Irish Satirical News Website", "Satire / Parody"),
    "thedailymash.co.uk": (15, "British Satirical News Outlet", "Satire / Parody"),
    "borowitzreport.com": (15, "Satirical Political Column", "Satire / Parody"),
    "duffelblog.com": (15, "Military Satire & Parody Publication", "Satire / Parody"),
    "satirewire.com": (15, "Satirical News Wire Service", "Satire / Parody"),
    "unrealtimes.com": (15, "Indian Satirical News Portal", "Satire / Parody"),
    "fakingnews.com": (15, "Indian Satirical News Portal", "Satire / Parody"),
    "newsthump.com": (15, "UK Satirical News & Current Affairs Outlet", "Satire / Parody"),
    "private-eye.co.uk": (30, "British Satirical & Current Affairs Magazine", "Satire / Parody"),
}

# Known Disinformation, Conspiracy, and Low-Credibility Outlets
LOW_TRUST_DOMAINS = {
    "breitbart.com": (25, "Partisan News Publisher - High Fact-Check Redirection", "Low Trust"),
    "infowars.com": (10, "Conspiracy Focus Outlet - Multiple Defamation & Falsehood Rulings", "Low Trust"),
    "naturalnews.com": (15, "Pseudo-Science Focus Outlet - Unverified Health Claims", "Low Trust"),
    "activistpost.com": (30, "Alternative Focus Blog - Unverified Editorial Content", "Low Trust"),
    "thegatewaypundit.com": (20, "Partisan Blog - Multiple False Claim Retractions", "Low Trust"),
    "worldnewsdailyreport.com": (5, "Fabricated Hoax & Fake News Factory", "Low Trust"),
    "beforeitsnews.com": (15, "Unmoderated Conspiracy Aggregator", "Low Trust"),
    "newspunch.com": (10, "Frequent Disinformation Publisher - Debunked by Fact Checkers", "Low Trust"),
    "yournewswire.com": (10, "Prolific Disinformation Generator", "Low Trust"),
    "empirenews.net": (10, "Fabricated Stories & Fake News Portal", "Low Trust"),
    "prntly.com": (10, "Fabricated Political Propaganda Site", "Low Trust"),
    "wnd.com": (25, "Conspiracy & Pseudoscience Aggregator", "Low Trust"),
    "globalresearch.ca": (20, "Conspiracy & Propaganda Aggregator", "Low Trust"),
    "veteranstoday.com": (15, "Conspiracy Focus Blog - Antisemitic & Disinformation History", "Low Trust"),
    "neonnettle.com": (15, "Conspiracy & Unverified Claims Blog", "Low Trust"),
    "bizpacreview.com": (30, "Hyper-Partisan Editorial Blog", "Low Trust"),
}

def get_domain_credibility(url: str) -> dict:
    """
    Parses a URL, extracts the canonical domain name, and returns a credibility rating dictionary.
    Handles raw domains, missing protocols, subdomains (e.g. edition.cnn.com, m.reuters.com),
    and port numbers cleanly.
    """
    if not url or not isinstance(url, str):
        return {"domain": "N/A", "score": 50, "status": "No URL provided", "badge": "Unverified"}
        
    try:
        url_clean = url.strip()
        if not url_clean:
            return {"domain": "N/A", "score": 50, "status": "Empty URL provided", "badge": "Unverified"}
            
        if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
            url_clean = "https://" + url_clean
            
        parsed_url = urlparse(url_clean)
        domain = parsed_url.netloc.lower() or parsed_url.path.split('/')[0].lower()
        
        # Strip port if present
        if ":" in domain:
            domain = domain.split(":")[0]
            
        # Strip standard prefixes: www., m., amp., edition.
        for prefix in ("www.", "m.", "amp.", "edition."):
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
                break
                
        # 1. Exact or subdomain match in SATIRE list (Check satire first!)
        for satire_d, (score, desc, badge) in SATIRE_DOMAINS.items():
            if domain == satire_d or domain.endswith("." + satire_d):
                return {"domain": domain, "score": score, "status": desc, "badge": badge}
                
        # 2. Exact or subdomain match in TRUSTED list
        for trusted_d, (score, desc, badge) in TRUSTED_DOMAINS.items():
            if domain == trusted_d or domain.endswith("." + trusted_d):
                return {"domain": domain, "score": score, "status": desc, "badge": badge}
                
        # 3. Exact or subdomain match in LOW-TRUST list
        for low_d, (score, desc, badge) in LOW_TRUST_DOMAINS.items():
            if domain == low_d or domain.endswith("." + low_d):
                return {"domain": domain, "score": score, "status": desc, "badge": badge}
            
        # 4. Pattern and TLD matches (e.g. *.gov, *.mil, *.edu, *.ac.uk, *.gov.in)
        if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".gov.in") or domain.endswith(".gov.au"):
            return {"domain": domain, "score": 98, "status": "Official Public Sector Government Domain", "badge": "Trusted"}
        if domain.endswith(".mil"):
            return {"domain": domain, "score": 98, "status": "Official Defense / Military Sector Domain", "badge": "Trusted"}
        if domain.endswith(".edu") or domain.endswith(".ac.uk") or domain.endswith(".edu.au"):
            return {"domain": domain, "score": 93, "status": "Academic Research & Educational Institution Domain", "badge": "Trusted"}
            
        # 5. Default fallback for unlisted domains
        return {
            "domain": domain,
            "score": 50,
            "status": "Independent or Generic Publisher (Standard Verification Required)",
            "badge": "Neutral"
        }
    except Exception as e:
        return {"domain": "Error", "score": 50, "status": f"Evaluation error: {str(e)}", "badge": "Unverified"}
