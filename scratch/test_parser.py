from bs4 import BeautifulSoup

with open("scratch/dump.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, 'html.parser')
links = soup.find_all('a', class_='result-link')
snippets = soup.find_all('td', class_='result-snippet')

print(f"Links found: {len(links)}, Snippets found: {len(snippets)}")
for i, (l, s) in enumerate(zip(links[:3], snippets[:3])):
    print(f"\n[{i+1}] Title: {l.get_text().strip()}")
    print(f"URL: {l['href']}")
    print(f"Snippet: {s.get_text().strip()}")
