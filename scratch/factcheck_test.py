import requests

def dump_html(query):
    url = "https://lite.duckduckgo.com/lite/"
    data = {"q": query + " fact check"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    r = requests.post(url, data=data, headers=headers, timeout=10)
    with open("scratch/dump.html", "w", encoding="utf-8") as f:
        f.write(r.text)
    print("Dumped html to scratch/dump.html. Length:", len(r.text))

if __name__ == "__main__":
    dump_html("Donald Trump is killed")
