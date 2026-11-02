"""Stage 0: ingestion. Download each UNC policy page once and save its text as Markdown.

We keep the headings (## Procedure, ### 2.1 ...) because chunking splits on them.
Output: data/policies.json. It's committed, so you only need to re-run this if the policies change.

    python ingest.py
"""

import json
import re

import requests
from bs4 import BeautifulSoup

BLOCKS = ["h2", "h3", "h4", "h5", "p", "li", "tr"]


def html_to_markdown(html: str) -> str:
    body = BeautifulSoup(html, "html.parser").find(id="ctl00_ctl00_cpContent_cpContent_divBody")
    lines = []
    for el in body.find_all(BLOCKS):
        # Skip blocks nested in another block (a <p> inside an <li>), or we'd get them twice.
        if el.find_parent(BLOCKS):
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append("\n" + "#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        elif el.name == "tr":
            lines.append(" | ".join(c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])))
        else:
            lines.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def main():
    sources = json.load(open("data/sources.json"))
    for doc in sources:
        print("fetching", doc["title"])
        html = requests.get(doc["source_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        doc["text"] = html_to_markdown(html)
    json.dump(sources, open("data/policies.json", "w"), indent=2, ensure_ascii=False)
    print(f"saved {len(sources)} policies to data/policies.json")


if __name__ == "__main__":
    main()
