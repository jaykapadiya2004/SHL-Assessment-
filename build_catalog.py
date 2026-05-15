#!/usr/bin/env python3
"""
Converts the official SHL product catalog JSON (from tcp-us-prod-rnd.shl.com)
into the format expected by main.py:
  - name, url, description, test_type, job_levels,
    remote_testing, adaptive, duration_minutes, languages, keywords

Run:  python build_catalog.py
Output: catalog.json (in the same directory)
"""

import json
import re
import urllib.request
from pathlib import Path

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = Path(__file__).parent / "catalog.json"

# ── Map "keys" array from the source to single-letter test_type codes ──────────
# When an item has multiple keys, pick the primary one via priority order.
KEY_PRIORITY = [
    "Personality & Behavior",
    "Ability & Aptitude",
    "Simulations",
    "Knowledge & Skills",
    "Biodata & Situational Judgment",
    "Competencies",
    "Development & 360",
    "Assessment Exercises",
]

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def pick_test_type(keys: list[str]) -> str:
    """Pick the best single test_type letter from a list of SHL 'keys'."""
    if not keys:
        return "K"  # fallback
    # Use priority order
    for priority_key in KEY_PRIORITY:
        for k in keys:
            if k.strip() == priority_key:
                return KEY_TO_CODE[priority_key]
    # Fallback: first recognised key
    for k in keys:
        code = KEY_TO_CODE.get(k.strip())
        if code:
            return code
    return "K"


def parse_duration(duration_str: str) -> int:
    """Extract integer minutes from strings like '17 minutes' or ''."""
    if not duration_str:
        return 0
    m = re.search(r"(\d+)", duration_str)
    return int(m.group(1)) if m else 0


def build_keywords(item: dict) -> list[str]:
    """Generate simple keyword tags from name, keys, and description snippet."""
    kws = set()
    # From test type keys
    for k in item.get("keys", []):
        kws.add(k.lower())
    # Name words (skip short stop-words)
    stop = {"new", "the", "and", "for", "with", "in", "of", "a", "an", "&"}
    for word in re.split(r"[\s\(\)\-\.\,]+", item.get("name", "")):
        w = word.lower().strip()
        if len(w) > 2 and w not in stop:
            kws.add(w)
    # First 8 significant words of description
    desc = item.get("description", "")
    for word in re.split(r"[\s\,\.\;\(\)]+", desc)[:60]:
        w = word.lower().strip()
        if len(w) > 3 and w not in stop:
            kws.add(w)
            if len(kws) >= 20:
                break
    return sorted(kws)[:15]  # cap at 15


def convert_item(raw: dict) -> dict:
    # The raw feed uses /products/ paths; the live SHL website uses /solutions/products/
    # Normalize to /solutions/products/ so URLs resolve correctly.
    url = raw.get("link", "").strip()
    url = url.replace("/products/product-catalog/", "/solutions/products/product-catalog/")
    
    return {
        "name": raw.get("name", "").strip(),
        "url": url.strip(),
        "description": raw.get("description", "").strip(),
        "test_type": pick_test_type(raw.get("keys", [])),
        "job_levels": raw.get("job_levels", []),
        "remote_testing": raw.get("remote", "").lower() == "yes",
        "adaptive": raw.get("adaptive", "").lower() == "yes",
        "duration_minutes": parse_duration(raw.get("duration", "")),
        "languages": raw.get("languages", []),
        "keywords": build_keywords(raw),
    }


def main():
    print(f"Fetching catalog from {CATALOG_URL} ...")
    req = urllib.request.Request(CATALOG_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_text = resp.read().decode("utf-8")
    # strict=False tolerates embedded control chars in the raw feed
    raw_data = json.loads(raw_text, strict=False)

    print(f"  Downloaded {len(raw_data)} raw entries")

    # Filter: only items with status=ok
    ok_items = [item for item in raw_data if item.get("status") == "ok"]
    print(f"  {len(ok_items)} items with status=ok")

    # Convert to our schema
    catalog = []
    seen_urls = set()
    for item in ok_items:
        converted = convert_item(item)
        # Deduplicate by URL
        if converted["url"] and converted["url"] not in seen_urls:
            seen_urls.add(converted["url"])
            catalog.append(converted)

    # Sort alphabetically by name for readability
    catalog.sort(key=lambda x: x["name"].lower())

    print(f"  Final catalog: {len(catalog)} unique items")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Written to {OUTPUT_PATH}")

    # Print type distribution
    from collections import Counter
    counts = Counter(item["test_type"] for item in catalog)
    print("\nTest type distribution:")
    labels = {"A": "Ability & Aptitude", "B": "Biodata/SJT", "C": "Competencies",
              "D": "Development & 360", "E": "Exercises", "K": "Knowledge & Skills",
              "P": "Personality", "S": "Simulations"}
    for code in sorted(counts):
        print(f"  {code} ({labels.get(code, '?')}): {counts[code]}")


if __name__ == "__main__":
    main()
