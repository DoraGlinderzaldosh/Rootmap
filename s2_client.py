import os
import json
import time
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()
S2_API_KEY = os.getenv("S2_API_KEY")
 
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
 
# Proactive throttling: enforce a minimum gap between outgoing requests
# so we stop triggering 429s in the first place, instead of only reacting after.
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests; lower this once you confirm your key is active
_last_request_time = [0.0]  # list so it's mutable inside the function below
 
 
def _throttle():
    elapsed = time.time() - _last_request_time[0]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time[0] = time.time()
 
PAPER_FIELDS = "title,abstract,year,authors,venue,externalIds,influentialCitationCount,citationCount,referenceCount,openAccessPdf"
REF_FIELDS = "title,abstract,year,venue,externalIds,influentialCitationCount,citationCount,isInfluential"
 
 
def _cache_path(key: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"
 
 
def _cached_get(url: str, params: dict, max_retries: int = 5) -> dict:
    cache_key = url + json.dumps(params, sort_keys=True)
    cpath = _cache_path(cache_key)
 
    if cpath.exists():
        return json.loads(cpath.read_text())
 
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
 
    for attempt in range(max_retries):
        _throttle()
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            wait = min(2 ** attempt, 30)
            print(f"[s2_client] 429 rate limited, retrying in {wait}s "
                  f"(attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        cpath.write_text(json.dumps(data))
        return data
 
    raise RuntimeError(f"Gave up after {max_retries} retries: {url}")
 
 
def search_paper(title: str) -> dict:
    """Resolve a title to a single best-match paper with metadata."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "fields": PAPER_FIELDS, "limit": 1}
    data = _cached_get(url, params)
    if not data.get("data"):
        raise ValueError(f"No paper found for title: {title}")
    return data["data"][0]
 
 
def get_paper_by_id(paper_id: str) -> dict:
    """Fetch metadata for a paper directly by its S2 paperId."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": PAPER_FIELDS}
    return _cached_get(url, params)
 
 
def get_references(paper_id: str, limit: int = 100) -> list[dict]:
    """Fetch the list of papers this paper cites, with influence flags."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params = {"fields": REF_FIELDS, "limit": limit}
    data = _cached_get(url, params)
    return data.get("data") or []
 