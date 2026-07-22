"""
summarize.py — Structured paper summaries via Llama 3.1 8B Instruct,
served through Hugging Face's free serverless Inference API.
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

CACHE_DIR = Path(".summary_cache")
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_client = InferenceClient(model=MODEL, token=HF_TOKEN) if HF_TOKEN else None

SYSTEM_PROMPT = """You summarize academic paper abstracts into structured, concise summaries.
Respond ONLY with a JSON object, no other text, no markdown code fences, matching exactly this shape:
{"problem": "...", "method": "...", "key_result": "...", "one_liner": "..."}

- problem: one sentence, what problem does this paper address
- method: one sentence, what approach/method does it use
- key_result: one sentence, the main finding or contribution
- one_liner: a single plain-English sentence summarizing the whole paper

Keep every field under 25 words. If the abstract is thin or missing, do your best from the title alone."""


def _cache_path(paper_id: str) -> Path:
    h = hashlib.sha256(paper_id.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _extract_json(text: str) -> dict | None:
    """Llama sometimes wraps JSON in prose or fences despite instructions; pull out the first {...} block."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def summarize_paper(paper_id: str, title: str, abstract: str | None, max_retries: int = 4) -> dict:
    cpath = _cache_path(paper_id)
    if cpath.exists():
        return json.loads(cpath.read_text())

    if _client is None:
        raise RuntimeError("HF_TOKEN not set — add it to your .env file.")

    user_content = f"Title: {title}\n\nAbstract: {abstract or '(no abstract available)'}"

    for attempt in range(max_retries):
        try:
            response = _client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            raw = response.choices[0].message.content
            result = _extract_json(raw)
            if result is None:
                result = {"problem": "", "method": "", "key_result": "", "one_liner": raw[:200]}
            cpath.write_text(json.dumps(result))
            return result
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "loading" in msg:
                wait = min(2 ** attempt, 20)
                print(f"[summarize] HF API busy, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise

    fallback = {"problem": "", "method": "", "key_result": "", "one_liner": "(summary unavailable)"}
    cpath.write_text(json.dumps(fallback))
    return fallback


def summarize_papers_batch(papers: list[dict]) -> dict[str, dict]:
    results = {}
    for p in papers:
        try:
            results[p["paper_id"]] = summarize_paper(p["paper_id"], p["title"], p.get("abstract"))
        except Exception as e:
            print(f"  [warn] summarization failed for {p['paper_id']}: {e}")
            results[p["paper_id"]] = {
                "problem": "", "method": "", "key_result": "",
                "one_liner": "(summary unavailable)"
            }
    return results