import os
import time
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import vector_store

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN) if HF_TOKEN else None


def _embed_one(text: str, max_retries: int = 5) -> list[float]:
    if _client is None:
        raise RuntimeError("HF_TOKEN not set — add it to your .env file.")

    for attempt in range(max_retries):
        try:
            vec = _client.feature_extraction(text)
            arr = np.array(vec)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            return arr.tolist()
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "loading" in msg:
                wait = min(2 ** attempt, 20)
                print(f"[embeddings] HF API busy/rate-limited, retrying in {wait}s "
                      f"(attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Gave up after {max_retries} retries embedding text via HF API.")


def embed_texts(texts: list[str], paper_ids: list[str] | None = None) -> list[list[float]]:
    results: list[list[float]] = []
    for i, text in enumerate(texts):
        text = text or ""
        cached = vector_store.get_cached(text)
        if cached is not None:
            results.append(cached)
        else:
            vec = _embed_one(text)
            payload = {"text": text[:500]}
            if paper_ids:
                payload["paper_id"] = paper_ids[i]
            vector_store.store(text, vec, payload)
            results.append(vec)
    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)