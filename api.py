"""
api.py — FastAPI backend: crawl the citation graph, summarize it, and
answer questions about it via the tool-using reasoning agent.

Run: uvicorn api:app --reload --port 8000
"""
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from traverse import traverse
from summarize import summarize_papers_batch
from reasoning_agent import ask_agent

app = FastAPI(title="Citation Graph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory store: graph_id -> networkx graph.
# Fine for local single-user dev; swap for Redis/a DB for multi-user deployment.
GRAPH_STORE: dict[str, object] = {}


class CrawlRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    graph_id: str
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/crawl")
def crawl(req: CrawlRequest):
    try:
        graph = traverse(req.title)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {e}")

    paper_list = [
        {
            "paper_id": graph.nodes[n]["paper"].paper_id,
            "title": graph.nodes[n]["paper"].title,
            "abstract": graph.nodes[n]["paper"].abstract,
        }
        for n in graph.nodes
    ]
    summaries = summarize_papers_batch(paper_list)

    # Attach summaries onto the actual Paper objects so the reasoning agent's
    # tools (which read straight from the graph) can see them.
    for n in graph.nodes:
        paper = graph.nodes[n]["paper"]
        paper.summary = summaries.get(paper.paper_id)

    graph_id = str(uuid.uuid4())
    GRAPH_STORE[graph_id] = graph

    nodes = []
    for node_id in graph.nodes:
        paper = graph.nodes[node_id]["paper"]
        nodes.append({
            "id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "venue": paper.venue,
            "abstract": paper.abstract,
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "depth": paper.depth,
            "is_influential_ref": paper.is_influential_ref,
            "summary": paper.summary,
        })

    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "is_influential": data.get("is_influential", False),
        })

    return {"graph_id": graph_id, "nodes": nodes, "edges": edges}


@app.post("/api/chat")
def chat(req: ChatRequest):
    graph = GRAPH_STORE.get(req.graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Unknown graph_id — build a graph first.")

    try:
        answer = ask_agent(graph, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    return {"answer": answer}