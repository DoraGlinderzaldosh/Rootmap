
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from traverse import traverse

app = FastAPI(title="Citation Graph API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrawlRequest(BaseModel):
    title: str


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
        })

    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "is_influential": data.get("is_influential", False),
        })

    return {"nodes": nodes, "edges": edges}