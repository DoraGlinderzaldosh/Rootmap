"""
reasoning_agent.py — A ReAct-style tool-using agent over Llama 3.1 8B Instruct
(via HF's free serverless Inference API), reasoning over a crawled citation graph.

Since the free serverless tier doesn't reliably support native function-calling
for this model, we implement ReAct manually: the model is instructed to reply
in a strict text format (Thought / Action / Action Input, or Final Answer),
we parse that format ourselves, execute the requested tool, and feed the
result back as the next message — looping until it gives a Final Answer.
"""
import os
import re
import json
import time
import networkx as nx
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import embeddings
import vector_store

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "prism-ml/Ternary-Bonsai-27B-gguf"
_client = InferenceClient(model=MODEL, token=HF_TOKEN) if HF_TOKEN else None

TOOL_DESCRIPTIONS = """You have access to these tools:

get_node_summary(paper_id) — title, year, venue, and structured summary for one paper.
get_neighbors(paper_id, direction) — direction is "cites" (papers this one references) or "cited_by" (papers that reference this one), within this graph only.
shortest_path(paper_id_a, paper_id_b) — the citation chain connecting two papers, ignoring direction.
semantic_search(query, top_k) — find papers in this graph most related to a natural-language query. top_k is optional, default 5.

To use a tool, respond in EXACTLY this format and nothing else:
Thought: <your reasoning about what to look up next>
Action: <tool name>
Action Input: <a single-line JSON object of arguments, e.g. {"paper_id": "abc123"}>

When you have enough information to answer, respond in EXACTLY this format instead:
Thought: <your final reasoning>
Final Answer: <your complete answer, citing paper TITLES not raw IDs, in plain readable prose>

Only ever output one Thought/Action pair or one Thought/Final Answer per turn. Never invent paper_ids — only use ones you've seen from tool results or the graph overview below."""


def _node_brief(graph: nx.DiGraph, paper_id: str) -> dict:
    if paper_id not in graph.nodes:
        return {"error": f"paper_id {paper_id} not found in this graph"}
    paper = graph.nodes[paper_id]["paper"]
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue,
        "depth": paper.depth,
        "summary": paper.summary,
    }


def _execute_tool(graph: nx.DiGraph, all_paper_ids: list[str], name: str, tool_input: dict) -> dict:
    try:
        if name == "get_node_summary":
            return _node_brief(graph, tool_input["paper_id"])

        if name == "get_neighbors":
            pid = tool_input["paper_id"]
            if pid not in graph.nodes:
                return {"error": f"paper_id {pid} not found"}
            direction = tool_input.get("direction", "cites")
            neighbor_ids = list(graph.successors(pid)) if direction == "cites" else list(graph.predecessors(pid))
            return {"neighbors": [_node_brief(graph, nid) for nid in neighbor_ids]}

        if name == "shortest_path":
            a, b = tool_input["paper_id_a"], tool_input["paper_id_b"]
            if a not in graph.nodes or b not in graph.nodes:
                return {"error": "one or both paper_ids not found in this graph"}
            undirected = graph.to_undirected()
            try:
                path = nx.shortest_path(undirected, a, b)
                return {"path": [_node_brief(graph, pid) for pid in path]}
            except nx.NetworkXNoPath:
                return {"error": "no path exists between these papers in this graph"}

        if name == "semantic_search":
            query = tool_input["query"]
            top_k = tool_input.get("top_k", 5)
            query_vec = embeddings.embed_texts([query])[0]
            results = vector_store.search_similar(query_vec, top_k=top_k, paper_ids=all_paper_ids)
            return {
                "results": [
                    {"paper_id": r.payload.get("paper_id"), "score": round(r.score, 3),
                     "text_snippet": r.payload.get("text")}
                    for r in results
                ]
            }

        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": f"tool execution failed: {e}"}


def _parse_response(text: str) -> dict:
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return {"type": "final", "answer": final_match.group(1).strip()}

    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*\})", text, re.DOTALL)

    if action_match and input_match:
        try:
            tool_input = json.loads(input_match.group(1).strip())
        except json.JSONDecodeError:
            tool_input = {}
        return {"type": "action", "tool": action_match.group(1).strip(), "input": tool_input}

    return {"type": "final", "answer": text.strip()}


def _call_model(messages: list[dict], max_retries: int = 4) -> str:
    if _client is None:
        raise RuntimeError("HF_TOKEN not set — add it to your .env file.")

    for attempt in range(max_retries):
        try:
            response = _client.chat_completion(messages=messages, max_tokens=500, temperature=0.2)
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "loading" in msg:
                wait = min(2 ** attempt, 20)
                print(f"[reasoning_agent] HF API busy, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gave up after repeated HF API failures.")


def ask_agent(graph: nx.DiGraph, question: str, max_turns: int = 6) -> str:
    all_paper_ids = list(graph.nodes)
    seed_id = next(pid for pid in graph.nodes if graph.nodes[pid]["paper"].depth == 0)
    seed_title = graph.nodes[seed_id]["paper"].title

    system_prompt = (
        f"You are analyzing a citation graph rooted at the paper '{seed_title}' "
        f"(paper_id: {seed_id}), containing {len(all_paper_ids)} papers total.\n\n"
        f"{TOOL_DESCRIPTIONS}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for _ in range(max_turns):
        raw = _call_model(messages)
        parsed = _parse_response(raw)

        if parsed["type"] == "final":
            return parsed["answer"]

        messages.append({"role": "assistant", "content": raw})

        result = _execute_tool(graph, all_paper_ids, parsed["tool"], parsed["input"])
        messages.append({
            "role": "user",
            "content": f"Observation: {json.dumps(result)}\n\nContinue with the next Thought/Action, or give your Final Answer if you have enough information."
        })

    return "I wasn't able to reach a final answer within the tool-call budget — try a narrower question."