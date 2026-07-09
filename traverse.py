import sys
import heapq
import itertools
import math
import networkx as nx

from s2_client import search_paper, get_references
from models import Paper

NODE_BUDGET = 15
MAX_DEPTH = 4
MAX_REFS_PER_NODE = 6


def relevance_score(ref_entry: dict, depth: int) -> float:
    cited = ref_entry.get("citedPaper") or {}
    influential_bonus = 1.5 if ref_entry.get("isInfluential") else 1.0
    citation_count = cited.get("citationCount") or 0
    citation_component = math.log1p(citation_count)
    depth_decay = 0.8 ** depth
    return influential_bonus * (1 + citation_component) * depth_decay


def traverse(seed_title: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited_ids = set()

    seed_raw = search_paper(seed_title)
    seed_paper = Paper.from_s2_dict(seed_raw, depth=0)
    graph.add_node(seed_paper.paper_id, paper=seed_paper)

    counter = itertools.count()
    heap = [(-1.0, next(counter), seed_paper.paper_id, seed_raw, 0, False, None)]

    while heap and len(visited_ids) < NODE_BUDGET:
        neg_priority, _, paper_id, paper_raw, depth, is_influential_ref, parent_id = heapq.heappop(heap)

        if paper_id in visited_ids:
            continue
        visited_ids.add(paper_id)

        paper_obj = Paper.from_s2_dict(paper_raw, depth=depth, is_influential_ref=is_influential_ref)
        graph.add_node(paper_id, paper=paper_obj)
        if parent_id is not None:
            graph.add_edge(parent_id, paper_id, is_influential=is_influential_ref)

        print(f"[visited depth={depth}] {paper_obj.title[:70]}")

        if depth >= MAX_DEPTH:
            continue

        try:
            refs = get_references(paper_id, limit=100)
        except Exception as e:
            print(f"  [warn] failed to fetch references for {paper_id}: {e}")
            continue

        scored = []
        for ref in refs:
            cited = ref.get("citedPaper") or {}
            cid = cited.get("paperId")
            if not cid or cid in visited_ids:
                continue
            score = relevance_score(ref, depth + 1)
            scored.append((score, ref, cited))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_refs = scored[:MAX_REFS_PER_NODE]

        for score, ref, cited in top_refs:
            heapq.heappush(
                heap,
                (-score, next(counter), cited["paperId"], cited,
                 depth + 1, bool(ref.get("isInfluential")), paper_id)
            )

    return graph


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python traverse.py "paper title"')
        sys.exit(1)

    title = sys.argv[1]
    g = traverse(title)

    print(f"\nDone. {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

    from collections import Counter
    depths = Counter(g.nodes[n]["paper"].depth for n in g.nodes)
    print("Depth distribution:", dict(sorted(depths.items())))