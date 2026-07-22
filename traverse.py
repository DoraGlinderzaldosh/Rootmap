import sys
import heapq
import itertools
import math
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

from s2_client import search_paper, get_references
from models import Paper
from embeddings import embed_texts, cosine_similarity

NODE_BUDGET = 60
MAX_DEPTH = 4
MAX_REFS_PER_NODE = 6
BATCH_SIZE = 5

SEMANTIC_WEIGHT = 0.65


def heuristic_component(ref_entry: dict, depth: int) -> float:
    cited = ref_entry.get("citedPaper") or {}
    influential_bonus = 1.5 if ref_entry.get("isInfluential") else 1.0
    citation_count = cited.get("citationCount") or 0
    citation_component = math.log1p(citation_count)
    depth_decay = 0.8 ** depth
    return influential_bonus * (1 + citation_component) * depth_decay


def score_candidates(seed_vector, refs: list[dict], depth: int) -> list[tuple[float, dict, dict]]:
    candidates = []
    for ref in refs:
        cited = ref.get("citedPaper") or {}
        if cited.get("paperId"):
            candidates.append((ref, cited))

    if not candidates:
        return []

    abstracts = [cited.get("abstract") or cited.get("title") or "" for _, cited in candidates]
    candidate_ids = [cited.get("paperId") for _, cited in candidates]
    vectors = embed_texts(abstracts, paper_ids=candidate_ids)

    scored = []
    for (ref, cited), vec in zip(candidates, vectors):
        similarity = cosine_similarity(seed_vector, vec)
        similarity = max(similarity, 0.0)
        heuristic = heuristic_component(ref, depth)
        final_score = (SEMANTIC_WEIGHT * similarity * 10) + ((1 - SEMANTIC_WEIGHT) * heuristic)
        scored.append((final_score, ref, cited))

    return scored


def traverse(seed_title: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited_ids = set()

    seed_raw = search_paper(seed_title)
    seed_paper = Paper.from_s2_dict(seed_raw, depth=0)
    graph.add_node(seed_paper.paper_id, paper=seed_paper)

    seed_text = seed_paper.abstract or seed_paper.title
    seed_vector = embed_texts([seed_text], paper_ids=[seed_paper.paper_id])[0]

    counter = itertools.count()
    heap = [(-1.0, next(counter), seed_paper.paper_id, seed_raw, 0, False, None)]

    executor = ThreadPoolExecutor(max_workers=BATCH_SIZE)

    while heap and len(visited_ids) < NODE_BUDGET:
        batch = []
        while heap and len(batch) < BATCH_SIZE and len(visited_ids) + len(batch) < NODE_BUDGET:
            neg_priority, _, paper_id, paper_raw, depth, is_influential_ref, parent_id = heapq.heappop(heap)
            if paper_id in visited_ids or any(b[0] == paper_id for b in batch):
                continue
            batch.append((paper_id, paper_raw, depth, is_influential_ref, parent_id))

        if not batch:
            break

        for paper_id, paper_raw, depth, is_influential_ref, parent_id in batch:
            visited_ids.add(paper_id)
            paper_obj = Paper.from_s2_dict(paper_raw, depth=depth, is_influential_ref=is_influential_ref)
            graph.add_node(paper_id, paper=paper_obj)
            if parent_id is not None:
                graph.add_edge(parent_id, paper_id, is_influential=is_influential_ref)
            print(f"[visited depth={depth}] {paper_obj.title[:70]}")

        to_expand = [(pid, depth) for pid, _, depth, _, _ in batch if depth < MAX_DEPTH]
        if not to_expand:
            continue

        futures = {
            executor.submit(get_references, pid, 100): (pid, depth)
            for pid, depth in to_expand
        }

        for future in as_completed(futures):
            paper_id, depth = futures[future]
            try:
                refs = future.result()
            except Exception as e:
                print(f"  [warn] failed to fetch references for {paper_id}: {e}")
                continue

            candidates = [
                ref for ref in refs
                if (ref.get("citedPaper") or {}).get("paperId") not in visited_ids
            ]

            try:
                scored = score_candidates(seed_vector, candidates, depth + 1)
            except Exception as e:
                print(f"  [warn] embedding scoring failed for children of {paper_id}: {e}")
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            top_refs = scored[:MAX_REFS_PER_NODE]

            for score, ref, cited in top_refs:
                heapq.heappush(
                    heap,
                    (-score, next(counter), cited["paperId"], cited,
                     depth + 1, bool(ref.get("isInfluential")), paper_id)
                )

    executor.shutdown(wait=True)
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