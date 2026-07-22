"""
models.py — shared data structures for the citation graph project.
"""
from dataclasses import dataclass


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    venue: str | None
    citation_count: int
    influential_citation_count: int
    depth: int
    is_influential_ref: bool = False
    summary: dict | None = None

    @classmethod
    def from_s2_dict(cls, d: dict, depth: int, is_influential_ref: bool = False) -> "Paper":
        return cls(
            paper_id=d.get("paperId"),
            title=d.get("title") or "UNKNOWN TITLE",
            abstract=d.get("abstract"),
            year=d.get("year"),
            venue=d.get("venue"),
            citation_count=d.get("citationCount") or 0,
            influential_citation_count=d.get("influentialCitationCount") or 0,
            depth=depth,
            is_influential_ref=is_influential_ref,
        )