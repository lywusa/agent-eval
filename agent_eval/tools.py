"""Deterministic, logged, mockable tools (research plan §4.2).

`search(query)`, `read(doc_id)`, `finish(answer, cited_doc_ids)`, plus the RQ5
perturbation wrapper that turns a call into an error with probability p.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from agent_eval.retrieval import BM25Index

SNIPPET_CHARS = 200
READ_MAX_CHARS = 2000
INJECTED_ERROR = "ERROR: the tool service is temporarily unavailable (HTTP 503). No result was returned."

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the Wikipedia-paragraph corpus with a keyword query. "
            "Returns the top 5 results as doc_id, title and a short snippet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the full paragraph of one document returned by search.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string", "description": "A doc_id such as d01234."}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Give the final answer (a short phrase, not a sentence) and the doc_ids "
            "of the documents that support it. Ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Short final answer."},
                    "cited_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "doc_ids of supporting documents you have read.",
                    },
                },
                "required": ["answer", "cited_doc_ids"],
            },
        },
    },
]

TOOL_NAMES = {"search", "read", "finish"}


@dataclass
class ToolResult:
    observation: str
    ok: bool
    perturbed: str | None = None
    doc_ids_returned: list[str] | None = None  # search hits
    doc_id_read: str | None = None


class ToolEnv:
    def __init__(
        self,
        docs: list[dict[str, Any]] | dict[str, dict[str, Any]],
        index: BM25Index,
        top_k: int = 5,
        perturb_p: float = 0.0,
        perturb_seed: int = 0,
    ):
        self.docs = docs if isinstance(docs, dict) else {d["doc_id"]: d for d in docs}
        self.index = index
        self.top_k = top_k
        self.perturb_p = perturb_p
        self.rng = random.Random(perturb_seed)

    def call(self, tool: str, args: dict[str, Any]) -> ToolResult:
        if tool == "finish":
            return ToolResult(observation="", ok=True)
        # Draw for every non-finish call, so the injection pattern for a given
        # (question, seed) depends only on the call index.
        if self.perturb_p > 0 and self.rng.random() < self.perturb_p:
            return ToolResult(observation=INJECTED_ERROR, ok=False, perturbed="error")
        if tool == "search":
            return self.search(str(args.get("query", "")))
        if tool == "read":
            return self.read(str(args.get("doc_id", "")))
        return ToolResult(observation=f"ERROR: unknown tool {tool!r}. Use search, read or finish.", ok=False)

    def search(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult(observation="ERROR: empty query.", ok=False)
        hits = self.index.search(query, k=self.top_k)
        if not hits:
            return ToolResult(observation=f"Search results for {query!r}:\nNo results.", ok=True, doc_ids_returned=[])
        lines = []
        for doc_id, _ in hits:
            d = self.docs[doc_id]
            snippet = d["text"][:SNIPPET_CHARS].replace("\n", " ")
            if len(d["text"]) > SNIPPET_CHARS:
                snippet += "..."
            lines.append(f"[{doc_id}] {d['title']}: {snippet}")
        return ToolResult(
            observation=f"Search results for {query!r}:\n" + "\n".join(lines),
            ok=True,
            doc_ids_returned=[h[0] for h in hits],
        )

    def read(self, doc_id: str) -> ToolResult:
        doc_id = doc_id.strip().strip("[]")
        d = self.docs.get(doc_id)
        if d is None:
            return ToolResult(observation=f"ERROR: no document with doc_id {doc_id!r}.", ok=False)
        return ToolResult(
            observation=f"Document [{doc_id}] {d['title']}\n{d['text'][:READ_MAX_CHARS]}",
            ok=True,
            doc_id_read=doc_id,
        )
