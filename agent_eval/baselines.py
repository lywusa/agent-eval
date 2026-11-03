"""Closed-book (contamination control) and single-shot RAG (simple baseline).

Both are logged in the same trajectory schema as ReAct so every scorer applies.
Single-shot RAG records one synthetic `retrieve` step whose returned documents
count as read, followed by a `finish` step.
"""

from __future__ import annotations

import time
from typing import Any

from agent_eval.agent import SCHEMA_VERSION, derive_seed
from agent_eval.configs import RunConfig
from agent_eval.llm import Backend, GenRequest
from agent_eval.parsing import parse_answer_block
from agent_eval.prompts import CLOSED_BOOK_SYSTEM, SINGLE_RAG_SYSTEM, SINGLE_RAG_USER
from agent_eval.retrieval import BM25Index
from agent_eval.tools import READ_MAX_CHARS


def _base_record(q: dict[str, Any], config: RunConfig, seed: int, backend: Backend) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": q["question_id"],
        "config_id": config.config_id,
        "model": config.model,
        "scaffold": config.scaffold,
        "thinking": False,
        "perturb_p": 0.0,
        "seed": seed,
        "question": q["question"],
        "backend": backend.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_baseline(
    questions: list[dict[str, Any]],
    config: RunConfig,
    seed: int,
    backend: Backend,
    docs: list[dict[str, Any]] | None = None,
    index: BM25Index | None = None,
) -> list[dict[str, Any]]:
    doc_map = {d["doc_id"]: d for d in docs or []}
    requests, retrieved = [], []
    for q in questions:
        if config.scaffold == "closed_book":
            msgs = [
                {"role": "system", "content": CLOSED_BOOK_SYSTEM},
                {"role": "user", "content": f"Question: {q['question']}"},
            ]
            hits: list[str] = []
        elif config.scaffold == "single_rag":
            assert index is not None, "single_rag needs the BM25 index"
            hits = [h for h, _ in index.search(q["question"], k=config.top_k_search)]
            block = "\n\n".join(
                f"[{h}] {doc_map[h]['title']}\n{doc_map[h]['text'][:READ_MAX_CHARS]}" for h in hits
            )
            msgs = [
                {"role": "system", "content": SINGLE_RAG_SYSTEM},
                {"role": "user", "content": SINGLE_RAG_USER.format(documents=block or "(none)", question=q["question"])},
            ]
        else:
            raise ValueError(f"not a baseline scaffold: {config.scaffold}")
        retrieved.append(hits)
        requests.append(
            GenRequest(
                messages=msgs,
                sampling=config.sampling,
                enable_thinking=False,
                seed=derive_seed(config.config_id, seed, q["question_id"], 0),
            )
        )
    gens = backend.generate(requests)
    out = []
    for q, hits, g in zip(questions, retrieved, gens):
        answer, cited, ok = parse_answer_block(g.visible)
        if config.scaffold == "closed_book":
            cited = []
        steps = []
        if config.scaffold == "single_rag":
            steps.append(
                {
                    "step": 1,
                    "thought": "",
                    "thought_tokens": 0,
                    "content": "",
                    "action": {"tool": "retrieve", "args": {"query": q["question"]}},
                    "observation": " ".join(f"[{h}]" for h in hits),
                    "parse_ok": True,
                    "parse_error": None,
                    "perturbed": None,
                    "tool_ok": True,
                    "doc_ids_returned": hits,
                    "doc_id_read": None,
                    "docs_read": hits,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "finish_reason": "stop",
                    "thinking_budget_exhausted": False,
                }
            )
        steps.append(
            {
                "step": len(steps) + 1,
                "thought": g.thought,
                "thought_tokens": g.thought_tokens,
                "content": g.visible,
                "action": {"tool": "finish", "args": {"answer": answer, "cited_doc_ids": cited}},
                "observation": "",
                "parse_ok": ok,
                "parse_error": None if ok else "no 'Answer:' line",
                "perturbed": None,
                "tool_ok": True,
                "doc_ids_returned": None,
                "doc_id_read": None,
                "latency_ms": round(g.latency_ms, 1),
                "prompt_tokens": g.prompt_tokens,
                "completion_tokens": g.completion_tokens,
                "finish_reason": g.finish_reason,
                "thinking_budget_exhausted": False,
            }
        )
        rec = _base_record(q, config, seed, backend)
        rec.update(
            final_answer=answer,
            cited_doc_ids=cited,
            terminated_by="finish" if g.finish_reason != "context_overflow" else "token_cap",
            n_steps=len(steps),
            total_prompt_tokens=g.prompt_tokens,
            total_completion_tokens=g.completion_tokens,
            total_thought_tokens=0,
            total_tokens=g.prompt_tokens + g.completion_tokens,
            wall_ms=round(g.latency_ms, 1),
            steps=steps,
        )
        out.append(rec)
    return out
