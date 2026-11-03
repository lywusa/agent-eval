"""Family B: deterministic trajectory-aware metrics (research plan §5.2). No LLM."""

from __future__ import annotations

import math
from typing import Any

from agent_eval.retrieval import tokenize

REDUNDANT_JACCARD = 0.8
NON_FINISH_TOOLS = {"search", "read", "retrieve"}


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _docs_read_at(step: dict[str, Any]) -> list[str]:
    if step.get("docs_read"):
        return list(step["docs_read"])
    if step.get("doc_id_read"):
        return [step["doc_id_read"]]
    return []


def _tool(step: dict[str, Any]) -> str | None:
    a = step.get("action")
    return a["tool"] if a else None


def _is_sensible_followup(failed: dict[str, Any], nxt: dict[str, Any] | None) -> tuple[bool, bool]:
    """Return (sensible, alternative) for the action after an injected tool error.

    sensible: any valid non-finish tool call (retry or alternative).
    alternative: the research plan's stricter rule -- a search with a changed
    query, or a read of a different document.
    """
    if nxt is None or not nxt.get("parse_ok") or _tool(nxt) not in NON_FINISH_TOOLS:
        return False, False
    ft, nt = _tool(failed), _tool(nxt)
    fa, na = failed["action"]["args"], nxt["action"]["args"]
    if nt == "search":
        alt = ft != "search" or jaccard(str(fa.get("query", "")), str(na.get("query", ""))) < 1.0
    elif nt == "read":
        alt = ft != "read" or str(fa.get("doc_id")) != str(na.get("doc_id"))
    else:
        alt = False
    return True, alt


def trajectory_metrics(traj: dict[str, Any], gold_doc_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_doc_ids)
    steps = traj["steps"]
    read_order: list[tuple[int, str]] = []
    queries: list[str] = []
    redundant_queries = 0
    reread = 0
    seen_read: set[str] = set()
    seen_in_search: set[str] = set()
    n_tool_calls = 0
    for s in steps:
        t = _tool(s)
        if t in NON_FINISH_TOOLS and s.get("parse_ok"):
            n_tool_calls += 1
        if t == "search" and s.get("parse_ok"):
            q = str(s["action"]["args"].get("query", ""))
            if any(jaccard(q, prev) > REDUNDANT_JACCARD for prev in queries):
                redundant_queries += 1
            queries.append(q)
        if t == "read" and s.get("parse_ok"):
            d = str(s["action"]["args"].get("doc_id", ""))
            if d in seen_read:
                reread += 1
        for d in s.get("doc_ids_returned") or []:
            seen_in_search.add(d)
        for d in _docs_read_at(s):
            seen_read.add(d)
            read_order.append((s["step"], d))

    gold_hits = [st for st, d in read_order if d in gold]
    cited = set(traj.get("cited_doc_ids") or [])
    cited_gold = cited & gold
    n_parse_fail = sum(1 for s in steps if not s.get("parse_ok"))

    perturbed = [i for i, s in enumerate(steps) if s.get("perturbed")]
    sensible = alt = 0
    for i in perturbed:
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        a, b = _is_sensible_followup(steps[i], nxt)
        sensible += a
        alt += b
    finished = traj.get("terminated_by") == "finish"

    last_tool_step = max((s["step"] for s in steps if _tool(s) in NON_FINISH_TOOLS), default=0)
    gold_recall = len(gold & seen_read) / len(gold) if gold else math.nan
    cit_prec = len(cited_gold) / len(cited) if cited else math.nan
    cit_rec = len(cited_gold) / len(gold) if gold else math.nan
    cit_f1 = (
        2 * cit_prec * cit_rec / (cit_prec + cit_rec)
        if cited and gold and (cit_prec + cit_rec) > 0
        else 0.0
    )
    grounded = bool(cited) and cited <= gold and len(cited_gold) >= 1
    redundant = redundant_queries + reread
    m = {
        "n_tool_calls": n_tool_calls,
        "n_docs_read": len(seen_read),
        "gold_evidence_recall": gold_recall,
        "gold_seen_in_search": len(gold & (seen_in_search | seen_read)) / len(gold) if gold else math.nan,
        "all_gold_read": bool(gold) and gold <= seen_read,
        "n_cited": len(cited),
        "citation_precision": cit_prec,
        "citation_recall": cit_rec,
        "citation_f1": cit_f1,
        "grounded": grounded,
        "grounded_strict": grounded and cited <= seen_read,
        "cited_unread": len(cited - seen_read),
        "steps_to_first_gold": gold_hits[0] if gold_hits else math.nan,
        "steps_after_last_gold": (last_tool_step - gold_hits[-1]) if gold_hits else math.nan,
        "redundant_queries": redundant_queries,
        "rereads": reread,
        "redundant_calls": redundant,
        "redundant_rate": redundant / n_tool_calls if n_tool_calls else 0.0,
        "parse_failures": n_parse_fail,
        "parse_failure_rate": n_parse_fail / len(steps) if steps else 0.0,
        "n_perturbed": len(perturbed),
        "recovery_sensible": (sensible / len(perturbed)) if perturbed else math.nan,
        "recovery_alternative": (alt / len(perturbed)) if perturbed else math.nan,
        "recovered": (sensible == len(perturbed) and finished) if perturbed else None,
        "thinking_budget_exhausted": sum(1 for s in steps if s.get("thinking_budget_exhausted")),
    }
    # Pre-registered Family-B composite used for RQ2 rankings: equal-weight mean of
    # evidence recall, grounding, and non-redundancy.
    m["b_composite"] = (
        (0.0 if math.isnan(gold_recall) else gold_recall) + float(grounded) + (1.0 - min(1.0, m["redundant_rate"]))
    ) / 3.0
    return m
