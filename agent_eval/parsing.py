"""Parse policy outputs into (thought, visible content, tool call)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent_eval.tools import TOOL_NAMES

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_DOC_ID_RE = re.compile(r"\bd\d{3,}\b")


@dataclass
class ParsedAction:
    ok: bool
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_tool_calls: int = 0


def split_thinking(text: str) -> tuple[str, str]:
    """Return (thought, visible). Handles outputs with or without an opening <think>."""
    if "</think>" in text:
        thought, visible = text.split("</think>", 1)
        thought = thought.replace("<think>", "").strip()
        return thought, visible.strip()
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "").strip(), ""
    return "", text.strip()


def _first_json_object(s: str) -> dict[str, Any] | None:
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = s.find("{", start + 1)
    return None


def _normalize_args(tool: str, args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"query": args} if tool == "search" else {"doc_id": args} if tool == "read" else {"answer": args}
    if not isinstance(args, dict):
        args = {}
    if tool == "finish":
        cited = args.get("cited_doc_ids", args.get("citations", []))
        if isinstance(cited, str):
            cited = _DOC_ID_RE.findall(cited)
        args = {"answer": str(args.get("answer", "")).strip(), "cited_doc_ids": [str(c).strip("[] ") for c in cited]}
    return args


def parse_tool_call(visible: str) -> ParsedAction:
    blocks = _TOOL_CALL_RE.findall(visible)
    candidates = blocks if blocks else [visible]
    for cand in candidates:
        obj = _first_json_object(cand)
        if obj is None:
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        if isinstance(name, dict):
            obj, name = name, name.get("name")
        args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
        if name not in TOOL_NAMES:
            return ParsedAction(ok=False, error=f"unknown tool {name!r}", n_tool_calls=len(blocks))
        return ParsedAction(ok=True, tool=name, args=_normalize_args(name, args), n_tool_calls=max(1, len(blocks)))
    return ParsedAction(ok=False, error="no tool call found", n_tool_calls=len(blocks))


_ANSWER_RE = re.compile(r"^\s*\**answer\**\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CITE_RE = re.compile(r"^\s*\**citations?\**\s*[:：]\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_answer_block(visible: str) -> tuple[str, list[str], bool]:
    """Parse the `Answer: ...` / `Citations: ...` format of the baselines.
    Returns (answer, cited_doc_ids, parse_ok)."""
    m = _ANSWER_RE.search(visible)
    ok = m is not None
    answer = m.group(1).strip() if m else visible.strip().splitlines()[0].strip() if visible.strip() else ""
    c = _CITE_RE.search(visible)
    cited = _DOC_ID_RE.findall(c.group(1)) if c else []
    return answer, cited, ok
