"""Hand-written ReAct loop (research plan §4.3).

The trajectory log is the dataset of the paper, so every step is recorded with
its thought, action, observation, parse status, injected perturbation and token
counts. Episodes of a shard are advanced in lock-step so the backend can batch
all active trajectories into one generation call per step.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from agent_eval.configs import RunConfig
from agent_eval.llm import Backend, Generation, GenRequest
from agent_eval.parsing import parse_tool_call
from agent_eval.prompts import (
    LAST_STEP_NUDGE,
    PARSE_ERROR_FEEDBACK,
    REACT_SYSTEM,
    REACT_USER,
    TOOL_FORMAT_FALLBACK,
)
from agent_eval.retrieval import BM25Index
from agent_eval.tools import TOOL_SCHEMAS, ToolEnv

SCHEMA_VERSION = 1


def derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@dataclass
class Episode:
    question: dict[str, Any]
    config: RunConfig
    seed: int
    env: ToolEnv
    messages: list[dict[str, Any]]
    steps: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    terminated_by: str | None = None
    final_answer: str = ""
    cited_doc_ids: list[str] = field(default_factory=list)
    consecutive_parse_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thought_tokens: int = 0
    wall_ms: float = 0.0

    def to_trajectory(self, backend_name: str) -> dict[str, Any]:
        c = self.config
        return {
            "schema_version": SCHEMA_VERSION,
            "question_id": self.question["question_id"],
            "config_id": c.config_id,
            "model": c.model,
            "scaffold": c.scaffold,
            "thinking": c.thinking,
            "perturb_p": c.perturb_p,
            "seed": self.seed,
            "question": self.question["question"],
            "final_answer": self.final_answer,
            "cited_doc_ids": self.cited_doc_ids,
            "terminated_by": self.terminated_by,
            "n_steps": len(self.steps),
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_thought_tokens": self.thought_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "wall_ms": round(self.wall_ms, 1),
            "backend": backend_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": self.steps,
        }


def initial_messages(question: str, config: RunConfig, native_tools: bool) -> list[dict[str, Any]]:
    system = REACT_SYSTEM.format(max_steps=config.max_steps)
    if not native_tools:
        system += TOOL_FORMAT_FALLBACK
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": REACT_USER.format(question=question)},
    ]


def _request(ep: Episode, native_tools: bool) -> GenRequest:
    c = ep.config
    return GenRequest(
        messages=ep.messages,
        sampling=c.sampling,
        enable_thinking=c.thinking,
        thinking_budget=c.thinking_budget if c.thinking else None,
        tools=TOOL_SCHEMAS if native_tools else None,
        seed=derive_seed(c.config_id, ep.seed, ep.question["question_id"], len(ep.steps)),
        stop=["</tool_call>"],
    )


def _advance(ep: Episode, g: Generation) -> None:
    c = ep.config
    step_no = len(ep.steps) + 1
    ep.prompt_tokens += g.prompt_tokens
    ep.completion_tokens += g.completion_tokens
    ep.thought_tokens += g.thought_tokens
    ep.wall_ms += g.latency_ms
    record: dict[str, Any] = {
        "step": step_no,
        "thought": g.thought,
        "thought_tokens": g.thought_tokens,
        "content": g.visible,
        "action": None,
        "observation": "",
        "parse_ok": False,
        "parse_error": None,
        "perturbed": None,
        "tool_ok": None,
        "doc_ids_returned": None,
        "doc_id_read": None,
        "latency_ms": round(g.latency_ms, 1),
        "prompt_tokens": g.prompt_tokens,
        "completion_tokens": g.completion_tokens,
        "finish_reason": g.finish_reason,
        "thinking_budget_exhausted": g.budget_exhausted,
    }
    if g.finish_reason == "context_overflow":
        record["parse_error"] = "context_overflow"
        ep.steps.append(record)
        ep.done, ep.terminated_by = True, "token_cap"
        return

    parsed = parse_tool_call(g.visible)
    ep.messages.append({"role": "assistant", "content": g.visible})
    if parsed.ok:
        ep.consecutive_parse_failures = 0
        record.update(parse_ok=True, action={"tool": parsed.tool, "args": parsed.args})
        if parsed.n_tool_calls > 1:
            record["parse_error"] = f"{parsed.n_tool_calls} tool calls; executed the first"
        if parsed.tool == "finish":
            ep.final_answer = parsed.args.get("answer", "")
            ep.cited_doc_ids = parsed.args.get("cited_doc_ids", [])
            record["tool_ok"] = True
            ep.steps.append(record)
            ep.done, ep.terminated_by = True, "finish"
            return
        res = ep.env.call(parsed.tool, parsed.args)
        record.update(
            observation=res.observation,
            tool_ok=res.ok,
            perturbed=res.perturbed,
            doc_ids_returned=res.doc_ids_returned,
            doc_id_read=res.doc_id_read,
        )
        feedback_role, feedback = "tool", res.observation
    else:
        ep.consecutive_parse_failures += 1
        record.update(parse_error=parsed.error, observation=PARSE_ERROR_FEEDBACK)
        feedback_role, feedback = "user", PARSE_ERROR_FEEDBACK
    ep.steps.append(record)

    if not parsed.ok and ep.consecutive_parse_failures >= c.max_consecutive_parse_failures:
        ep.done, ep.terminated_by = True, "parse_failure"
    elif ep.completion_tokens >= c.token_cap:
        ep.done, ep.terminated_by = True, "token_cap"
    elif step_no >= c.max_steps:
        ep.done, ep.terminated_by = True, "max_steps"
    else:
        if step_no == c.max_steps - 1:
            feedback = f"{feedback}\n\n{LAST_STEP_NUDGE}"
        ep.messages.append({"role": feedback_role, "content": feedback})


def run_react(
    questions: list[dict[str, Any]],
    config: RunConfig,
    seed: int,
    backend: Backend,
    docs: list[dict[str, Any]],
    index: BM25Index,
) -> list[dict[str, Any]]:
    native = backend.supports_native_tools
    doc_map = {d["doc_id"]: d for d in docs}
    episodes = []
    for q in questions:
        env = ToolEnv(
            doc_map,
            index,
            top_k=config.top_k_search,
            perturb_p=config.perturb_p,
            perturb_seed=derive_seed("perturb", config.config_id, seed, q["question_id"]),
        )
        episodes.append(Episode(q, config, seed, env, initial_messages(q["question"], config, native)))
    active = list(episodes)
    while active:
        gens = backend.generate([_request(ep, native) for ep in active])
        for ep, g in zip(active, gens):
            _advance(ep, g)
        active = [ep for ep in active if not ep.done]
    return [ep.to_trajectory(backend.name) for ep in episodes]
