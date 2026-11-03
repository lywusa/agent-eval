"""LLM judges (research plan §5.1, §5.3).

- correctness: reference-guided CORRECT/INCORRECT (Family A).
- steps: ternary step labels + 1-5 holistic score, reference-free (Family B′).
- pairwise: which of two trajectories on the same question has the better
  process, asked in both orders (position-bias test).
- verbosity: holistic re-score after padding thoughts with neutral filler.
- A second-family judge re-runs `steps` on the same subsample (self-preference).

Judges see the blinded rendering only (no config, model, seed or token counts).
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable

from agent_eval.configs import JUDGE_SAMPLING, SamplingConfig
from agent_eval.llm import Backend, GenRequest
from agent_eval.parsing import _first_json_object
from agent_eval.rubric import rubric_text

MAX_THOUGHT_CHARS = 800
MAX_OBS_CHARS = 700
FILLER = (
    " Let me restate the task carefully to make sure I am being thorough. "
    "I will proceed step by step and keep the goal in mind."
)


def traj_key(t: dict[str, Any]) -> str:
    return f"{t['config_id']}|{t['seed']}|{t['question_id']}"


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def render_trajectory(
    traj: dict[str, Any],
    include_thoughts: bool = True,
    pad_filler: int = 0,
    max_thought_chars: int = MAX_THOUGHT_CHARS,
    max_obs_chars: int = MAX_OBS_CHARS,
) -> str:
    """Blinded markdown rendering shared by LLM judges and human annotators."""
    out = [f"Question: {traj['question']}", ""]
    for s in traj["steps"]:
        out.append(f"### Step {s['step']}")
        thought = s.get("thought") or ""
        if pad_filler:
            thought = (thought + FILLER * pad_filler).strip()
        if include_thoughts and thought:
            out.append(f"Thought: {_clip(thought, max_thought_chars + len(FILLER) * pad_filler)}")
        a = s.get("action")
        if a is None:
            out.append(f"Action: (could not be parsed) {_clip(s.get('content', ''), 300)}")
        elif a["tool"] == "retrieve":
            out.append(f"Action: retrieve(query={a['args'].get('query', '')!r})  [single-shot retrieval]")
        else:
            out.append(f"Action: {a['tool']}({json.dumps(a['args'], ensure_ascii=False)})")
        if s.get("observation"):
            out.append(f"Observation: {_clip(s['observation'], max_obs_chars)}")
        out.append("")
    out.append(f"Final answer: {traj.get('final_answer') or '(none)'}")
    out.append(f"Cited documents: {', '.join(traj.get('cited_doc_ids') or []) or '(none)'}")
    out.append(f"Ended by: {traj.get('terminated_by')}")
    return "\n".join(out)


CORRECTNESS_PROMPT = """You are grading a short answer to a question against a gold answer.
Judge semantic equivalence: ignore case, punctuation, articles and harmless extra words; an answer that adds a wrong claim or names a different entity is INCORRECT.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Reply with exactly one word: CORRECT or INCORRECT."""

STEPS_PROMPT = """You are evaluating the research PROCESS of an AI agent that answers multi-hop questions by searching and reading Wikipedia paragraphs. You are not given the gold answer; judge whether each step is a sound research action and whether the final answer is supported by what the agent actually read.

{rubric}

Trajectory:
{trajectory}

Return JSON only, with one label per step (there are {n_steps} steps):
{{"step_labels": [{{"step": 1, "label": 1}}, ...], "holistic": <1-5>, "rationale": "<one sentence>"}}"""

PAIRWISE_PROMPT = """Two AI agents answered the same multi-hop question by searching and reading Wikipedia paragraphs. Which trajectory shows the better research PROCESS (relevant queries, reading the needed evidence, answer supported by what was read, no wasted steps)? Do not prefer a trajectory because it is longer.

=== Trajectory A ===
{a}

=== Trajectory B ===
{b}

Answer with exactly one of: A, B, TIE."""


def _req(content: str, sampling: SamplingConfig = JUDGE_SAMPLING) -> GenRequest:
    return GenRequest(messages=[{"role": "user", "content": content}], sampling=sampling, enable_thinking=False, seed=0)


def parse_correctness(text: str) -> bool | None:
    t = text.strip().upper()
    if re.search(r"\bINCORRECT\b", t):
        return False
    if re.search(r"\bCORRECT\b", t):
        return True
    return None


def parse_steps(text: str, n_steps: int) -> dict[str, Any]:
    obj = _first_json_object(text) or {}
    labels: list[int | None] = [None] * n_steps
    for item in obj.get("step_labels", []) or []:
        try:
            i, lab = int(item["step"]) - 1, int(item["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n_steps and lab in (-1, 0, 1):
            labels[i] = lab
    try:
        holistic = int(obj.get("holistic"))
        holistic = holistic if 1 <= holistic <= 5 else None
    except (TypeError, ValueError):
        holistic = None
    return {
        "step_labels": labels,
        "holistic": holistic,
        "rationale": str(obj.get("rationale", ""))[:500],
        "parse_ok": holistic is not None and all(lab is not None for lab in labels),
    }


def step_label_summary(labels: list[int | None]) -> dict[str, Any]:
    valid = [lab for lab in labels if lab is not None]
    first_err = next((i + 1 for i, lab in enumerate(labels) if lab == -1), None)
    return {
        "first_error_step": first_err,
        "harmful_fraction": (sum(1 for lab in valid if lab == -1) / len(valid)) if valid else None,
        "advancing_fraction": (sum(1 for lab in valid if lab == 1) / len(valid)) if valid else None,
    }


def _batched(items: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def judge_correctness(
    trajs: list[dict[str, Any]], gold: dict[str, str], backend: Backend, judge_key: str, batch: int = 256
) -> list[dict[str, Any]]:
    out = []
    for chunk in _batched(trajs, batch):
        reqs = [
            _req(
                CORRECTNESS_PROMPT.format(
                    question=t["question"], gold=gold[t["question_id"]], pred=t.get("final_answer") or "(no answer)"
                ),
                SamplingConfig(0.0, 1.0, -1, max_tokens=8),
            )
            for t in chunk
        ]
        for t, g in zip(chunk, backend.generate(reqs)):
            out.append(
                {
                    "key": traj_key(t),
                    "judge": judge_key,
                    "judge_correct": parse_correctness(g.visible),
                    "raw": g.visible[:50],
                    "prompt_tokens": g.prompt_tokens,
                    "completion_tokens": g.completion_tokens,
                }
            )
    return out


def judge_steps(
    trajs: list[dict[str, Any]],
    backend: Backend,
    judge_key: str,
    pad_filler: int = 0,
    batch: int = 64,
    tag: str = "steps",
) -> list[dict[str, Any]]:
    out = []
    for chunk in _batched(trajs, batch):
        reqs = [
            _req(
                STEPS_PROMPT.format(
                    rubric=rubric_text(),
                    trajectory=render_trajectory(t, pad_filler=pad_filler),
                    n_steps=len(t["steps"]),
                )
            )
            for t in chunk
        ]
        for t, g in zip(chunk, backend.generate(reqs)):
            parsed = parse_steps(g.visible, len(t["steps"]))
            out.append(
                {
                    "key": traj_key(t),
                    "judge": judge_key,
                    "kind": tag,
                    "pad_filler": pad_filler,
                    **parsed,
                    **step_label_summary(parsed["step_labels"]),
                    "prompt_tokens": g.prompt_tokens,
                    "completion_tokens": g.completion_tokens,
                }
            )
    return out


def make_pairs(trajs: list[dict[str, Any]], n_pairs: int, seed: int = 0) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pairs of trajectories on the same question and seed from two different configs."""
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for t in trajs:
        groups.setdefault((t["question_id"], t["seed"]), []).append(t)
    cands = [g for _, g in sorted(groups.items()) if len({t["config_id"] for t in g}) >= 2]
    rng.shuffle(cands)
    pairs = []
    for g in cands[:n_pairs]:
        a, b = rng.sample(sorted(g, key=traj_key), 2)
        while a["config_id"] == b["config_id"]:
            a, b = rng.sample(g, 2)
        pairs.append((a, b))
    return pairs


def parse_pairwise(text: str) -> str | None:
    t = text.strip().upper()
    m = re.search(r"\b(TIE|A|B)\b", t)
    return m.group(1) if m else None


def judge_pairwise(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], backend: Backend, judge_key: str, batch: int = 32
) -> list[dict[str, Any]]:
    out = []
    for chunk in _batched(pairs, batch):
        reqs = []
        for a, b in chunk:
            ra, rb = render_trajectory(a), render_trajectory(b)
            reqs.append(_req(PAIRWISE_PROMPT.format(a=ra, b=rb)))
            reqs.append(_req(PAIRWISE_PROMPT.format(a=rb, b=ra)))
        gens = backend.generate(reqs)
        for j, (a, b) in enumerate(chunk):
            g1, g2 = gens[2 * j], gens[2 * j + 1]
            first, swapped = parse_pairwise(g1.visible), parse_pairwise(g2.visible)
            # Map the swapped verdict back to the original labels.
            swapped_mapped = {"A": "B", "B": "A", "TIE": "TIE"}.get(swapped) if swapped else None
            out.append(
                {
                    "key_a": traj_key(a),
                    "key_b": traj_key(b),
                    "judge": judge_key,
                    "verdict_original": first,
                    "verdict_swapped": swapped_mapped,
                    "consistent": first is not None and first == swapped_mapped,
                    "prompt_tokens": g1.prompt_tokens + g2.prompt_tokens,
                    "completion_tokens": g1.completion_tokens + g2.completion_tokens,
                }
            )
    return out


def bias_subsample(trajs: list[dict[str, Any]], n: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    react = sorted([t for t in trajs if t["scaffold"] == "react" and t.get("perturb_p", 0) == 0], key=traj_key)
    return rng.sample(react, min(n, len(react)))
