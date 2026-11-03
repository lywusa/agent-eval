"""Build the per-trajectory score table that every analysis reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agent_eval.io import iter_jsonl, read_jsonl
from agent_eval.judge import traj_key
from agent_eval.scoring.outcome import exact_match, f1_score
from agent_eval.scoring.trajectory import trajectory_metrics


def load_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    trajs = []
    for p in sorted(Path(runs_dir).glob("*/seed*.jsonl")):
        trajs.extend(iter_jsonl(p))
    # Deduplicate by key (a resumed shard may contain a repeated chunk); keep the first.
    seen, out = set(), []
    for t in trajs:
        k = traj_key(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def cell_label(correct: bool, grounded: bool) -> str:
    return ("RA" if correct else "WA") + "-" + ("GP" if grounded else "BP")


def _load_judge(judge_dir: Path, judge_key: str, name: str) -> dict[str, dict[str, Any]]:
    p = judge_dir / judge_key / f"{name}.jsonl"
    return {r["key"]: r for r in iter_jsonl(p)} if p.exists() else {}


def build_score_table(
    runs_dir: str | Path,
    questions_path: str | Path,
    judge_dir: str | Path | None = None,
    judge_key: str = "qwen3-14b-awq",
    correct_def: str = "auto",
) -> pd.DataFrame:
    """correct_def: 'auto' (LLM judge where available, else EM), 'em', 'judge', or 'f1>=X'."""
    qs = {q["question_id"]: q for q in read_jsonl(questions_path)}
    trajs = [t for t in load_runs(runs_dir) if t["question_id"] in qs]
    jdir = Path(judge_dir) if judge_dir else None
    corr = _load_judge(jdir, judge_key, "correctness") if jdir else {}
    steps = _load_judge(jdir, judge_key, "steps") if jdir else {}
    rows = []
    for t in trajs:
        q = qs[t["question_id"]]
        k = traj_key(t)
        pred = t.get("final_answer") or ""
        row: dict[str, Any] = {
            "key": k,
            "question_id": t["question_id"],
            "config_id": t["config_id"],
            "model": t["model"],
            "scaffold": t["scaffold"],
            "thinking": bool(t["thinking"]),
            "perturb_p": t.get("perturb_p", 0.0),
            "seed": t["seed"],
            "qtype": q.get("type"),
            "final_answer": pred,
            "gold_answer": q["answer"],
            "terminated_by": t["terminated_by"],
            "n_steps": t["n_steps"],
            "total_tokens": t["total_tokens"],
            "completion_tokens": t["total_completion_tokens"],
            "thought_tokens": t.get("total_thought_tokens", 0),
            "wall_ms": t.get("wall_ms"),
            "em": exact_match(pred, q["answer"]),
            "f1": f1_score(pred, q["answer"]),
        }
        row.update(trajectory_metrics(t, q["gold_doc_ids"]))
        row["first_query"] = next(
            (
                str(s["action"]["args"].get("query", ""))
                for s in t["steps"]
                if s.get("action") and s["action"]["tool"] in ("search", "retrieve")
            ),
            None,
        )
        jc = corr.get(k, {}).get("judge_correct")
        row["judge_correct"] = jc
        if correct_def == "em":
            row["correct"] = bool(row["em"])
        elif correct_def == "judge":
            row["correct"] = bool(jc)
        elif correct_def.startswith("f1>="):
            row["correct"] = row["f1"] >= float(correct_def[4:])
        else:
            row["correct"] = bool(jc) if jc is not None else bool(row["em"])
        row["correct_source"] = "judge" if (correct_def in ("auto", "judge") and jc is not None) else correct_def
        s = steps.get(k)
        row["llm_holistic"] = s.get("holistic") if s else None
        row["llm_harmful_fraction"] = s.get("harmful_fraction") if s else None
        row["llm_first_error_step"] = s.get("first_error_step") if s else None
        row["llm_judge_tokens"] = (s["prompt_tokens"] + s["completion_tokens"]) if s else None
        row["cell"] = cell_label(row["correct"], row["grounded"])
        rows.append(row)
    return pd.DataFrame(rows)
