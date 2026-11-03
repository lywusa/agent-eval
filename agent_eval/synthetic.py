"""Fake data for the week-3 pre-registration milestone ("analysis code skeleton
with fake data") and for tests. Produces every artifact the real pipeline does
(questions, corpus, run JSONL, judge outputs, annotation batches with labels), so
``score`` -> ``analyze`` can be exercised end to end before any GPU time is spent.

The generative model is arbitrary; the numbers it produces mean nothing.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from agent_eval.annotation import export_batch, derived_batch, stratified_sample, trajs_index
from agent_eval.configs import GRID, MAIN_CONFIGS
from agent_eval.io import write_jsonl
from agent_eval.judge import traj_key
from agent_eval.scoring.table import build_score_table

SKILL = {"qwen3-1.7b": 0.35, "qwen3-4b": 0.55, "qwen3-8b": 0.68}


def _step(n: int, tool: str, args: dict, obs: str = "", **kw: Any) -> dict[str, Any]:
    base = {
        "step": n, "thought": "", "thought_tokens": 0, "content": "", "action": {"tool": tool, "args": args},
        "observation": obs, "parse_ok": True, "parse_error": None, "perturbed": None, "tool_ok": True,
        "doc_ids_returned": None, "doc_id_read": None, "latency_ms": 100.0, "prompt_tokens": 800,
        "completion_tokens": 120, "finish_reason": "stop", "thinking_budget_exhausted": False,
    }
    base.update(kw)
    return base


def make_fake(out: str | Path, n_questions: int = 60, seeds: int = 3, seed: int = 0, with_perturb: bool = True) -> dict[str, Path]:
    out = Path(out)
    rng = random.Random(seed)
    data_dir, runs_dir, judge_dir, ann_dir = out / "data", out / "runs", out / "judge", out / "annotation"
    docs = [{"doc_id": f"d{i:05d}", "title": f"Title {i}", "text": f"Paragraph {i} text."} for i in range(n_questions * 4)]
    qs = []
    for i in range(n_questions):
        qs.append({
            "question_id": f"q{i:04d}", "question": f"Fake question {i}?", "answer": f"answer {i}",
            "type": "bridge" if i % 5 else "comparison", "level": "hard",
            "gold_titles": [f"Title {2 * i}", f"Title {2 * i + 1}"], "gold_doc_ids": [f"d{2 * i:05d}", f"d{2 * i + 1:05d}"],
            "supporting_facts": [],
        })
    write_jsonl(data_dir / "corpus.jsonl", docs)
    write_jsonl(data_dir / "questions_eval.jsonl", qs)
    known = {q["question_id"]: rng.random() < 0.35 for q in qs}  # "in pretraining data"

    configs = list(MAIN_CONFIGS) + ([c for c in GRID if c.endswith("-perturb")] if with_perturb else [])
    all_trajs = []
    for cid in configs:
        cfg = GRID[cid]
        skill = SKILL[cfg.model] + (0.05 if cfg.thinking else 0.0)
        for s in range(seeds):
            rows = []
            for q in qs:
                gold = q["gold_doc_ids"]
                distract = rng.choice(docs)["doc_id"]
                steps, cited, read = [], [], []
                n = 0
                if cfg.scaffold == "react":
                    for g in gold:
                        n += 1
                        steps.append(_step(n, "search", {"query": f"{q['question']} {g}"}, "Search results", doc_ids_returned=[g, distract]))
                        if cfg.perturb_p and rng.random() < cfg.perturb_p:
                            n += 1
                            steps.append(_step(n, "read", {"doc_id": g}, "ERROR", tool_ok=False, perturbed="error"))
                        if rng.random() < skill + 0.2:
                            n += 1
                            steps.append(_step(n, "read", {"doc_id": g}, "Document", doc_id_read=g))
                            read.append(g)
                    if rng.random() < 0.3:
                        n += 1
                        steps.append(_step(n, "search", {"query": f"{q['question']} {gold[0]}"}, "Search results", doc_ids_returned=[distract]))
                    if rng.random() < 0.1 * (1 - skill):
                        n += 1
                        steps.append(_step(n, "read", {}, "", parse_ok=False, action=None, parse_error="no tool call found"))
                    cited = read if rng.random() < 0.8 else read + [distract]
                elif cfg.scaffold == "single_rag":
                    hit = [g for g in gold if rng.random() < 0.6] + [distract]
                    n += 1
                    steps.append(_step(n, "retrieve", {"query": q["question"]}, "", doc_ids_returned=hit, docs_read=hit))
                    read, cited = hit, [h for h in hit if rng.random() < 0.7]
                p = 0.15 + 0.6 * skill * (len(set(read) & set(gold)) / 2) + (0.35 if known[q["question_id"]] else 0.0)
                if cfg.scaffold == "closed_book":
                    p = 0.1 + (0.55 if known[q["question_id"]] else 0.0)
                correct = rng.random() < min(p, 0.97)
                ans = q["answer"] if correct else f"wrong {rng.randint(0, 9)}"
                n += 1
                steps.append(_step(n, "finish", {"answer": ans, "cited_doc_ids": cited}))
                think = 900 if cfg.thinking else 0
                for st in steps:
                    st["thought_tokens"] = think
                    st["completion_tokens"] += think
                tot_c = sum(st["completion_tokens"] for st in steps)
                tot_p = sum(st["prompt_tokens"] for st in steps)
                rows.append({
                    "schema_version": 1, "question_id": q["question_id"], "config_id": cid, "model": cfg.model,
                    "scaffold": cfg.scaffold, "thinking": cfg.thinking, "perturb_p": cfg.perturb_p, "seed": s,
                    "question": q["question"], "final_answer": ans, "cited_doc_ids": cited, "terminated_by": "finish",
                    "n_steps": len(steps), "total_prompt_tokens": tot_p, "total_completion_tokens": tot_c,
                    "total_thought_tokens": think * len(steps), "total_tokens": tot_p + tot_c, "wall_ms": 100.0 * len(steps),
                    "backend": "synthetic", "created_at": "1970-01-01T00:00:00Z", "steps": steps,
                })
            write_jsonl(runs_dir / cid / f"seed{s}.jsonl", rows)
            all_trajs += rows

    # Judges.
    gold_ans = {q["question_id"]: q["answer"] for q in qs}
    for jk, noise in (("qwen3-14b-awq", 0.6), ("llama3.1-8b", 0.9)):
        corr, steps_out, verb, pw = [], [], [], []
        for t in all_trajs:
            k = traj_key(t)
            right = t["final_answer"] == gold_ans[t["question_id"]]
            if jk == "qwen3-14b-awq":
                corr.append({"key": k, "judge": jk, "judge_correct": right if rng.random() > 0.03 else not right,
                             "raw": "", "prompt_tokens": 120, "completion_tokens": 2})
            if t["scaffold"] != "react":
                continue
            n_read = sum(1 for s in t["steps"] if s.get("doc_id_read"))
            base = 1 + n_read * 1.3 + (1 if right else 0)
            hol = max(1, min(5, round(base + rng.gauss(0.3 if jk.startswith("qwen") else 0, noise))))
            labels = [1 if (s.get("doc_id_read") or s["action"] and s["action"]["tool"] == "finish") else (-1 if not s["parse_ok"] else 0) for s in t["steps"]]
            steps_out.append({"key": k, "judge": jk, "kind": "steps", "pad_filler": 0, "step_labels": labels, "holistic": hol,
                              "rationale": "", "parse_ok": True, "first_error_step": None, "harmful_fraction": labels.count(-1) / len(labels),
                              "advancing_fraction": labels.count(1) / len(labels), "prompt_tokens": 2500, "completion_tokens": 200})
            if jk == "qwen3-14b-awq" and rng.random() < 0.2:
                verb.append({**steps_out[-1], "kind": "verbosity", "pad_filler": 3, "holistic": min(5, hol + (rng.random() < 0.3))})
        if jk == "qwen3-14b-awq":
            react = [t for t in all_trajs if t["scaffold"] == "react" and t["perturb_p"] == 0]
            for _ in range(100):
                a, b = rng.sample(react, 2)
                v = rng.choice(["A", "B"])
                pw.append({"key_a": traj_key(a), "key_b": traj_key(b), "judge": jk, "verdict_original": v,
                           "verdict_swapped": v if rng.random() < 0.7 else ("B" if v == "A" else "A"),
                           "consistent": None, "prompt_tokens": 5000, "completion_tokens": 2})
            for r in pw:
                r["consistent"] = r["verdict_original"] == r["verdict_swapped"]
            write_jsonl(judge_dir / jk / "correctness.jsonl", corr)
            write_jsonl(judge_dir / jk / "verbosity.jsonl", verb)
            write_jsonl(judge_dir / jk / "pairwise.jsonl", pw)
        write_jsonl(judge_dir / jk / "steps.jsonl", steps_out)

    # Human labels via the real annotation export path.
    scores = build_score_table(runs_dir, data_dir / "questions_eval.jsonl", judge_dir)
    sample = stratified_sample(scores, n=min(150, len(scores) // 4), seed=seed)
    idx = trajs_index(all_trajs)
    bmain = export_batch(sample["key"].tolist(), idx, ann_dir, "main", seed=seed)
    by_key = scores.set_index("key")

    def fake_labels(bdir: Path, annotator: str, noise: float) -> None:
        key = {r["anon_id"]: r["key"] for r in csv.DictReader((bdir / "key.csv").open())}
        rows = []
        for a, k in key.items():
            r = by_key.loc[k]
            t = idx[k]
            h = 1 + 2 * (r["gold_evidence_recall"] if r["gold_evidence_recall"] == r["gold_evidence_recall"] else 0) + float(r["grounded"]) + 0.5 * float(r["correct"])
            labels = [1 if s.get("doc_id_read") else (0 if s["parse_ok"] else -1) for s in t["steps"]]
            labels = [x if rng.random() > noise / 3 else rng.choice([-1, 0, 1]) for x in labels]
            rows.append({"anon_id": a, "annotator": annotator, "step_labels": labels,
                         "holistic": max(1, min(5, round(h + rng.gauss(0, noise)))),
                         "error_category": "none" if r["grounded"] else rng.choice(["missed_evidence", "hallucination", "poor_query"]),
                         "notes": "", "seconds": 300 + rng.random() * 200, "labelled_at": "1970-01-01T00:00:00Z"})
        write_jsonl(bdir / f"labels_{annotator}.jsonl", rows)

    fake_labels(bmain, "student", 0.5)
    bret = derived_batch(bmain, idx, ann_dir, "retest", n=min(30, len(sample)), seed=seed + 1)
    fake_labels(bret, "student", 0.6)
    bext = derived_batch(bmain, idx, ann_dir, "external", n=min(40, len(sample)), seed=seed + 2)
    fake_labels(bext, "external1", 0.8)
    return {"data": data_dir, "runs": runs_dir, "judge": judge_dir, "annotation": ann_dir}
