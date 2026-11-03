"""CPU end-to-end smoke test: real corpus builder, BM25, tools, ReAct loop,
baselines, runner, judges, scorer, annotation export and analysis, driven by the
mock backend on a tiny HotpotQA-format dataset. Run this before spending GPU time."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from agent_eval.analysis import run_all
from agent_eval.annotation import export_batch, load_human_labels, stratified_sample, trajs_index
from agent_eval.configs import GRID, MAIN_CONFIGS
from agent_eval.data import build_all, load_questions
from agent_eval.io import append_jsonl, read_jsonl, write_jsonl
from agent_eval.judge import bias_subsample, judge_correctness, judge_pairwise, judge_steps, make_pairs
from agent_eval.llm import MockBackend, heuristic_judge
from agent_eval.runner import run_shard
from agent_eval.scoring.table import build_score_table, load_runs


def mini_hotpot(n: int = 24, seed: int = 0) -> list[dict[str, Any]]:
    """Bridge questions in the official HotpotQA JSON format."""
    rng = random.Random(seed)
    people = [f"Person {i}" for i in range(n)]
    cities = [f"Town {i}" for i in range(n)]
    countries = [f"Country {i % 7}" for i in range(n)]
    exs = []
    for i in range(n):
        gold = [
            [people[i], [f"{people[i]} is a writer.", f" {people[i]} was born in {cities[i]}."]],
            [cities[i], [f"{cities[i]} is a town.", f" {cities[i]} is located in {countries[i]}."]],
        ]
        others = rng.sample([j for j in range(n) if j != i], 4)
        distract = [[people[j], [f"{people[j]} is a writer.", f" {people[j]} was born in {cities[j]}."]] for j in others]
        ctx = gold + distract
        rng.shuffle(ctx)
        exs.append(
            {
                "_id": f"mini{i:03d}",
                "question": f"In which country is the town where {people[i]} was born?",
                "answer": countries[i],
                "type": "bridge" if i % 4 else "comparison",
                "level": "hard",
                "supporting_facts": [[people[i], 1], [cities[i], 1]],
                "context": ctx,
            }
        )
    return exs


def run_smoke(out: str | Path = "smoke", seeds: tuple[int, ...] = (0, 1, 2)) -> Path:
    out = Path(out)
    raw = out / "mini_hotpot.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(mini_hotpot()))
    data = out / "data"
    build_all(raw, data, n_eval=16, n_pilot=4, seed=1)
    questions = load_questions(data / "questions_eval.jsonl")
    runs = out / "runs"
    backend = MockBackend()
    for cid in list(MAIN_CONFIGS) + [c for c in GRID if c.endswith("-perturb")]:
        for s in seeds:
            run_shard(GRID[cid], s, questions, data, runs, backend, chunk_size=8)

    trajs = load_runs(runs)
    jb = MockBackend(heuristic_judge)
    jdir = out / "judge"
    gold = {q["question_id"]: q["answer"] for q in questions}
    for jk in ("qwen3-14b-awq", "llama3.1-8b"):
        react = [t for t in trajs if t["scaffold"] == "react"]
        if jk == "qwen3-14b-awq":
            write_jsonl(jdir / jk / "correctness.jsonl", judge_correctness(trajs, gold, jb, jk))
            write_jsonl(jdir / jk / "steps.jsonl", judge_steps(react, jb, jk))
            write_jsonl(jdir / jk / "verbosity.jsonl", judge_steps(bias_subsample(trajs, 20), jb, jk, pad_filler=3, tag="verbosity"))
            main = [t for t in react if t["perturb_p"] == 0]
            write_jsonl(jdir / jk / "pairwise.jsonl", judge_pairwise(make_pairs(main, 20), jb, jk))
        else:
            write_jsonl(jdir / jk / "steps.jsonl", judge_steps(bias_subsample(trajs, 20), jb, jk))

    scores = build_score_table(runs, data / "questions_eval.jsonl", jdir)
    sample = stratified_sample(scores, n=24, seed=0)
    ann = out / "annotation"
    if (ann / "main").exists():
        import shutil

        shutil.rmtree(ann)
    bdir = export_batch(sample["key"].tolist(), trajs_index(trajs), ann, "main", seed=0)
    # Stand-in "human" labels so the reliability and RQ3 code paths run.
    rng = random.Random(0)
    rows = []
    for it in read_jsonl(bdir / "items.jsonl"):
        rows.append({"anon_id": it["anon_id"], "annotator": "smoke", "step_labels": [rng.choice([-1, 0, 1]) for _ in range(it["n_steps"])],
                     "holistic": rng.randint(1, 5), "error_category": "none", "notes": "", "seconds": 1.0, "labelled_at": ""})
    append_jsonl(bdir / "labels_smoke.jsonl", rows)
    q_gold = {q["question_id"]: q["gold_doc_ids"] for q in questions}
    return run_all(scores, out / "results", human=load_human_labels(ann), judge_dir=jdir, trajs=trajs, gold=q_gold, n_boot=50)
