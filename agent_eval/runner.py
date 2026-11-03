"""Run one (config, seed) shard with checkpointing and resume (research plan §6).

Questions are processed in chunks (default 50). After each chunk the
trajectories are appended and fsynced to ``runs/<config>/seed<k>.jsonl`` and an
optional sync command (e.g. ``kaggle datasets version ...``) is executed, so a
killed 12-hour session never loses more than one chunk. Re-running the same
command skips questions already present in the output file.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_eval.agent import run_react
from agent_eval.baselines import run_baseline
from agent_eval.configs import RunConfig, get_config, get_model
from agent_eval.data import fingerprint, load_corpus, load_questions
from agent_eval.io import append_jsonl, iter_jsonl, write_json
from agent_eval.llm import Backend, make_backend
from agent_eval.retrieval import load_or_build_index


def shard_path(runs_dir: str | Path, config_id: str, seed: int) -> Path:
    return Path(runs_dir) / config_id / f"seed{seed}.jsonl"


def completed_ids(path: Path) -> set[str]:
    return {t["question_id"] for t in iter_jsonl(path)}


def run_shard(
    config: RunConfig,
    seed: int,
    questions: list[dict[str, Any]],
    data_dir: str | Path,
    runs_dir: str | Path,
    backend: Backend,
    chunk_size: int = 50,
    sync_cmd: str | None = None,
    max_hours: float | None = None,
) -> Path:
    out = shard_path(runs_dir, config.config_id, seed)
    done = completed_ids(out)
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"[run] {config.config_id} seed={seed}: {len(done)} done, {len(todo)} to go -> {out}")
    meta = {
        "config": config.to_dict(),
        "model_hf_id": get_model(config.model).hf_id,
        "seed": seed,
        "backend": backend.name,
        "questions_fingerprint": fingerprint([q["question_id"] for q in questions]),
        "n_questions": len(questions),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out.with_suffix(".meta.json"), meta)
    if not todo:
        return out

    docs = load_corpus(data_dir)
    index = load_or_build_index(data_dir, docs) if config.scaffold != "closed_book" else None
    t_start = time.time()
    for i in range(0, len(todo), chunk_size):
        if max_hours is not None and (time.time() - t_start) / 3600 > max_hours:
            print(f"[run] stopping after {max_hours} h; re-run the same command to resume")
            break
        chunk = todo[i : i + chunk_size]
        t0 = time.time()
        if config.scaffold == "react":
            trajs = run_react(chunk, config, seed, backend, docs, index)
        else:
            trajs = run_baseline(chunk, config, seed, backend, docs, index)
        append_jsonl(out, trajs)
        toks = sum(t["total_completion_tokens"] for t in trajs)
        dt = time.time() - t0
        print(
            f"[run] chunk {i // chunk_size + 1}: {len(trajs)} trajectories, "
            f"{toks} completion tokens, {dt:.0f}s ({toks / max(dt, 1e-6):.0f} tok/s)"
        )
        if sync_cmd:
            subprocess.run(shlex.split(sync_cmd), check=False)
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["n_completed"] = len(completed_ids(out))
    write_json(out.with_suffix(".meta.json"), meta)
    return out


def run_many(
    config_ids: list[str],
    seeds: list[int],
    questions_path: str | Path,
    data_dir: str | Path,
    runs_dir: str | Path,
    backend_kind: str = "vllm",
    limit: int | None = None,
    chunk_size: int = 50,
    sync_cmd: str | None = None,
    max_hours: float | None = None,
    **backend_kwargs: Any,
) -> list[Path]:
    """Run several configs that share one model, loading the model only once."""
    configs = [get_config(c) for c in config_ids]
    models = {c.model for c in configs}
    if len(models) != 1:
        raise ValueError(f"run_many loads one model per process; got {sorted(models)}")
    spec = get_model(models.pop())
    questions = load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    backend = make_backend(backend_kind, spec, seed=0, **backend_kwargs)
    t_start = time.time()
    paths = []
    for c in configs:
        for s in seeds:
            remaining = None if max_hours is None else max_hours - (time.time() - t_start) / 3600
            if remaining is not None and remaining <= 0:
                print("[run] time budget exhausted before starting", c.config_id, s)
                continue
            paths.append(
                run_shard(c, s, questions, data_dir, runs_dir, backend, chunk_size, sync_cmd, remaining)
            )
    return paths
