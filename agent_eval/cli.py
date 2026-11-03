"""Command-line entry point: ``python -m agent_eval <command>`` (or ``agent-eval``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_eval.configs import DEFAULT_JUDGE, GRID, SECOND_JUDGE, get_model


def _csv(s: str | None) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


# --------------------------------------------------------------------------- data


def cmd_data(a: argparse.Namespace) -> None:
    from agent_eval.data import build_all, download_hotpot_dev
    from agent_eval.retrieval import BM25Index
    from agent_eval.data import load_corpus

    src = a.hotpot
    if a.download or not src:
        src = download_hotpot_dev(Path(a.out) / "raw")
    manifest = build_all(src, a.out, n_eval=a.n_eval, n_pilot=a.n_pilot, seed=a.seed, bridge_fraction=a.bridge_frac)
    print(json.dumps(manifest, indent=2))
    docs = load_corpus(a.out)
    BM25Index().fit(docs).save(Path(a.out) / "bm25.pkl")
    print(f"[data] BM25 index over {len(docs)} docs -> {Path(a.out) / 'bm25.pkl'}")


def cmd_configs(a: argparse.Namespace) -> None:
    for cid, c in GRID.items():
        if a.json:
            print(json.dumps(c.to_dict()))
        else:
            spec = get_model(c.model)
            print(f"{cid:34s} model={spec.hf_id:24s} scaffold={c.scaffold:12s} thinking={c.thinking!s:5s} perturb_p={c.perturb_p}")


# ---------------------------------------------------------------------------- run


def _backend_kwargs(a: argparse.Namespace) -> dict:
    kw = {}
    if getattr(a, "enforce_eager", False):
        kw["enforce_eager"] = True
    return kw


def cmd_run(a: argparse.Namespace) -> None:
    from agent_eval.runner import run_many

    run_many(
        _csv(a.configs),
        [int(s) for s in _csv(a.seeds)],
        a.questions,
        a.data,
        a.runs,
        backend_kind=a.backend,
        limit=a.limit,
        chunk_size=a.chunk,
        sync_cmd=a.sync_cmd,
        max_hours=a.max_hours,
        **(_backend_kwargs(a) if a.backend == "vllm" else {}),
    )


def cmd_bench(a: argparse.Namespace) -> None:
    from agent_eval.bench import measure, project_gpu_hours

    rows = measure(a.model, a.backend, tuple(int(x) for x in _csv(a.batch_sizes)), a.max_tokens, a.thinking)
    best = max(r["tok_per_s"] for r in rows)
    print(json.dumps({"best_tok_per_s": best, **project_gpu_hours(best)}, indent=2))


# -------------------------------------------------------------------------- judge


def cmd_judge(a: argparse.Namespace) -> None:
    from agent_eval import judge as J
    from agent_eval.io import append_jsonl, iter_jsonl, read_jsonl
    from agent_eval.llm import make_backend
    from agent_eval.scoring.table import load_runs

    trajs = load_runs(a.runs)
    if a.configs:
        keep = set(_csv(a.configs))
        trajs = [t for t in trajs if t["config_id"] in keep]
    out = Path(a.out) / a.judge
    out_file = out / f"{a.kind}.jsonl"
    key_field = "key_a" if a.kind == "pairwise" else "key"
    done = {r[key_field] for r in iter_jsonl(out_file)}
    backend = make_backend(a.backend, get_model(a.judge), role="judge", **(_backend_kwargs(a) if a.backend == "vllm" else {}))

    if a.kind == "correctness":
        gold = {q["question_id"]: q["answer"] for q in read_jsonl(a.questions)}
        todo = [t for t in trajs if J.traj_key(t) not in done and t["question_id"] in gold]
        for i in range(0, len(todo), 1024):
            append_jsonl(out_file, J.judge_correctness(todo[i : i + 1024], gold, backend, a.judge))
    elif a.kind in ("steps", "verbosity"):
        react = [t for t in trajs if t["scaffold"] == "react"]
        if a.kind == "verbosity" or a.subsample:
            react = J.bias_subsample(react, n=a.subsample or 200, seed=a.seed)
        todo = [t for t in react if J.traj_key(t) not in done]
        pad = a.pad if a.kind == "verbosity" else 0
        for i in range(0, len(todo), 256):
            append_jsonl(out_file, J.judge_steps(todo[i : i + 256], backend, a.judge, pad_filler=pad, tag=a.kind))
    elif a.kind == "pairwise":
        main = [t for t in trajs if t["scaffold"] == "react" and t.get("perturb_p", 0) == 0]
        pairs = [p for p in J.make_pairs(main, a.n_pairs, seed=a.seed) if J.traj_key(p[0]) not in done]
        append_jsonl(out_file, J.judge_pairwise(pairs, backend, a.judge))
    print(f"[judge] {a.kind} -> {out_file} ({sum(1 for _ in iter_jsonl(out_file))} rows)")


# -------------------------------------------------------------------------- score


def cmd_score(a: argparse.Namespace) -> None:
    from agent_eval.scoring.table import build_score_table

    df = build_score_table(a.runs, a.questions, a.judge_dir, a.judge, a.correct_def)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summary = df.groupby("config_id").agg(n=("key", "size"), acc=("correct", "mean"), em=("em", "mean"), grounded=("grounded", "mean"), tokens=("total_tokens", "mean"))
    print(summary.to_string())
    print(f"[score] {len(df)} trajectories -> {a.out}")


def _read_scores(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    for c in ("correct", "grounded", "thinking", "grounded_strict", "all_gold_read"):
        if c in df:
            df[c] = df[c].astype(bool)
    df["judge_correct"] = df["judge_correct"].map(lambda v: None if pd.isna(v) else str(v) in ("True", "1", "1.0"))
    df["recovered"] = df["recovered"].map(lambda v: None if pd.isna(v) else str(v) in ("True", "1", "1.0"))
    return df


# ----------------------------------------------------------------------- annotate


def cmd_annotate(a: argparse.Namespace) -> None:
    from agent_eval import annotation as A
    from agent_eval.scoring.table import load_runs

    if a.action == "sample":
        scores = _read_scores(a.scores)
        sample = A.stratified_sample(scores, n=a.n, seed=a.seed)
        idx = A.trajs_index(load_runs(a.runs))
        bdir = A.export_batch(sample["key"].tolist(), idx, a.out, a.batch or "main", seed=a.seed)
        sample.to_csv(bdir / "sample_strata_PRIVATE.csv", index=False)
        print(sample.groupby(["config_id", "cell"]).size().unstack(fill_value=0).to_string())
        print(f"[annotate] exported {len(sample)} items -> {bdir} (do not open key.csv or sample_strata_PRIVATE.csv)")
    elif a.action in ("retest", "external"):
        idx = A.trajs_index(load_runs(a.runs))
        n = a.n or (30 if a.action == "retest" else 40)
        bdir = A.derived_batch(a.main, idx, a.out, a.batch or a.action, n=n, seed=a.seed, require_labelled=a.action == "retest")
        print(f"[annotate] exported {n} items -> {bdir}")
    elif a.action == "label":
        A.label_batch(a.batch_dir, a.annotator)
    elif a.action == "import":
        n = A.import_sheet(a.batch_dir, a.sheet, a.annotator)
        print(f"[annotate] imported {n} rows")
    elif a.action == "status":
        if not Path(a.out).is_dir():
            print(f"[annotate] no annotation directory at {a.out} yet")
            return
        h = A.load_human_labels(a.out)
        for bdir in sorted(Path(a.out).iterdir()):
            if (bdir / "items.jsonl").exists():
                n_items = sum(1 for _ in open(bdir / "items.jsonl"))
                lab = h[h["batch"] == bdir.name].groupby("annotator").size().to_dict() if len(h) else {}
                print(f"{bdir.name:12s} items={n_items:4d} labelled={lab}")


# ------------------------------------------------------------------------ analyze


def cmd_analyze(a: argparse.Namespace) -> None:
    from agent_eval.analysis import run_all
    from agent_eval.annotation import load_human_labels
    from agent_eval.io import read_jsonl
    from agent_eval.scoring.table import load_runs

    scores = _read_scores(a.scores)
    human = load_human_labels(a.annotation) if a.annotation and Path(a.annotation).exists() else None
    trajs = gold = None
    if a.runs and a.questions:
        trajs = load_runs(a.runs)[:2000]
        gold = {q["question_id"]: q["gold_doc_ids"] for q in read_jsonl(a.questions)}
    report = run_all(
        scores, a.out, human=human, judge_dir=a.judge_dir, judge_key=a.judge,
        second_judge_key=a.second_judge, trajs=trajs, gold=gold, n_boot=a.n_boot, cost_budget=a.cost_budget,
    )
    print(f"[analyze] report -> {report}")


def cmd_synth(a: argparse.Namespace) -> None:
    from agent_eval.synthetic import make_fake

    paths = make_fake(a.out, n_questions=a.n_questions, seeds=a.seeds, seed=a.seed)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def cmd_smoke(a: argparse.Namespace) -> None:
    from agent_eval.smoke import run_smoke

    print(f"[smoke] report -> {run_smoke(a.out)}")


# ------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build the frozen corpus, BM25 index and question samples")
    d.add_argument("--hotpot", help="path to hotpot_dev_distractor_v1.json (or HF-format .jsonl)")
    d.add_argument("--download", action="store_true", help="download the official dev distractor file")
    d.add_argument("--out", default="data")
    d.add_argument("--n-eval", type=int, default=400)
    d.add_argument("--n-pilot", type=int, default=50)
    d.add_argument("--seed", type=int, default=2026)
    d.add_argument("--bridge-frac", type=float, default=0.8)
    d.set_defaults(fn=cmd_data)

    c = sub.add_parser("configs", help="list the frozen experiment grid")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_configs)

    r = sub.add_parser("run", help="run (config, seed) shards that share one model")
    r.add_argument("--configs", required=True, help="comma-separated config ids (same model)")
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--questions", default="data/questions_eval.jsonl")
    r.add_argument("--data", default="data")
    r.add_argument("--runs", default="runs")
    r.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "mock"])
    r.add_argument("--limit", type=int)
    r.add_argument("--chunk", type=int, default=50)
    r.add_argument("--sync-cmd", help="shell command run after every chunk (e.g. kaggle datasets version ...)")
    r.add_argument("--max-hours", type=float, help="stop starting new chunks after this many hours")
    r.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs (saves memory on T4)")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench", help="measure generation throughput")
    b.add_argument("--model", default="qwen3-4b")
    b.add_argument("--backend", default="vllm")
    b.add_argument("--batch-sizes", default="1,16,64")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--thinking", action="store_true")
    b.set_defaults(fn=cmd_bench)

    j = sub.add_parser("judge", help="run an LLM judge over trajectories")
    j.add_argument("kind", choices=["correctness", "steps", "pairwise", "verbosity"])
    j.add_argument("--judge", default=DEFAULT_JUDGE)
    j.add_argument("--runs", default="runs")
    j.add_argument("--questions", default="data/questions_eval.jsonl")
    j.add_argument("--out", default="judge")
    j.add_argument("--backend", default="vllm", choices=["vllm", "transformers", "mock"])
    j.add_argument("--configs", help="restrict to these config ids")
    j.add_argument("--subsample", type=int, help="judge only a seeded subsample of N ReAct trajectories")
    j.add_argument("--n-pairs", type=int, default=200)
    j.add_argument("--pad", type=int, default=3, help="filler repetitions per thought (verbosity test)")
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--enforce-eager", action="store_true")
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("score", help="build the per-trajectory score table (Family A + B + B′)")
    s.add_argument("--runs", default="runs")
    s.add_argument("--questions", default="data/questions_eval.jsonl")
    s.add_argument("--judge-dir", default="judge")
    s.add_argument("--judge", default=DEFAULT_JUDGE)
    s.add_argument("--correct-def", default="auto", help="auto | em | judge | f1>=0.8")
    s.add_argument("--out", default="results/scores.csv")
    s.set_defaults(fn=cmd_score)

    an = sub.add_parser("annotate", help="blinded human annotation (Family C)")
    an.add_argument("action", choices=["sample", "retest", "external", "label", "import", "status"])
    an.add_argument("--scores", default="results/scores.csv")
    an.add_argument("--runs", default="runs")
    an.add_argument("--out", default="annotation")
    an.add_argument("--batch", help="batch name (default: main / retest / external)")
    an.add_argument("--main", default="annotation/main", help="main batch directory (retest/external)")
    an.add_argument("--batch-dir", help="batch directory (label/import)")
    an.add_argument("--annotator", default="student")
    an.add_argument("--sheet", help="filled sheet.csv (import)")
    an.add_argument("--n", type=int, default=None)
    an.add_argument("--seed", type=int, default=2026)
    an.set_defaults(fn=cmd_annotate)

    z = sub.add_parser("analyze", help="RQ1-RQ5 tables, figures and report.md")
    z.add_argument("--scores", default="results/scores.csv")
    z.add_argument("--annotation", default="annotation")
    z.add_argument("--judge-dir", default="judge")
    z.add_argument("--judge", default=DEFAULT_JUDGE)
    z.add_argument("--second-judge", default=SECOND_JUDGE)
    z.add_argument("--runs", default="runs", help="used only to time deterministic metrics (RQ3 cost)")
    z.add_argument("--questions", default="data/questions_eval.jsonl")
    z.add_argument("--out", default="results")
    z.add_argument("--n-boot", type=int, default=1000)
    z.add_argument("--cost-budget", type=float, help="matched-cost budget in total tokens (default: ReAct median)")
    z.set_defaults(fn=cmd_analyze)

    y = sub.add_parser("synth", help="generate fake runs/judges/labels to dry-run the analysis")
    y.add_argument("--out", default="fake")
    y.add_argument("--n-questions", type=int, default=60)
    y.add_argument("--seeds", type=int, default=3)
    y.add_argument("--seed", type=int, default=0)
    y.set_defaults(fn=cmd_synth)

    m = sub.add_parser("smoke", help="CPU end-to-end run of the real agent loop with the mock backend")
    m.add_argument("--out", default="smoke")
    m.set_defaults(fn=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
