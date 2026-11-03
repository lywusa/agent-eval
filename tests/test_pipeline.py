import csv

from agent_eval.agent import run_react
from agent_eval.annotation import import_sheet
from agent_eval.cli import main
from agent_eval.configs import get_config
from agent_eval.data import build_all, load_corpus, load_questions
from agent_eval.io import read_jsonl
from agent_eval.llm import MockBackend
from agent_eval.retrieval import load_or_build_index
from agent_eval.runner import run_shard
from agent_eval.smoke import mini_hotpot, run_smoke
from agent_eval.synthetic import make_fake


def _data(tmp_path):
    import json

    raw = tmp_path / "mini.json"
    raw.write_text(json.dumps(mini_hotpot()))
    build_all(raw, tmp_path / "data", n_eval=12, n_pilot=4, seed=3)
    return tmp_path / "data"


def test_data_build_disjoint_pilot(tmp_path):
    d = _data(tmp_path)
    ev = {q["question_id"] for q in load_questions(d / "questions_eval.jsonl")}
    pi = {q["question_id"] for q in load_questions(d / "questions_pilot.jsonl")}
    assert len(ev) == 12 and len(pi) == 4 and not ev & pi
    q = load_questions(d / "questions_eval.jsonl")[0]
    assert len(q["gold_doc_ids"]) == 2


def test_react_loop_max_steps_and_parse_failure(tmp_path):
    d = _data(tmp_path)
    docs, qs = load_corpus(d), load_questions(d / "questions_eval.jsonl")[:3]
    idx = load_or_build_index(d, docs)
    cfg = get_config("react-qwen3-4b-nothink")
    loop = MockBackend(lambda r: '<tool_call>{"name": "search", "arguments": {"query": "Person"}}</tool_call>')
    trajs = run_react(qs, cfg, 0, loop, docs, idx)
    assert all(t["terminated_by"] == "max_steps" and t["n_steps"] == cfg.max_steps for t in trajs)
    garbage = MockBackend(lambda r: "I do not know how to call tools.")
    trajs = run_react(qs, cfg, 0, garbage, docs, idx)
    assert all(t["terminated_by"] == "parse_failure" and t["n_steps"] == cfg.max_consecutive_parse_failures for t in trajs)


def test_runner_resume(tmp_path):
    d = _data(tmp_path)
    qs = load_questions(d / "questions_eval.jsonl")
    cfg = get_config("react-qwen3-4b-nothink")
    out = run_shard(cfg, 0, qs[:5], d, tmp_path / "runs", MockBackend(), chunk_size=2)
    assert len(read_jsonl(out)) == 5
    run_shard(cfg, 0, qs, d, tmp_path / "runs", MockBackend(), chunk_size=4)
    rows = read_jsonl(out)
    assert len(rows) == len(qs) and len({r["question_id"] for r in rows}) == len(qs)


def test_smoke_end_to_end(tmp_path):
    report = run_smoke(tmp_path / "smoke", seeds=(0, 1))
    text = report.read_text()
    assert "## RQ1" in text and "## RQ5" in text


def test_synthetic_cli_and_sheet_import(tmp_path):
    paths = make_fake(tmp_path / "fake", n_questions=24, seeds=3)
    scores = tmp_path / "scores.csv"
    main(["score", "--runs", str(paths["runs"]), "--questions", str(paths["data"] / "questions_eval.jsonl"),
          "--judge-dir", str(paths["judge"]), "--out", str(scores)])
    main(["analyze", "--scores", str(scores), "--annotation", str(paths["annotation"]), "--judge-dir", str(paths["judge"]),
          "--runs", str(paths["runs"]), "--questions", str(paths["data"] / "questions_eval.jsonl"),
          "--out", str(tmp_path / "results"), "--n-boot", "50"])
    text = (tmp_path / "results" / "report.md").read_text()
    assert "intra_rater_step_kappa" in text and "H4" in text

    bdir = paths["annotation"] / "external"
    items = read_jsonl(bdir / "items.jsonl")
    sheet = tmp_path / "filled.csv"
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "n_steps", "step_labels", "holistic", "error_category", "notes"])
        for it in items:
            w.writerow([it["anon_id"], it["n_steps"], ",".join(["1"] * it["n_steps"]), 4, "none", ""])
        w.writerow([items[0]["anon_id"], 1, "1,1,1,1,1,1,1,1,1,1,1,1", 9, "none", "bad row"])
    assert import_sheet(bdir, sheet, "ext2") == len(items)
