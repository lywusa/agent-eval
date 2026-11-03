import math

import pytest

from agent_eval.annotation import allocate
from agent_eval.parsing import parse_answer_block, parse_tool_call, split_thinking
from agent_eval.retrieval import BM25Index
from agent_eval.scoring.outcome import exact_match, f1_score, pass_at_k, pass_hat_k
from agent_eval.scoring.trajectory import trajectory_metrics
from agent_eval.stats import cohen_kappa, mcnemar_exact, pareto_front
from agent_eval.tools import INJECTED_ERROR, ToolEnv


def test_em_f1():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("Paris France", "Paris") == pytest.approx(2 / 3)


def test_pass_hat_k():
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(2, 3, 3) == 0.0
    assert pass_hat_k(2, 3, 1) == pytest.approx(2 / 3)
    assert pass_hat_k(2, 3, 2) == pytest.approx(1 / 3)
    assert pass_at_k(1, 3, 3) == 1.0


def test_split_thinking_and_parse():
    thought, vis = split_thinking('<think>\nhmm\n</think>\n\n<tool_call>\n{"name": "search", "arguments": {"query": "a b"}}\n</tool_call>')
    assert thought == "hmm"
    p = parse_tool_call(vis)
    assert p.ok and p.tool == "search" and p.args == {"query": "a b"}
    p = parse_tool_call('<tool_call>{"name": "finish", "arguments": {"answer": "X", "cited_doc_ids": "d00001, d00002"}}</tool_call>')
    assert p.ok and p.args["cited_doc_ids"] == ["d00001", "d00002"]
    assert not parse_tool_call("I think the answer is X").ok
    assert not parse_tool_call('<tool_call>{"name": "browse", "arguments": {}}</tool_call>').ok
    # Missing closing tag (stopped by max_tokens) still parses.
    assert parse_tool_call('<tool_call>\n{"name": "read", "arguments": {"doc_id": "d00003"}}').ok


def test_parse_answer_block():
    a, c, ok = parse_answer_block("Answer: Paris\nCitations: d00001, d00002")
    assert (a, c, ok) == ("Paris", ["d00001", "d00002"], True)
    a, c, ok = parse_answer_block("Paris")
    assert a == "Paris" and not ok


def _docs():
    return [
        {"doc_id": "d00000", "title": "Apple", "text": "Apple is a fruit grown in orchards."},
        {"doc_id": "d00001", "title": "Banana", "text": "Banana is a yellow fruit."},
        {"doc_id": "d00002", "title": "Car", "text": "A car is a vehicle with wheels."},
    ]


def test_bm25_and_tools(tmp_path):
    idx = BM25Index().fit(_docs())
    assert idx.search("yellow fruit", 2)[0][0] == "d00001"
    idx.save(tmp_path / "i.pkl")
    assert BM25Index.load(tmp_path / "i.pkl").search("wheels", 1)[0][0] == "d00002"
    env = ToolEnv(_docs(), idx)
    r = env.call("search", {"query": "vehicle"})
    assert r.ok and r.doc_ids_returned[0] == "d00002"
    assert env.call("read", {"doc_id": "[d00000]"}).doc_id_read == "d00000"
    assert not env.call("read", {"doc_id": "nope"}).ok
    always = ToolEnv(_docs(), idx, perturb_p=1.0, perturb_seed=1)
    r = always.call("search", {"query": "vehicle"})
    assert r.perturbed == "error" and r.observation == INJECTED_ERROR


def _step(n, tool, args, **kw):
    s = {"step": n, "action": {"tool": tool, "args": args}, "parse_ok": True, "observation": "", "perturbed": None,
         "doc_ids_returned": None, "doc_id_read": None}
    s.update(kw)
    return s


def test_trajectory_metrics():
    traj = {
        "terminated_by": "finish",
        "cited_doc_ids": ["d1", "d2"],
        "steps": [
            _step(1, "search", {"query": "alpha beta"}, doc_ids_returned=["d1", "d9"]),
            _step(2, "read", {"doc_id": "d1"}, doc_id_read="d1"),
            _step(3, "search", {"query": "alpha beta"}, doc_ids_returned=["d1"]),
            _step(4, "read", {"doc_id": "d1"}, doc_id_read="d1"),
            _step(5, "read", {"doc_id": "d2"}, perturbed="error"),
            _step(6, "read", {"doc_id": "d2"}, doc_id_read="d2"),
            _step(7, "finish", {"answer": "x", "cited_doc_ids": ["d1", "d2"]}),
        ],
    }
    m = trajectory_metrics(traj, ["d1", "d2"])
    assert m["gold_evidence_recall"] == 1.0
    assert m["grounded"] and m["grounded_strict"]
    assert m["redundant_queries"] == 1 and m["rereads"] == 1
    assert m["steps_to_first_gold"] == 2
    assert m["steps_after_last_gold"] == 0
    assert m["recovery_sensible"] == 1.0 and m["recovery_alternative"] == 0.0 and m["recovered"]
    traj["cited_doc_ids"] = ["d1", "d9"]
    assert not trajectory_metrics(traj, ["d1", "d2"])["grounded"]
    traj["cited_doc_ids"] = []
    m = trajectory_metrics(traj, ["d1", "d2"])
    assert not m["grounded"] and math.isnan(m["citation_precision"])


def test_allocate():
    a = allocate(10, {"a": 2, "b": 1, "c": 1}, {"a": 100, "b": 100, "c": 1})
    assert sum(a.values()) == 10 and a["c"] == 1 and a["a"] > a["b"]
    assert sum(allocate(50, {"a": 1, "b": 1}, {"a": 3, "b": 4}).values()) == 7


def test_stats():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    mc = mcnemar_exact([True] * 10 + [False] * 10, [False] * 10 + [False] * 10)
    assert mc["a_only"] == 10 and mc["p_value"] < 0.01
    assert pareto_front({"a": (1, 0.5), "b": (2, 0.4), "c": (3, 0.9)}) == ["a", "c"]
