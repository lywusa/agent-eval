"""Frozen corpus and evaluation-question sample (research plan §4.2).

Corpus = every paragraph of the HotpotQA distractor-setting dev set, deduplicated
by title. The 400 evaluation questions are a seeded, type-stratified sample of the
same dev set, so their gold paragraphs are guaranteed to be in the corpus while the
other ~7,000 questions' paragraphs act as distractors.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from agent_eval.io import read_jsonl, write_json, write_jsonl

HOTPOT_DEV_DISTRACTOR_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HF_DEV_DISTRACTOR_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
)


def download_hotpot_dev(raw_dir: str | Path) -> Path:
    """Official JSON if reachable, otherwise the identical Hugging Face parquet."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    official = raw_dir / "hotpot_dev_distractor_v1.json"
    hf = raw_dir / "hotpot_dev_distractor_hf.parquet"
    for p in (official, hf):
        if p.exists():
            return p
    try:
        print(f"[data] downloading {HOTPOT_DEV_DISTRACTOR_URL}")
        urllib.request.urlretrieve(HOTPOT_DEV_DISTRACTOR_URL, official)
        return official
    except OSError as e:
        official.unlink(missing_ok=True)
        print(f"[data] official host failed ({e}); using Hugging Face mirror {HF_DEV_DISTRACTOR_URL}")
    urllib.request.urlretrieve(HF_DEV_DISTRACTOR_URL, hf)
    return hf


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    def plain(v: Any) -> Any:
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v

    return [{k: plain(v) for k, v in row.items()} for row in pd.read_parquet(path).to_dict("records")]


def _normalize_example(ex: dict[str, Any]) -> dict[str, Any]:
    """Accept both the official JSON format and the Hugging Face `hotpot_qa` format."""
    if "_id" in ex:  # official
        context = [(t, list(s)) for t, s in ex["context"]]
        sfacts = [(t, int(i)) for t, i in ex["supporting_facts"]]
        qid = ex["_id"]
    else:  # HF datasets
        context = list(zip(ex["context"]["title"], ex["context"]["sentences"]))
        sfacts = list(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
        qid = ex["id"]
    return {
        "question_id": qid,
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type", "bridge"),
        "level": ex.get("level", "hard"),
        "context": context,
        "supporting_facts": sfacts,
    }


def load_hotpot(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = read_jsonl(path)
    elif path.suffix == ".parquet":
        raw = _read_parquet(path)
    else:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    return [_normalize_example(ex) for ex in raw]


def paragraph_text(sentences: list[str]) -> str:
    return "".join(sentences).strip() if all(s.startswith(" ") for s in sentences[1:]) else " ".join(
        s.strip() for s in sentences
    )


def build_corpus(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    """Return (docs, title->doc_id, stats). doc_ids are assigned in sorted-title order,
    so they do not depend on the order of the input file."""
    by_title: dict[str, list[str]] = {}
    conflicted: set[str] = set()
    for ex in examples:
        for title, sents in ex["context"]:
            if title in by_title:
                if by_title[title] != sents:
                    conflicted.add(title)
                continue
            by_title[title] = sents
    conflicts = len(conflicted)
    docs = []
    title_to_id: dict[str, str] = {}
    width = max(5, len(str(len(by_title))))
    for i, title in enumerate(sorted(by_title)):
        doc_id = f"d{i:0{width}d}"
        title_to_id[title] = doc_id
        docs.append({"doc_id": doc_id, "title": title, "text": paragraph_text(by_title[title])})
    stats = {"n_examples": len(examples), "n_docs": len(docs), "title_text_conflicts": conflicts,
             "conflicted_titles": sorted(conflicted)}
    return docs, title_to_id, stats


def build_questions(
    examples: list[dict[str, Any]], title_to_id: dict[str, str], conflicted: set[str] | None = None
) -> list[dict[str, Any]]:
    """`gold_conflict` marks questions whose gold paragraph title has more than one
    text variant in the dev set (the corpus keeps only one); they are excluded from
    the evaluation sample."""
    conflicted = conflicted or set()
    out = []
    for ex in examples:
        gold_titles = sorted({t for t, _ in ex["supporting_facts"]})
        out.append(
            {
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "gold_titles": gold_titles,
                "gold_doc_ids": [title_to_id[t] for t in gold_titles if t in title_to_id],
                "supporting_facts": ex["supporting_facts"],
                "gold_conflict": any(t in conflicted for t in gold_titles),
            }
        )
    return out


def sample_questions(
    questions: list[dict[str, Any]],
    n: int = 400,
    seed: int = 2026,
    bridge_fraction: float = 0.8,
    level: str | None = "hard",
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Seeded sample stratified by question type (bridge/comparison).

    The HotpotQA dev set contains only `hard` questions, so the `level` filter is a
    no-op there; it matters if a train-split file is supplied instead.
    """
    exclude_ids = exclude_ids or set()
    pool = [
        q
        for q in questions
        if (level is None or q["level"] == level)
        and len(q["gold_doc_ids"]) >= 2
        and not q.get("gold_conflict", False)
        and q["question_id"] not in exclude_ids
    ]
    rng = random.Random(seed)
    bridge = sorted([q for q in pool if q["type"] == "bridge"], key=lambda q: q["question_id"])
    comp = sorted([q for q in pool if q["type"] != "bridge"], key=lambda q: q["question_id"])
    n_bridge = min(len(bridge), round(n * bridge_fraction))
    n_comp = min(len(comp), n - n_bridge)
    n_bridge = min(len(bridge), n - n_comp)
    chosen = rng.sample(bridge, n_bridge) + rng.sample(comp, n_comp)
    rng.shuffle(chosen)
    return chosen


def fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def build_all(
    hotpot_path: str | Path,
    out_dir: str | Path,
    n_eval: int = 400,
    n_pilot: int = 50,
    seed: int = 2026,
    bridge_fraction: float = 0.8,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    examples = load_hotpot(hotpot_path)
    docs, title_to_id, stats = build_corpus(examples)
    questions = build_questions(examples, title_to_id, set(stats.pop("conflicted_titles")))
    eval_qs = sample_questions(questions, n=n_eval, seed=seed, bridge_fraction=bridge_fraction)
    # The pilot uses questions disjoint from the evaluation sample so that pilot
    # bug-fixing cannot tune prompts on evaluation questions.
    pilot_qs = sample_questions(
        questions,
        n=n_pilot,
        seed=seed + 1,
        bridge_fraction=bridge_fraction,
        exclude_ids={q["question_id"] for q in eval_qs},
    )
    write_jsonl(out_dir / "corpus.jsonl", docs)
    write_jsonl(out_dir / "questions_all.jsonl", questions)
    write_jsonl(out_dir / "questions_eval.jsonl", eval_qs)
    write_jsonl(out_dir / "questions_pilot.jsonl", pilot_qs)
    manifest = {
        **stats,
        "n_eval": len(eval_qs),
        "n_pilot": len(pilot_qs),
        "seed": seed,
        "bridge_fraction": bridge_fraction,
        "eval_types": dict(Counter(q["type"] for q in eval_qs)),
        "eval_fingerprint": fingerprint([q["question_id"] for q in eval_qs]),
        "pilot_fingerprint": fingerprint([q["question_id"] for q in pilot_qs]),
        "source": str(hotpot_path),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_corpus(data_dir: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(data_dir) / "corpus.jsonl")


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
