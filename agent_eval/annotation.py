"""Family C: blinded single-annotator human evaluation (research plan §5.4).

Workflow
--------
1. ``sample``  : stratified sample of 150 ReAct trajectories over the four A×B cells
                 and all 6 ReAct configs, 4B over-sampled.
2. ``export``  : writes a blinded batch: ``items.jsonl`` (what the annotator sees),
                 ``trajectories.html`` (read-only view), ``sheet.csv`` (spreadsheet
                 template) and ``key.csv`` (anon_id -> trajectory; never open it
                 while annotating).
3. ``label``   : terminal labeler that reads only ``items.jsonl`` and appends to
                 ``labels.jsonl`` after every item (resumable).
4. ``retest``  : 30 already-labelled items re-exported with fresh anon ids and order,
                 to be labelled >= 14 days later without looking at the first labels.
5. ``external``: 40 items for the external second annotator (spreadsheet or labeler).
6. ``import``  : spreadsheet -> ``labels.jsonl``.
"""

from __future__ import annotations

import csv
import html
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from agent_eval.configs import ANCHOR_MODEL
from agent_eval.io import append_jsonl, iter_jsonl, read_jsonl, write_jsonl
from agent_eval.judge import render_trajectory, traj_key
from agent_eval.rubric import ERROR_CATEGORIES, ERROR_DESCRIPTIONS, HOLISTIC, STEP_LABELS

CELLS = ["RA-GP", "RA-BP", "WA-GP", "WA-BP"]


# ----------------------------------------------------------------------- sampling


def allocate(n: int, weights: dict[str, float], available: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of n items over strata, capped by availability,
    redistributing any shortfall to strata that still have items."""
    alloc = {k: 0 for k in weights}
    remaining = n
    while remaining > 0:
        active = [k for k in sorted(weights) if alloc[k] < available.get(k, 0) and weights[k] > 0]
        if not active:
            break
        total_w = sum(weights[k] for k in active)
        quotas = {k: remaining * weights[k] / total_w for k in active}
        adds = {k: min(int(quotas[k]), available[k] - alloc[k]) for k in active}
        if sum(adds.values()) == 0:
            for k in sorted(active, key=lambda k: (-quotas[k], k)):
                if remaining == 0:
                    break
                alloc[k] += 1
                remaining -= 1
            continue
        for k, v in adds.items():
            alloc[k] += v
        remaining -= sum(adds.values())
    return alloc


def stratified_sample(
    scores: pd.DataFrame,
    n: int = 150,
    seed: int = 2026,
    anchor_model: str = ANCHOR_MODEL,
    anchor_weight: float = 2.0,
) -> pd.DataFrame:
    df = scores[(scores["scaffold"] == "react") & (scores["perturb_p"] == 0)].copy()
    configs = sorted(df["config_id"].unique())
    w_cfg = {c: (anchor_weight if df.loc[df.config_id == c, "model"].iloc[0] == anchor_model else 1.0) for c in configs}
    strata_w, avail = {}, {}
    for c in configs:
        for cell in CELLS:
            k = f"{c}::{cell}"
            strata_w[k] = w_cfg[c] / len(CELLS)
            avail[k] = int(((df.config_id == c) & (df.cell == cell)).sum())
    alloc = allocate(n, strata_w, avail)
    rng = random.Random(seed)
    picked = []
    for k in sorted(alloc):
        if alloc[k] == 0:
            continue
        c, cell = k.split("::")
        keys = sorted(df.loc[(df.config_id == c) & (df.cell == cell), "key"])
        picked += rng.sample(keys, alloc[k])
    out = df[df.key.isin(picked)][["key", "config_id", "model", "thinking", "cell"]].copy()
    return out.sort_values("key").reset_index(drop=True)


# ------------------------------------------------------------------------- export


def _anon_ids(n: int, rng: random.Random) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("t" + "".join(rng.choice("0123456789abcdef") for _ in range(8)))
    return sorted(ids, key=lambda _: rng.random())


def export_batch(
    keys: list[str],
    trajs_by_key: dict[str, dict[str, Any]],
    out_dir: str | Path,
    batch: str,
    seed: int,
    meta: dict[str, Any] | None = None,
) -> Path:
    bdir = Path(out_dir) / batch
    if (bdir / "items.jsonl").exists():
        raise FileExistsError(f"{bdir} already exists; refusing to overwrite a batch")
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    anon = _anon_ids(len(keys), rng)
    items, key_rows = [], []
    for a, k in zip(anon, keys):
        t = trajs_by_key[k]
        items.append(
            {
                "anon_id": a,
                "question": t["question"],
                "n_steps": len(t["steps"]),
                "rendered": render_trajectory(t, max_thought_chars=2000, max_obs_chars=2000),
            }
        )
        key_rows.append({"anon_id": a, "key": k})
    write_jsonl(bdir / "items.jsonl", items)
    with (bdir / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anon_id", "key"])
        w.writeheader()
        w.writerows(key_rows)
    with (bdir / "sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "n_steps", "step_labels", "holistic", "error_category", "notes"])
        for it in items:
            w.writerow([it["anon_id"], it["n_steps"], "", "", "", ""])
    _write_html(bdir / "trajectories.html", items, batch)
    (bdir / "batch.json").write_text(
        json.dumps({"batch": batch, "n": len(items), "seed": seed, "created_at": _now(), **(meta or {})}, indent=2)
    )
    return bdir


def _write_html(path: Path, items: list[dict[str, Any]], batch: str) -> None:
    cats = "".join(f"<li><b>{c}</b>: {html.escape(ERROR_DESCRIPTIONS[c])}</li>" for c in ERROR_CATEGORIES)
    steps = "".join(f"<li><b>{k:+d}</b>: {html.escape(v)}</li>" for k, v in STEP_LABELS.items())
    hol = "".join(f"<li><b>{k}</b>: {html.escape(v)}</li>" for k, v in HOLISTIC.items())
    body = [
        f"<html><head><meta charset='utf-8'><title>Annotation batch {batch}</title>",
        "<style>body{font-family:sans-serif;max-width:60em;margin:auto}pre{white-space:pre-wrap;"
        "background:#f6f6f6;padding:1em;border-radius:6px}h2{border-top:2px solid #999;padding-top:1em}</style>",
        f"</head><body><h1>Annotation batch: {batch}</h1>",
        "<p>Fill in <code>sheet.csv</code>: <code>step_labels</code> as a comma-separated list with one "
        "value per step (1, 0, -1), <code>holistic</code> 1-5, one <code>error_category</code>.</p>",
        f"<h3>Step labels</h3><ul>{steps}</ul><h3>Holistic</h3><ul>{hol}</ul><h3>Error categories</h3><ul>{cats}</ul>",
    ]
    for it in items:
        body.append(f"<h2>{it['anon_id']} ({it['n_steps']} steps)</h2><pre>{html.escape(it['rendered'])}</pre>")
    body.append("</body></html>")
    path.write_text("\n".join(body), encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -------------------------------------------------------------------------- label


def _ask(prompt: str, valid: Callable[[str], bool]) -> str:
    while True:
        v = input(prompt).strip()
        if v.lower() in ("q", "quit"):
            raise KeyboardInterrupt
        if valid(v):
            return v
        print("  invalid input, try again (q to quit)")


def label_batch(batch_dir: str | Path, annotator: str) -> None:
    """Interactive terminal labeler. Saves after every trajectory; re-run to resume."""
    bdir = Path(batch_dir)
    items = read_jsonl(bdir / "items.jsonl")
    labels_path = bdir / f"labels_{annotator}.jsonl"
    done = {r["anon_id"] for r in iter_jsonl(labels_path)}
    todo = [it for it in items if it["anon_id"] not in done]
    print(f"{len(done)} labelled, {len(todo)} to go. Enter q at any prompt to stop.\n")
    cat_menu = "  ".join(f"{i}={c}" for i, c in enumerate(ERROR_CATEGORIES))
    try:
        for it in todo:
            print("=" * 100)
            print(f"[{it['anon_id']}]")
            print(it["rendered"])
            print("-" * 100)
            t0 = time.time()
            n = it["n_steps"]
            labs = _ask(
                f"Step labels ({n} values from 1/0/-1, comma-separated): ",
                lambda v: len([x for x in v.split(",") if x.strip()]) == n
                and all(x.strip() in ("1", "0", "-1", "+1") for x in v.split(",")),
            )
            hol = _ask("Holistic 1-5: ", lambda v: v in {"1", "2", "3", "4", "5"})
            print(cat_menu)
            cat = _ask("Error category number: ", lambda v: v.isdigit() and int(v) < len(ERROR_CATEGORIES))
            notes = input("Notes (optional): ").strip()
            append_jsonl(
                labels_path,
                [
                    {
                        "anon_id": it["anon_id"],
                        "annotator": annotator,
                        "step_labels": [int(x) for x in labs.split(",")],
                        "holistic": int(hol),
                        "error_category": ERROR_CATEGORIES[int(cat)],
                        "notes": notes,
                        "seconds": round(time.time() - t0, 1),
                        "labelled_at": _now(),
                    }
                ],
            )
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; progress saved")


def import_sheet(batch_dir: str | Path, sheet: str | Path, annotator: str) -> int:
    bdir = Path(batch_dir)
    n_steps = {it["anon_id"]: it["n_steps"] for it in read_jsonl(bdir / "items.jsonl")}
    rows, errors = [], []
    with open(sheet, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["anon_id"].strip()
            if not r.get("holistic", "").strip():
                continue
            try:
                labs = [int(x) for x in r["step_labels"].replace(";", ",").split(",") if x.strip()]
                hol = int(r["holistic"])
                cat = r.get("error_category", "").strip() or "none"
                assert a in n_steps, "unknown anon_id"
                assert len(labs) == n_steps[a], f"expected {n_steps[a]} step labels, got {len(labs)}"
                assert all(x in (-1, 0, 1) for x in labs), "step labels must be -1/0/1"
                assert 1 <= hol <= 5, "holistic must be 1-5"
                assert cat in ERROR_CATEGORIES, f"unknown error category {cat!r}"
            except (AssertionError, ValueError) as e:
                errors.append(f"{a}: {e}")
                continue
            rows.append(
                {
                    "anon_id": a,
                    "annotator": annotator,
                    "step_labels": labs,
                    "holistic": hol,
                    "error_category": cat,
                    "notes": r.get("notes", ""),
                    "seconds": None,
                    "labelled_at": _now(),
                }
            )
    if errors:
        print("rows with errors (not imported):\n  " + "\n  ".join(errors))
    write_jsonl(bdir / f"labels_{annotator}.jsonl", rows)
    return len(rows)


# ---------------------------------------------------------------- re-test/external


def _labelled_keys(batch_dir: Path) -> list[str]:
    key = {r["anon_id"]: r["key"] for r in csv.DictReader((batch_dir / "key.csv").open())}
    labelled = set()
    for p in batch_dir.glob("labels_*.jsonl"):
        labelled |= {r["anon_id"] for r in iter_jsonl(p)}
    return sorted(key[a] for a in labelled if a in key)


def derived_batch(
    main_batch_dir: str | Path,
    trajs_by_key: dict[str, dict[str, Any]],
    out_dir: str | Path,
    batch: str,
    n: int,
    seed: int,
    require_labelled: bool = True,
) -> Path:
    mdir = Path(main_batch_dir)
    if require_labelled:
        keys = _labelled_keys(mdir)
    else:
        keys = sorted(r["key"] for r in csv.DictReader((mdir / "key.csv").open()))
    if len(keys) < n:
        raise ValueError(f"only {len(keys)} eligible items in {mdir}, need {n}")
    chosen = random.Random(seed).sample(keys, n)
    return export_batch(chosen, trajs_by_key, out_dir, batch, seed + 7919, meta={"derived_from": mdir.name})


# --------------------------------------------------------------------------- load


def load_human_labels(annotation_dir: str | Path) -> pd.DataFrame:
    """All labels from all batches, de-anonymized via each batch's key.csv."""
    rows = []
    for bdir in sorted(p for p in Path(annotation_dir).iterdir() if (p / "key.csv").exists()):
        key = {r["anon_id"]: r["key"] for r in csv.DictReader((bdir / "key.csv").open())}
        for lp in sorted(bdir.glob("labels_*.jsonl")):
            for r in iter_jsonl(lp):
                rows.append({**r, "batch": bdir.name, "key": key.get(r["anon_id"])})
    if not rows:
        return pd.DataFrame(columns=["anon_id", "annotator", "step_labels", "holistic", "error_category", "batch", "key"])
    return pd.DataFrame(rows)


def trajs_index(trajs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {traj_key(t): t for t in trajs}

