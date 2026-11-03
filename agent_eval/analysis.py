"""Analyses that answer RQ1-RQ5 plus evaluator-reliability checks (research plan §5.5).

Every function takes the per-trajectory score table (``scoring.table``) and, where
needed, human labels and judge outputs, and returns DataFrames. ``run_all`` writes
CSV tables, PNG figures and ``report.md`` with a pre-registered hypothesis check.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent_eval.configs import GRID, POLICY_MODELS
from agent_eval.io import iter_jsonl
from agent_eval.scoring.outcome import pass_hat_k
from agent_eval.stats import (
    cluster_bootstrap,
    cohen_kappa,
    kendall_tau,
    mcnemar_exact,
    pareto_front,
    spearman,
    spearman_ci,
)

CELLS = ["RA-GP", "RA-BP", "WA-GP", "WA-BP"]


def _main(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[scores["perturb_p"] == 0]


def config_order(configs: list[str]) -> list[str]:
    def k(c: str) -> tuple:
        cfg = GRID.get(c)
        if cfg is None:
            return (9, 9, c)
        scaffold = {"react": 0, "single_rag": 1, "closed_book": 2}[cfg.scaffold]
        size = POLICY_MODELS.index(cfg.model) if cfg.model in POLICY_MODELS else 9
        return (scaffold, size, int(cfg.thinking), c)

    return sorted(configs, key=k)


def _nanmean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if x.notna().any() else math.nan


# ----------------------------------------------------------------------------- RQ1


def rq1_disagreement(scores: pd.DataFrame, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2x2 (A-correct x B-grounded) per config, RA-BP rate with CI, and the
    parametric-recall test against the same model's closed-book accuracy."""
    df = _main(scores)
    df = df[df["scaffold"] != "closed_book"]
    cb = _main(scores)
    cb = cb[cb["scaffold"] == "closed_book"].groupby(["model", "question_id"])["correct"].mean().rename("cb_acc")
    rows, mech = [], []
    for c in config_order(list(df["config_id"].unique())):
        d = df[df["config_id"] == c]
        counts = d["cell"].value_counts().reindex(CELLS, fill_value=0)
        n_correct = int(d["correct"].sum())

        def ra_bp(x: pd.DataFrame) -> float:
            k = x["correct"].sum()
            return float(((x["cell"] == "RA-BP").sum()) / k) if k else math.nan

        def wa_gp(x: pd.DataFrame) -> float:
            k = (~x["correct"]).sum()
            return float(((x["cell"] == "WA-GP").sum()) / k) if k else math.nan

        r, rlo, rhi = cluster_bootstrap(d, ra_bp, n_boot=n_boot)
        w, wlo, whi = cluster_bootstrap(d, wa_gp, n_boot=n_boot)
        rows.append(
            {
                "config_id": c,
                "n": len(d),
                **{f"n_{k}": int(v) for k, v in counts.items()},
                "accuracy": float(d["correct"].mean()),
                "grounded_rate": float(d["grounded"].mean()),
                "ra_bp_rate_among_correct": r,
                "ra_bp_ci_lo": rlo,
                "ra_bp_ci_hi": rhi,
                "wa_gp_rate_among_wrong": w,
                "wa_gp_ci_lo": wlo,
                "wa_gp_ci_hi": whi,
                "n_correct": n_correct,
            }
        )
        if len(cb):
            dd = d.join(cb, on=["model", "question_id"])
            dd = dd[dd["correct"] & dd["cb_acc"].notna()]
            if len(dd):

                def gap(x: pd.DataFrame) -> float:
                    a = x.loc[x["cell"] == "RA-BP", "cb_acc"]
                    b = x.loc[x["cell"] == "RA-GP", "cb_acc"]
                    return float(a.mean() - b.mean()) if len(a) and len(b) else math.nan

                g, glo, ghi = cluster_bootstrap(dd, gap, n_boot=n_boot)
                mech.append(
                    {
                        "config_id": c,
                        "cb_acc_given_RA_BP": _nanmean(dd.loc[dd["cell"] == "RA-BP", "cb_acc"]),
                        "cb_acc_given_RA_GP": _nanmean(dd.loc[dd["cell"] == "RA-GP", "cb_acc"]),
                        "gap": g,
                        "gap_ci_lo": glo,
                        "gap_ci_hi": ghi,
                        "n_RA_BP": int((dd["cell"] == "RA-BP").sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(mech)


# ----------------------------------------------------------------------------- RQ2


def human_primary(human: pd.DataFrame, batch: str = "main") -> pd.DataFrame:
    if human is None or human.empty:
        return pd.DataFrame(columns=["key", "human_holistic", "human_step_labels", "human_error"])
    h = human[human["batch"] == batch].dropna(subset=["key"]).drop_duplicates("key", keep="first")
    return h.rename(
        columns={"holistic": "human_holistic", "step_labels": "human_step_labels", "error_category": "human_error"}
    )[["key", "human_holistic", "human_step_labels", "human_error"]]


def _poststratified_human(d: pd.DataFrame) -> float:
    """Human mean per config, re-weighted from the stratified sample back to the
    config's true cell distribution (the sample over-represents rare cells)."""
    lab = d[d["human_holistic"].notna()]
    if lab.empty:
        return math.nan
    p_cell = d["cell"].value_counts(normalize=True)
    fallback = lab["human_holistic"].mean()
    total = 0.0
    for cell, p in p_cell.items():
        x = lab.loc[lab["cell"] == cell, "human_holistic"]
        total += p * (x.mean() if len(x) else fallback)
    return float(total)


def config_summary(scores: pd.DataFrame, human: pd.DataFrame | None, cost_budget: float | None = None) -> tuple[pd.DataFrame, float]:
    df = _main(scores).copy()
    react = df[df["scaffold"] == "react"]
    budget = float(cost_budget) if cost_budget else float(react["total_tokens"].median() if len(react) else df["total_tokens"].median())
    if human is not None and not human.empty:
        df = df.merge(human_primary(human), on="key", how="left")
    else:
        df["human_holistic"] = np.nan
    rows = []
    for c in config_order(list(df["config_id"].unique())):
        d = df[df["config_id"] == c]
        rows.append(
            {
                "config_id": c,
                "scaffold": d["scaffold"].iloc[0],
                "n": len(d),
                "accuracy": float(d["correct"].mean()),
                "em": float(d["em"].mean()),
                "f1": float(d["f1"].mean()),
                "b_composite": float(d["b_composite"].mean()),
                "gold_evidence_recall": _nanmean(d["gold_evidence_recall"]),
                "grounded": float(d["grounded"].mean()),
                "redundant_calls": float(d["redundant_calls"].mean()),
                "parse_failure_rate": float(d["parse_failure_rate"].mean()),
                "mean_total_tokens": float(d["total_tokens"].mean()),
                "acc_at_budget": float((d["correct"] & (d["total_tokens"] <= budget)).mean()),
                "human_holistic_poststrat": _poststratified_human(d),
                "n_human": int(d["human_holistic"].notna().sum()),
                **{f"term_{k}": float(v) for k, v in d["terminated_by"].value_counts(normalize=True).items()},
            }
        )
    return pd.DataFrame(rows), budget


RANKERS = {
    "A_accuracy": "accuracy",
    "B_composite": "b_composite",
    "C_human": "human_holistic_poststrat",
    "A_at_matched_cost": "acc_at_budget",
}


def rq2_rankings(scores: pd.DataFrame, human: pd.DataFrame | None, n_boot: int = 500, seed: int = 0, cost_budget: float | None = None) -> dict[str, Any]:
    summ, budget = config_summary(scores, human, cost_budget)
    react = summ[summ["scaffold"] == "react"].set_index("config_id")
    pairs = [(a, b) for i, a in enumerate(RANKERS) for b in list(RANKERS)[i + 1 :]]
    df = _main(scores)
    df = df[df["scaffold"] == "react"]
    if human is not None and not human.empty:
        df = df.merge(human_primary(human), on="key", how="left")
    else:
        df = df.assign(human_holistic=np.nan)

    def taus(sub: pd.DataFrame) -> dict[tuple[str, str], float]:
        per = {}
        for c, d in sub.groupby("config_id"):
            per[c] = {
                "accuracy": d["correct"].mean(),
                "b_composite": d["b_composite"].mean(),
                "human_holistic_poststrat": _poststratified_human(d),
                "acc_at_budget": (d["correct"] & (d["total_tokens"] <= budget)).mean(),
            }
        t = pd.DataFrame(per).T
        out = {}
        for a, b in pairs:
            x, y = t[RANKERS[a]].astype(float), t[RANKERS[b]].astype(float)
            m = x.notna() & y.notna()
            out[(a, b)] = kendall_tau(x[m], y[m]) if m.sum() >= 3 else math.nan
        return out

    point = taus(df)
    groups = list(df.reset_index(drop=True).groupby("question_id").indices.values())
    dfr = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    boots: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for p, v in taus(dfr.iloc[idx]).items():
            if not math.isnan(v):
                boots[p].append(v)
    rows = []
    for p in pairs:
        b = boots[p]
        rows.append(
            {
                "ranking_a": p[0],
                "ranking_b": p[1],
                "kendall_tau": point[p],
                "ci_lo": float(np.percentile(b, 2.5)) if b else math.nan,
                "ci_hi": float(np.percentile(b, 97.5)) if b else math.nan,
                "n_configs": int(react[[RANKERS[p[0]], RANKERS[p[1]]]].dropna().shape[0]),
            }
        )
    ranks = pd.DataFrame(
        {name: react[col].rank(ascending=False, method="min") for name, col in RANKERS.items() if react[col].notna().any()}
    )
    pts = {c: (r["mean_total_tokens"], r["accuracy"]) for c, r in summ.set_index("config_id").iterrows()}
    return {
        "summary": summ,
        "budget": budget,
        "taus": pd.DataFrame(rows),
        "ranks": ranks.reset_index(),
        "pareto": pareto_front(pts),
    }


def paired_comparisons(scores: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """McNemar on per-(question, seed) correctness for config pairs (e.g. thinking on vs off)."""
    df = _main(scores)
    rows = []
    for a, b in pairs:
        da = df[df.config_id == a].set_index(["question_id", "seed"])["correct"]
        db = df[df.config_id == b].set_index(["question_id", "seed"])["correct"]
        j = pd.concat([da.rename("a"), db.rename("b")], axis=1).dropna()
        if j.empty:
            continue
        rows.append({"config_a": a, "config_b": b, "n_pairs": len(j), **mcnemar_exact(j["a"].astype(bool), j["b"].astype(bool))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- RQ3

AUTO_METRICS = {
    # name: (column, sign so that higher = better process, cost kind)
    "outcome_correct": ("correct", 1, "llm_correctness"),
    "outcome_em": ("em", 1, "free"),
    "outcome_f1": ("f1", 1, "free"),
    "gold_evidence_recall": ("gold_evidence_recall", 1, "free"),
    "grounded": ("grounded", 1, "free"),
    "citation_f1": ("citation_f1", 1, "free"),
    "redundant_calls": ("redundant_calls", -1, "free"),
    "steps_after_last_gold": ("steps_after_last_gold", -1, "free"),
    "parse_failure_rate": ("parse_failure_rate", -1, "free"),
    "b_composite": ("b_composite", 1, "free"),
    "llm_holistic": ("llm_holistic", 1, "llm_steps"),
    "llm_harmful_fraction": ("llm_harmful_fraction", -1, "llm_steps"),
}


def rq3_validity(
    scores: pd.DataFrame,
    human: pd.DataFrame | None,
    judge_steps: dict[str, dict[str, Any]] | None = None,
    correctness_tokens_per_traj: float | None = None,
    cpu_seconds_per_traj: float | None = None,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if human is None or human.empty:
        return pd.DataFrame(), {}
    df = _main(scores).merge(human_primary(human), on="key", how="inner")
    df = df[df["human_holistic"].notna()]
    if df.empty:
        return pd.DataFrame(), {}
    llm_tokens = _nanmean(df["llm_judge_tokens"])
    rows = []
    for name, (col, sign, kind) in AUTO_METRICS.items():
        if col not in df or pd.to_numeric(df[col], errors="coerce").notna().sum() < 3:
            continue
        x = sign * pd.to_numeric(df[col], errors="coerce").astype(float)
        rho, lo, hi = spearman_ci(x.values, df["human_holistic"].astype(float).values, n_boot=n_boot)
        cost_tokens = {"free": 0.0, "llm_steps": llm_tokens, "llm_correctness": correctness_tokens_per_traj or math.nan}[kind]
        rows.append(
            {
                "metric": name,
                "spearman_rho_vs_human": rho,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": int(x.notna().sum()),
                "cost_kind": kind,
                "judge_tokens_per_traj": cost_tokens,
                "cpu_ms_per_traj": 1000 * cpu_seconds_per_traj if (kind == "free" and cpu_seconds_per_traj is not None) else math.nan,
            }
        )
    step_agree: dict[str, Any] = {}
    if judge_steps:
        hs, ls = [], []
        for _, r in df.iterrows():
            j = judge_steps.get(r["key"])
            if not j or r["human_step_labels"] is None:
                continue
            for h, l in zip(r["human_step_labels"], j["step_labels"]):
                if h is not None and l is not None:
                    hs.append(h)
                    ls.append(l)
        if hs:
            step_agree = {"n_steps": len(hs), "kappa_llm_vs_human_steps": cohen_kappa(hs, ls)}
    return pd.DataFrame(rows).sort_values("spearman_rho_vs_human", ascending=False), step_agree


def time_deterministic_metrics(trajs: list[dict[str, Any]], gold: dict[str, list[str]]) -> float:
    from agent_eval.scoring.trajectory import trajectory_metrics

    t0 = time.process_time()
    for t in trajs:
        trajectory_metrics(t, gold.get(t["question_id"], []))
    return (time.process_time() - t0) / max(1, len(trajs))


# ----------------------------------------------------------------------------- RQ4


def rq4_reliability(scores: pd.DataFrame, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _main(scores)
    rows, div = [], []
    for c in config_order(list(df["config_id"].unique())):
        d = df[df["config_id"] == c]
        per_q = d.groupby("question_id")["correct"].agg(["sum", "count"])
        n = int(per_q["count"].min())
        per_q = per_q[per_q["count"] >= n]
        row: dict[str, Any] = {"config_id": c, "n_questions": len(per_q), "n_seeds": n}
        for k in range(1, n + 1):
            vals = pd.DataFrame(
                {"question_id": per_q.index, "v": [pass_hat_k(int(s), n, k) for s in per_q["sum"]]}
            )
            p, lo, hi = cluster_bootstrap(vals, lambda x: float(x["v"].mean()), n_boot=n_boot)
            row[f"pass^{k}"], row[f"pass^{k}_lo"], row[f"pass^{k}_hi"] = p, lo, hi
        for k in range(2, n + 1):
            row[f"ratio_pass^{k}/pass^1"] = row[f"pass^{k}"] / row["pass^1"] if row["pass^1"] else math.nan
        row["frac_questions_mixed"] = float(((per_q["sum"] > 0) & (per_q["sum"] < n)).mean())
        rows.append(row)
        # Divergence sources on questions where seeds disagree.
        for qid, g in d.groupby("question_id"):
            if g["correct"].all() or not g["correct"].any():
                continue
            ok, bad = g[g["correct"]], g[~g["correct"]]
            ref = ok.iloc[0]
            for _, f in bad.iterrows():
                div.append({"config_id": c, "question_id": qid, "seed": f["seed"], "source": _divergence_source(ref, f),
                            "first_query_differs": ref.get("first_query") != f.get("first_query")})
    return pd.DataFrame(rows), pd.DataFrame(div)


def _divergence_source(ok: pd.Series, bad: pd.Series) -> str:
    if bad["terminated_by"] == "parse_failure" or bad["parse_failures"] > ok["parse_failures"]:
        return "parse_failure"
    if bad["terminated_by"] in ("max_steps", "token_cap"):
        return "step_or_token_limit"
    rec_ok = 0.0 if pd.isna(ok["gold_evidence_recall"]) else ok["gold_evidence_recall"]
    rec_bad = 0.0 if pd.isna(bad["gold_evidence_recall"]) else bad["gold_evidence_recall"]
    if bad["n_tool_calls"] < ok["n_tool_calls"] and rec_bad < rec_ok:
        return "early_finish"
    if rec_bad < rec_ok:
        return "query_choice"
    return "answer_from_same_evidence"


# ----------------------------------------------------------------------------- RQ5


def rq5_robustness(scores: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    pert = scores[scores["perturb_p"] > 0]
    for pc in sorted(pert["config_id"].unique()):
        base = pc.removesuffix("-perturb")
        a = scores[(scores.config_id == base)].set_index(["question_id", "seed"])
        b = scores[(scores.config_id == pc)].set_index(["question_id", "seed"])
        j = a[["correct", "grounded", "gold_evidence_recall"]].merge(
            b[["correct", "grounded", "gold_evidence_recall", "recovery_sensible", "recovery_alternative", "recovered", "n_perturbed"]],
            left_index=True, right_index=True, suffixes=("_base", "_pert"), how="inner",
        ).reset_index()
        if j.empty:
            continue
        out: dict[str, Any] = {"config_id": base, "n_pairs": len(j)}
        for m in ("correct", "grounded", "gold_evidence_recall"):
            p, lo, hi = cluster_bootstrap(
                j, lambda x, m=m: float(pd.to_numeric(x[f"{m}_pert"]).mean() - pd.to_numeric(x[f"{m}_base"]).mean()), n_boot=n_boot
            )
            out[f"delta_{m}"], out[f"delta_{m}_lo"], out[f"delta_{m}_hi"] = p, lo, hi
        hit = j[j["n_perturbed"] > 0]
        out["frac_with_injection"] = float(len(hit) / len(j))
        out["recovery_sensible"] = _nanmean(hit["recovery_sensible"])
        out["recovery_alternative"] = _nanmean(hit["recovery_alternative"])
        out["recovered_rate"] = _nanmean(hit["recovered"].astype(float))
        # Degradation visible only to trajectory metrics: still correct, lost grounding.
        both = j[j["correct_base"].astype(bool) & j["correct_pert"].astype(bool)]
        out["hidden_degradation_rate"] = float((both["grounded_base"].astype(bool) & ~both["grounded_pert"].astype(bool)).mean()) if len(both) else math.nan
        mc = mcnemar_exact(j["correct_base"].astype(bool), j["correct_pert"].astype(bool))
        out["mcnemar_p"] = mc["p_value"]
        rows.append(out)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- judge bias


def judge_bias(scores: pd.DataFrame, judge_dir: str | Path | None, judge_key: str, second_key: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    s = scores[scores["judge_correct"].notna()]
    if len(s):
        out["correctness_judge_vs_em_kappa"] = cohen_kappa(s["judge_correct"].astype(bool).tolist(), s["em"].astype(bool).tolist())
        out["correctness_judge_vs_em_agreement"] = float((s["judge_correct"].astype(bool) == s["em"].astype(bool)).mean())
        out["judge_correct_but_not_em"] = float((s["judge_correct"].astype(bool) & ~s["em"].astype(bool)).mean())
    if not judge_dir:
        return out
    jdir = Path(judge_dir)
    pw = list(iter_jsonl(jdir / judge_key / "pairwise.jsonl"))
    if pw:
        valid = [r for r in pw if r["verdict_original"] and r["verdict_swapped"]]
        out["pairwise_n"] = len(pw)
        out["pairwise_consistency"] = float(np.mean([r["consistent"] for r in valid])) if valid else math.nan
        # Under the swapped order the original "A" is shown second; a judge that always
        # picks the first slot yields verdict_original=A and verdict_swapped=B.
        out["pairwise_first_slot_preference"] = (
            float(np.mean([(r["verdict_original"] == "A") and (r["verdict_swapped"] == "B") for r in valid])) if valid else math.nan
        )
    base = {r["key"]: r for r in iter_jsonl(jdir / judge_key / "steps.jsonl")}
    verb = list(iter_jsonl(jdir / judge_key / "verbosity.jsonl"))
    if verb:
        d = [(r["holistic"], base[r["key"]]["holistic"]) for r in verb if r["key"] in base and r["holistic"] and base[r["key"]]["holistic"]]
        if d:
            arr = np.array(d, dtype=float)
            out["verbosity_n"] = len(d)
            out["verbosity_mean_drift"] = float((arr[:, 0] - arr[:, 1]).mean())
            from scipy.stats import wilcoxon

            diff = arr[:, 0] - arr[:, 1]
            out["verbosity_wilcoxon_p"] = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    if second_key:
        sec = {r["key"]: r for r in iter_jsonl(jdir / second_key / "steps.jsonl")}
        common = [k for k in sec if k in base and sec[k]["holistic"] and base[k]["holistic"]]
        if common:
            a = [base[k]["holistic"] for k in common]
            b = [sec[k]["holistic"] for k in common]
            out["second_judge_n"] = len(common)
            out["second_judge_holistic_spearman"] = spearman(a, b)
            out["second_judge_holistic_kappa_quadratic"] = cohen_kappa(a, b, weights="quadratic")
            # Positive = the Qwen judge rates Qwen policies higher than the other family does.
            out["self_preference_mean_diff"] = float(np.mean(np.array(a, float) - np.array(b, float)))
            sa, sb = [], []
            for k in common:
                for x, y in zip(base[k]["step_labels"], sec[k]["step_labels"]):
                    if x is not None and y is not None:
                        sa.append(x)
                        sb.append(y)
            if sa:
                out["second_judge_step_kappa"] = cohen_kappa(sa, sb)
    return out


# -------------------------------------------------------------- human reliability


def _pair_batches(human: pd.DataFrame, batch_a: str, batch_b: str, same_annotator: bool) -> pd.DataFrame:
    a = human[human["batch"] == batch_a]
    b = human[human["batch"] == batch_b]
    j = a.merge(b, on="key", suffixes=("_a", "_b"))
    if same_annotator:
        j = j[j["annotator_a"] == j["annotator_b"]]
    else:
        j = j[j["annotator_a"] != j["annotator_b"]]
    return j


def human_reliability(human: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if human is None or human.empty:
        return out
    if human["seconds"].notna().any():
        out["median_seconds_per_traj"] = float(human["seconds"].dropna().median())
    for name, batch, same in (("intra_rater", "retest", True), ("external", "external", False)):
        j = _pair_batches(human, "main", batch, same)
        if j.empty:
            continue
        sa, sb = [], []
        for _, r in j.iterrows():
            for x, y in zip(r["step_labels_a"], r["step_labels_b"]):
                sa.append(x)
                sb.append(y)
        out[f"{name}_n_traj"] = len(j)
        out[f"{name}_step_kappa"] = cohen_kappa(sa, sb)
        out[f"{name}_holistic_spearman"] = spearman(j["holistic_a"], j["holistic_b"])
        out[f"{name}_holistic_kappa_quadratic"] = cohen_kappa(j["holistic_a"].tolist(), j["holistic_b"].tolist(), weights="quadratic")
        out[f"{name}_error_category_kappa"] = cohen_kappa(j["error_category_a"].tolist(), j["error_category_b"].tolist())
    return out


# ------------------------------------------------------------------------ figures


def _figures(out: Path, rq1: pd.DataFrame, rq2: dict[str, Any], rq3: pd.DataFrame, rq4: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    if not rq1.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        bottom = np.zeros(len(rq1))
        colors = {"RA-GP": "#4c9a2a", "RA-BP": "#e8a33d", "WA-GP": "#5b8bd6", "WA-BP": "#c44e52"}
        for cell in CELLS:
            v = rq1[f"n_{cell}"] / rq1["n"]
            ax.bar(rq1["config_id"], v, bottom=bottom, label=cell, color=colors[cell])
            bottom += v.values
        ax.set_ylabel("fraction of trajectories")
        ax.set_title("RQ1: outcome (RA/WA) x process (GP/BP)")
        ax.legend(ncol=4, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "rq1_cells.png", dpi=150)
        plt.close(fig)
        made.append("rq1_cells.png")
    summ = rq2["summary"]
    if not summ.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, r in summ.iterrows():
            m = {"react": "o", "single_rag": "s", "closed_book": "^"}.get(r["scaffold"], "x")
            ax.scatter(r["mean_total_tokens"], r["accuracy"], marker=m, s=50)
            ax.annotate(r["config_id"], (r["mean_total_tokens"], r["accuracy"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        front = summ.set_index("config_id").loc[rq2["pareto"]]
        ax.plot(front["mean_total_tokens"], front["accuracy"], "k--", lw=1, label="Pareto frontier")
        ax.axvline(rq2["budget"], color="grey", lw=0.8, ls=":", label="matched-cost budget")
        ax.set_xscale("log")
        ax.set_xlabel("mean total tokens per question (log)")
        ax.set_ylabel("accuracy")
        ax.set_title("RQ2: accuracy vs. cost")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "rq2_pareto.png", dpi=150)
        plt.close(fig)
        made.append("rq2_pareto.png")
    if not rq3.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = rq3["judge_tokens_per_traj"].fillna(0) + 1
        ax.errorbar(x, rq3["spearman_rho_vs_human"], yerr=[rq3["spearman_rho_vs_human"] - rq3["ci_lo"], rq3["ci_hi"] - rq3["spearman_rho_vs_human"]], fmt="o")
        for xi, (_, r) in zip(x, rq3.iterrows()):
            ax.annotate(r["metric"], (xi, r["spearman_rho_vs_human"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("judge tokens per trajectory + 1 (log; deterministic metrics = 1)")
        ax.set_ylabel("Spearman ρ with human holistic score")
        ax.set_title("RQ3: validity vs. cost")
        fig.tight_layout()
        fig.savefig(out / "rq3_validity_cost.png", dpi=150)
        plt.close(fig)
        made.append("rq3_validity_cost.png")
    if not rq4.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for _, r in rq4.iterrows():
            ks = [k for k in range(1, int(r["n_seeds"]) + 1)]
            ax.plot(ks, [r[f"pass^{k}"] for k in ks], marker="o", label=r["config_id"])
        ax.set_xlabel("k")
        ax.set_ylabel("pass^k")
        ax.set_title("RQ4: pass^k (all k runs correct)")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "rq4_passk.png", dpi=150)
        plt.close(fig)
        made.append("rq4_passk.png")
    return made


# ------------------------------------------------------------------------- report


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def _md(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    d = df[cols] if cols else df
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def _is_react(c: str) -> bool:
    return GRID.get(c) is not None and GRID[c].scaffold == "react"


def hypothesis_checks(
    rq1: pd.DataFrame,
    rq2: dict[str, Any],
    rq3: pd.DataFrame,
    rq4: pd.DataFrame,
    bias: dict[str, Any],
    mech: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    react1 = rq1[rq1["config_id"].map(_is_react)] if not rq1.empty else rq1
    if not react1.empty:
        rows.append({"hypothesis": "H1a: RA-BP >= 10% of correct answers (ReAct configs)",
                     "result": f"{(react1['ra_bp_rate_among_correct'] >= 0.10).sum()}/{len(react1)} configs",
                     "supported": bool((react1["ra_bp_rate_among_correct"] >= 0.10).all())})
    if mech is not None and not mech.empty:
        m = mech[mech["config_id"].map(_is_react)]
        if not m.empty:
            ok = (m["gap"] > 0) & (m["gap_ci_lo"] > 0)
            rows.append({"hypothesis": "H1a-mech: RA-BP explained by parametric recall (closed-book gap > 0, CI excludes 0)",
                         "result": ", ".join(f"{c}={g:.2f} [{lo:.2f}, {hi:.2f}]" for c, g, lo, hi in
                                             zip(m["config_id"], m["gap"], m["gap_ci_lo"], m["gap_ci_hi"])),
                         "supported": bool(ok.all())})
        by_size = react1.assign(model=react1["config_id"].map(lambda c: GRID[c].model)).groupby("model")["ra_bp_rate_among_correct"].mean()
        by_size = by_size.reindex([m for m in POLICY_MODELS if m in by_size.index])
        mono = bool(len(by_size) >= 2 and all(np.diff(by_size.values) < 0))
        rows.append({"hypothesis": "H1b: RA-BP rate decreases with model size",
                     "result": ", ".join(f"{m}={v:.3f}" for m, v in by_size.items()), "supported": mono})
    t = rq2.get("taus", pd.DataFrame())
    if not t.empty:
        r = t[(t.ranking_a == "C_human") & (t.ranking_b == "A_at_matched_cost")]
        if r.empty:
            r = t[(t.ranking_a == "A_at_matched_cost") & (t.ranking_b == "C_human")]
        if not r.empty and not math.isnan(r["kendall_tau"].iloc[0]):
            v = r.iloc[0]
            rows.append({"hypothesis": "H2: tau(human, accuracy at matched cost) < 0.8",
                         "result": f"tau={v['kendall_tau']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]",
                         "supported": bool(v["kendall_tau"] < 0.8)})
    if not rq3.empty:
        det = rq3[rq3["metric"].isin(["gold_evidence_recall", "redundant_calls"])]["spearman_rho_vs_human"].max()
        llm = rq3[rq3["metric"] == "llm_holistic"]["spearman_rho_vs_human"]
        if len(llm) and not math.isnan(det):
            rows.append({"hypothesis": "H3a: best rule-based metric rho >= LLM step-judge rho",
                         "result": f"rule={det:.3f}, llm={llm.iloc[0]:.3f}", "supported": bool(det >= llm.iloc[0])})
    if bias:
        biased = []
        if "pairwise_consistency" in bias:
            biased.append(bias["pairwise_consistency"] < 0.9)
        if "verbosity_wilcoxon_p" in bias:
            biased.append(bias["verbosity_wilcoxon_p"] < 0.05)
        if "self_preference_mean_diff" in bias:
            biased.append(abs(bias["self_preference_mean_diff"]) > 0.25)
        if biased:
            rows.append({"hypothesis": "H3b: LLM judge shows measurable position/verbosity/self-preference bias",
                         "result": f"{sum(biased)}/{len(biased)} tests flag bias", "supported": bool(any(biased))})
    if not rq4.empty:
        col = "ratio_pass^3/pass^1"
        react4 = rq4[rq4["config_id"].map(_is_react)]
        if col in rq4.columns and not react4.empty:
            rows.append({"hypothesis": f"H4: {col.removeprefix('ratio_')} < 0.7 for every ReAct config",
                         "result": ", ".join(f"{c}={v:.2f}" for c, v in zip(react4["config_id"], react4[col])),
                         "supported": bool((react4[col] < 0.7).all())})
    return pd.DataFrame(rows)


def run_all(
    scores: pd.DataFrame,
    out_dir: str | Path,
    human: pd.DataFrame | None = None,
    judge_dir: str | Path | None = None,
    judge_key: str = "qwen3-14b-awq",
    second_judge_key: str | None = "llama3.1-8b",
    trajs: list[dict[str, Any]] | None = None,
    gold: dict[str, list[str]] | None = None,
    n_boot: int = 1000,
    cost_budget: float | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)

    rq1, mech = rq1_disagreement(scores, n_boot=n_boot)
    rq2 = rq2_rankings(scores, human, n_boot=max(100, n_boot // 2), cost_budget=cost_budget)
    judge_steps = {r["key"]: r for r in iter_jsonl(Path(judge_dir) / judge_key / "steps.jsonl")} if judge_dir else None
    corr_rows = list(iter_jsonl(Path(judge_dir) / judge_key / "correctness.jsonl")) if judge_dir else []
    corr_tok = float(np.mean([r["prompt_tokens"] + r["completion_tokens"] for r in corr_rows])) if corr_rows else None
    cpu = time_deterministic_metrics(trajs, gold) if trajs and gold else None
    rq3, step_agree = rq3_validity(scores, human, judge_steps, corr_tok, cpu, n_boot=n_boot)
    rq4, div = rq4_reliability(scores, n_boot=n_boot)
    rq5 = rq5_robustness(scores, n_boot=n_boot)
    bias = judge_bias(scores, judge_dir, judge_key, second_judge_key)
    rel = human_reliability(human)
    react_pairs = []
    for m in POLICY_MODELS:
        a, b = f"react-{m}-think", f"react-{m}-nothink"
        if a in set(scores.config_id) and b in set(scores.config_id):
            react_pairs.append((a, b))
        if b in set(scores.config_id) and f"rag-{m}" in set(scores.config_id):
            react_pairs.append((b, f"rag-{m}"))
    paired = paired_comparisons(scores, react_pairs)
    hyp = hypothesis_checks(rq1, rq2, rq3, rq4, bias, mech)

    tables = {
        "rq1_cells.csv": rq1, "rq1_mechanism.csv": mech, "rq2_config_summary.csv": rq2["summary"],
        "rq2_kendall_tau.csv": rq2["taus"], "rq2_ranks.csv": rq2["ranks"], "rq3_validity.csv": rq3,
        "rq4_passk.csv": rq4, "rq4_divergence.csv": div, "rq5_robustness.csv": rq5,
        "paired_mcnemar.csv": paired, "hypotheses.csv": hyp,
    }
    for name, t in tables.items():
        t.to_csv(out / name, index=False)
    figs = _figures(out, rq1, rq2, rq3, rq4)

    div_summary = (
        div.groupby(["config_id", "source"]).size().unstack(fill_value=0).reset_index() if not div.empty else pd.DataFrame()
    )
    lines = [
        "# Results report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
        f"{len(scores)} trajectories, {scores['config_id'].nunique()} configs, "
        f"{scores['question_id'].nunique()} questions. Correctness source: "
        f"{', '.join(f'{k}={v}' for k, v in scores['correct_source'].value_counts().items())}.",
        "",
        "## Pre-registered hypotheses",
        _md(hyp),
        "## RQ1: outcome vs. process disagreement",
        _md(rq1, ["config_id", "n", "accuracy", "grounded_rate", "n_RA-GP", "n_RA-BP", "n_WA-GP", "n_WA-BP",
                  "ra_bp_rate_among_correct", "ra_bp_ci_lo", "ra_bp_ci_hi", "wa_gp_rate_among_wrong"]),
        "Parametric-recall test (closed-book accuracy of the same model on the same question; "
        "a positive gap means RA-BP answers come from questions the model already knows):",
        "",
        _md(mech),
        "## RQ2: rankings",
        f"Matched-cost budget: {rq2['budget']:.0f} total tokens per question (trajectories over budget count as failures). "
        f"Pareto frontier: {', '.join(rq2['pareto'])}.",
        "",
        _md(rq2["summary"], ["config_id", "n", "accuracy", "em", "b_composite", "grounded", "mean_total_tokens",
                             "acc_at_budget", "human_holistic_poststrat", "n_human"]),
        _md(rq2["taus"]),
        "Paired McNemar tests (thinking on vs. off; ReAct vs. single-shot RAG):",
        "",
        _md(paired),
        "## RQ3: which automatic metric tracks human judgment",
        _md(rq3),
        f"Step-level agreement: {step_agree or 'n/a'}",
        "",
        "## RQ4: reliability (pass^k)",
        _md(rq4, [c for c in rq4.columns if not c.endswith(("_lo", "_hi"))] if not rq4.empty else None),
        "Divergence sources on questions where seeds disagree:",
        "",
        _md(div_summary),
        "## RQ5: robustness to injected tool errors",
        _md(rq5),
        "## Judge bias",
        _md(pd.DataFrame([bias])) if bias else "_no judge outputs_\n",
        "## Human-label reliability",
        _md(pd.DataFrame([rel])) if rel else "_no human labels_\n",
        "## Figures",
        *[f"![{f}]({f})" for f in figs],
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "report.md"
