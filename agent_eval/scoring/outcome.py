"""Family A: outcome-only metrics (EM, token F1, pass^k)."""

from __future__ import annotations

import re
import string
from collections import Counter
from math import comb


def normalize_answer(s: str) -> str:
    """Official HotpotQA/SQuAD normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # HotpotQA rule: yes/no/noanswer must match exactly.
    if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def pass_hat_k(n_correct: int, n_trials: int, k: int) -> float:
    """τ-bench pass^k: unbiased estimate of P(all k i.i.d. trials succeed) = C(c,k)/C(n,k)."""
    if k > n_trials:
        raise ValueError("k must be <= number of trials")
    return comb(n_correct, k) / comb(n_trials, k)


def pass_at_k(n_correct: int, n_trials: int, k: int) -> float:
    """pass@k: P(at least one of k trials succeeds) = 1 - C(n-c,k)/C(n,k)."""
    if k > n_trials:
        raise ValueError("k must be <= number of trials")
    return 1.0 - comb(n_trials - n_correct, k) / comb(n_trials, k)
