"""Week-2 throughput measurement: replace the guessed tok/s in research plan §6
with measured numbers, and project GPU-hours for the full grid."""

from __future__ import annotations

import time
from typing import Any

from agent_eval.configs import NON_THINKING_SAMPLING, THINKING_SAMPLING, SamplingConfig, get_model
from agent_eval.llm import GenRequest, make_backend

BENCH_Q = [
    "Which magazine was started first, Arthur's Magazine or First for Women?",
    "The Oberoi family is part of a hotel company that has a head office in what city?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
]


def measure(model_key: str, backend_kind: str = "vllm", batch_sizes: tuple[int, ...] = (1, 16, 64), max_tokens: int = 256, thinking: bool = False) -> list[dict[str, Any]]:
    spec = get_model(model_key)
    backend = make_backend(backend_kind, spec)
    base = THINKING_SAMPLING if thinking else NON_THINKING_SAMPLING
    sampling = SamplingConfig(base.temperature, base.top_p, base.top_k, max_tokens=max_tokens)
    rows = []
    for bs in batch_sizes:
        reqs = [
            GenRequest(
                messages=[{"role": "user", "content": BENCH_Q[i % len(BENCH_Q)] + " Explain step by step."}],
                sampling=sampling,
                enable_thinking=thinking,
                thinking_budget=max_tokens if thinking else None,
                seed=i,
            )
            for i in range(bs)
        ]
        backend.generate(reqs[:1])  # warm-up
        t0 = time.time()
        gens = backend.generate(reqs)
        dt = time.time() - t0
        toks = sum(g.completion_tokens for g in gens)
        rows.append({"model": model_key, "thinking": thinking, "batch": bs, "completion_tokens": toks, "seconds": round(dt, 2), "tok_per_s": round(toks / dt, 1)})
        print(rows[-1])
    return rows


def project_gpu_hours(tok_per_s: float, n_questions: int = 400, seeds: int = 3, tokens_nothink: int = 1000, tokens_think: int = 7000) -> dict[str, float]:
    """Research plan §6 arithmetic with measured throughput. Token counts per
    trajectory should be replaced by pilot medians."""
    per_model = n_questions * seeds * (tokens_nothink + tokens_think)
    react = 3 * per_model
    baselines = 6 * n_questions * seeds * 300
    return {
        "react_tokens": react,
        "baseline_tokens": baselines,
        "react_gpu_hours": react / tok_per_s / 3600,
        "baseline_gpu_hours": baselines / tok_per_s / 3600,
    }
