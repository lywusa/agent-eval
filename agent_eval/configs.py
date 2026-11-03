"""Frozen experiment grid (research plan §4.1, §4.3).

Edit this file only before pre-registration (week 3). After that, every change
must be recorded in ``prereg/PREREGISTRATION.md`` under "Deviations".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

Scaffold = Literal["react", "closed_book", "single_rag"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    dtype: str = "float16"  # T4 has no native bf16
    tensor_parallel_size: int = 1
    quantization: str | None = None
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    family: str = "qwen3"


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 512
    presence_penalty: float = 0.0


# Qwen's recommended sampling; greedy decoding is discouraged for Qwen3 policies.
THINKING_SAMPLING = SamplingConfig(temperature=0.6, top_p=0.95, top_k=20, max_tokens=512)
NON_THINKING_SAMPLING = SamplingConfig(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)
JUDGE_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024)


MODELS: dict[str, ModelSpec] = {
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
    "qwen3-4b": ModelSpec("qwen3-4b", "Qwen/Qwen3-4B"),
    # fp16 with TP=2 on T4x2; swap to "qwen3-8b-awq" if TP=2 is unstable.
    "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", tensor_parallel_size=2),
    "qwen3-8b-awq": ModelSpec("qwen3-8b-awq", "Qwen/Qwen3-8B-AWQ", quantization="awq"),
    # Judges.
    "qwen3-14b-awq": ModelSpec(
        "qwen3-14b-awq", "Qwen/Qwen3-14B-AWQ", quantization="awq", max_model_len=12288
    ),
    # Second-family judge for the self-preference test (200-trajectory subsample only).
    # Llama 3.1 runs in fp16 on T4 (Gemma 3 is unstable in fp16; gpt-oss MXFP4 needs newer GPUs).
    "llama3.1-8b": ModelSpec(
        "llama3.1-8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=2,
        max_model_len=12288,
        family="llama",
    ),
}

POLICY_MODELS = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
ANCHOR_MODEL = "qwen3-4b"
DEFAULT_JUDGE = "qwen3-14b-awq"
SECOND_JUDGE = "llama3.1-8b"


@dataclass(frozen=True)
class RunConfig:
    config_id: str
    model: str
    scaffold: Scaffold
    thinking: bool = False
    thinking_budget: int = 1024
    max_steps: int = 10
    token_cap: int = 16000  # completion tokens per trajectory
    top_k_search: int = 5
    max_consecutive_parse_failures: int = 3
    perturb_p: float = 0.0  # RQ5: probability a tool call returns an injected error
    sampling: SamplingConfig = field(default=NON_THINKING_SAMPLING)

    def to_dict(self) -> dict:
        return asdict(self)


def _react(model: str, thinking: bool) -> RunConfig:
    tag = "think" if thinking else "nothink"
    return RunConfig(
        config_id=f"react-{model}-{tag}",
        model=model,
        scaffold="react",
        thinking=thinking,
        sampling=THINKING_SAMPLING if thinking else NON_THINKING_SAMPLING,
    )


def _baseline(model: str, scaffold: Scaffold) -> RunConfig:
    prefix = {"closed_book": "cb", "single_rag": "rag"}[scaffold]
    return RunConfig(
        config_id=f"{prefix}-{model}",
        model=model,
        scaffold=scaffold,
        thinking=False,
        max_steps=1,
        sampling=NON_THINKING_SAMPLING,
    )


def build_grid() -> dict[str, RunConfig]:
    grid: dict[str, RunConfig] = {}
    for m in POLICY_MODELS:
        for thinking in (False, True):
            c = _react(m, thinking)
            grid[c.config_id] = c
    for m in POLICY_MODELS:
        for s in ("closed_book", "single_rag"):
            c = _baseline(m, s)
            grid[c.config_id] = c
    # RQ5 (optional, gated at week 9): one perturbation type on two configs.
    for base in ("react-qwen3-4b-nothink", "react-qwen3-8b-nothink"):
        c = grid[base]
        grid[f"{base}-perturb"] = replace(c, config_id=f"{base}-perturb", perturb_p=0.2)
    return grid


GRID = build_grid()
MAIN_CONFIGS = [c for c in GRID if not c.endswith("-perturb")]
REACT_CONFIGS = [c for c in MAIN_CONFIGS if GRID[c].scaffold == "react"]


def get_config(config_id: str) -> RunConfig:
    if config_id not in GRID:
        raise KeyError(f"unknown config {config_id!r}; known: {sorted(GRID)}")
    return GRID[config_id]


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODELS)}")
    return MODELS[key]
