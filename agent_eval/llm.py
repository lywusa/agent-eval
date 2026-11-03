"""Generation backends behind one batched interface.

- ``VLLMBackend``: the real backend on Kaggle (T4/P100, fp16 or AWQ).
- ``TransformersBackend``: slow, sequential; for debugging without vLLM.
- ``MockBackend``: deterministic scripted policy/judge for tests and CPU smoke runs.

Thinking budget: Qwen3-2504 has no native budget knob, so thinking-mode requests
are generated in two phases. Phase 1 stops at ``</think>`` or after
``thinking_budget`` tokens; if the budget ran out, Qwen's early-exit sentence and
``</think>`` are appended; phase 2 generates the visible answer.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_eval.configs import ModelSpec, SamplingConfig
from agent_eval.parsing import split_thinking
from agent_eval.prompts import THINKING_EARLY_EXIT


@dataclass
class GenRequest:
    messages: list[dict[str, Any]]
    sampling: SamplingConfig
    enable_thinking: bool = False
    thinking_budget: int | None = None
    tools: list[dict[str, Any]] | None = None
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class Generation:
    text: str
    thought: str
    visible: str
    prompt_tokens: int
    completion_tokens: int
    thought_tokens: int
    latency_ms: float
    finish_reason: str  # stop | length | context_overflow
    budget_exhausted: bool = False


class Backend:
    name = "base"
    supports_native_tools = True

    def generate(self, requests: list[GenRequest]) -> list[Generation]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# --------------------------------------------------------------------------- vLLM


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, spec: ModelSpec, seed: int = 0, enforce_eager: bool = False, **llm_kwargs: Any):
        from vllm import LLM  # imported lazily: not installed on CPU machines

        self.spec = spec
        kwargs = dict(
            model=spec.hf_id,
            dtype=spec.dtype,
            tensor_parallel_size=spec.tensor_parallel_size,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
            seed=seed,
            enforce_eager=enforce_eager,
        )
        if spec.quantization:
            kwargs["quantization"] = spec.quantization
        kwargs.update(llm_kwargs)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, req: GenRequest) -> str:
        kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if self.spec.family == "qwen3":
            kwargs["enable_thinking"] = req.enable_thinking
        if req.tools:
            kwargs["tools"] = req.tools
        return self.tokenizer.apply_chat_template(req.messages, **kwargs)

    def _params(self, s: SamplingConfig, max_tokens: int, stop: list[str], seed: int | None):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            min_p=s.min_p,
            presence_penalty=s.presence_penalty,
            max_tokens=max(1, max_tokens),
            stop=stop or None,
            include_stop_str_in_output=True,
            seed=seed,
        )

    def generate(self, requests: list[GenRequest]) -> list[Generation]:
        t0 = time.time()
        results: list[Generation | None] = [None] * len(requests)
        prompts, params, owners = [], [], []
        prompt_lens: dict[int, int] = {}
        for i, req in enumerate(requests):
            prompt = self.render(req)
            n_prompt = self.count_tokens(prompt)
            prompt_lens[i] = n_prompt
            room = self.spec.max_model_len - n_prompt
            phase1_tokens = req.thinking_budget if req.enable_thinking and req.thinking_budget else req.sampling.max_tokens
            if room < min(64, phase1_tokens):
                results[i] = Generation("", "", "", n_prompt, 0, 0, 0.0, "context_overflow")
                continue
            if req.enable_thinking and req.thinking_budget:
                p = self._params(req.sampling, min(req.thinking_budget, room), ["</think>"], req.seed)
            else:
                p = self._params(req.sampling, min(req.sampling.max_tokens, room), req.stop, req.seed)
            prompts.append(prompt)
            params.append(p)
            owners.append(i)
        outs = self.llm.generate(prompts, params, use_tqdm=False) if prompts else []

        phase2_prompts, phase2_params, phase2_owners, phase1 = [], [], [], {}
        for i, prompt, out in zip(owners, prompts, outs):
            req = requests[i]
            o = out.outputs[0]
            if req.enable_thinking and req.thinking_budget:
                text1 = o.text
                exhausted = "</think>" not in text1 and o.finish_reason == "length"
                if exhausted:
                    text1 = text1 + THINKING_EARLY_EXIT
                elif "</think>" not in text1:
                    # Ended with EOS inside thinking; nothing visible to continue.
                    phase1[i] = (text1, len(o.token_ids), False)
                    results[i] = self._finish(text1, prompt_lens[i], len(o.token_ids), len(o.token_ids), o.finish_reason, False)
                    continue
                phase1[i] = (text1, len(o.token_ids), exhausted)
                p2 = prompt + text1
                room = self.spec.max_model_len - prompt_lens[i] - self.count_tokens(text1)
                if room < 16:
                    results[i] = self._finish(text1, prompt_lens[i], len(o.token_ids), len(o.token_ids), "context_overflow", exhausted)
                    continue
                seed2 = None if req.seed is None else req.seed + 1
                phase2_prompts.append(p2)
                phase2_params.append(self._params(req.sampling, min(req.sampling.max_tokens, room), req.stop, seed2))
                phase2_owners.append(i)
            else:
                results[i] = self._finish(o.text, prompt_lens[i], len(o.token_ids), 0, o.finish_reason, False)
        if phase2_prompts:
            outs2 = self.llm.generate(phase2_prompts, phase2_params, use_tqdm=False)
            for i, out in zip(phase2_owners, outs2):
                o = out.outputs[0]
                text1, n1, exhausted = phase1[i]
                results[i] = self._finish(text1 + o.text, prompt_lens[i], n1 + len(o.token_ids), n1, o.finish_reason, exhausted)
        latency = (time.time() - t0) * 1000
        for g in results:
            assert g is not None
            g.latency_ms = latency
        return results  # type: ignore[return-value]

    @staticmethod
    def _finish(text: str, n_prompt: int, n_completion: int, n_thought: int, reason: str | None, exhausted: bool) -> Generation:
        thought, visible = split_thinking(text)
        return Generation(text, thought, visible, n_prompt, n_completion, n_thought, 0.0, reason or "stop", exhausted)


# --------------------------------------------------------------------- transformers


class TransformersBackend(Backend):
    """Sequential Hugging Face generation. Slow; use only to debug prompts without vLLM."""

    name = "transformers"

    def __init__(self, spec: ModelSpec, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.spec = spec
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float16, device_map="auto")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _gen(self, prompt: str, s: SamplingConfig, max_new: int, stop: list[str], seed: int | None) -> tuple[str, int, str]:
        torch = self.torch
        torch.manual_seed(self.seed if seed is None else seed)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kwargs: dict[str, Any] = dict(max_new_tokens=max_new, do_sample=s.temperature > 0)
        if s.temperature > 0:
            kwargs.update(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k if s.top_k > 0 else None)
        if stop:
            kwargs.update(stop_strings=stop, tokenizer=self.tokenizer)
        out = self.model.generate(**ids, **kwargs)[0][ids["input_ids"].shape[1] :]
        text = self.tokenizer.decode(out, skip_special_tokens=True)
        reason = "length" if len(out) >= max_new else "stop"
        return text, len(out), reason

    def generate(self, requests: list[GenRequest]) -> list[Generation]:
        results = []
        for req in requests:
            t0 = time.time()
            kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
            if self.spec.family == "qwen3":
                kwargs["enable_thinking"] = req.enable_thinking
            if req.tools:
                kwargs["tools"] = req.tools
            prompt = self.tokenizer.apply_chat_template(req.messages, **kwargs)
            n_prompt = self.count_tokens(prompt)
            if req.enable_thinking and req.thinking_budget:
                t1, n1, r1 = self._gen(prompt, req.sampling, req.thinking_budget, ["</think>"], req.seed)
                exhausted = "</think>" not in t1 and r1 == "length"
                if exhausted:
                    t1 += THINKING_EARLY_EXIT
                t2, n2, r2 = self._gen(prompt + t1, req.sampling, req.sampling.max_tokens, req.stop, req.seed)
                text, n_c, n_t, reason = t1 + t2, n1 + n2, n1, r2
            else:
                text, n_c, reason = self._gen(prompt, req.sampling, req.sampling.max_tokens, req.stop, req.seed)
                n_t, exhausted = 0, False
            thought, visible = split_thinking(text)
            results.append(Generation(text, thought, visible, n_prompt, n_c, n_t, (time.time() - t0) * 1000, reason, exhausted))
        return results


# ----------------------------------------------------------------------------- mock

_DOC_RE = re.compile(r"\[(d\d+)\]")


def heuristic_policy(req: GenRequest) -> str:
    """A deterministic stand-in policy: search the question, read the top two hits,
    finish with the title of the first read document. Exercises every code path."""
    msgs = req.messages
    system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    question = user.split("Question:", 1)[-1].strip()
    thought = "<think>\nI should look up the entities in the question.\n</think>\n\n" if req.enable_thinking else ""
    if "Answer: <short phrase>" in system and "Citations" not in system:
        return thought + "Answer: unknown"
    if "Citations" in system:
        ids = _DOC_RE.findall(user)
        m = re.search(r"\[(d\d+)\] ([^\n]+)\n", user)
        first_title = m.group(2) if m else "unknown"
        return thought + f"Answer: {first_title}\nCitations: {', '.join(ids[:2])}"
    tool_msgs = [m["content"] for m in msgs if m["role"] == "tool"]
    reads = [t for t in tool_msgs if t.startswith("Document [")]
    read_ids = [_DOC_RE.search(t).group(1) for t in reads]
    seen_ids: list[str] = []
    for t in tool_msgs:
        if t.startswith("Search results"):
            seen_ids += [i for i in _DOC_RE.findall(t) if i not in seen_ids]
    unread = [i for i in seen_ids if i not in read_ids]
    # Seed-dependent quirks so that runs diverge across seeds like a sampled policy.
    rng = random.Random(req.seed)
    want_reads = 1 if rng.random() < 0.25 else 2
    if not seen_ids:
        call = {"name": "search", "arguments": {"query": question}}
    elif len(read_ids) < want_reads and unread:
        pick = unread[1] if len(unread) > 1 and rng.random() < 0.2 else unread[0]
        call = {"name": "read", "arguments": {"doc_id": pick}}
    else:
        last = reads[-1].split("\n", 1)[-1].strip().rstrip(".") if reads else ""
        answer = " ".join(last.split()[-2:]) if last else "unknown"
        call = {"name": "finish", "arguments": {"answer": answer, "cited_doc_ids": read_ids}}
    return thought + "<tool_call>\n" + json.dumps(call) + "\n</tool_call>"


def heuristic_judge(req: GenRequest) -> str:
    content = req.messages[-1]["content"]
    if "CORRECT or INCORRECT" in content:
        m = re.search(r"Gold answer: (.*)\nPredicted answer: (.*)", content)
        ok = bool(m) and m.group(1).strip().lower() in m.group(2).strip().lower()
        return "CORRECT" if ok else "INCORRECT"
    if "Which trajectory" in content:
        return "A"
    n_steps = len(re.findall(r"^### Step \d+", content, re.MULTILINE))
    labels = [{"step": i + 1, "label": 1 if i % 3 == 0 else 0} for i in range(n_steps)]
    return json.dumps({"step_labels": labels, "holistic": 3, "rationale": "mock"})


class MockBackend(Backend):
    name = "mock"
    supports_native_tools = False

    def __init__(self, fn: Callable[[GenRequest], str] | None = None, spec: ModelSpec | None = None):
        self.fn = fn or heuristic_policy
        self.spec = spec

    def generate(self, requests: list[GenRequest]) -> list[Generation]:
        out = []
        for req in requests:
            text = self.fn(req)
            thought, visible = split_thinking(text)
            n_prompt = sum(self.count_tokens(str(m.get("content", ""))) for m in req.messages)
            out.append(
                Generation(text, thought, visible, n_prompt, self.count_tokens(text), self.count_tokens(thought), 1.0, "stop")
            )
        return out


def make_backend(kind: str, spec: ModelSpec | None, seed: int = 0, role: str = "policy", **kwargs: Any) -> Backend:
    if kind == "vllm":
        assert spec is not None
        return VLLMBackend(spec, seed=seed, **kwargs)
    if kind == "transformers":
        assert spec is not None
        return TransformersBackend(spec, seed=seed)
    if kind == "mock":
        return MockBackend(heuristic_judge if role == "judge" else heuristic_policy, spec)
    raise ValueError(f"unknown backend {kind!r}")
