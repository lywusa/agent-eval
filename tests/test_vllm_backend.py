"""Exercise VLLMBackend's two-phase thinking-budget logic against a stub `vllm`."""

import sys
import types
from dataclasses import dataclass

from agent_eval.configs import THINKING_SAMPLING, ModelSpec
from agent_eval.llm import GenRequest
from agent_eval.prompts import THINKING_EARLY_EXIT


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "|".join(m["content"] for m in messages) + f"|think={kw.get('enable_thinking')}|"

    def encode(self, text, add_special_tokens=False):
        return text.split()


@dataclass
class _Out:
    text: str
    token_ids: list
    finish_reason: str


class _Req:
    def __init__(self, o):
        self.outputs = [o]


class _SP:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _LLM:
    calls: list = []

    def __init__(self, **kw):
        pass

    def get_tokenizer(self):
        return _Tok()

    def generate(self, prompts, params, use_tqdm=False):
        out = []
        for p, sp in zip(prompts, params):
            _LLM.calls.append((p, sp))
            if sp.stop == ["</think>"]:
                if "LONG" in p:
                    out.append(_Req(_Out("<think>\n" + "word " * sp.max_tokens, [0] * sp.max_tokens, "length")))
                else:
                    out.append(_Req(_Out("<think>\nshort\n</think>", [0] * 3, "stop")))
            else:
                out.append(_Req(_Out('\n\n<tool_call>{"name": "search", "arguments": {"query": "q"}}</tool_call>', [0] * 9, "stop")))
        return out


def test_thinking_budget(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = _LLM, _SP
    monkeypatch.setitem(sys.modules, "vllm", fake)
    from agent_eval.llm import VLLMBackend

    be = VLLMBackend(ModelSpec("x", "x"))
    mk = lambda c, think: GenRequest([{"role": "user", "content": c}], THINKING_SAMPLING, think, 64 if think else None, seed=1, stop=["</tool_call>"])
    gens = be.generate([mk("LONG question", True), mk("short question", True), mk("plain", False)])
    long_, short, plain = gens
    assert long_.budget_exhausted and long_.thought_tokens == 64 and long_.completion_tokens == 73
    assert long_.visible.startswith("<tool_call>")
    phase2_prompts = [p for p, sp in _LLM.calls if sp.stop == ["</tool_call>"] and "LONG" in p]
    assert phase2_prompts and phase2_prompts[0].endswith(THINKING_EARLY_EXIT)
    assert not short.budget_exhausted and short.thought == "short" and short.thought_tokens == 3
    assert plain.thought_tokens == 0 and plain.visible.startswith("<tool_call>")
