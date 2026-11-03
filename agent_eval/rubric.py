"""Shared rubric for the LLM step judge (Family B′) and human annotation (Family C).

Freeze after the week-6 calibration pass; ``annotation/RUBRIC.md`` is the
human-readable version of the same definitions.
"""

STEP_LABELS = {
    1: "advances: the step moves toward a correct, well-supported answer "
    "(a relevant query, reading a needed document, a correct supported finish)",
    0: "neutral: reasonable exploration that neither helps nor hurts",
    -1: "harmful: a wrong or misleading action (irrelevant or repeated query, "
    "malformed call, ignoring or contradicting evidence, fabricating, finishing prematurely "
    "or with an unsupported answer)",
}

HOLISTIC = {
    5: "excellent: targeted queries, reads exactly the needed evidence, answer follows from it, no waste",
    4: "good: sound process with minor inefficiency",
    3: "mixed: reaches relevant evidence but with clear waste, weak use of evidence, or partial support",
    2: "poor: mostly irrelevant or repetitive actions; answer weakly connected to evidence",
    1: "very poor: no meaningful research, fabricated or unsupported answer, or broken tool use",
}

ERROR_CATEGORIES = [
    "none",
    "formatting_error",
    "poor_query",
    "missed_evidence",
    "ignored_evidence",
    "hallucination",
    "reasoning_error",
    "premature_finish",
    "looping",
    "answer_format",
    "tool_error_unrecovered",
    "other",
]

ERROR_DESCRIPTIONS = {
    "none": "no process error",
    "formatting_error": "malformed or unparseable tool call",
    "poor_query": "search queries irrelevant, too broad, or missing the key entity",
    "missed_evidence": "a needed document appeared in results but was never read",
    "ignored_evidence": "evidence was read but the answer contradicts or ignores it",
    "hallucination": "answer or claims not supported by any observation",
    "reasoning_error": "wrong bridge entity / wrong comparison despite the right evidence",
    "premature_finish": "finished before gathering the evidence needed",
    "looping": "repeated identical or near-identical actions",
    "answer_format": "right content in the wrong form (sentence, extra words)",
    "tool_error_unrecovered": "did not respond sensibly to a tool error",
    "other": "anything else (describe in notes)",
}


def rubric_text() -> str:
    lines = ["Step labels:"]
    lines += [f"  {k:+d} = {v}" for k, v in STEP_LABELS.items()]
    lines.append("Holistic process score (1-5):")
    lines += [f"  {k} = {v}" for k, v in HOLISTIC.items()]
    return "\n".join(lines)
