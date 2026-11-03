# Annotation rubric (Family C)

Status: **DRAFT until the week-6 calibration pass.** After calibrating on 20 trajectories
(not part of the 150), set the status to FROZEN with the date and the git commit, and do
not edit it again. The same definitions are used by the LLM step judge
(`agent_eval/rubric.py`); keep the two in sync before freezing.

You see only the question, the steps (thought / action / observation) and the final
answer with its citations. You do **not** see the gold answer, the model, the config, the
seed, token counts or any automatic score. Judge the *research process*, not whether the
answer happens to be right.

## 1. Step labels (one per step)

| Label | Meaning |
|---|---|
| **+1** advances | moves toward a correct, well-supported answer: a relevant query, reading a needed document, a correct supported `finish` |
| **0** neutral | reasonable exploration that neither helps nor hurts (a plausible query that returns nothing useful, reading a document that turns out irrelevant but was a sensible choice) |
| **−1** harmful | wrong or misleading: irrelevant or repeated query, re-reading the same document, malformed call, ignoring/contradicting evidence, fabricating, finishing prematurely or with an unsupported answer |

Rules of thumb
- Label a step by what the agent could know *at that time*, not by hindsight.
- A `finish` step is +1 only if the answer follows from documents the agent actually read.
- An unparseable step is always −1.
- After an injected tool error ("HTTP 503"), a retry or a changed action is 0 or +1; giving up is −1.

## 2. Holistic process score (1–5)

| Score | Meaning |
|---|---|
| 5 | targeted queries, reads exactly the needed evidence, answer follows from it, no waste |
| 4 | sound process with minor inefficiency |
| 3 | reaches relevant evidence but with clear waste, weak use of evidence, or partial support |
| 2 | mostly irrelevant or repetitive actions; answer weakly connected to evidence |
| 1 | no meaningful research, fabricated or unsupported answer, or broken tool use |

## 3. Primary error category (one per trajectory; `none` if the process is sound)

`formatting_error` · `poor_query` · `missed_evidence` · `ignored_evidence` · `hallucination` ·
`reasoning_error` · `premature_finish` · `looping` · `answer_format` · `tool_error_unrecovered` ·
`other` (describe in notes) · `none`

Descriptions: see `agent_eval/rubric.py::ERROR_DESCRIPTIONS` (adapted from TRAIL's taxonomy,
scaled down to this task).

## 4. Procedure

1. `python -m agent_eval annotate label --batch-dir annotation/main --annotator <you>`
   (or fill `sheet.csv` while reading `trajectories.html`).
2. Never open `key.csv` or `sample_strata_PRIVATE.csv` before all batches are labelled.
3. Re-test: ≥ 14 days after finishing `main`, label `annotation/retest` without looking at
   your earlier labels.
4. External rater: 15-minute briefing using this file only, then `annotation/external`.
5. Record time per trajectory (the terminal labeler does it automatically).
