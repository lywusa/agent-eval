# Pre-registration (week 3)

Fill in the bracketed fields, commit, and tag the commit `prereg-v1` **before** any main
run (pilot runs are allowed only for bug-fixing). Afterwards, never edit sections 1–6;
append changes to section 7.

- Date: [YYYY-MM-DD]
- Git commit of the code used for all main runs: [sha]
- `data/manifest.json` `eval_fingerprint`: [16 hex chars] (400 questions, seed 2026, 320 bridge / 80 comparison)
- `data/manifest.json` `pilot_fingerprint`: [16 hex chars] (50 disjoint pilot questions)
- Model family decision: Qwen3-2504 hybrid checkpoints (not Qwen3.5)
- Measured throughput (week 2, `agent-eval bench`): 1.7B [..] tok/s, 4B [..] tok/s, 8B [..] tok/s

## 1. Research questions and hypotheses

| ID | Hypothesis | Test | Decision rule |
|---|---|---|---|
| H1a | RA-BP ≥ 10% of correct answers | `rq1_cells.csv`: `ra_bp_rate_among_correct` per ReAct config | supported if the point estimate ≥ 0.10 for every ReAct config |
| H1a-mech | RA-BP is driven by parametric recall | `rq1_mechanism.csv`: closed-book accuracy gap (RA-BP − RA-GP) | supported if gap > 0 and its 95% CI excludes 0 |
| H1b | RA-BP rate decreases with model size | mean RA-BP rate per model size | supported if strictly decreasing 1.7B > 4B > 8B |
| H2 | Human ranking ≠ accuracy-at-matched-cost ranking | Kendall τ over the 6 ReAct configs (`rq2_kendall_tau.csv`) | supported if τ < 0.8 |
| H3a | Rule-based metrics ≥ LLM step judge in predicting human score | Spearman ρ with human holistic (`rq3_validity.csv`) | supported if max(ρ gold-evidence recall, ρ redundant calls) ≥ ρ LLM holistic |
| H3b | LLM judge biases are measurable | swap consistency, verbosity drift, cross-family difference | supported if any: consistency < 0.9, verbosity Wilcoxon p < 0.05, \|self-preference diff\| > 0.25 |
| H4 | pass^3 < 0.7 × pass^1 | `rq4_passk.csv` | supported if the ratio < 0.7 for every ReAct config |
| H5 (optional) | Trajectory metrics detect injected-error degradation that outcome misses | `rq5_robustness.csv`: `hidden_degradation_rate`, Δgrounded | descriptive; run only if the week-9 gate passes |

## 2. Primary definitions
- **Correct (Family A)**: Qwen3-14B-AWQ reference-guided judge verdict (`--correct-def auto`); EM and F1 reported alongside, judge–EM κ reported.
- **Good process (Family B)**: `grounded` = ≥ 1 cited doc, all cited docs gold, ≥ 1 gold doc cited.
- **B composite**: mean of gold-evidence recall, grounded (0/1), 1 − redundant-call rate.
- **Matched cost**: budget = median `total_tokens` of all ReAct trajectories; a trajectory over budget counts as a failure.
- **Human score**: holistic 1–5 from the `main` batch; per-config means are post-stratified to each config's true cell distribution.
- **CIs**: 95% percentile bootstrap resampling questions (1,000 resamples).

## 3. Grid (frozen; see `agent-eval configs`)
3 models (Qwen3-1.7B, 4B, 8B) × ReAct thinking on/off (6 configs) + closed-book and single-shot RAG baselines per model (6 configs) = 12 configs × 400 questions × 3 seeds. Sampling: Qwen defaults; thinking budget 1,024 tokens; max 10 steps; 16,000 completion-token cap; BM25 top-5.

## 4. Human evaluation
150 stratified trajectories (4 cells × 6 ReAct configs; 4B weighted ×2), 30-item re-test after ≥ 14 days, 40-item external rater. Rubric frozen in week 6 (`annotation/RUBRIC.md`).

## 5. Fallback order if quota runs out
Drop 8B thinking → drop 1.7B thinking → N = 300. Never drop closed-book, the 4B anchor, or human annotation.

## 6. Minimum viable paper
RQ1 + RQ4 on the 4B model with Family A and deterministic Family B.

## 7. Deviations (append-only)

| Date | Change | Reason | Affects which result |
|---|---|---|---|
| | | | |
