# Does the Final Answer Measure the Agent? Outcome-Only vs. Trajectory-Aware vs. Human Evaluation of a Research-Assistant Agent

Semester research plan for a master's GenAI course, scoped for **one student working alone**. Open-weight models only (Qwen3 family), all compute on a single Kaggle free-tier account.

Single-student scoping in one line: 3 models × thinking on/off ReAct (6 configs) + 6 cheap baselines, N=400 questions, 3 seeds, one blinded self-annotation pass with a small external reliability check. Roughly 40 GPU-hours total against a 30 GPU-h/week quota `[COMPUTED, on guessed throughput]`.

Claim tags used throughout: `[KNOWN]` training fact (`[KNOWN, verified]` = re-checked against the primary source while drafting this plan), `[COMPUTED]`, `[INFERRED]`, `[COMMON]`, `[GUESS]`. Confidence: HIGH ≥80%, MED 50–80%, LOW 20–50%.

---

## 0. Problems with the draft, and what this plan changes

The draft is a good project. It is not yet a research design. Four objections, each of which changes the plan.

**Objection 1: the stated question has a known answer.** "Does evaluating only the final answer accurately measure agent quality?" Nobody in the literature claims it does; the ACL 2026 evaluation survey, HAL, TRACE, AgentProcessBench and TELBench all *start* from that premise `[KNOWN, verified]`. A semester paper that concludes "no" adds nothing. The question must be quantitative and decision-relevant: *how much* do outcome-only and trajectory-aware evaluations disagree, *in which direction*, *for which decisions* (ranking systems, picking a config, deploying), and *which one tracks human judgment better per unit of evaluation cost*. Section 2 restates it that way with falsifiable hypotheses.

**Objection 2: human evaluation is not a free gold standard.** TRAIL reports ~110 annotation minutes per (long) trace and MAST needed six expert annotators to reach κ=0.88 `[KNOWN, verified]`. A single student annotating trajectories produced by their own agent is the weakest link in the design, and there is no second team to blind against. The plan therefore treats human labels as a *third measurement with its own reliability estimate* (script-enforced blinding, intra-rater re-test, a small external second-annotator sample for κ), not as ground truth by fiat. Section 5.4 is explicit about how much weaker this is than a multi-annotator study.

**Objection 3: live web search kills reproducibility and inflates variance.** Results from an agent hitting the live web cannot be re-run. The plan uses a **frozen local corpus** (Section 4.2). This costs realism, which the survey lists as a gap `[KNOWN, verified]`; we accept that trade explicitly and state it as a limitation rather than pretending otherwise.

**Objection 4: "cost: very low" is true for dollars and false for the binding constraint.** Kaggle gives ~30 GPU-hours/week per account, 12-hour session cap, T4x2 (2×16 GB) or P100 (16 GB) `[KNOWN, verified]`. Qwen3 thinking mode can emit thousands of tokens per step. Token budget, not money, is the constraint, and for one student there is no pooling of quotas across accounts; Section 6 budgets it and Section 7 serializes the work so GPU runs never wait on the student.

Two more confounds the draft does not name:

- **Contamination.** HotpotQA/2Wiki/MuSiQue are almost certainly in Qwen3's pretraining data `[INFERRED, HIGH]`. A closed-book (no-tool) baseline is mandatory; otherwise "answer correct, trajectory useless" cannot be distinguished from "trajectory not needed".
- **Scaffold failures vs. reasoning failures.** With 0.6B–4B models, a large share of failures will be malformed tool calls and parsing errors `[INFERRED, HIGH]`. These must be tagged separately or they will dominate every trajectory metric and make the study about JSON formatting.

---

## 1. One-paragraph summary

Build a ReAct-style research-assistant agent (search → retrieve → read → reason → answer) over a frozen Wikipedia-paragraph corpus using Qwen3 open-weight models (1.7B, 4B, 8B; thinking on/off) on Kaggle. Run it over 400 multi-hop questions with gold supporting documents, 3 seeds per configuration, logging full trajectories. Score every trajectory three ways: (A) outcome-only (EM/F1 + reference-guided LLM judge), (B) trajectory-aware (gold-evidence recall, step efficiency, unnecessary calls, recovery after injected tool failures, evidence-grounding of the answer, step-level LLM judge), and (C) blinded single-annotator human scoring on a stratified subset, with a re-test and a small second-annotator sample to bound its reliability. The paper's contribution is not the agent; it is the measured disagreement between A, B and C, its effect on system rankings under cost control, and the reliability of each evaluator.

---

## 2. Research questions and hypotheses

**RQ1 (disagreement).** On identical trajectories, how often do outcome-only and trajectory-aware evaluations disagree, and what are the two off-diagonal populations: *right answer / bad process* (RA-BP) and *wrong answer / good process* (WA-GP)?

- H1a: RA-BP rate is non-trivial (≥10% of "correct" answers) and is driven mainly by parametric recall (correct without retrieving gold evidence) `[GUESS, LOW]`. Test: compare RA-BP rate against closed-book accuracy per question.
- H1b: RA-BP rate *decreases* with model size (larger models retrieve properly *and* know the answer), so outcome-only evaluation is *more* misleading for small models `[GUESS, LOW]`.

**RQ2 (ranking stability).** Do the three evaluation families rank the same set of agent configurations the same way?

- H2: Rankings by outcome-only accuracy and by human preference differ (Kendall τ < 0.8) once configurations are compared *at matched token cost* (Pareto framing from Kapoor et al. `[KNOWN, verified]`) `[GUESS, LOW]`.

**RQ3 (evaluator validity).** Which automated signal best predicts human trajectory judgment, and at what cost?

- H3a: Cheap deterministic trajectory metrics (gold-evidence recall, redundant-call count) correlate with human process scores at least as well as an LLM step-judge `[GUESS, LOW]`.
- H3b: Open-weight LLM judges show measurable position, verbosity and self-preference bias on trajectories, replicating Zheng et al. (2023) `[KNOWN, verified]` in the agentic setting `[INFERRED, MED]`.

**RQ4 (reliability).** How much does single-run accuracy overstate reliability?

- H4: pass^k (τ-bench definition: all k runs correct `[KNOWN, verified]`) at k=3 is < 70% of pass^1 for every configuration `[GUESS, LOW]`. (k=3 rather than 5 is a quota concession; pass^3 is a weaker test than pass^5 and the paper must say so.) Outcome-only single-run evaluation cannot see this; trajectory logs can explain *why* runs diverge.

**RQ5 (robustness, optional for one student).** Under injected tool errors, does trajectory-aware evaluation detect degradation that outcome-only evaluation misses or misattributes? Reduced to a single perturbation type on two configs; dropped entirely if week 9 arrives without RQ1–RQ4 data complete.

Every hypothesis above is tagged `[GUESS]` deliberately: they are pre-registered predictions to be written down in week 3 *before* full runs, so the paper can report which were wrong. A plan whose hypotheses all confirm is a red flag.

**Minimum viable paper (MVP).** RQ1 + RQ4 on the 4B model alone, with Family A and deterministic Family B only, is a complete, defensible short paper. Everything else is layered on top. A single student must have the MVP tables by week 9.

---

## 3. Related work (what to read, what to reuse)

Grouped by the role it plays in this project. arXiv IDs are given so the student can verify; do not cite anything here without opening it. Reading load for one person: the six items marked **core** below are required in weeks 1–2; the rest are read on demand during writing.

### 3.1 Why agent evaluation is an open problem
- **[core]** Yehudai et al., *A Survey on Evaluation of LLM-based Agents*, arXiv:2503.16416, Findings of ACL 2026 `[KNOWN, verified]`. Identifies cost-efficiency, safety, robustness, fine-grained and scalable evaluation as gaps; distinguishes reference-based vs. reference-free trajectory assessment. **Use as the framing citation and for the taxonomy of trajectory evaluators.**
- **[core]** Kapoor, Stroebl, Siegel, Nadgir, Narayanan, *AI Agents That Matter*, arXiv:2407.01502, TMLR 2025 `[KNOWN, verified]`. Cost-controlled evaluation, Pareto frontiers, simple baselines beating complex agents, holdout hygiene. **Use its Pareto methodology directly.**
- Kapoor et al., *Holistic Agent Leaderboard (HAL)*, arXiv:2510.11977, ICLR 2026 `[KNOWN, verified]`. 21,730 rollouts; finds higher reasoning effort *reduced* accuracy in most runs; LLM-aided log inspection surfaces unreported behaviors. **Directly relevant to the thinking-on/off factor and to the log-inspection methodology.**

### 3.2 Trajectory- and process-level evaluation (the "B" family)
- Zhuge et al., *Agent-as-a-Judge*, arXiv:2410.10934 `[KNOWN, verified]`. Judge agent inspects intermediate artifacts; reports alignment with human judgment on code tasks. Assumes judge can interact with the environment post hoc.
- Lù et al., *AgentRewardBench*, arXiv:2504.08942 `[KNOWN, verified]`. Benchmark for judging web-agent trajectories (success, side effects, repetition); notes rule-based evaluation *underestimates* success. **Template for our judge-vs-human agreement analysis.**
- Deshpande et al., *TRAIL*, arXiv:2505.08638 `[KNOWN, verified]`. Error taxonomy (>20 types), 148 annotated traces built on GAIA and SWE-bench, 841 errors, ~110 min/trace. **Borrow the taxonomy; scale down the annotation.**
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail? (MAST)*, arXiv:2503.13657, NeurIPS 2025 `[KNOWN, verified]`. Grounded-theory taxonomy, 14 failure modes, κ=0.88, validated LLM-judge annotator. **Model for how to build and validate an annotation scheme.**
- **[core]** *AgentProcessBench*, arXiv:2603.14465 (2026) `[KNOWN, verified]`. 1,000 tool-use trajectories, 8,509 step labels (+1/0/−1), First-Error Accuracy; finds weak policies show *inflated* correct-step ratios due to early termination. **Reuse the ternary step-label scheme and the early-termination warning.**
- **[core]** *TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents*, arXiv:2602.21230 (2026) `[KNOWN, verified]`. "High-score illusion" of Pass@1; hierarchical trajectory utility (efficiency, evidence grounding). **Closest prior work to this project's thesis; the paper must position against it.**
- Wang et al., *Where Do Deep-Research Agents Go Wrong? (TELBench / DRIFT)*, arXiv:2606.02060 (2026) `[KNOWN, verified]`. Span-level error localization; claim-centric auditing. **Optional stretch: run a simplified claim-audit as one of the B evaluators.**

### 3.3 LLM-as-judge validity
- **[core]** Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685, NeurIPS 2023 `[KNOWN, verified]`. Position bias (GPT-4 consistent only 65% under swap), verbosity bias, self-enhancement bias; mitigations: swap positions, reference-guided judging. **Replicate the swap test on trajectory judgments.**

### 3.4 Task and data
- Yang et al., *HotpotQA*, 2018 `[KNOWN]`. Distractor setting: 10 paragraphs/question, 2 gold, with sentence-level supporting-fact labels. **Primary source of gold-evidence labels for tool-use accuracy.**
- Ho et al., *2WikiMultiHopQA*, 2020; Trivedi et al., *MuSiQue*, 2022; Press et al., *Bamboogle* (in *Measuring and Narrowing the Compositionality Gap*), 2022 `[KNOWN]`. Standard multi-hop set, exactly the seven-dataset suite used by Search-R1 `[KNOWN, verified]`. MuSiQue includes unanswerable variants, useful for RQ5 `[KNOWN, MED]`.
- Krishna et al., *FRAMES*, arXiv:2409.12941 `[KNOWN, verified]`. 824 questions needing 2–15 Wikipedia articles, gold articles provided. Out of scope for one student (corpus construction alone is a week); cite as future work.
- Jin et al., *Search-R1*, arXiv:2503.09516 `[KNOWN, verified]`. Qwen2.5-3B/7B agents with a local retriever over a 2018 Wikipedia dump; outcome-only EM reward. **Reuse its retrieval setup and its multi-hop evaluation suite; its outcome-only reward is exactly the assumption this project interrogates.**
- Wei et al., *BrowseComp*, arXiv:2504.12516 `[KNOWN, verified]` and Mialon et al., *GAIA*, 2023 `[KNOWN]`. Out of scope for small models on a frozen corpus (near-zero accuracy expected `[INFERRED, HIGH]`); cite to delimit scope.

### 3.5 Reliability metric
- **[core]** Yao et al., *τ-bench*, arXiv:2406.12045 `[KNOWN, verified]`. pass^k = probability all k i.i.d. trials succeed; gpt-4o fell from >60% pass^1 to <25% pass^8. **Use pass^k for RQ4.**

### 3.6 Agent scaffold
- Yao et al., *ReAct*, 2022 `[KNOWN]`. The scaffold.
- Hugging Face `smolagents` (`ToolCallingAgent`, `CodeAgent`, local `TransformersModel`) `[KNOWN, verified]`. Acceptable, but see Section 4.3: a ~200-line hand-written loop is preferred for logging control.

---

## 4. System design

### 4.1 Models (Qwen3, Apache 2.0)

Qwen3 (April 2025) dense sizes 0.6B/1.7B/4B/8B/14B/32B plus MoE 30B-A3B and 235B-A22B; the original "2504" checkpoints support thinking and non-thinking mode in one checkpoint via `enable_thinking` or `/think` `/no_think` `[KNOWN, verified]`. The later 2507 refresh split 4B and 30B-A3B into separate `-Instruct-2507` and `-Thinking-2507` checkpoints `[KNOWN, HIGH]`. **Use the hybrid 2504 checkpoints** so that "thinking on/off" is a within-checkpoint factor, not a checkpoint change.

Qwen3.5 small models (0.8B/2B/4B/9B, Apache 2.0, March 2026, multimodal, 262K context) exist `[KNOWN, verified]`. They are an acceptable substitute if the course wants the newest weights, but they change the thinking-mode mechanics and add a vision encoder we do not use; the plan assumes Qwen3-2504 for the cleaner factorial design. Decide in week 1 and do not switch afterwards.

Kaggle hardware constraints `[KNOWN, verified]`: T4x2 = 2×16 GB, P100 = 16 GB, ~30 GPU-h/week/account, 12 h/session. T4 has no native bf16; run fp16 `[COMMON]`. Model ladder:

| Config | Precision | Fits on | Note |
|---|---|---|---|
| Qwen3-1.7B, 4B | fp16 | one T4 | policy ladder (0.6B dropped: too many parse failures to be informative `[INFERRED, MED]`; keep it only as a footnote run if quota remains) |
| Qwen3-8B | fp16 with TP=2, or official AWQ-int4 on one T4 | T4x2 | top of policy ladder |
| Qwen3-14B-AWQ | int4 | one T4 | **judge only**; 32B dropped for one student (TP=2 int4 fit is `[INFERRED, MED]` and not worth a week of debugging) |
| Second-family judge (e.g., gpt-oss-20b or a Gemma open-weight model) | native | T4x2 | self-preference test on a 200-trajectory subsample only `[KNOWN, MED]` |

Serve with vLLM (supports T4, sm_75 `[KNOWN, HIGH]`). Use Qwen's recommended sampling (thinking: T=0.6, top-p=0.95, top-k=20; non-thinking: T=0.7, top-p=0.8; never greedy) `[KNOWN, verified]`. Sampling is *required*, which is also why multi-seed runs are non-negotiable.

### 4.2 Frozen corpus and tools

**Corpus.** Pool all paragraphs from the HotpotQA distractor-setting dev set (7,405 questions × 10 paragraphs ≈ 74K paragraphs `[COMPUTED]`), deduplicated by title. This is large enough that the agent must actually search, small enough for an in-memory BM25 index (`bm25s` or `rank_bm25`), and every evaluation question has gold paragraphs *inside* the corpus. **Single-student scope: HotpotQA only.** The 400 evaluation questions are sampled from the dev set stratified by HotpotQA's `hard` level and bridge/comparison type; the remaining ~7,000 questions' paragraphs stay in the corpus as distractors. MuSiQue/2Wiki and dense retrieval are future work.

**Tools** (deterministic, logged, mockable):
- `search(query) -> [{doc_id, title, snippet}]` (top-k=5)
- `read(doc_id) -> full paragraph`
- `finish(answer, cited_doc_ids)`

`finish` requires citations. This is what makes evidence-grounding measurable without an LLM.

**Perturbation harness** (RQ5, optional): wrap tools so that with probability p a call returns an error string. Seeded and logged, so "recovered from injected failure" is a deterministic label. Empty-result and corrupted-retrieval variants are dropped from the single-student scope; the harness is a 30-line wrapper, so build it, but only run it if the week-9 gate in Section 7 is passed.

### 4.3 Agent loop

ReAct with structured tool calls, max 10 steps, hard token cap per trajectory. Write the loop by hand (~200 lines) rather than adopting a framework: the *trajectory log is the dataset of this paper* and it must be complete and stable. Each step is recorded as:

```json
{"step": 3, "thought_tokens": 412, "action": {"tool": "search", "args": {"query": "..."}},
 "observation": "...", "parse_ok": true, "perturbed": null,
 "latency_ms": 1830, "prompt_tokens": 2210, "completion_tokens": 468}
```

plus per-trajectory: `question_id, config_id, seed, final_answer, cited_doc_ids, terminated_by (finish|max_steps|parse_failure|token_cap), total_tokens, wall_ms`.

**Configurations (fixed in week 3, not pruned later):**
- Model size: 1.7B, 4B, 8B
- Thinking: on (with `thinking_budget` cap of 1,024 tokens) / off
- Scaffold: ReAct (main); plus two baselines — *closed-book* (no tools; contamination control) and *single-shot RAG* (one BM25 retrieval, then answer; the "simple baseline" Kapoor et al. insist on `[KNOWN, verified]`). Baselines run in non-thinking mode only.

That is 6 ReAct configs + 6 baseline configs = 12 configs, of which the 6 baselines cost almost nothing (one generation each). No plan-then-execute variant, no 0.6B. The 4B model is the anchor: every analysis is first completed on 4B (the MVP), then extended to 1.7B and 8B. A week-4 pilot on 50 questions × 1 seed is used only to fix parsing and prompt bugs, not to choose configs, so that the week-3 pre-registration stays honest.

---

## 5. Evaluation design (the actual contribution)

### 5.1 Family A — outcome-only
- Exact match and token-F1 against gold (standard multi-hop protocol `[COMMON]`).
- Reference-guided LLM judge (correct/incorrect given gold answer), Qwen3-14B-AWQ, non-thinking, T=0. Report agreement with EM.
- pass^1 and pass^k for k=1..3 (RQ4).

### 5.2 Family B — trajectory-aware, deterministic (no LLM required)
- **Gold-evidence recall**: fraction of gold supporting paragraphs that appeared in any `read` observation (HotpotQA supporting-fact labels).
- **Citation precision/recall**: `cited_doc_ids` vs. gold paragraphs. Answer is *grounded* if all cited docs are gold and ≥1 gold doc is cited.
- **Steps to first gold hit**; **steps after last gold hit** (proxy for unnecessary calls).
- **Redundant calls**: repeated identical/near-identical queries (Jaccard on query tokens > 0.8) or re-reading a doc.
- **Parse failure rate**; **termination reason** distribution. These separate scaffold failures from reasoning failures (Objection 4-bis).
- **Recovery rate**: given an injected tool failure at step t, did the agent issue a sensible alternative action at t+1 (retry with changed query, or read a different doc) and still finish? Rule-based first, LLM-judged second.
- **Cost**: total tokens (primary), wall-clock (secondary; only comparable within one hardware type and one session).

### 5.3 Family B′ — trajectory-aware, LLM-judged
- Step-level ternary labels (+1 advances / 0 neutral-exploratory / −1 harmful), following AgentProcessBench `[KNOWN, verified]`; report **First-Error step** and fraction of harmful steps.
- Holistic trajectory score 1–5 with a rubric (relevance of queries, use of retrieved evidence, absence of fabrication, efficiency).
- Bias controls, scaled for one student: (i) pairwise judgments with position swap on a 200-pair subsample, report consistency rate; (ii) the same 200 trajectories judged by Qwen3-14B and a second-family judge, report agreement; (iii) verbosity test on the same 200: pad thoughts with neutral filler and measure score drift. Running these on the full set is unnecessary; the bias estimate does not need N=7,200.

### 5.4 Family C — human (single annotator, with reliability bounds)

Be clear about what this is: one annotator who also built the agent. It cannot produce an inter-annotator κ on its own and the paper must not call it a gold standard. It can still be made *auditable*:

- Stratified sample: 150 trajectories covering all four A×B cells (RA-GP, RA-BP, WA-GP, WA-BP), all 6 ReAct configs, and both thinking settings; 4B over-sampled because it is the MVP anchor.
- **Script-enforced blinding**: an annotation notebook shuffles trajectories, strips `config_id`, model name, seed, token counts and all automatic scores, and presents only question, steps (thought/action/observation) and final answer. Thinking-mode trajectories are visibly longer, so blinding to that factor is imperfect `[INFERRED, HIGH]`; report it.
- Rubric: same as B′ (ternary step labels, 1–5 holistic score) plus one TRAIL-style error category per failed trajectory. Freeze the rubric in week 6 after a 20-trajectory calibration pass; do not edit it afterwards.
- **Intra-rater reliability**: re-annotate 30 randomly chosen trajectories two weeks after the first pass, without access to the first labels; report κ (step labels) and Spearman ρ (holistic score). This is the minimum reliability evidence a single-annotator study can offer.
- **External second annotator, small**: recruit one classmate, friend, or the instructor to label 40 trajectories using the frozen rubric and a 15-minute briefing; report Cohen's κ against the student's labels. If nobody is available, say so in the limitations; do not fabricate a second rater by using an LLM and calling it human.
- Budget: ~7 min per trajectory `[GUESS, LOW]` → 150 + 30 re-test ≈ 21 hours of the student's time `[COMPUTED on the guess]`, spread across weeks 7–9. The external rater's 40 trajectories are ~5 hours of their time.

### 5.5 Analyses that answer the RQs
- **RQ1:** 2×2 confusion (A-correct × B-grounded) per config; RA-BP rate vs. closed-book accuracy on the same questions (if RA-BP questions are exactly those the closed-book model already gets right, the mechanism is parametric recall).
- **RQ2:** Rank configs by (i) A accuracy, (ii) B composite, (iii) C mean score, (iv) A accuracy at matched token cost (Pareto). Kendall τ between rankings with bootstrap CIs.
- **RQ3:** Spearman correlation of each automated metric with human trajectory score; plus cost (tokens/GPU-seconds) of computing each metric. Plot validity vs. cost.
- **RQ4:** pass^k curves per config; use trajectory diffs across seeds of the same question to categorize divergence sources (query choice, parse failure, early finish).
- **RQ5 (if run):** Δaccuracy and Δgrounding under injected tool errors; recovery rate, 4B vs. 8B.
- **Power** `[COMPUTED]`: detecting a 10-point accuracy difference between two configs at ~50% baseline with α=0.05, power 0.8, unpaired, needs ≈390 questions per config; all comparisons here are paired on identical questions (McNemar / paired bootstrap), which needs fewer. N=400 is adequate for 10-point effects and underpowered for 5-point effects; the paper reports CIs, not just point estimates, so small differences show up as inconclusive rather than as false findings.

---

## 6. Compute budget

All throughput numbers below are `[GUESS, LOW]` and must be replaced by measured values in week 2; the arithmetic is `[COMPUTED]` on those guesses.

- Assume vLLM on one T4, Qwen3-4B fp16, batched agent rollouts: ~300–500 output tok/s aggregate.
- Non-thinking trajectory: ~6 steps × ~150 completion tokens ≈ 1K tokens. Thinking trajectory with a 1,024 budget: ~6 × ~1.2K ≈ 7K tokens.
- ReAct grid: 6 configs × 400 questions × 3 seeds = 7,200 trajectories. Half non-thinking (~1K) and half thinking (~7K) → ~4K average → 28.8M tokens → at 400 tok/s ≈ 20 GPU-hours.
- Baselines (closed-book, single-shot RAG): 6 configs × 400 × 3 seeds × ~300 tokens ≈ 2.2M tokens ≈ 1.5 GPU-hours.
- Judging: Family A correctness on 14,400 answers (short, ~1 h); Family B′ step-judging of 7,200 trajectories with Qwen3-14B-AWQ, ~3K prompt + ~300 generated tokens each, prefill-dominated ≈ 8 GPU-hours `[GUESS, LOW]`; bias tests and second-family judge on the 200-trajectory subsample ≈ 2 GPU-hours.
- Optional RQ5: 2 configs × 200 questions × 3 seeds × ~4K ≈ 4.8M tokens ≈ 3.5 GPU-hours.
- Optional pass^5 on the 4B anchor only: 2 extra seeds × 2 configs × 400 × ~4K ≈ 6.4M tokens ≈ 4.5 GPU-hours. Worth doing if quota remains, because it makes the MVP's RQ4 result much stronger.
- Pilot and debugging overhead (weeks 2 and 4): ~8 GPU-hours `[GUESS]`.
- **Total ≈ 40 GPU-hours core, ≈ 50 with both optionals.** One account has ~30 GPU-h/week; the run window in Section 7 is weeks 5–8, i.e. ~120 GPU-h available. **Quota is not the binding constraint for a single student at this scope; the student's serial attention is.** Thinking mode without the `thinking_budget` cap would change that (HAL found more reasoning effort reduced accuracy in most runs `[KNOWN, verified]`; the cap is a design choice, not a compromise).

Fallbacks, in order: drop 8B thinking; drop 1.7B thinking; reduce N to 300. Never drop the closed-book baseline, the 4B anchor, or the human annotation.

Practical rules for a solo operator: shard runs by `(config, seed)` into ≤10-hour Kaggle jobs and **queue them the night before**, so GPU time runs while the student writes scoring code; write JSONL to a Kaggle Dataset after every 50 trajectories (a 12-hour session kill must never lose more than 50); keep the corpus, BM25 index and model weights as Kaggle Datasets so no session re-downloads them; disable the GPU while writing analysis code (quota is charged on enabled-session wall time, not utilization `[KNOWN, MED]`); version the analysis notebooks in git from week 2, because there is no teammate to reconstruct what a notebook did.

---

## 7. Semester schedule (14 teaching weeks, one student)

The schedule is serial with one rule: **GPU runs are always queued before the student starts the next coding task**, so compute and human work overlap even with one person. Week 9 is a hard gate.

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Read the six **[core]** papers; pick Qwen3-2504 vs. Qwen3.5; upload weights + build corpus/BM25 index as Kaggle Datasets | 1-page reading memo; corpus dataset |
| 2 | Agent loop + tools + JSONL logging; measure real tok/s for 1.7B/4B/8B on T4; git repo for all notebooks | 20 trajectories end-to-end; §6 rewritten with measured numbers |
| 3 | **Pre-registration**: freeze RQs, hypotheses, metrics, N=400 question sample (seeded), 12 configs, analysis code skeleton with fake data | Pre-registration doc committed; question IDs committed |
| 4 | Pilot: 50 questions × 12 configs × 1 seed, bug-fixing only (parsing, prompts, token caps). Queue 4B main runs at end of week | Pilot report |
| 5 | 4B main runs (2 configs × 400 × 3 seeds) in background; write Family A + deterministic Family B scorers against pilot data | Scorers pass unit tests on pilot JSONL |
| 6 | 1.7B and 8B main runs in background; annotation rubric + 20-trajectory calibration; freeze rubric; MVP tables on 4B (RQ1, RQ4) | Rubric v-final; MVP tables draft |
| 7 | Baseline runs (cheap) in background; human annotation pass 1 (75 of 150) | Half of human labels |
| 8 | Family B′ judging queued in background; human annotation pass 1 (remaining 75); recruit external rater | All first-pass human labels |
| 9 | **Gate**: RQ1–RQ4 tables complete for all configs? If yes, run RQ5 and pass^5-on-4B optionals. If no, cut them and fix what is missing. Intra-rater re-test (30) starts | Gate decision recorded in repo |
| 10 | Analysis: RQ1–RQ4 (+RQ5 if run); bias tests on 200-subsample; external rater's 40 labels in; κ / ρ computed | Figures v1 |
| 11 | Full draft; request one round of feedback from the instructor or a peer in the course, using a workshop review form | Draft v1 + one written review |
| 12–13 | Revision; release code, trajectories, labels, pre-registration diff | Camera-ready draft |
| 14 | Presentation | 8-page paper (ACL/NeurIPS workshop format) + artifact |

Week 3 pre-registration is the point of the course. Hypotheses that flip after seeing data are reported as flipped. Slack: none is scheduled explicitly; the two optionals in week 9 *are* the slack, and the MVP definition in Section 2 is what a bad semester still delivers.

---

## 8. Deliverables and grading emphasis

1. **Paper** (8 pages + appendix). Graded on the experimental design and the honesty of the analysis, not on the agent's accuracy.
2. **Artifact**: code, frozen corpus recipe, all trajectories (JSONL), human labels, judge outputs, analysis notebooks. A reader must be able to recompute every table on Kaggle.
3. **Pre-registration diff**: what changed between week 3 and the final paper, and why.

Rubric weights (suggested): design and pre-registration 30%, execution and artifact 25%, analysis and statistics 25%, writing and positioning against §3 20%. For a solo project, "execution" is graded against the pre-registered scope and the week-9 gate decision, not against the full wish-list: a student who cut RQ5 at the gate and documented why has executed correctly.

---

## 9. Expected failure modes of the project itself

- Small models never produce a valid tool call → the study becomes about parsing. Mitigation: constrained decoding / JSON schema enforcement in vLLM; report parse failures separately; keep 4B as the anchor model.
- Contamination makes closed-book accuracy high → trajectory metrics look useless. That is a finding, not a failure; but sample the 400 questions from HotpotQA's `hard` level and over-weight bridge questions to keep headroom `[INFERRED, MED]`.
- Intra-rater κ is low or the external rater disagrees → the human "gold standard" collapses. Report it; RQ3 then becomes "no evaluator is reliable, including this human," which is publishable if measured properly and fatal if hidden.
- Single-annotator bias toward the annotator's own agent → the strongest threat in this design. Mitigations already in §5.4 (script blinding, frozen rubric, re-test, external sample). What cannot be mitigated is stated in the limitations.
- Quota exhaustion → follow the §6 fallback order; never silently reduce seeds without documenting it.
- Judge = policy family → self-preference. Mitigated by the second-family judge on the subsample; if unavailable, state it as a limitation.
- **Single point of failure.** Illness or a lost fortnight has no teammate to absorb it. Mitigations: the MVP definition (§2), the week-9 gate (§7), nightly JSONL checkpoints to Kaggle Datasets, and everything in git. A semester that ends with only the 4B MVP tables and an honest limitations section is still a pass; a semester that ends with 12 configs half-run and no analysis is not.
- Scope creep is the solo-student default failure `[COMMON]`. Section 10 exists so that ideas go into a list instead of into the codebase.

---

## 10. Stretch directions (parking lot; none of these are expected from one student in one semester)

- Second datasets (MuSiQue, 2Wiki, FRAMES) and the 0.6B / 14B policy models cut from this scope.
- Full perturbation suite (empty results, corrupted retrieval, unanswerable questions).
- Simplified claim-centric audit (DRIFT-style `[KNOWN, verified]`) as an additional B′ evaluator.
- Use trajectory-aware scores as a reward signal for best-of-n selection at inference and test whether it beats outcome-only selection (AgentProcessBench reports process signals help test-time scaling `[KNOWN, verified]`).
- Replace BM25 with the Search-R1 E5/Wikipedia-2018 retriever and check whether conclusions survive a corpus change.
- Swap in Qwen3.5-4B/9B and test whether the A/B disagreement pattern is model-generation-stable.

---

## 11. Minimal Kaggle bootstrap (illustrative)

```python
# Kaggle notebook, accelerator: GPU T4 x2. Internet on for first run only; then attach as Datasets.
!pip -q install vllm bm25s datasets

from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen3-4B", dtype="float16", tensor_parallel_size=1,
          max_model_len=8192, gpu_memory_utilization=0.9)
non_thinking = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)

# Chat template with thinking disabled for the policy; flip enable_thinking for the factor.
msgs = [{"role": "system", "content": SYSTEM_PROMPT_WITH_TOOL_SCHEMA},
        {"role": "user", "content": question}]
prompt = llm.get_tokenizer().apply_chat_template(
    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = llm.generate([prompt], non_thinking)[0].outputs[0]
# parse tool call -> execute -> append observation -> loop (max 10 steps) -> log JSONL
```

`enable_thinking` in `apply_chat_template` and the `/think` `/no_think` soft switches are the documented mechanisms `[KNOWN, verified]`. Whether `max_model_len=8192` with fp16 4B leaves enough KV cache for batching on a 16 GB T4 is `[INFERRED, MED]`; measure in week 2.
