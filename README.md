# agent-eval

**Does the final answer measure the agent?** This repo compares outcome-only, trajectory-aware and human evaluation of a ReAct research-assistant agent. It uses Qwen3 open-weight models on HotpotQA and runs on Kaggle's free GPUs.

The repo contains the complete code for the study in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) and [`proposal/proposal.tex`](proposal/proposal.tex) (CS 6180 project proposal):

| Stage | Plan § | Module | Command |
|---|---|---|---|
| Frozen corpus, BM25 index, 400-question eval sample + 50-question pilot | 4.2 | `agent_eval/data.py`, `retrieval.py` | `agent-eval data` |
| Tools `search` / `read` / `finish` + injected-error wrapper (RQ5) | 4.2 | `tools.py` | – |
| Hand-written ReAct loop with full step logging, Qwen3 thinking on/off + 1,024-token thinking budget | 4.3 | `agent.py`, `llm.py`, `parsing.py`, `prompts.py` | `agent-eval run` |
| Closed-book and single-shot RAG baselines | 4.3 | `baselines.py` | `agent-eval run` |
| Sharded, resumable runs with checkpoints every 50 questions | 6 | `runner.py` | `agent-eval run` |
| Throughput benchmark + GPU-hour projection | 6, week 2 | `bench.py` | `agent-eval bench` |
| **Family A**: EM, F1, reference-guided LLM judge, pass^k | 5.1 | `scoring/outcome.py`, `judge.py` | `agent-eval judge correctness` |
| **Family B**: gold-evidence recall, citation P/R, grounding, redundancy, parse failures, recovery, cost | 5.2 | `scoring/trajectory.py` | `agent-eval score` |
| **Family B′**: LLM step labels (+1/0/−1), holistic 1–5, position / verbosity / second-family bias tests | 5.3 | `judge.py`, `rubric.py` | `agent-eval judge steps\|pairwise\|verbosity` |
| **Family C**: stratified sample, script-enforced blinding, terminal labeler, spreadsheet import, re-test and external batches | 5.4 | `annotation.py` | `agent-eval annotate …` |
| RQ1–RQ5 analyses, bootstrap CIs, Kendall τ, Pareto, κ, McNemar, hypothesis table, figures, `report.md` | 5.5 | `analysis.py`, `stats.py` | `agent-eval analyze` |
| Fake data for the week-3 "analysis skeleton", CPU smoke test | 7 | `synthetic.py`, `smoke.py` | `agent-eval synth`, `agent-eval smoke` |

Everything except the GPU stages runs on a laptop CPU. The whole pipeline (data build → agent loop → judges → scoring → annotation export → analysis) is tested end to end on CPU with a mock model (`pytest`, `agent-eval smoke`). The vLLM path has only been tested against a stub. **Your first GPU session (week 2) must confirm it on a real T4**; see [Kaggle notes](#kaggle-notes).

---

## 1. Where the plan and the proposal disagree, and what the code does

The proposal is a shortened version of the plan. Where they differ, the code follows the more specific and more defensible option:

| Topic | Research plan | Proposal | Implemented choice |
|---|---|---|---|
| Configurations ranked in RQ2 | 12 configs (6 ReAct + 6 baselines) | "the 6 configurations" | Rank correlations (Kendall τ) are computed over the **6 ReAct configs**, the only ones with human scores. All 12 appear in the accuracy and Pareto tables. |
| Retriever | BM25 over the HotpotQA paragraph corpus | "reproducible local retriever, which I reuse" from Search-R1 (E5 over Wikipedia-2018) | **BM25** on the frozen HotpotQA corpus: deterministic, CPU-only, and every gold paragraph is guaranteed to be in the corpus. Search-R1's retriever stays a stretch goal (plan §10). |
| H1 | H1a (≥10% RA-BP, parametric recall) and H1b (decreases with size) | merged into one H1 | Kept separate: H1a, H1a-mech and H1b, each with its own decision rule (`prereg/PREREGISTRATION.md`). |
| H3 | H3a (rules ≥ LLM judge) and H3b (judge bias) | merged | Kept separate: H3a and H3b. |
| RQ5 (robustness) | optional, gated at week 9 | not mentioned | Implemented (`*-perturb` configs, 20% injected 503 errors on 4B and 8B non-thinking). Run it only if the week-9 gate passes. |
| "Correct" answer | EM/F1 plus LLM judge | "outcome-only score" | Primary = **Qwen3-14B-AWQ reference-guided judge** (small models' answers are often correct but not exact-match). EM and F1 are always reported, together with judge–EM κ. Switch with `--correct-def em`. |
| Qwen3 vs Qwen3.5 | decide in week 1 | Qwen3 | **Qwen3-2504 hybrid checkpoints**, so thinking on/off happens within one checkpoint. |
| Second-family judge | "gpt-oss-20b or a Gemma model" | – | **Llama-3.1-8B-Instruct** (fp16, TP=2). gpt-oss's MXFP4 weights and Gemma-3's fp16 instability are both risky on T4. Change `SECOND_JUDGE` in `configs.py` if you prefer another. |
| Question sample | "hard level, over-weight bridge" | 400 questions | The dev set is 100% `hard` and ~80% bridge, so the sample is a seeded 320 bridge / 80 comparison split (80 comparison questions are enough for a subgroup CI). It excludes the 63 dev questions whose gold paragraph has conflicting text variants. |

Other decisions the plan left open, all pre-registerable in `prereg/PREREGISTRATION.md`:
- **"Good process"** for the 2×2 table means `grounded`: at least one doc cited, every cited doc is gold, and at least one gold doc is cited. `grounded_strict` also requires the cited docs to have been read.
- **B composite** (for RQ2) is the mean of gold-evidence recall, grounded, and 1 − redundant-call rate.
- **Matched cost** uses a budget equal to the median `total_tokens` of all ReAct trajectories; any trajectory over budget counts as a failure. Override with `--cost-budget`.
- **Human per-config score** is post-stratified. The 150-item sample over-represents rare cells (RA-BP, WA-GP), so each config's mean is re-weighted to that config's real cell distribution.
- **Recovery after an injected error** is reported two ways: `recovery_sensible` (any valid non-finish action, including a retry) and `recovery_alternative` (the plan's stricter rule: a changed query or a different doc).
- **The pilot uses 50 questions disjoint from the 400**, so prompt fixes made during the pilot cannot overfit the evaluation set.

---

## 2. Install

```bash
git clone https://github.com/lywusa/agent-eval.git && cd agent-eval
pip install -e ".[dev]"          # CPU: data, scoring, judges-with-mock, annotation, analysis
pytest -q                        # 14 tests, ~10 s
agent-eval smoke --out smoke     # full pipeline on CPU with a mock model; open smoke/results/report.md
```

On a GPU machine, also install `pip install -e ".[gpu]"`, or pin vLLM as described in [Kaggle notes](#kaggle-notes). `python -m agent_eval <cmd>` is equivalent to `agent-eval <cmd>`.

Repository layout:

```
agent_eval/        package (see table above)
kaggle/            notebook scripts: 01 data (CPU), 02 policy runs (GPU), 03 judges (GPU), 04 score/annotate/analyze (CPU)
prereg/            PREREGISTRATION.md (week 3) and GATE.md (week 9) templates
annotation/        RUBRIC.md (+ batches created by `agent-eval annotate`, git-ignored until done)
tests/             unit + end-to-end tests
presentation/      week-2 progress deck with speaker notes; rebuild with `python3 presentation/build_deck.py`
RESEARCH_PLAN.md   full semester plan;  proposal/  course proposal
```

---

## 3. Step-by-step guide (follows the plan's §7 schedule)

Default paths are `data/`, `runs/`, `judge/`, `annotation/`, `results/`. Every command takes `--help`.

### Week 1: data (CPU, once)

```bash
agent-eval data --download --out data          # ~5 s after download
cat data/manifest.json
```
- This downloads the HotpotQA dev distractor set. If the official CMU host is down, it falls back to the identical Hugging Face parquet. A local file also works: `--hotpot path/to/hotpot_dev_distractor_v1.json`.
- It writes `corpus.jsonl` (66,581 deduplicated paragraphs), `bm25.pkl`, `questions_all.jsonl`, `questions_eval.jsonl` (400, seed 2026), `questions_pilot.jsonl` (50, disjoint) and `manifest.json` with SHA fingerprints of both samples.
- On Kaggle, run `kaggle/01_build_data.py` (CPU session) and save `/kaggle/working/data` as the Dataset **`agent-eval-data`**.
- Sanity check from development: querying BM25 with the raw question puts at least one gold paragraph in the top 5 for ~93% of eval questions, but both for only ~51% (~48% of bridge questions; mean gold recall@5 ≈ 0.72). The second hop really requires searching.

### Week 2: measure throughput and confirm the agent on GPU

On Kaggle (T4 x2):
```bash
agent-eval bench --model qwen3-4b --batch-sizes 1,16,64           # also --thinking, and qwen3-1.7b / qwen3-8b
agent-eval run --configs react-qwen3-4b-nothink,react-qwen3-4b-think --seeds 0 \
    --questions data/questions_pilot.jsonl --runs pilot_runs --limit 20
```
- `bench` prints tok/s and projects GPU-hours for the full grid. Put the measured numbers into `RESEARCH_PLAN.md` §6 and `prereg/PREREGISTRATION.md`.
- Read 5–10 trajectories in `pilot_runs/<config>/seed0.jsonl`. Each line is one trajectory; the step format is in [§5](#5-data-formats). Check `parse_ok`, `terminated_by`, `thinking_budget_exhausted` and the token counts.

### Week 3: pre-registration

1. Run the analysis on fake data so that every table and figure exists before you see real data:
   ```bash
   agent-eval synth --out fake
   agent-eval score   --runs fake/runs --questions fake/data/questions_eval.jsonl --judge-dir fake/judge --out fake/results/scores.csv
   agent-eval analyze --scores fake/results/scores.csv --annotation fake/annotation --judge-dir fake/judge \
                      --runs fake/runs --questions fake/data/questions_eval.jsonl --out fake/results
   ```
   The numbers are meaningless; the point is that `fake/results/report.md` shows every pre-registered test.
2. Fill in `prereg/PREREGISTRATION.md` (fingerprints from `data/manifest.json`, code commit, throughput), commit, and `git tag prereg-v1`.
3. From now on, any change to `configs.py`, `prompts.py`, scoring definitions or the question sample goes into its "Deviations" table.

### Week 4: pilot (bug-fixing only)

```bash
# one Kaggle session per model; 50 pilot questions × 1 seed × 4 configs of that model
agent-eval run --configs react-qwen3-4b-nothink,react-qwen3-4b-think,cb-qwen3-4b,rag-qwen3-4b \
    --seeds 0 --questions data/questions_pilot.jsonl --runs pilot_runs
agent-eval score --runs pilot_runs --questions data/questions_pilot.jsonl --judge-dir none --out pilot/scores.csv
```
Look at `parse_failure_rate`, `terminated_by` and the share of `thinking_budget_exhausted`. Fix parsing and prompt bugs only, and do not choose configs from pilot accuracy. If 1.7B parse failures dominate, keep them; they are reported separately (plan objection 4-bis). Re-tag if the code changed: `prereg-v1.1` plus a deviation entry.

### Weeks 5–6: main policy runs (GPU, in the background)

One Kaggle session per model (`kaggle/02_policy_runs.py`, "Save & Run All (Commit)"):

| Session | `--configs` | Seeds |
|---|---|---|
| 4B (MVP anchor, first) | `react-qwen3-4b-nothink,react-qwen3-4b-think` | `0,1,2` |
| 1.7B | `react-qwen3-1.7b-nothink,react-qwen3-1.7b-think` | `0,1,2` |
| 8B | `react-qwen3-8b-nothink,react-qwen3-8b-think` | `0,1,2` |
| Baselines (cheap; one session per model or appended to the above) | `cb-qwen3-4b,rag-qwen3-4b` (etc.) | `0,1,2` |

```bash
agent-eval run --configs react-qwen3-4b-nothink,react-qwen3-4b-think --seeds 0,1,2 \
    --questions data/questions_eval.jsonl --data data --runs runs --chunk 50 --max-hours 10.5
```
- Output: `runs/<config_id>/seed<k>.jsonl` plus `seed<k>.meta.json` (config, model id, question fingerprint, timestamps).
- **Checkpoint and resume.** Trajectories are fsynced after every 50 questions, and `--max-hours` stops cleanly before Kaggle's 12-hour kill. Re-running the same command skips finished questions. Between sessions, save `/kaggle/working/runs` as a new version of the Dataset **`agent-eval-runs`**; the notebook copies it back in to resume. Optionally pass `--sync-cmd "kaggle datasets version -p /kaggle/working/runs -m ckpt --dir-mode zip"` to push after every chunk. This needs a Kaggle API token as a secret and a `dataset-metadata.json` in `/kaggle/working/runs` naming the existing `agent-eval-runs` dataset.
- If quota runs short, follow the plan's fallback order (drop 8B thinking, then 1.7B thinking, then N=300) and log it in the deviations table.
- If TP=2 for 8B is unstable, use `qwen3-8b-awq`: edit `RunConfig.model` for the 8B configs and record a deviation.
- Week 5 (CPU, while the GPU runs): `agent-eval score` on the first finished shards (no judge yet; `correct` falls back to EM).
- **MVP tables (week 6):** as soon as the 4B runs are done, `agent-eval score` + `agent-eval analyze` give RQ1 and RQ4 for 4B.

### Week 6: rubric calibration

1. Edit `annotation/RUBRIC.md` and `agent_eval/rubric.py` together.
2. Calibrate on 20 trajectories that will not be in the main sample, for example by exporting a throwaway batch from the pilot runs:
   ```bash
   agent-eval score --runs pilot_runs --questions data/questions_pilot.jsonl --judge-dir none --out pilot/scores.csv
   agent-eval annotate sample --scores pilot/scores.csv --runs pilot_runs --out pilot/annotation --batch calib --n 20
   agent-eval annotate label --batch-dir pilot/annotation/calib --annotator me
   ```
3. Mark the rubric FROZEN (date + commit). The LLM step judge uses the same `rubric.py` text, so run `judge steps` only after freezing.

### Weeks 7–8: judges (GPU, in the background) and human annotation (you)

**Judges** (`kaggle/03_judges.py`; Qwen3-14B-AWQ fits on one T4):
```bash
agent-eval judge correctness --judge qwen3-14b-awq          # all trajectories (Family A)
agent-eval judge steps       --judge qwen3-14b-awq          # all ReAct trajectories (Family B′)
agent-eval judge pairwise    --judge qwen3-14b-awq --n-pairs 200                # position-swap test
agent-eval judge verbosity   --judge qwen3-14b-awq --subsample 200 --pad 3      # filler-padding test
agent-eval judge steps       --judge llama3.1-8b   --subsample 200              # second-family judge
```
Outputs go to `judge/<judge>/<kind>.jsonl`. All judge commands resume and skip done keys. Judges only see the blinded rendering, with no config, model, seed or token counts. `verbosity` and the second judge use the same seeded 200-trajectory subsample.

**Score table** (CPU):
```bash
agent-eval score --runs runs --questions data/questions_eval.jsonl --judge-dir judge --out results/scores.csv
```
This writes one row per trajectory: Family A (`em`, `f1`, `judge_correct`, `correct`), every Family B metric, the B′ judge columns, cost, and the `cell` (RA-GP / RA-BP / WA-GP / WA-BP).

**Human annotation** (CPU, blinded):
```bash
agent-eval annotate sample --scores results/scores.csv --runs runs --n 150     # -> annotation/main/
agent-eval annotate label  --batch-dir annotation/main --annotator me          # resumable; q to stop
agent-eval annotate status
```
- `annotation/main/` contains:
  - `items.jsonl`: what the labeler shows, shuffled under anonymous IDs.
  - `trajectories.html`: a read-only view of the same items.
  - `sheet.csv`: a spreadsheet template.
  - `key.csv` and `sample_strata_PRIVATE.csv`: **do not open these until all labelling is done.**
- The sample is stratified over 4 cells × 6 ReAct configs, with 4B weighted ×2. If a cell has too few trajectories, its share is redistributed.
- To label in a spreadsheet instead, fill `sheet.csv` (with `step_labels` such as `1,0,-1`) and run `agent-eval annotate import --batch-dir annotation/main --sheet sheet.csv --annotator me`. The import validates every row.
- The external rater gets their own batch: `agent-eval annotate external --main annotation/main --n 40`. Send them `annotation/external/trajectories.html`, `sheet.csv` and `annotation/RUBRIC.md`. Import their sheet with `--annotator ext1`.

### Week 9: gate

Fill in `prereg/GATE.md`. If the gate passes:
```bash
# RQ5: injected tool errors (p=0.2) on two configs
agent-eval run --configs react-qwen3-4b-nothink-perturb --seeds 0,1,2 --limit 200
agent-eval run --configs react-qwen3-8b-nothink-perturb --seeds 0,1,2 --limit 200
# pass^5 on the 4B anchor: two extra seeds
agent-eval run --configs react-qwen3-4b-nothink,react-qwen3-4b-think --seeds 3,4
```
`analyze` picks up the extra seeds automatically (pass^1…pass^5); H4 stays tested on pass^3 as pre-registered.

Start the intra-rater re-test at least 14 days after finishing `main`:
```bash
agent-eval annotate retest --main annotation/main --n 30          # fresh anon IDs and order
agent-eval annotate label --batch-dir annotation/retest --annotator me
```

### Weeks 10–13: analysis and writing

```bash
agent-eval score   --runs runs --questions data/questions_eval.jsonl --judge-dir judge --out results/scores.csv
agent-eval analyze --scores results/scores.csv --annotation annotation --judge-dir judge \
                   --runs runs --questions data/questions_eval.jsonl --out results
```
`results/report.md` collects everything; the individual tables are in `results/*.csv`:

| File | Answers | Key columns |
|---|---|---|
| `hypotheses.csv` | pre-registered H1a, H1a-mech, H1b, H2, H3a, H3b, H4 | `supported` |
| `rq1_cells.csv`, `rq1_cells.png` | RQ1 2×2 per config | `ra_bp_rate_among_correct` (+95% CI), `wa_gp_rate_among_wrong` |
| `rq1_mechanism.csv` | is RA-BP parametric recall? | closed-book accuracy of the same model on RA-BP vs RA-GP questions (`gap`, CI) |
| `rq2_config_summary.csv`, `rq2_pareto.png` | per-config A / B / C / cost, Pareto frontier | `accuracy`, `b_composite`, `human_holistic_poststrat`, `acc_at_budget` |
| `rq2_kendall_tau.csv`, `rq2_ranks.csv` | RQ2 ranking agreement | τ with question-bootstrap CI for every pair of rankings |
| `paired_mcnemar.csv` | thinking on vs off; ReAct vs RAG | exact McNemar on identical (question, seed) pairs |
| `rq3_validity.csv`, `rq3_validity_cost.png` | RQ3: which metric tracks humans, at what cost | Spearman ρ (+CI) vs human holistic; judge tokens / CPU ms per trajectory; step-level κ (LLM vs human) in `report.md` |
| `rq4_passk.csv`, `rq4_passk.png`, `rq4_divergence.csv` | RQ4 pass^k and why seeds diverge | `pass^k` (+CI), `ratio_pass^3/pass^1`; divergence source per failing seed (parse failure, step/token limit, early finish, query choice, answer from same evidence) |
| `rq5_robustness.csv` | RQ5 | Δaccuracy, Δgrounded, Δevidence recall (+CI), recovery rates, `hidden_degradation_rate` (still correct but lost grounding) |
| report §Judge bias | H3b | judge–EM κ, swap consistency, first-slot preference, verbosity drift (Wilcoxon), cross-family ρ / κ and self-preference difference |
| report §Human-label reliability | §5.4 | intra-rater and external κ (steps, error category), ρ and weighted κ (holistic), median seconds per trajectory |

All CIs are 95% percentile bootstraps that resample **questions**, because seeds and configs on the same question are not independent. Use `--n-boot` to trade precision for speed.

### Week 12–14: release

Upload `data/`, `runs/`, `judge/` and the annotation batches (`git add -f annotation/` once labelling is finished) as the artifact, together with `results/`. Then diff `prereg/PREREGISTRATION.md` against tag `prereg-v1` for the paper's pre-registration section.

---

## 4. Configuration reference

`agent-eval configs` lists the frozen grid (`agent_eval/configs.py`):

| Config id | Model | Scaffold | Thinking |
|---|---|---|---|
| `react-qwen3-{1.7b,4b,8b}-{nothink,think}` | Qwen/Qwen3-{1.7B,4B,8B} | ReAct, ≤10 steps, BM25 top-5 | off: T=0.7, top-p 0.8; on: T=0.6, top-p 0.95, 1,024-token thinking budget |
| `cb-qwen3-*` | same | closed-book, one generation | off |
| `rag-qwen3-*` | same | one BM25 retrieval (top 5 paragraphs), then answer with citations | off |
| `react-qwen3-{4b,8b}-nothink-perturb` | same | ReAct with 20% injected tool errors (RQ5) | off |

- **Thinking budget.** Qwen3-2504 has no native budget knob, so thinking turns are generated in two phases. Phase 1 stops at `</think>` or after 1,024 tokens. If the budget ran out, Qwen's documented early-exit sentence plus `</think>` is appended, and phase 2 generates the visible tool call.
- **History.** Thoughts are logged but stripped from the conversation history, as Qwen recommends for multi-turn use.
- **Other limits.** 16,000 completion tokens per trajectory (`token_cap`); 3 consecutive unparseable outputs end the episode (`parse_failure`).
- **Seeds.** Every request gets a seed derived from (config, seed, question, step), so a re-run reproduces the same samples on the same software stack. Injected errors are seeded per (config, seed, question).
- **Models.** Each model's `ModelSpec` holds the dtype (fp16 on T4), tensor parallelism, quantization and `max_model_len`. Judges: `qwen3-14b-awq` (default) and `llama3.1-8b` (second family).

---

## 5. Data formats

**Trajectory** (one JSON line in `runs/<config>/seed<k>.jsonl`):
```json
{"question_id": "...", "config_id": "react-qwen3-4b-think", "model": "qwen3-4b", "scaffold": "react",
 "thinking": true, "perturb_p": 0.0, "seed": 0, "question": "...",
 "final_answer": "...", "cited_doc_ids": ["d01234"], "terminated_by": "finish|max_steps|parse_failure|token_cap",
 "n_steps": 4, "total_prompt_tokens": 5210, "total_completion_tokens": 3120, "total_thought_tokens": 2900,
 "total_tokens": 8330, "wall_ms": 15200.0, "steps": [
   {"step": 1, "thought": "...", "thought_tokens": 412, "content": "<tool_call>...</tool_call>",
    "action": {"tool": "search", "args": {"query": "..."}}, "observation": "Search results for ...",
    "parse_ok": true, "parse_error": null, "perturbed": null, "tool_ok": true,
    "doc_ids_returned": ["d01234", "..."], "doc_id_read": null, "latency_ms": 1830.0,
    "prompt_tokens": 2210, "completion_tokens": 468, "finish_reason": "stop", "thinking_budget_exhausted": false}]}
```
`latency_ms` is the wall time of the batched generation call the step belonged to. It is comparable only within one hardware type and session (plan §5.2); use tokens as the primary cost.

**Question** (`data/questions_eval.jsonl`): `question_id, question, answer, type, level, gold_titles, gold_doc_ids, supporting_facts, gold_conflict`.

**Human label** (`annotation/<batch>/labels_<annotator>.jsonl`): `anon_id, annotator, step_labels[], holistic, error_category, notes, seconds, labelled_at`.

---

## Kaggle notes

Every step can run on Kaggle. Only the policy runs and judges use GPU quota:

| Stage | Notebook script | Accelerator | Output Dataset |
|---|---|---|---|
| Data, BM25 index, question samples | `kaggle/01_build_data.py` | None (CPU) | `agent-eval-data` |
| Throughput bench, pilot, main runs, RQ5 / pass^5 optionals | `kaggle/02_policy_runs.py` (one model per session) | GPU T4 x2 | `agent-eval-runs` |
| LLM judges | `kaggle/03_judges.py` | GPU T4 x2 | `agent-eval-judge` |
| Scoring, annotation export/import, analysis and report | `kaggle/04_score_annotate_analyze.py` | None (CPU) | download `results/` |
| Human labelling | offline: `trajectories.html` + `sheet.csv`, imported in 04 | none | `agent-eval-annotation` |

The terminal labeler (`annotate label`) also works in an interactive Kaggle CPU session, but files in `/kaggle/working` disappear when an interactive session ends. Save a version, or download the labels, after every sitting. Labelling offline in the spreadsheet is less fragile.

- **Getting the code into Kaggle.** The notebooks `git clone` this repo. If the repo is private, upload a zip of the repo as a Kaggle Dataset (`agent-eval-code`) instead and run `pip install -e /kaggle/input/agent-eval-code/...`.
- **Hardware.** Use **GPU T4 x2**; vLLM does not support the P100 (sm_60). T4 has no bf16, so everything runs in fp16 or AWQ. Quota is about 30 GPU-hours per week with a 12-hour session cap.
- **vLLM version.** Qwen3 needs vLLM ≥ 0.8.5. The notebooks pin `vllm==0.9.2` and set `VLLM_USE_V1=0`, because older V1 engines did not support Turing GPUs. Treat this pin as the first thing to verify in week 2: if a newer vLLM runs on T4 in your session, use it and record the version in the pre-registration. Add `--enforce-eager` if CUDA-graph capture runs out of memory.
- **Fallback backend.** If vLLM cannot be made to work, `--backend transformers` runs the identical agent loop with Hugging Face `generate`. It is sequential and much slower, but fine for the 4B MVP on a reduced N.
- **Weights.** Turn Internet on for the first session of each model, then attach the weights as a Kaggle Model or Dataset so later sessions do not re-download them. Llama-3.1 is gated: accept the license and add `HF_TOKEN` as a Kaggle secret.
- **Quota.** Quota is charged for enabled-session wall time, so turn the GPU off while writing analysis code. Scoring, annotation and analysis are CPU-only.
- **Memory.** If KV-cache memory is short on one T4, lower `max_model_len` or `gpu_memory_utilization` in `configs.py`, or the batch size (`--chunk`). Record any change as a deviation.

## Testing

```bash
pytest -q                 # unit tests, mock-model end-to-end pipeline, stubbed vLLM thinking-budget path
agent-eval smoke          # real corpus builder, BM25, ReAct loop, baselines, judges, scoring, annotation, analysis on CPU
```

## Third-party licenses

HotpotQA is CC BY-SA 4.0. Qwen3 weights are Apache-2.0. Llama-3.1 is under the Llama 3.1 Community License. Add a `LICENSE` file for this repository's code before the artifact release.
