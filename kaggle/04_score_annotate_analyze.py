# %% [markdown]
# # 04 - Scoring, annotation export and analysis (CPU only; no GPU quota)
# Accelerator: None. Internet: On (for pip install of this repo only).
# Inputs: `agent-eval-data`, `agent-eval-runs`, `agent-eval-judge` (optional until judges ran),
#         `agent-eval-annotation` (optional; your labelled batches, uploaded as a Dataset).
# Outputs: /kaggle/working/results (report.md, CSV tables, PNG figures)
#          /kaggle/working/annotation (newly exported batches, to download and label)

# %%
!git clone https://github.com/lywusa/agent-eval.git /kaggle/working/agent-eval
%cd /kaggle/working/agent-eval
!pip -q install -e .

# %%
import os, shutil
DATA = "/kaggle/input/agent-eval-data"
RUNS = "/kaggle/input/agent-eval-runs/runs"
JUDGE = "/kaggle/input/agent-eval-judge/judge"
if not os.path.isdir(JUDGE):
    JUDGE = "/kaggle/working/no-judge"  # scoring falls back to exact match for "correct"
ANN = "/kaggle/working/annotation"
prev_ann = "/kaggle/input/agent-eval-annotation/annotation"
if os.path.isdir(prev_ann):
    shutil.copytree(prev_ann, ANN, dirs_exist_ok=True)

# %% Score every trajectory (Family A + B + B').
!python -m agent_eval score --runs {RUNS} --questions {DATA}/questions_eval.jsonl \
    --judge-dir {JUDGE} --out /kaggle/working/results/scores.csv

# %% [markdown]
# ## Annotation export (run once, after the judges, weeks 7-8)
# Uncomment the batch you need. Download /kaggle/working/annotation/<batch>/trajectories.html
# and sheet.csv, label offline, then put the filled sheet back with `annotate import`
# (locally or in this notebook) and upload the whole `annotation/` folder as the Dataset
# `agent-eval-annotation`. Do not open key.csv or sample_strata_PRIVATE.csv.

# %%
# !python -m agent_eval annotate sample   --scores /kaggle/working/results/scores.csv --runs {RUNS} --out {ANN} --n 150
# !python -m agent_eval annotate retest   --main {ANN}/main --runs {RUNS} --out {ANN} --n 30
# !python -m agent_eval annotate external --main {ANN}/main --runs {RUNS} --out {ANN} --n 40
# !python -m agent_eval annotate import   --batch-dir {ANN}/main --sheet /kaggle/input/<your-sheet-dataset>/sheet.csv --annotator me
!python -m agent_eval annotate status --out {ANN} || true

# %% Analysis: RQ1-RQ5, judge bias, human reliability, hypotheses, figures.
!python -m agent_eval analyze --scores /kaggle/working/results/scores.csv --annotation {ANN} \
    --judge-dir {JUDGE} --runs {RUNS} --questions {DATA}/questions_eval.jsonl --out /kaggle/working/results
!cat /kaggle/working/results/report.md
