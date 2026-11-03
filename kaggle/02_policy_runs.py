# %% [markdown]
# # 02 - Policy runs: one model per notebook session (GPU T4 x2)
# Inputs: Datasets `agent-eval-data` (from 01) and, optionally, `agent-eval-runs`
# (previous outputs, for resuming). Internet: On for the first run of each model
# (downloads weights), then attach the weights as a Dataset / Kaggle Model.
# Run with "Save Version -> Save & Run All (Commit)" so it runs in the background.

# %%
!git clone https://github.com/lywusa/agent-eval.git /kaggle/working/agent-eval
%cd /kaggle/working/agent-eval
# Pin a vLLM version that you verified on T4 in week 2 (see README "Kaggle notes").
!pip -q install -e . "vllm==0.9.2"

# %%
import os, shutil
# T4 (sm_75) may need the V0 engine on older vLLM releases; remove if V1 works.
os.environ.setdefault("VLLM_USE_V1", "0")
# Resume: copy previous shard files (if attached) into the working runs directory.
prev = "/kaggle/input/agent-eval-runs/runs"
if os.path.isdir(prev):
    shutil.copytree(prev, "/kaggle/working/runs", dirs_exist_ok=True)

# %%
# Edit these two lines per session. One model per session; configs must share the model.
CONFIGS = "react-qwen3-4b-nothink,react-qwen3-4b-think"   # plus cb-qwen3-4b,rag-qwen3-4b when running baselines
SEEDS = "0,1,2"

!python -m agent_eval run --configs {CONFIGS} --seeds {SEEDS} \
    --questions /kaggle/input/agent-eval-data/questions_eval.jsonl \
    --data /kaggle/input/agent-eval-data --runs /kaggle/working/runs \
    --chunk 50 --max-hours 10.5

# %% [markdown]
# Pilot (week 4): replace `questions_eval.jsonl` by `questions_pilot.jsonl`, use
# `--seeds 0`, and write to `--runs /kaggle/working/pilot_runs`.
#
# After the commit finishes: create/update the Dataset `agent-eval-runs` from
# /kaggle/working/runs (New Dataset Version). If the session hit the time limit,
# simply re-run: finished questions are skipped.
