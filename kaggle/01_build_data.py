# %% [markdown]
# # 01 - Build the frozen corpus, BM25 index and question samples (CPU, run once)
# Accelerator: None. Internet: On.
# Output: /kaggle/working/data -> save as a Kaggle Dataset named `agent-eval-data`.

# %%
!git clone https://github.com/lywusa/agent-eval.git /kaggle/working/agent-eval
%cd /kaggle/working/agent-eval
!pip -q install -e .

# %%
!python -m agent_eval data --download --out /kaggle/working/data --n-eval 400 --n-pilot 50 --seed 2026
!cat /kaggle/working/data/manifest.json

# %% [markdown]
# Commit `data/manifest.json` (fingerprints) and `data/questions_eval.jsonl` question IDs
# to git as part of the week-3 pre-registration. Then "Save Version" and create the
# Dataset `agent-eval-data` from this notebook's output.
