# %% [markdown]
# # 03 - LLM judges (GPU T4 x2)
# Inputs: `agent-eval-data`, `agent-eval-runs`, optionally `agent-eval-judge` (resume).
# Output: /kaggle/working/judge -> Dataset `agent-eval-judge`.

# %%
!git clone https://github.com/lywusa/agent-eval.git /kaggle/working/agent-eval
%cd /kaggle/working/agent-eval
!pip -q install -e . "vllm==0.9.2"

# %%
import os, shutil
os.environ.setdefault("VLLM_USE_V1", "0")
prev = "/kaggle/input/agent-eval-judge/judge"
if os.path.isdir(prev):
    shutil.copytree(prev, "/kaggle/working/judge", dirs_exist_ok=True)
RUNS = "/kaggle/input/agent-eval-runs/runs"
Q = "/kaggle/input/agent-eval-data/questions_eval.jsonl"

# %% Main judge (Qwen3-14B-AWQ): Family A correctness, Family B' steps, bias tests.
# Each command loads the model again; run them as separate sessions if memory is tight.
!python -m agent_eval judge correctness --judge qwen3-14b-awq --runs {RUNS} --questions {Q} --out /kaggle/working/judge
!python -m agent_eval judge steps       --judge qwen3-14b-awq --runs {RUNS} --out /kaggle/working/judge
!python -m agent_eval judge pairwise    --judge qwen3-14b-awq --runs {RUNS} --out /kaggle/working/judge --n-pairs 200
!python -m agent_eval judge verbosity   --judge qwen3-14b-awq --runs {RUNS} --out /kaggle/working/judge --subsample 200 --pad 3

# %% Second-family judge on the same 200-trajectory subsample (self-preference test).
# Llama 3.1 is gated: accept the license on Hugging Face and add HF_TOKEN as a Kaggle secret.
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!python -m agent_eval judge steps --judge llama3.1-8b --runs {RUNS} --out /kaggle/working/judge --subsample 200
