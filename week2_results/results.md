1.7B, no thinking:

{'model': 'qwen3-1.7b', 'thinking': False, 'batch': 1, 'completion_tokens': 256, 'seconds': 4.19, 'tok_per_s': 61.1}
{'model': 'qwen3-1.7b', 'thinking': False, 'batch': 16, 'completion_tokens': 3643, 'seconds': 6.43, 'tok_per_s': 566.3}
{'model': 'qwen3-1.7b', 'thinking': False, 'batch': 64, 'completion_tokens': 13903, 'seconds': 12.56, 'tok_per_s': 1106.8}
{
  "best_tok_per_s": 1106.8,
  "react_tokens": 28800000,
  "baseline_tokens": 2160000,
  "react_gpu_hours": 7.228044813877846,
  "baseline_gpu_hours": 0.5421033610408384
}


1.7B, thinking:

{'model': 'qwen3-1.7b', 'thinking': True, 'batch': 1, 'completion_tokens': 460, 'seconds': 7.64, 'tok_per_s': 60.2}
{'model': 'qwen3-1.7b', 'thinking': True, 'batch': 16, 'completion_tokens': 7721, 'seconds': 15.89, 'tok_per_s': 486.1}
{'model': 'qwen3-1.7b', 'thinking': True, 'batch': 64, 'completion_tokens': 31478, 'seconds': 40.25, 'tok_per_s': 782.1}
{
  "best_tok_per_s": 782.1,
  "react_tokens": 28800000,
  "baseline_tokens": 2160000,
  "react_gpu_hours": 10.22887098836466,
  "baseline_gpu_hours": 0.7671653241273494
}


4b, no thinking
