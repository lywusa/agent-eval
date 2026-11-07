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


4B, no thinking:

{'model': 'qwen3-4b', 'thinking': False, 'batch': 1, 'completion_tokens': 256, 'seconds': 15.51, 'tok_per_s': 16.5}
{'model': 'qwen3-4b', 'thinking': False, 'batch': 16, 'completion_tokens': 3986, 'seconds': 15.08, 'tok_per_s': 264.3}
{'model': 'qwen3-4b', 'thinking': False, 'batch': 64, 'completion_tokens': 15674, 'seconds': 25.3, 'tok_per_s': 619.6}
{
  "best_tok_per_s": 619.6,
  "react_tokens": 28800000,
  "baseline_tokens": 2160000,
  "react_gpu_hours": 12.911555842479018,
  "baseline_gpu_hours": 0.9683666881859263
}


4B, thinking:

{'model': 'qwen3-4b', 'thinking': True, 'batch': 1, 'completion_tokens': 487, 'seconds': 35.71, 'tok_per_s': 13.6}
{'model': 'qwen3-4b', 'thinking': True, 'batch': 16, 'completion_tokens': 7749, 'seconds': 32.73, 'tok_per_s': 236.8}
{'model': 'qwen3-4b', 'thinking': True, 'batch': 64, 'completion_tokens': 31287, 'seconds': 66.93, 'tok_per_s': 467.4}
{
  "best_tok_per_s": 467.4,
  "react_tokens": 28800000,
  "baseline_tokens": 2160000,
  "react_gpu_hours": 17.115960633290545,
  "baseline_gpu_hours": 1.2836970474967908
}


8B, no thinking:

{'model': 'qwen3-8b', 'thinking': False, 'batch': 1, 'completion_tokens': 256, 'seconds': 30.27, 'tok_per_s': 8.5}
{'model': 'qwen3-8b', 'thinking': False, 'batch': 16, 'completion_tokens': 4085, 'seconds': 12.87, 'tok_per_s': 317.4}
{'model': 'qwen3-8b', 'thinking': False, 'batch': 64, 'completion_tokens': 16324, 'seconds': 23.15, 'tok_per_s': 705.2}
{
  "best_tok_per_s": 705.2,
  "react_tokens": 28800000,
  "baseline_tokens": 2160000,
  "react_gpu_hours": 11.344299489506522,
  "baseline_gpu_hours": 0.8508224617129893
}


8B, thinking:

