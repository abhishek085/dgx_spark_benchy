# dgx_spark_benchy leaderboard

_generated from `results/spark_bench_plus.csv`, 14 labeled run(s)_

## Overall quality ranking

*General-purpose accuracy across 22 graded tasks (tool use, coding, safety, instruction-following, and more). `LocalScore` is one 0-100 number combining how often the model got the task right (Quality), how consistent it was across repeats (Reliability), how good its accuracy-per-token was (Efficiency), and how fast it started responding (Responsiveness).*

| model | label | LocalScore | Quality | Reliability | Efficiency | Responsiveness | tokens/sec | last run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | 85.9 | 74.1 | 100.0 | 100.0 | 87.7 | 29.9 | 2026-07-19T01:30:41 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 67.1 | 37.7 | 92.3 | 100.0 | 85.1 | 51.6 | 2026-07-19T02:15:45 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | 61.8 | 47.8 | 94.2 | 100.0 | 33.4 | 1080.4 | 2026-07-19T02:05:27 |
| deepreinforce-ai/Ornith-1.0-35B | deepreinforce-ai-ornith-10-35b | 54.1 | 43.3 | 98.3 | 100.0 | 0.0 | 24.1 | 2026-07-19T03:17:54 |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | 52.9 | 42.6 | 94.9 | 100.0 | 0.0 | 123.9 | 2026-07-19T00:25:30 |
| nvidia/Qwen3.6-27B-NVFP4 | qwen27b-nvidia-nvfp4 | — | — | — | — | — | 682666775.5 | 2026-07-17T23:39:01 |
| unsloth/Qwen3.6-27B-NVFP4 | qwen27b-unsloth-nvfp4 | — | — | — | — | — | 85333415.3 | 2026-07-17T23:54:02 |
| unsloth/Qwen3.6-27B-NVFP4 | unsloth-qwen36-27b-nvfp4 | — | — | — | — | — | 88.5 | 2026-07-18T14:30:08 |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | — | — | — | — | — | 237.9 | 2026-07-18T16:28:43 |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | — | — | — | — | — | 4.7 | 2026-07-18T16:45:34 |
| unsloth/gemma-4-31B-it-NVFP4 | unsloth-gemma-4-31b-it-nvfp4 | — | — | — | — | — | 6.0 | 2026-07-18T17:01:10 |
| unsloth/gemma-4-26B-A4B-it-NVFP4 | unsloth-gemma-4-26b-a4b-it-nvfp4 | — | — | — | — | — | 32.5 | 2026-07-18T17:14:25 |
| Qwen/Qwen3.6-27B | qwen-qwen36-27b | — | — | — | — | — | 14.5 | 2026-07-18T18:21:25 |
| google/gemma-4-31B-it | google-gemma-4-31b-it | — | — | — | — | — | 2.6 | 2026-07-18T19:16:42 |

## How many people can use it at once

*Each "ceiling" is the largest number of simultaneous conversations this box sustained on that kind of traffic before answers got measurably worse or slower — the real answer to "how many users can share this model." `orchestrator` is multi-step tool-chain traffic (the shape an autonomous agent sends), `coding_agent` is code-generation requests, `chat_agent` is casual back-and-forth conversation.*

| model | label | tool-chain agents | coding agents | chat sessions | tool-calling works? |
|---|---|---:|---:|---:|:---:|
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 1 | 4 | 2 | ❌ |
| deepreinforce-ai/Ornith-1.0-35B | deepreinforce-ai-ornith-10-35b | 1 | 4 | — | ❌ |
| nvidia/Qwen3.6-27B-NVFP4 | qwen27b-nvidia-nvfp4 | — | — | — | ✅ |
| unsloth/Qwen3.6-27B-NVFP4 | qwen27b-unsloth-nvfp4 | — | — | — | ✅ |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | — | — | — | ✅ |
| unsloth/Qwen3.6-27B-NVFP4 | unsloth-qwen36-27b-nvfp4 | — | — | — | ✅ |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | — | 2 | — | ✅ |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | — | — | — | ✅ |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | — | — | — | ❌ |
| unsloth/gemma-4-31B-it-NVFP4 | unsloth-gemma-4-31b-it-nvfp4 | — | — | — | ❌ |
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | — | 2 | — | ✅ |
| unsloth/gemma-4-26B-A4B-it-NVFP4 | unsloth-gemma-4-26b-a4b-it-nvfp4 | — | — | — | ❌ |
| Qwen/Qwen3.6-27B | qwen-qwen36-27b | — | — | — | ✅ |
| google/gemma-4-31B-it | google-gemma-4-31b-it | — | — | — | ❌ |

## Hermes Benchmark — best model for a personal-agent harness

*One 0-100 number for "how good would this model be behind a Hermes-style personal agent": 50% how well it actually completes real agent tasks (tool use, web search, remembering things across a conversation), 30% how many concurrent sessions it sustains, 20% how quickly it starts responding.*

| model | label | Hermes Score | Quality | Capacity | Responsiveness | approx size |
|---|---|---:|---:|---:|---:|---:|
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | 75.2 | 77.6 | 100.0 | 31.8 | 26B |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | 69.0 | 77.6 | 100.0 | 1.0 | 35B |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | 66.7 | 33.3 | 100.0 | 100.0 | 27B |
| unsloth/gemma-4-31B-it-NVFP4 | unsloth-gemma-4-31b-it-nvfp4 | 53.3 | 51.5 | 25.0 | 100.0 | 31B |
| unsloth/gemma-4-26B-A4B-it-NVFP4 | unsloth-gemma-4-26b-a4b-it-nvfp4 | 53.3 | 51.5 | 25.0 | 100.0 | 26B |
| google/gemma-4-31B-it | google-gemma-4-31b-it | 53.3 | 51.5 | 25.0 | 100.0 | 31B |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | 48.0 | 48.5 | 12.5 | 100.0 | 31B |
| nvidia/Qwen3.6-27B-NVFP4 | qwen27b-nvidia-nvfp4 | 46.8 | 33.3 | 100.0 | 0.8 | 27B |
| unsloth/Qwen3.6-27B-NVFP4 | unsloth-qwen36-27b-nvfp4 | 44.5 | 28.8 | 100.0 | 0.8 | 27B |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | 31.2 | 28.8 | 50.0 | 8.9 | 35B |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 29.0 | 14.3 | 6.2 | 100.0 | 30B |
| deepreinforce-ai/Ornith-1.0-35B | deepreinforce-ai-ornith-10-35b | 29.0 | 14.3 | 6.2 | 100.0 | 35B |
| unsloth/Qwen3.6-27B-NVFP4 | qwen27b-unsloth-nvfp4 | 28.3 | 40.5 | 25.0 | 3.0 | 27B |
| Qwen/Qwen3.6-27B | qwen-qwen36-27b | 22.9 | 30.3 | 25.0 | 1.2 | 27B |

**Smallest model that's actually good enough (70+/100):** nvidia/Gemma-4-26B-A4B-NVFP4 (~26B, score 75.2) — the one to reach for if you're tight on memory.  
**Largest model tested:** nvidia/Gemma-4-26B-A4B-NVFP4 (~26B, score 75.2) — the highest-quality option this box can run.

