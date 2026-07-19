# dgx_spark_benchy leaderboard

_generated from `results/spark_bench_plus.csv`, 6 labeled run(s)_

## Best model for...

*Different from the composite scores above on purpose -- these pick a winner for one specific thing you might care about, straight from the same raw measurements, no single number trying to represent everything at once. A model can win here and rank modestly on LocalScore/Hermes Score, or vice versa -- that's the point.*

| best for... | winner | why | size |
|---|---|---|---:|
| **Fastest (with real answers, not just fast garbage)** | nvidia/Gemma-4-26B-A4B-NVFP4 (30 tok/s) | Highest peak decode speed among models that still cleared a basic quality bar (Quality ≥ 50/100) -- the pick when speed is what you care about most, as long as it's not getting things wrong to get there. | 26B |
| **Most accurate (that isn't painfully slow)** | nvidia/Gemma-4-26B-A4B-NVFP4 (74/100 quality) | Highest Quality score among models still doing at least 20 tokens/sec -- the pick when correctness matters most and you just need it to not crawl. | 26B |
| **Best at coding** | nvidia/Qwen3.6-35B-A3B-NVFP4 (100% coding accuracy) | Highest code-generation accuracy (exec-verified against test cases), speed not considered at all -- the pick for a coding assistant specifically. | 35B |
| **Best at tool calling** | nvidia/Qwen3.6-35B-A3B-NVFP4 (83/100 tool-use quality) | Highest tool-use accuracy from the graded eval suite -- the pick for anything agentic that lives or dies on correctly calling functions. | 35B |
| **Best for a Hermes-style personal agent** | nvidia/Gemma-4-26B-A4B-NVFP4 (75/100 Hermes Score) | Highest Hermes Score -- see the dedicated section below for the full breakdown. | 26B |

## Overall quality ranking

*General-purpose accuracy across 22 graded tasks (tool use, coding, safety, instruction-following, and more). `LocalScore` is one 0-100 number combining how often the model got the task right (Quality), how consistent it was across repeats (Reliability), how good its accuracy-per-token was (Efficiency), and how fast it started responding (Responsiveness).*

| model | label | LocalScore | Quality | Reliability | Efficiency | Responsiveness | tokens/sec | last run |
|---|---|---:|---:|---:|---:|---:|---:|---|
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | 73.9 | 74.1 | 100.0 | 40.1 | 87.6 | 32.0 | 2026-07-19T19:33:57 |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 64.7 | 44.1 | 94.1 | 76.7 | 77.2 | 51.5 | 2026-07-19T13:11:25 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | 61.7 | 46.8 | 94.3 | 100.0 | 32.6 | 271.9 | 2026-07-19T12:31:16 |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | 61.3 | 78.2 | 100.0 | 10.6 | 44.9 | 7.6 | 2026-07-19T18:31:33 |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | 60.4 | 47.7 | 92.0 | 100.0 | 25.7 | 234.1 | 2026-07-19T14:08:37 |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | 47.6 | 46.1 | 86.8 | 69.0 | 0.0 | 67.9 | 2026-07-19T16:48:19 |

## How many people can use it at once

*Each "ceiling" is the largest number of simultaneous conversations this box sustained on that kind of traffic before answers got measurably worse or slower — the real answer to "how many users can share this model." `orchestrator` is multi-step tool-chain traffic (the shape an autonomous agent sends), `coding_agent` is code-generation requests, `chat_agent` is casual back-and-forth conversation.*

| model | label | tool-chain agents | coding agents | chat sessions | tool-calling works? |
|---|---|---:|---:|---:|:---:|
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 1 | 4 | — | ❌ |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | — | 2 | — | ✅ |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | — | 2 | — | ✅ |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | — | — | — | ✅ |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | — | — | — | ✅ |
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | — | 2 | — | ✅ |

## Hermes Benchmark — best model for a personal-agent harness

*One 0-100 number for "how good would this model be behind a Hermes-style personal agent": 50% how well it actually completes real agent tasks (tool use, web search, remembering things across a conversation), 30% how many concurrent sessions it sustains, 20% how quickly it starts responding.*

| model | label | Hermes Score | Quality | Capacity | Responsiveness | approx size |
|---|---|---:|---:|---:|---:|---:|
| nvidia/Gemma-4-26B-A4B-NVFP4 | nvidia-gemma-4-26b-a4b-nvfp4 | 75.2 | 77.6 | 100.0 | 31.7 | 26B |
| nvidia/Gemma-4-31B-IT-NVFP4 | nvidia-gemma-4-31b-it-nvfp4 | 71.7 | 77.6 | 100.0 | 14.4 | 31B |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | nvidia-qwen36-35b-a3b-nvfp4 | 69.0 | 77.6 | 100.0 | 1.0 | 35B |
| unsloth/Qwen3.6-35B-A3B-NVFP4 | unsloth-qwen36-35b-a3b-nvfp4 | 69.0 | 77.6 | 100.0 | 0.9 | 35B |
| nvidia/Qwen3.6-27B-NVFP4 | nvidia-qwen36-27b-nvfp4 | 68.9 | 77.6 | 100.0 | 0.6 | 27B |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | nvidia-nvidia-nemotron-3-nano-30b-a3b-nvfp4 | 29.0 | 14.3 | 6.2 | 100.0 | 30B |

**Smallest model that's actually good enough (70+/100):** nvidia/Gemma-4-26B-A4B-NVFP4 (~26B, score 75.2) — the one to reach for if you're tight on memory.  
**Largest model tested:** nvidia/Gemma-4-31B-IT-NVFP4 (~31B, score 71.7) — the highest-quality option this box can run.

