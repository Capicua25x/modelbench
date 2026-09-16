## dsv41exl3-stock-20260916 (2026-09-16T15:31:21Z)

Mia TP2 EXL3 kit stock (600k ctx, MAX_NUM_SEQS=2, batched=1024, DSpark k=3, text-only). Levels >=3 are queue-bound by MAX_NUM_SEQS=2.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 34.58 | 38.13 | 0.387 |
| C2 | 53.78 | 31.11 | 0.502 |
| C4 | 53.33 | 30.39 | 3.317 |
| C6 | 53.13 | 29.17 | 6.277 |

### Per-stream tok/s by category

| category | C1 | C2 | C4 | C6 |
|---|---|---|---|---|
| coding | 41.12 | 38.19 | 34.29 | 32.39 |
| json | 41.08 | 31.19 | 32.72 | 31.24 |
| narrative | 24.38 | 17.74 | 18.88 | 18.8 |
| prose | 26.75 | 19.6 | 19.48 | 19.33 |
| math | 49.06 | 41.08 | 40.13 | 38.6 |
| reasoning | 41.55 | 33.67 | 32.35 | 34.39 |
| summary | 29.53 | 24.36 | 24.46 | 21.67 |
| format | 51.57 | 43.06 | 40.85 | 36.92 |
| ceiling_count | 39.23 | 35.81 | 36.01 | 35.8 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 3.6 | 819.5 |
| 8000 | 11592 | 13.761 | 842.4 |
| 32000 | 46810 | 55.876 | 837.7 |
| 64000 | 93335 | 113.393 | 823.1 |
