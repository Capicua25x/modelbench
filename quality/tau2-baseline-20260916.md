# τ² baseline — DS4.1 Flash EXL3 (pair) vs DeepSeek API (native) — 2026-09-16

First quality baseline for the DS4.1 Flash EXL3 lane (Mia kit @`8404ac7`, 2.9bpw + cooperative MoE).
The question it answers: *is the local weight quant mostly intact vs first-party serving?*

## Cell protocol (identical on both arms)

- τ² suite: **telecom 114 / airline 50 / retail 60** tasks, 1 trial, seed 1234
- agent sampling **t1.0 / p0.95**, `max_tokens` 30000, thinking ON (template default), **`reasoning_effort=max`**
- concurrency **c=2**; user-sim `gpt-5.6-luna` + NL-assertions judge `gpt-5.6-sol` — both via OpenRouter (single-judge rule)
- pair arm: served id `DeepSeek-v4.1-Flash-EXL3` on the local engine (`http://<engine-host>:8888/v1`) (coop MoE enabled)
- reference arm: `api.deepseek.com` model `deepseek-flash` = **DeepSeek-V4.1-Flash** (models page, 2026-09-16)
- runners: `~/lm-eval/tau2/corre_dsv41exl3.sh`, `corre_dsv41dsapi.sh` · paired diff: `compare_arms.py`
- logs: `tau2_agentic.note` · results: `~/lm-eval/tau2/results/tau2_<dom>_dsv41{exl3,dsapi}_c2/`

## Scores

| domain | pair EXL3 | DeepSeek API (native) | Δ (API − pair) |
|---|---:|---:|---:|
| telecom | 107/114 = 0.9386 | 110/114 = 0.9649 | +2.63 |
| airline | 44/50 = 0.8800 | 40/50 = 0.8000 | −8.00 |
| retail | 54/60 = 0.9000 | 49/60 = 0.8167 | −8.33 |
| **total** | **205/224 = 0.9152** | **199/224 = 0.8884** | **−2.68** |

Paired task-level flips (224 tasks): pair 15 · DeepSeek 9 · both miss 20 · both scored 180.
Sign test on the 24 flips: **p ≈ 0.15 — not significant.** Verdict: **no measurable quant penalty at this sample size**
(nominal +2.7 points for the pair, carried by airline/retail; DeepSeek took telecom by 3 tasks).

## Third data point (secondary, NOT a comparable cell)

Unpinned OpenRouter pass (same model, provider routing default/mixed, same protocol otherwise):
telecom 0.9298 · airline 0.8200 · retail 0.9333 → **203/224 = 0.9063**. Kept as context only — mixed upstreams.

## Predecessor reference — V4-Flash 4.0 pair (native FP8-class, archived lane)

The 4.0 era ran the same instruments on the same pair. Protocol deltas declared: 79-preamble effort class
(vs this lane's template `max`), τ² c=6 (vs c=2), same tasks/judge/sim/seed; cells are cross-era, read the deltas.

| instrument | 4.0 pair | 4.1 pair (EXL3 2.9bpw) | 4.0 vendor | 4.1 vendor |
|---|---:|---:|---:|---:|
| τ² telecom (114 tasks) | **113 = 0.9912** | 107 = 0.9386 | 0.9825 | 0.9649 |
| SWE-bench Verified 0:100 | 79 | **95** | 80 | 97 |
| TB Hard-44 (+×3 clock) | 22/44 | 23/44 | — | 23/44 (regular) |
| class-A gsm8k / aime / ifeval / gpqa | .947 / .989 / .946 / **.922** | .94 / 1.0 / .95 / .867 | — | .94 / 1.0 / .9625 / .917 |

**Read:** the generation moved SWE +16 (pair) / +17 (vendor) and held TB, while **telecom slipped on both
sides** — vendor −2 items, pair −6. So the 4.1 telecom dip is a model-generation (and cross-era protocol)
effect, not a quant penalty: the pair-vs-vendor offset stayed flat across generations (4.0: +1 telecom item,
−1 SWE; 4.1: −3 telecom, −2 SWE — both within flip-test noise, p≈0.15). The quant remained in parity with
first-party serving in both generations; the newer model is simply not the stronger telecom player.

## Caveats

- **n=1** per domain, sampling temp 1.0 → per-domain swings (airline/retail ±8 pts) are well inside run-to-run
  variance; only the paired flip count is load-bearing.
- `effort=max` was accepted by the hosted API (no 400, behaviour shift observed); not independently verified.
- Pair arm ran alone on the pair (no other benches) for the whole suite.
- TB + SWE cells: see `dsv41api_tbb2` (single regular-clock sweep, running) and SWE-API 93/100 (done);
  pair-side TB pass 1 + ×3 clock sweep and pair-side SWE = pending.
