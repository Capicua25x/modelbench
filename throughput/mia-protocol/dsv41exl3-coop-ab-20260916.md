# DS4.1 EXL3 + cooperative MoE — local repin, GPU gate, and stock↔coop A/B (2026-09-16)

## Why a local repin

Upstream merged the cooperative-MoE extension (PR #8, wesleyyoung, 09:05Z) and then
restructured the opt-in (PR #13, Mia, 13:08Z) to stage a GPU-validated
`cooperative_moe.so` — **but the validated binary is not distributed** (no release,
no `artifacts/` file). Clean `nvcc` rebuilds are documented as *not* bit-identical
(GNU build-id + CUDA `-lineinfo` metadata), confirmed by mrexodia on PR #8 and
reproduced here (four distinct hashes across runs: `d699b07e`, `848deb55`, plus
mrexodia's `b0cf244c`, `cf6ff1e2` — none matched the pin `a09a589c`).
Upstream's sanctioned path for a lab: rebuild → **repeat the GPU gate** → explicit
pin update.

## What was done (node-a, kit at main 8404ac7)

- Built with the official tools: `extensions/cooperative_moe/archive_upstream.sh`
  (host-side git archive of the pinned ExLlamaV3 commit) + the git-free `build.sh`
  in the pinned recipe image (no network, 2 CPUs).
- **Repin (local, documented):**
  - `cooperative_moe.so` = `848deb55d1491ffbed7f1bc956b15217f7c98683299e9d1451a04d8d42d0d182`
    (the built binary).
  - `runtime.py` — embedded load-time `SHA256` updated to the same hash;
    new adapter digest `4296d2fedb0f0b2b63b96bde5b9bae3ea8fdcb7dabf44c87be097de62ab4ffcc`.
  - `prepare_profile.py` — `BINARY_SHA`/`ADAPTER_SHA` updated accordingly.
  - Generated overlay `exl3-cooperative.py` = `e2d044ed…` (host digest == in-container
    mount digest on both ranks), staged to `~/.cache/vllm-dsv41-flash-exl3/KS6f6X/` on
    both nodes with `sha256sum -c SHA256SUMS` clean.
- **GPU gate (packaged 54-case integration test): PASS on both nodes** (head exit 0,
  worker exit 0; retained strict-difference counts match the published report:
  6,124,491 raw / 3,912,196 post-BF16).
- Rollback backups: `runtime.py.bak-pre-localrepin-*`, `prepare_profile.py.bak-pre-localrepin-*`,
  `~/.cache/dsv41-coop-run-dir` → `logs/cooperative-moe.KS6f6X/original.env`, `GATE-RESULTS.txt`.

## Config for the A/B

Mia's documented comparison settings, both arms (text-only, seqs=2, k=3, 600k):

    MAX_NUM_BATCHED_TOKENS=3072
    LONG_PREFILL_TOKEN_THRESHOLD=2816
    EXL3_TEMP_ROWS_FUSED=8
    KV_CACHE_MEMORY_BYTES=2596000000   (~749k tokens)

## A/B results — Mia's protocol, same config, medians (3 reps / 3 seeds)

| cell | stock | cooperative | Δ | her published stock→coop |
|---|---:|---:|---:|---:|
| Poetry decode | 25.03 | **32.18** | **+28.6%** | 23.62 → 29.26 |
| Coding decode | 42.80 | **55.15** | **+28.9%** | 38.76 → 42.96 |
| C1 decode | 33.69 | **42.33** | **+25.6%** | 31.45 → 40.23 |
| C2 aggregate | 51.58 | **67.53** | **+30.9%** | 45.87 → 61.06 |
| Uncached 32K prefill | ~1024 (recheck) | 1130.7 | ≈unchanged (noise) | 1138 → ~unchanged |

Incident: unmeasurable under the content-time formula on this build (thinking-on
consumes the whole 512-token cap in reasoning; parser sends it to `reasoning`, not
`content`). Stock reasoning-inclusive: ~34.7 tok/s.

Independent confirmation: mrexodia's post-merge revalidation on another pair saw
C1 +25.3% / C2 +35.2% — our numbers land in the same band.

## Operational notes (load-bearing)

- **All starts must pass `SKIP_BUILD=1 SKIP_PULL=1 SKIP_SHIP=1`** (plus `SKIP_SYNC=1`
  on this rsync setup): the upstream #13 upgrade moved the recipe stamp, so a bare
  `./start.sh` **rebuilds the image** (unvalidated) and takes 30+ min. The healer and
  `pair-model.sh dsv41` wrappers were updated accordingly.
- The upstream upgrade (b9c49e9 → 8404ac7) was applied with our two local patches kept:
  `download.sh` (+`config.json` in ENGRAM_FILES) and `start.sh` (rsync preflight honors
  `SKIP_SYNC`/existing worker copy). Both are still needed upstream as of 8404ac7.
- Rollback out of coop: comment `EXL3_OVERLAY_HOST` in `.env`, restart with the SKIP
  flags. Full original config: `logs/cooperative-moe.KS6f6X/original.env`.

Results JSONs: `mia-protocol/bench-dsv41exl3-{stock,coop}-herconfig.json`.
