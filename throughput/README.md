# throughput/ — serving-envelope benchmarks

Serving-speed side of the bench: how fast an arm decodes, and how many concurrent
requests it holds before it preempts. Accuracy/agentic harnesses live in `../harness/`
and `../lengths/`.

> **The concurrency protocol is `mia_protocol.py`** (MiaAI-Lab's cooperative-MoE protocol;
> operator 2026-10-02 — estate-wide, prod `:8011` included). The in-house v4 sweep and Tony's
> V4.1 bench were **retired** to `retired/` (receipts stay). One protocol, comparable to her
> published cells: don't compare across protocols, and since its reference C1 is
> fixed-prompt/temp-0, quote the sampled workloads (poetry/code/incident) for acceptance claims.

- **`mia_protocol.py`** — Mia's measurement protocol, workloads verbatim from
  `extensions/cooperative_moe/benchmarks/workloads.json`: poetry/code/incident × seeds 11/23/47
  (512-cap, temp 1, top_p .95), reference C1/C2 (hash-map prompt, temp 0, 400 outputs, C2 with a
  start barrier), a cold 32K prefill probe, and spec-decode acceptance from `/metrics`. Formulas
  exactly as documented (C1 = (ct−1)/(last−first **content**); C2 summed over the shared window).
  Caveat: with a reasoning parser, the thinking-ON incident case lands in `reasoning`, so the
  content-time formula yields nothing — use the reasoning-inclusive rate, or match parser config.
  Run: `python3 mia_protocol.py --base http://127.0.0.1:8888/v1 --model <id> --out <dir> --label <l> [--reps 3]`.
  Prod `:8011`: `--base http://localhost:8011/v1 --model qwen`.

## Retired (2026-10-02) → `retired/`

- `concurrency-bench.sh` — the in-house v4 tok/s sweep (rotating-topic essay + trivial + acc/draft).
- `concurrency-test-arm.sh` — per-arm envelope test (run after a model swap, before the accuracy bench).
- `tony-v41bench.py` — Tony's (`tonyd2wild`) V4.1 benchmark, kept verbatim for 1:1 comparability.

The private `master` lineage additionally carried `bench_sweep_apx.py`, `render-conc-matrix.py` and
`run-cap*-conc-*.sh` — same retirement there. **Receipts stay where they were**: `apx-results/`,
`conc-logs/`, `tony-v41/`, `mia-protocol/`.
