# throughput/ — serving-envelope benchmarks

Serving-speed side of the bench: how fast an arm decodes, and how many concurrent
requests it holds before it preempts. Accuracy/agentic harnesses live in `../harness/`
and `../lengths/`.

- **`concurrency-bench.sh`** — the tok/s sweep driver (v4: replay-proof essay workload +
  per-cell spec-decode acceptance). Levels, `--trivial` (community peak-finder) and
  `--prompt-tokens N` (context-bound shape), `--think on|off|raw`. A tok/s is
  uninterpretable without its `acc/draft` column and prompt shape.
- **`concurrency-test-arm.sh`** — per-arm envelope test. Run after a model swap,
  BEFORE an accuracy bench; the aggregate-tok/s knee and the preemption Δ set the safe
  bench concurrency.

- **`tony-v41bench.py`** — Tony's (`tonyd2wild`) V4.1 benchmark, kept **verbatim** from
  `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` → `bench/v41bench.py` (adopted
  2026-09-16) so our cells compare 1:1 with his published results. Fixed prompt set v1
  (8 categories + a counting ceiling) × `--levels` × `--prefill`; temp 0, thinking off;
  tokens from the server's usage block; TTFT = first content delta; each request carries
  a unique front tag (defeats prefix cache).
  Run: `python3 tony-v41bench.py --base <url>/v1 --model <id> --label <l> --out out/` →
  `bench-<label>.{json,md}`.

- **`mia_protocol.py`** — runner for **MiaAI-Lab's cooperative-MoE measurement protocol**
  (kit `docs/cooperative-moe.md`; workloads verbatim from the kit's
  `extensions/cooperative_moe/benchmarks/workloads.json`): poetry/code/incident × seeds
  11/23/47 (512-cap, temp 1, top_p .95), reference C1/C2 (hash-map prompt, temp 0, 400
  outputs, C2 with a start barrier), a cold 32K prefill probe, and spec-decode acceptance
  from `/metrics`. Formulas exactly as documented (C1 = (ct−1)/(last−first **content**);
  C2 summed over the shared window). Caveat: on builds with a reasoning parser the
  thinking-ON incident case lands in `reasoning`, so the content-time formula yields
  nothing — report the reasoning-inclusive rate, or match the parser config.
  First published run (2026-09-16, stock vs cooperative on the same two-Spark box):
  `mia-protocol/` — see the summary md there.

> Hostnames/ports/paths in these scripts and results are private working values;
> sanitize before publishing anything derived from them.
