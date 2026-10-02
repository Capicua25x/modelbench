# retired/ — concurrency benches we no longer run

Retired **2026-10-02** (operator): the estate consolidated on **one** concurrency protocol —
`../mia_protocol.py`, MiaAI-Lab's cooperative-MoE measurement protocol — **prod `:8011` included**.
Three protocols invited wrong comparisons; only one is quoted from now on.

Moved here (public line):

- `concurrency-bench.sh` — the in-house v4 tok/s sweep (rotating-topic essay + trivial + acc/draft).
- `concurrency-test-arm.sh` — the per-arm serving envelope (short/6k/preemption probe).
- `tony-v41bench.py` — Tony's (`tonyd2wild`) V4.1 bench, kept verbatim for 1:1 comparability with his cells.

The private `master` lineage additionally carried `bench_sweep_apx.py`, `render-conc-matrix.py` and
`run-cap*-conc-*.sh` — same retirement there.

**Receipts were NOT moved** (provenance for past claims): `../apx-results/`, `../conc-logs/`,
`../tony-v41/`, `../mia-protocol/`.

When quoting legacy cells: keep the protocol label on the number, never compare across protocols, and
remember the fixed-prompt/temp-0 shapes (including Mia's reference C1) are replay-prone for stateful
drafters — for acceptance-sensitive claims quote the sampled workloads (poetry/code/incident).
