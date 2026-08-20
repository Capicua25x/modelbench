# modelbench — LLM serving + accuracy harness for the RDNA4 vLLM port

The measurement harness behind the numbers published on
[Capicua25x/vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) (`RDNA4-PORT.md`) and its
[Docker Hub page](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4). Everything targets any
OpenAI-compatible `/v1` endpoint; nothing here is model- or vendor-specific.

## Throughput

```bash
# per-user + aggregate tok/s at several concurrency levels (defaults: levels "1 16", :8011)
throughput/concurrency-bench.sh --url http://localhost:8011 --model qwen \
  --think on --levels "1 4 8 16 32 64"                  # short prompts (~30 tok)
throughput/concurrency-bench.sh ... --prompt-tokens 6000 --levels "1 4 8 16 32"   # long prompts

# the full per-configuration battery (short + 6k sweeps + long-context preemption probe):
throughput/concurrency-test-arm.sh <tag> [url]
```

Conventions that keep numbers comparable: fixed 256-token completions, thinking state pinned per run,
single-run cells, pair configurations back-to-back in the same quiet window (cross-window comparisons on a
busy box mis-measure by 15–30 %), and report per-user AND aggregate — one without the other hides the knee.

## Accuracy

```bash
# GSM8K / IFEval / GPQA / AIME via lm-evaluation-harness against the same endpoint (examples):
lm_eval --model local-chat-completions \
  --model_args model=qwen,base_url=http://HOST:8011/v1/chat/completions,num_concurrent=12 \
  --tasks gsm8k --limit 50 --apply_chat_template --log_samples \
  --gen_kwargs '{"temperature":1.0,"top_p":0.95,"top_k":20,"until":[],"chat_template_kwargs":{"enable_thinking":true},"max_gen_toks":16000}' \
  --output_path results/gsm8k_s1234 --seed 1234

# long-context reasoning (100 items, LLM-judged) and HLE (120 items):
python3 harness/aa_lcr_runner.py <tag> http://HOST:8011/v1/chat/completions <api-key-or-dummy> 100 1234
python3 harness/hle_runner.py    <tag> http://HOST:8011/v1/chat/completions <api-key-or-dummy> 120 1234
python3 harness/termination_rate.py results/<run-dir>      # denominators: empties/truncations count as wrong
python3 harness/rejudge.py lcr results/aa-lcr/<tag>_s1234.jsonl <judge-model>
```

Judged rows need an OpenAI-compatible judge endpoint: set `JUDGE_URL` and `JUDGE_KEY`. Use the SAME judge
for every column you compare. Sampling is ON-SPEC from the model card (`BENCH_TEMP`/`BENCH_TOPP` envs);
no client-side stop strings (`until:[]`) — stops inside reasoning silently truncate deliberation.

## Reporting rules (the part that matters)

* Publish the FIRST run per cell; never substitute reruns.
* State n beside every rate; ±2 items on n=50–120 is the noise band.
* Empty/truncated generations count as WRONG (report the null-rate row).
* Never compare rates produced by different harnesses or different windows.

Apache-2.0.

## Reasoning-length / loop-rate runs (single hard item)

`lengths/length_loop_run.py` is the instrument behind the serving-stack length-divergence
investigation (n-rep runs of one deliberation-heavy item, outcome = TERMINATES / LOOP /
UNDECIDED, loop-start token recorded). The request body is deliberately explicit — the
request IS the experiment; read the module docstring before using it.

Dependency: `pip install datasets` (pulls the HLE item; the dataset is gated — accept its
terms on HF and `huggingface-cli login` once).

Reproduce the published self-hosted row (n=18, cap 100k, t1.0/p1.0, no effort preamble):

```bash
BENCH_URL=http://localhost:8888/v1/chat/completions \
CAP=100000 N_REP=18 SLOTS=6 TEMP=1.0 TOPP=1.0 EFFORT=none CUT=1 TAG=spec_t10p10 \
python3 lengths/length_loop_run.py
```

Rules that carry over from the rest of this repo, plus two of its own:
- **Validate with `CUT=0` first** on a configuration whose rate you already know — zero
  false positives (sentenced-yet-terminated) before you trust `CUT=1`.
- **Concurrency is part of the arm** (`SLOTS`): never pool runs taken at different
  concurrency.
- **The effort preamble is part of the arm**: `EFFORT=none` renders no preamble on the
  stock DeepSeek-V4-Flash vLLM template; only `xhigh`/`max` reach the template (+79
  tokens, the vendor's default render). Hold it constant across arms — or run both.
