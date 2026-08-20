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
