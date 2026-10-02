#!/usr/bin/env bash
# Per-arm concurrency test — RUN AFTER SWAP, BEFORE THE BENCH (operator 2026-08-16 10:15).
# Measures the serving envelope so we (a) know the concurrency ceiling and (b) set the bench/HLE conc
# BELOW the preemption cliff (the HLE-thrash lesson: 4×60K on a 255k pool = 1,979 preemptions).
# Usage: ./concurrency-test-arm.sh <tag>   (server for <tag> must be up on :8011)
set -u; TAG="${1:?tag}"; URL="${2:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="${BENCH_BIN:-$HERE/concurrency-bench.sh}"   # single source; override with BENCH_BIN
S="${S:-$HERE/conc-logs}"   # per-arm sweep logs
mkdir -p "$S"; say(){ echo "[$(date +%H:%M:%S)] [$TAG-conc] $*"; }
say "server: $(curl -sm5 $URL/v1/models | python3 -c 'import sys,json;print([m[\"id\"] for m in json.load(sys.stdin)[\"data\"]])' 2>/dev/null)"
# baseline preemption counter
P0=$(curl -sm8 $URL/metrics | awk '/^vllm:num_preemptions_total/{print $NF}')
say "preemptions at start: $P0"
# 1) short-prompt sweep (throughput ceiling, think ON = production shape)
say "short-prompt sweep think ON, levels 1 4 8 16 32 64"
"$BENCH" --url $URL --model qwen --think on --levels "1 4 8 16 32 64" > "$S/conc_${TAG}_short_thinkon.log" 2>&1
say "short sweep done -> $S/conc_${TAG}_short_thinkon.log"
# 2) realistic 6k-prompt sweep (where the KV pool actually bites)
say "6k-prompt sweep think ON, levels 1 4 8 16 32"
"$BENCH" --url $URL --model qwen --think on --prompt-tokens 6000 --levels "1 4 8 16 32" > "$S/conc_${TAG}_6k_thinkon.log" 2>&1
say "6k sweep done -> $S/conc_${TAG}_6k_thinkon.log"
# 2b) BLOCK-ALIGNED sweep (added 2026-08-18). The 6k cell above is NOT a fair fp8-KV-vs-bf16-KV control:
#     fp8 KV halves bytes/token, so vLLM's mamba-aligned attention block doubles (800 -> 1600), and prefix
#     caching reuses whole blocks only. At --prompt-tokens 6000 (7,299 tokens, 7,263 shared) a block-1600
#     arm recomputes 899 prompt tokens per request against 99 for a block-800 arm -- a 9x handicap that has
#     nothing to do with fp8 arithmetic, and which accounted for roughly half the measured gap at c8.
#     --prompt-tokens 5329 (6,492 total / 6,456 shared) recomputes 92 tokens on BOTH block sizes, so the
#     arms differ only in kernel/format. ALWAYS report the pair; the 6k cell alone overstates the fp8-KV
#     penalty. (Verified with the checkpoint tokenizer; other aligned lengths: 4000 and 6651.)
say "block-aligned 5329-prompt sweep think ON, levels 1 4 8 16 32"
"$BENCH" --url $URL --model qwen --think on --prompt-tokens 5329 --levels "1 4 8 16 32" > "$S/conc_${TAG}_5329_thinkon.log" 2>&1
say "5329 sweep done -> $S/conc_${TAG}_5329_thinkon.log"
# 3) long-context probe: how many ~100k-token requests fit before preemption (the LCR/HLE regime)
P1=$(curl -sm8 $URL/metrics | awk '/^vllm:num_preemptions_total/{print $NF}')
POOL=$(docker logs ${TAG_CONTAINER:-$(docker ps --format '{{.Names}}' | grep -iE 'q38|vllm' | head -1)} 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | head -1)
say "preemptions after sweeps: $P1 (Δ $((P1-P0)))  |  $POOL"
say "CONC TEST COMPLETE — read the agg tok/s knee + preemption Δ; set bench/HLE conc below the knee"
