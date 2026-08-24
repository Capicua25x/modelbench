#!/bin/bash
# vLLM Concurrency Bench v4 — CONCURRENCY SWEEP
# Measures decode throughput at multiple concurrency levels, reporting BOTH:
#   • per-user tok/s  — what a single user *feels* at that load (the UX number)
#   • aggregate tok/s — total system output (the capacity number)
# plus average request latency and, when the server exposes spec-decode counters,
# per-cell acceptance (accepted/draft) and tokens-per-step. N=1 is the single-stream figure.
#
# Default target is a local vLLM server (OpenAI-compatible /v1/completions).
#
# v4 (2026-08-24) — CONTENT-REPLAY FIX + ACCEPTANCE VISIBILITY. v3's essay workload used
# ONE fixed prompt at temperature 0, so every request (and every warm re-run) regenerated
# the same text. On servers with speculative decoding that is a double confound:
#   1. REPLAY INFLATION — a stateful context-window drafter (DSpark-style) drafts text it
#      has recently seen far better than novel text (measured up to 4.2 vs ~1.9
#      accepted/draft for the same prompt shape), so warm re-runs inflate with the
#      server's content history (boot freshness, run order, which workload ran last).
#   2. CONTENT VARIANCE — prose acceptance is strongly topic-dependent (measured 1.3-4.2
#      accepted/draft across topics at n=1, same server, minutes apart), so any
#      single-prompt number is one draw from a wide distribution.
# Measured on a GB10 pair: the SAME server config scored 87-153 aggregate tok/s at n=6
# with zero config change — a spread previously misattributed to a serving-patch
# regression. v4 therefore (a) rotates distinct essay topics per request with a
# per-invocation nonce and temp 0.7, so no request regenerates text the drafter has seen;
# (b) prints per-cell accepted/draft and tok/step from /metrics so acceptance effects are
# visible instead of latent. Run MANY cells and average — topic variance is intrinsic.
# Essay numbers from v4 are NOT comparable to v3 logs (v3's were replay-inflated on
# spec-decode servers). --trivial is deliberately unchanged (community comparability).
#
# --prompt-tokens N pads a SHARED prefix to ~N tokens (identical across requests, so it's
# prefix-cacheable) with a unique tail per request — models a schema/tool-heavy system
# prompt. Use it to get the KV/context-bound concurrency (the real chatbot shape), since
# the default short prompt only shows the compute-bound ceiling.
#
# --trivial swaps the workload for the community peak-finder ("count to 300" per Tony/hdub):
# 19-token prompt, natural termination, temp=0, MTP acceptance ~99%. This is what the pair
# reports as ~86 tok/s at n=1 and what published headlines from peer stacks (DGX Spark, etc.)
# mean when they cite a single tok/s figure. Use it to cross-check against community numbers;
# do NOT read it as a realistic decode rate — see conc-logs/ds4-pair-patchA-2026-08-23.md
# for the workload-mismatch lesson (compare like-for-like before investigating a perceived
# gap: 30 seconds of `--trivial` up front saves an afternoon of chasing).
#
# Measured 2026-05-31 (Qwen3.6-35B-A3B MXFP4, vLLM TP2, dual R9700):
#   short-prompt (compute-bound): n1=64, n16=46/u, n96=21/u, n128=17/u; ceiling ~96.
#   KV pool 387,552 tok @ max_model_len 16384. Real ~6K-prompt KV knee ~64 (no prefix cache).
#   Production capped at --max-num-seqs 64 with --enable-prefix-caching (re-measure with
#   --prompt-tokens 6000 for the true context-aware curve).
#
# Usage:
#   ./concurrency-bench.sh                                  # default sweep (n1, n16) on :8011
#   ./concurrency-bench.sh --levels "1 16 32 64"            # custom levels
#   ./concurrency-bench.sh --ceiling                        # wide sweep 1->128
#   ./concurrency-bench.sh --prompt-tokens 6000 --levels "1 16 32 48 64"   # realistic prompt
#   ./concurrency-bench.sh --trivial                        # community peak-finder (count-to-300)
#   ./concurrency-bench.sh --trivial --levels "1 4 8"       # trivial-workload concurrency curve
#   ./concurrency-bench.sh --url http://localhost:8000 --model qwen --max-tokens 256

URL="${LLAMA_URL:-http://localhost:8000}"
MODEL=""
MAX_TOKENS=256
PROMPT_TOKENS=0          # 0 = short prompt; >0 pads a shared (prefix-cacheable) prefix to ~N tokens
LEVELS="1 16"            # default: single-stream and a light-concurrency sanity check
FLOOR=20                # per-user tok/s floor for the "practical ceiling" call
THINK="raw"             # raw = /v1/completions raw prompt (historical); on|off = /v1/chat/completions + chat_template_kwargs.enable_thinking
WORKLOAD="essay"        # essay (default: rotating-topic prose, temp 0.7, ignore_eos, sustained decode — replay-proof, see v4 note) | trivial (--trivial: count-to-300, natural stop, temp=0 — community peak-finder)

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)           URL="$2";           shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --levels)        LEVELS="$2";        shift 2 ;;
        --floor)         FLOOR="$2";         shift 2 ;;
        --think)         THINK="$2";         shift 2 ;;   # raw | on | off
        --trivial)       WORKLOAD="trivial"; MAX_TOKENS=4000; PROMPT_TOKENS=0; shift ;;
        --ceiling)       LEVELS="1 4 8 16 24 32 48 64 96 128"; shift ;;
        *) shift ;;
    esac
done

# GB10 hard cap (DGX Spark class: 140W SoC, 1.13L, firmware-controlled cooling, thermal trip ~104.8C).
# Decode is bandwidth-bound on this SoC: one stream already saturates the memory subsystem, so levels
# past ~6 measure queueing, not capacity — and sustained max-concurrency is the documented
# thermal-shutdown regime. When the bench runs ON a GB10 box, levels above 6 are removed.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "GB10"; then
    NEWLEVELS=""; CLAMPED=0
    for L in $LEVELS; do if [ "$L" -gt 6 ] 2>/dev/null; then CLAMPED=1; else NEWLEVELS="$NEWLEVELS $L"; fi; done
    if [ "$CLAMPED" = 1 ]; then
        case " $NEWLEVELS " in *" 6 "*) : ;; *) NEWLEVELS="$NEWLEVELS 6" ;; esac
        LEVELS="${NEWLEVELS# }"
        echo "  ⚠ GB10 detected: levels >6 removed (thermal envelope; cap the serve with --max-num-seqs 6 too). Levels: $LEVELS"
    fi
fi

if [ -z "$MODEL" ]; then
    MODEL=$(curl -s --max-time 5 "$URL/v1/models" | python3 -c "import sys,json
try: print(json.load(sys.stdin)['data'][0]['id'])
except: print('')" 2>/dev/null)
    [ -z "$MODEL" ] && { echo "❌ Could not detect a model at $URL/v1/models — is the server up?"; exit 1; }
fi

echo "=================================================================="
echo "  vLLM Concurrency Bench v3 — concurrency sweep"
echo "  Server: $URL   Model: $MODEL   max_tokens: $MAX_TOKENS   think: $THINK   workload: $WORKLOAD"
echo "  Levels: $LEVELS   |   floor: ${FLOOR} tok/s   |   prompt: ${PROMPT_TOKENS} tok (0=short)"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

URL="$URL" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" PROMPT_TOKENS="$PROMPT_TOKENS" \
LEVELS="$LEVELS" FLOOR="$FLOOR" THINK="$THINK" WORKLOAD="$WORKLOAD" python3 - <<'PY'
import os, json, time, urllib.request, concurrent.futures, itertools, random, string

URL = os.environ["URL"]; MODEL = os.environ["MODEL"]; THINK = os.environ.get("THINK", "raw")
WORKLOAD = os.environ.get("WORKLOAD", "essay")
MAXTOK = int(os.environ["MAX_TOKENS"]); FLOOR = float(os.environ["FLOOR"])
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "0"))
LEVELS = [int(x) for x in os.environ["LEVELS"].split()]

# Distinct essay topics, rotated per request via a global counter, tagged with a
# per-invocation nonce, generated at temp 0.7 — see the v4 header note. A stateful
# context-window drafter (DSpark-style) replays text it has recently seen; a bench
# that lets two requests generate the same text measures the drafter's memory, not
# the server's throughput.
_TOPICS = [
    "the history and architecture of distributed database systems",
    "how container orchestration schedulers make placement decisions",
    "the evolution of instruction set architectures since the 1970s",
    "consensus protocols and why they are hard to implement correctly",
    "the design trade-offs of columnar versus row-oriented storage",
    "how modern compilers decide what to inline and what to vectorize",
    "the engineering history of undersea communication cables",
    "memory allocators and the fragmentation problems they solve",
    "the development of public-key cryptography and its deployment",
    "how time synchronization works across datacenters",
    "the architecture of modern content delivery networks",
    "garbage collection strategies in managed language runtimes",
    "the design of fault-tolerant filesystems for commodity hardware",
    "how query optimizers estimate cost and why they get it wrong",
    "the evolution of GPU architectures for general-purpose compute",
    "network congestion control from TCP Reno to BBR",
    "the engineering behind high-frequency trading infrastructure",
    "schema migration strategies in continuously deployed systems",
]
_NONCE = "".join(random.choices(string.ascii_lowercase, k=6))
_REQ_COUNTER = itertools.count()

def _essay_prompt():
    i = next(_REQ_COUNTER)
    topic = _TOPICS[i % len(_TOPICS)]
    return f"[{_NONCE}-{i}] Write a long, detailed essay about {topic}:"
# Community peak-finder (Tony/hdub): trivial prompt, digits out, natural termination — high
# MTP acceptance (~99%) exposes the compute-bound headline number. NOT a realistic decode rate.
_TRIVIAL = "Count from 1 to 300, separated by commas. Numbers only."
# Shared prefix (~PROMPT_TOKENS tokens) modeling a schema/tool-heavy system prompt —
# identical across requests (prefix-cacheable); each request appends a unique tail. Trivial
# workload forces PROMPT_TOKENS=0 (bench-shell already does this) so PREFIX stays empty for it.
_FILLER = ("System context: a generic multi-table schema with orders, customers, "
           "inventory, invoices, and demand forecasts. ")
PREFIX = ((_FILLER * (PROMPT_TOKENS * 4 // len(_FILLER) + 1))[:PROMPT_TOKENS * 4]
          if PROMPT_TOKENS > 0 else "")

def one(uid):
    if WORKLOAD == "trivial":
        # Community peak-finder — match count300.py / Tony's shape exactly: chat/completions,
        # no chat_template_kwargs override (server's baked-in default applies), natural
        # termination, temp=0. --think is IGNORED here; the whole point of --trivial is
        # comparability with the published number, so the wire shape is fixed. Per-request
        # tag defeats prefix-cache dedup at n>1 without changing decode rate (19-token
        # prompt, prefill is trivial either way).
        prompt = f"[req {uid}] " + _TRIVIAL
        body = json.dumps({"model": MODEL,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": MAXTOK, "temperature": 0.0}).encode()
        req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    elif THINK == "raw":
        essay = _essay_prompt()
        prompt = (PREFIX + "\n" + essay) if PROMPT_TOKENS > 0 else essay
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAXTOK,
                           "ignore_eos": True, "temperature": 0.7, "top_p": 0.95}).encode()
        req = urllib.request.Request(URL + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    else:
        # chat endpoint so the template's thinking switch applies (production sampling).
        essay = _essay_prompt()
        prompt = (PREFIX + "\n" + essay) if PROMPT_TOKENS > 0 else essay
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": MAXTOK, "ignore_eos": True,
                           "temperature": 0.6, "top_p": 0.95,
                           "chat_template_kwargs": {"enable_thinking": THINK == "on"}}).encode()
        req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", MAXTOK)
    return ct, dt

def spec_counters():
    """Spec-decode counters (drafts, accepted, generated) or None if not exposed."""
    try:
        m = urllib.request.urlopen(URL.rsplit("/v1", 1)[0] + "/metrics", timeout=3).read().decode()
    except Exception:
        return None
    vals = {}
    for l in m.splitlines():
        for k in ("spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
                  "generation_tokens_total"):
            if l.startswith(f"vllm:{k}{{"):
                vals[k] = vals.get(k, 0.0) + float(l.rsplit(" ", 1)[1])
    if "spec_decode_num_drafts_total" not in vals:
        return None
    return (vals["spec_decode_num_drafts_total"],
            vals.get("spec_decode_num_accepted_tokens_total", 0.0),
            vals.get("generation_tokens_total", 0.0))

print(f"  warming up... (prompt ~{PROMPT_TOKENS or 30} tok, workload nonce {_NONCE})"); one(0); one(1)
print()
print(f"  {'users':>5} | {'per-user tok/s':>14} | {'aggregate tok/s':>15} | {'avg latency':>11} | {'acc/draft':>9} | {'tok/step':>8}")
print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*15}-+-{'-'*11}-+-{'-'*9}-+-{'-'*8}")

def settle():
    """Wait for the server to drain between levels — stragglers from the previous
    cell otherwise bleed into the next cell's wall clock (observed as a mid-grid
    dip, e.g. a c4 cell scoring below c6). Polls vLLM's /metrics for zero running
    requests when exposed; falls back to a fixed pause."""
    base = URL.rsplit("/v1", 1)[0]
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            m = urllib.request.urlopen(base + "/metrics", timeout=3).read().decode()
            runn = sum(float(l.split()[-1]) for l in m.splitlines()
                       if l.startswith("vllm:num_requests_running"))
            if runn == 0:
                time.sleep(3)   # scheduler cool-down after drain
                return
            time.sleep(2)
        except Exception:
            time.sleep(10)      # no metrics endpoint: fixed settle
            return

practical_max = LEVELS[0]
for n in LEVELS:
    settle()
    c0 = spec_counters()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        w0 = time.time()
        res = list(ex.map(one, range(n)))
        wall = time.time() - w0
    c1 = spec_counters() if c0 else None
    total = sum(r[0] for r in res)
    per_user = sum(r[0] / r[1] for r in res) / len(res)
    agg = total / wall
    lat = sum(r[1] for r in res) / len(res)
    acc_s, tps_s = "-", "-"
    if c0 and c1 and c1[0] > c0[0]:
        dd = c1[0] - c0[0]
        acc_s = f"{(c1[1] - c0[1]) / dd:.2f}"
        tps_s = f"{(c1[2] - c0[2]) / dd:.2f}"
    flag = "  ← below usable floor" if per_user < FLOOR else ""
    if per_user >= FLOOR:
        practical_max = n
    print(f"  {n:>5} | {per_user:>14.1f} | {agg:>15.0f} | {lat:>9.2f}s | {acc_s:>9} | {tps_s:>8}{flag}")

print()
print(f"  ➤ Practical ceiling (per-user stays ≥ {FLOOR:.0f} tok/s): ~{practical_max} concurrent users")
PY
echo "=================================================================="
