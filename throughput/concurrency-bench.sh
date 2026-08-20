#!/bin/bash
# LLM Benchmark v3 — CONCURRENCY SWEEP
# Measures decode throughput at multiple concurrency levels, reporting BOTH:
#   • per-user tok/s  — what a single user *feels* at that load (the UX number)
#   • aggregate tok/s — total system output (the capacity number)
# plus average request latency. N=1 is the single-stream figure.
#
# Default target is the production vLLM unit (Qwen on :8011, OpenAI /v1/completions).
#
# --prompt-tokens N pads a SHARED prefix to ~N tokens (identical across requests, so it's
# prefix-cacheable) with a unique tail per request — models the ~6K Anchor-scoped schema
# prompt. Use it to get the KV/context-bound concurrency (the real chatbot shape), since
# the default short prompt only shows the compute-bound ceiling.
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
#   ./concurrency-bench.sh --url http://localhost:8011 --model qwen --max-tokens 256

URL="${LLAMA_URL:-http://localhost:8011}"
MODEL=""
MAX_TOKENS=256
PROMPT_TOKENS=0          # 0 = short prompt; >0 pads a shared (prefix-cacheable) prefix to ~N tokens
LEVELS="1 16"            # default: the two the business cares about
FLOOR=20                # per-user tok/s floor for the "practical ceiling" call
THINK="raw"             # raw = /v1/completions raw prompt (historical); on|off = /v1/chat/completions + chat_template_kwargs.enable_thinking

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)           URL="$2";           shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
        --levels)        LEVELS="$2";        shift 2 ;;
        --floor)         FLOOR="$2";         shift 2 ;;
        --think)         THINK="$2";         shift 2 ;;   # raw | on | off
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
echo "  LLM Benchmark v3 — concurrency sweep"
echo "  Server: $URL   Model: $MODEL   max_tokens: $MAX_TOKENS   think: $THINK"
echo "  Levels: $LEVELS   |   floor: ${FLOOR} tok/s   |   prompt: ${PROMPT_TOKENS} tok (0=short)"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

URL="$URL" MODEL="$MODEL" MAX_TOKENS="$MAX_TOKENS" PROMPT_TOKENS="$PROMPT_TOKENS" \
LEVELS="$LEVELS" FLOOR="$FLOOR" THINK="$THINK" python3 - <<'PY'
import os, json, time, urllib.request, concurrent.futures

URL = os.environ["URL"]; MODEL = os.environ["MODEL"]; THINK = os.environ.get("THINK", "raw")
MAXTOK = int(os.environ["MAX_TOKENS"]); FLOOR = float(os.environ["FLOOR"])
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "0"))
LEVELS = [int(x) for x in os.environ["LEVELS"].split()]

_BASE = ("Escribe un ensayo largo y detallado sobre la logistica de una fabrica "
         "de materiales de construccion en Republica Dominicana:")
# Shared prefix (~PROMPT_TOKENS tokens) modeling the Anchor-scoped schema — identical
# across requests (prefix-cacheable); each request appends a unique tail.
_FILLER = ("Contexto del esquema ia.*: v_ventas_detalle (ventas, monto_neto_rd), "
           "v_cxc_detalle (saldo), v_inventario_diario, v_pronostico_demanda, v_churn_clientes. ")
PREFIX = ((_FILLER * (PROMPT_TOKENS * 4 // len(_FILLER) + 1))[:PROMPT_TOKENS * 4]
          if PROMPT_TOKENS > 0 else "")

def one(uid):
    prompt = (PREFIX + f"\n[solicitud {uid}] " + _BASE) if PROMPT_TOKENS > 0 else _BASE
    if THINK == "raw":
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": MAXTOK,
                           "ignore_eos": True, "temperature": 0}).encode()
        req = urllib.request.Request(URL + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    else:
        # chat endpoint so the template's thinking switch applies (production sampling)
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

print(f"  warming up... (prompt ~{PROMPT_TOKENS or 30} tok)"); one(0); one(1)
print()
print(f"  {'users':>5} | {'per-user tok/s':>14} | {'aggregate tok/s':>15} | {'avg latency':>11}")
print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*15}-+-{'-'*11}")

practical_max = LEVELS[0]
for n in LEVELS:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        w0 = time.time()
        res = list(ex.map(one, range(n)))
        wall = time.time() - w0
    total = sum(r[0] for r in res)
    per_user = sum(r[0] / r[1] for r in res) / len(res)
    agg = total / wall
    lat = sum(r[1] for r in res) / len(res)
    flag = "  ← below usable floor" if per_user < FLOOR else ""
    if per_user >= FLOOR:
        practical_max = n
    print(f"  {n:>5} | {per_user:>14.1f} | {agg:>15.0f} | {lat:>9.2f}s{flag}")

print()
print(f"  ➤ Practical ceiling (per-user stays ≥ {FLOOR:.0f} tok/s): ~{practical_max} concurrent users")
PY
echo "=================================================================="
