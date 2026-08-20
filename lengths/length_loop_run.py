#!/usr/bin/env python3
"""Reasoning-length + loop-rate instrument for a single hard item (default: HLE item 970).

WHY THIS INSTRUMENT EXISTS
    A runaway run burns the full token cap (100k) and ~65 min while producing
    nothing after the collapse point. A capture measured where the failure
    actually happens: text novelty drops to ZERO around token ~7,853 / ~13,773
    and from there the model repeats verbatim until the cap. So this measures
    the LOOP instead of the CAP: once novelty collapses in a sustained way, the
    run is sentenced and (optionally) cut.

WHY THIS DOES NOT REPEAT THE 40k-CAP MISTAKE
    An earlier battery was discarded for counting "hit the cap" as failure,
    when 64% of the vendor's HEALTHY runs exceed 40k. The error was the OUTCOME
    VARIABLE, not the cap. Here the variable is "did it loop?", read from the
    text, not from the budget. A run that hits the cap with novelty still alive
    is NOT a failure: it is reported as UNDECIDED, separately.

OUTCOMES
    TERMINATES   emitted final content: a success, with its length.
    LOOP         3 consecutive novelty windows < 2% new 8-grams: sentenced.
                 The loop-start token is recorded — arms can be compared by
                 WHERE they start failing, not only how often.
    UNDECIDED    hit the cap without terminating and without collapsing.
                 Reported separately, never counted as failure.

MANDATORY CONTROL BEFORE USING IT ON AN ARM
    Run it once with CUT=0 on a configuration whose rate you already know, and
    check (a) it reproduces the known rate and (b) ZERO false positives — no
    run that was sentenced went on to terminate. Only then use CUT=1.
    (Validated on the original stack: sentences 3 captured runaways at tokens
    7,853 / 13,773 / 29,698; 0 false positives across the terminating controls.)

REQUEST SHAPE (kept deliberately explicit — the request IS the experiment):
    POST {BENCH_URL}  with temperature/top_p/seed/max_tokens/stream(+usage),
    frequency_penalty/presence_penalty/repetition_penalty, and NO
    chat_template_kwargs unless EFFORT is set (EFFORT=none sends none, so the
    stock template renders NO effort preamble; on the stock DeepSeek-V4-Flash
    vLLM template only xhigh/max reach the template and add the +79-token
    preamble that the vendor renders by default).

ENV
    BENCH_URL   endpoint (default http://localhost:8888/v1/chat/completions)
    MODEL       served model name (default deepseek-v4-flash)
    ITEM        HLE item index (default 970)
    CAP         max_tokens (default 45000)
    N_REP       repetitions (default 6)
    SLOTS       concurrent requests (default 6) — concurrency is PART of the arm
    TAG         label for the output file (default "run")
    SEED0       first seed; rep k uses SEED0+k (default 3000)
    CUT         1 = cut sentenced runs, 0 = detect but let them run (default 1)
    TEMP/TOPP   sampling (default 0.6 / 0.95)
    FREQ_PEN/PRES_PEN/REP_PEN  penalties (defaults 0 / 0 / 1.0)
    EFFORT      none | low | medium | high | xhigh | max (default none)
    BENCH_OUT_DIR  output dir (default ./results)
"""
import json
import os
import re
import time
import urllib.request
import concurrent.futures as cf

from datasets import load_dataset

ITEM = int(os.environ.get("ITEM", "970"))
CAP = int(os.environ.get("CAP", "45000"))
N = int(os.environ.get("N_REP", "6"))
SLOTS = int(os.environ.get("SLOTS", "6"))
TAG = os.environ.get("TAG", "run")
SEED0 = int(os.environ.get("SEED0", "3000"))
CUT = os.environ.get("CUT", "1") == "1"
TEMP = float(os.environ.get("TEMP", "0.6"))
TOPP = float(os.environ.get("TOPP", "0.95"))
FREQ_PEN = float(os.environ.get("FREQ_PEN", "0"))
PRES_PEN = float(os.environ.get("PRES_PEN", "0"))
REP_PEN = float(os.environ.get("REP_PEN", "1.0"))
EFFORT = os.environ.get("EFFORT", "none")
URL = os.environ.get("BENCH_URL", "http://localhost:8888/v1/chat/completions")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
OUT = os.environ.get("BENCH_OUT_DIR", "./results")
os.makedirs(OUT, exist_ok=True)

WINDOW = 4000      # chars per novelty window
THRESHOLD = 0.02   # novelty below this = exhausted
STREAK = 3         # consecutive dry windows to sentence

SYS = ("Your response should be in the following format:\nExplanation: {your explanation}\n"
       "Exact Answer: {your succinct, final answer}\nConfidence: {your confidence score}%")

ds = load_dataset("cais/hle", split="test")
QUESTION = ds[ITEM]["question"]


def one(seed):
    body = {"model": MODEL, "temperature": TEMP, "top_p": TOPP,
            "seed": seed, "max_tokens": CAP, "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": FREQ_PEN, "presence_penalty": PRES_PEN,
            "repetition_penalty": REP_PEN,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": QUESTION}]}
    if EFFORT != "none":
        body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": EFFORT}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    cont, reason, usage = [], [], {}
    seen, dry, mark, chars = set(), 0, 0, 0
    loop_at = None
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=10800)
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            p = line[5:].strip()
            if p == "[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta") or {}
                rc = dl.get("reasoning") or dl.get("reasoning_content")
                if rc:
                    reason.append(rc)
                    chars += len(rc)
                if dl.get("content"):
                    cont.append(dl["content"])
            # close a novelty window?
            if chars - mark >= WINDOW:
                chunk = "".join(reason)[mark:]
                mark = chars
                words = re.findall(r"\w+", chunk.lower())
                sh = {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
                nov = len(sh - seen) / len(sh) if sh else 1.0
                seen |= sh
                dry = dry + 1 if nov < THRESHOLD else 0
                if dry >= STREAK and loop_at is None:
                    # the loop started at the first of the dry windows
                    loop_at = mark - WINDOW * STREAK
                    if CUT:
                        r.close()  # sentenced: don't burn 30k more tokens
                        break
    except Exception as e:
        if loop_at is None:
            return dict(seed=seed, outcome="TRANSPORT", err=type(e).__name__,
                        s=round(time.time() - t0))
    text = "".join(reason)
    gen = usage.get("completion_tokens", 0)
    # without usage (we cut the stream) estimate via the item's chars/token ratio
    cpt = (len(text) / gen) if gen else 3.8
    # Order matters: ask FIRST whether it terminated. With CUT=0 a run can carry
    # a loop verdict AND still terminate — that is exactly the false positive the
    # control looks for, and checking the loop first would hide it.
    loop_tok = int(loop_at / cpt) if loop_at is not None else None
    if "".join(cont).strip():
        return dict(seed=seed, outcome="TERMINATES", tok=gen,
                    false_positive=loop_at is not None, loop_tok=loop_tok,
                    s=round(time.time() - t0))
    if loop_at is not None:
        return dict(seed=seed, outcome="LOOP", loop_tok=loop_tok,
                    tok=gen or int(len(text) / cpt), s=round(time.time() - t0))
    return dict(seed=seed, outcome="UNDECIDED", tok=gen,
                s=round(time.time() - t0))


if __name__ == "__main__":
    print(f"   item {ITEM} · {N} reps · cap {CAP:,} · {SLOTS} slots · "
          f"t{TEMP}/p{TOPP} · effort={EFFORT} · cut={int(CUT)} · [{TAG}]"
          f"   (outcome = LOOP, not = cap)\n", flush=True)
    res = []
    with open(f"{OUT}/loop_rate_{TAG}.jsonl", "a") as out, \
         cf.ThreadPoolExecutor(max_workers=SLOTS) as ex:
        futs = [ex.submit(one, SEED0 + k) for k in range(N)]
        for fut in cf.as_completed(futs):
            r = fut.result()
            res.append(r)
            out.write(json.dumps({**r, "cap": CAP, "ts": time.strftime("%F %T")}) + "\n")
            out.flush()
            mk = {"TERMINATES": "OK ", "LOOP": "LOOP", "UNDECIDED": "??? ",
                  "TRANSPORT": "ERR "}[r["outcome"]]
            extra = (f"  loop from ~{r['loop_tok']:,}" if r.get("loop_tok") is not None
                     else "")
            print(f"   {mk} seed {r['seed']}  {r.get('tok', 0):>7,} tok  "
                  f"{r['s']:>5d}s  {r['outcome']}{extra}", flush=True)

    term = sum(1 for r in res if r["outcome"] == "TERMINATES")
    loop = sum(1 for r in res if r["outcome"] == "LOOP")
    und = sum(1 for r in res if r["outcome"] == "UNDECIDED")
    tr = sum(1 for r in res if r["outcome"] == "TRANSPORT")
    print(f"\n   -- TERMINATES {term}  ·  LOOP {loop}  ·  UNDECIDED {und}"
          f"{'  ·  TRANSPORT ' + str(tr) if tr else ''}")
    if term + loop:
        print(f"   -- success rate over decided runs: {term}/{term+loop} = "
              f"{term/(term+loop):.1%}")
    if und:
        print(f"   -- NOTE: {und} undecided. They do NOT count as failures. "
              f"If there are many, raise CAP.")
    starts = sorted(r["loop_tok"] for r in res
                    if r["outcome"] == "LOOP" and r.get("loop_tok") is not None)
    if starts:
        print(f"   -- loop start (tokens): {starts}")

    fp = [r for r in res if r.get("false_positive")]
    if not CUT:
        print(f"\n   -- DETECTOR VALIDATION (CUT=0, every run went to the cap)")
        if fp:
            print(f"      X {len(fp)} FALSE POSITIVES: sentenced yet terminated -> "
                  f"{[(r['seed'], r['loop_tok'], r['tok']) for r in fp]}")
            print(f"      The detector is NOT ready. Raise STREAK or lower THRESHOLD "
                  f"before using CUT=1.")
        else:
            print(f"      OK zero false positives across {term} successes: no run "
                  f"that terminated had been sentenced.")
    elif fp:
        print(f"\n   X {len(fp)} false positives despite CUT=1 (should not happen)")
