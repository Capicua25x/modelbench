#!/usr/bin/env python3
"""Mia's cooperative-MoE benchmark protocol — reproduced runner (2026-09-16).

Implements exactly what MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks documents
(docs/cooperative-moe.md "Measurement protocol"), so our pair's numbers sit next to
extensions/cooperative_moe/benchmarks/results.json:

  Sampled workloads  poetry / code / incident — seeds 11/23/47, max_tokens 512,
                     temp 1.0, top_p 0.95, streaming with usage block; a 32-token
                     warmup precedes each workload; thinking off except incident.
  Reference C1       one stream: the hash-map prompt, temp 0, top_p 1, thinking off,
                     400 outputs. decode = (ct-1)/(last_content - first_content).
  Reference C2       two streams with a start barrier, prompts suffixed
                     " (stream 1/2)" / " (stream 2/2)".
                     aggregate = sum(ct-1) over [earliest first content, latest last].
  Cold prefill       32K unique prefix, reply OK: prompt_tokens / TTFT.
  Acceptance         vllm spec-decode counters (accepted/proposed) around each phase.

usage:
  mia_protocol.py --base http://127.0.0.1:8888/v1 --model DeepSeek-v4.1-Flash-EXL3 \
                  --out ~/logs --label dsv41exl3-stock [--reps 3]
"""
import argparse
import json
import re
import statistics as st
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKLOADS = Path.home() / "dsv41-exl3/extensions/cooperative_moe/benchmarks/workloads.json"
HASHMAP = ("Write a detailed step-by-step explanation of how a hash map works, "
           "including collision handling, resizing, and time complexity. Be thorough.")


def metrics(base):
    root = base.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    g = lambda n: sum(float(x) for x in re.findall(
        r"^%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)$" % re.escape(n), text, re.M)) or 0.0
    return {"acc": g("vllm:spec_decode_num_accepted_tokens_total"),
            "draft": g("vllm:spec_decode_num_draft_tokens_total")}


def stream_chat(base, model, prompt, max_tokens, temperature, top_p, thinking,
                seed=None, timeout=1800):
    """One streaming request; per-phase first/last CONTENT times (her formula) plus
    first/last of ANY piece (content or reasoning) for context."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    c_first = c_last = a_first = a_last = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or "")
                if not piece:
                    continue
                now = time.perf_counter()
                if a_first is None:
                    a_first = now
                a_last = now
                if (d.get("content") or ""):
                    if c_first is None:
                        c_first = now
                    c_last = now
    ct = int((usage or {}).get("completion_tokens", 0))
    pt = int((usage or {}).get("prompt_tokens", 0))
    dec = ((ct - 1) / (c_last - c_first)) if (c_first and c_last and ct > 1 and c_last > c_first) else None
    return {"ct": ct, "pt": pt, "decode": dec,
            "ttft_content": (c_first - t0) if c_first else None,
            "ttft_any": (a_first - t0) if a_first else None,
            "c_first": c_first, "c_last": c_last, "total": time.perf_counter() - t0}


def c2_batch(base, model, max_tokens=400, timeout=1800):
    """Two streams, start barrier; aggregate over the shared content window."""
    c = 2
    barrier = threading.Barrier(c)
    def one(i):
        p = f"{HASHMAP} (stream {i + 1}/{c})"
        barrier.wait()
        return stream_chat(base, model, p, max_tokens, 0, 1, False, timeout=timeout)
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(one, range(c)))
    first = min(r["c_first"] for r in res)
    last = max(r["c_last"] for r in res)
    toks = sum(r["ct"] - 1 for r in res)
    agg = toks / (last - first) if last > first else None
    return {"agg_tok_s": agg, "per_stream": [r["decode"] for r in res],
            "ct": [r["ct"] for r in res]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="DeepSeek-v4.1-Flash-EXL3")
    ap.add_argument("--out", default=str(Path.home() / "logs"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    W = json.loads(WORKLOADS.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"label": args.label, "model": args.model, "protocol": "mia-cooperative-moe",
           "prompt_set": str(WORKLOADS), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "sampled": [], "c1": [], "c2": [], "prefill": [], "acceptance": []}

    def snap(tag):
        m = metrics(args.base)
        return {"tag": tag, **m}

    # ---- sampled workloads: poetry / code / incident x seeds -----------------
    for name, spec in W["cases"].items():
        a0 = snap(f"{name}:start")
        print(f"[{name}] warmup", flush=True)
        stream_chat(args.base, args.model, W["warmup"]["prompt"],
                    W["warmup"]["max_tokens"], W["temperature"], W["top_p"],
                    W["warmup"].get("thinking", False), seed=W["warmup"].get("seed"))
        per_seed = []
        for seed in W["seeds"]:
            r = stream_chat(args.base, args.model, spec["prompt"], W["max_tokens"],
                            W["temperature"], W["top_p"], spec.get("thinking", False),
                            seed=seed)
            per_seed.append(round(r["decode"], 2) if r["decode"] else None)
            print(f"  {name} seed={seed}: decode {r['decode'] and round(r['decode'], 2)} tok/s "
                  f"(ct={r['ct']})", flush=True)
        a1 = snap(f"{name}:end")
        decs = [x for x in per_seed if x]
        row = {"workload": name, "per_seed": per_seed,
               "median": round(st.median(decs), 2) if decs else None,
               "acceptance": round(a1["acc"] / a1["draft"], 4) if a1.get("draft") else None}
        res["sampled"].append(row)
        res["acceptance"].append({"phase": name, "start": a0, "end": a1})
        print(f"  {name}: median {row['median']} tok/s | acc {row['acceptance']}", flush=True)

    # ---- reference C1 / C2 ---------------------------------------------------
    for i in range(args.reps):
        a0 = snap(f"c1[{i}]:start")
        r = stream_chat(args.base, args.model, HASHMAP, 400, 0, 1, False)
        a1 = snap(f"c1[{i}]:end")
        res["c1"].append({"rep": i, "tok_s": round(r["decode"], 2) if r["decode"] else None})
        res["acceptance"].append({"phase": f"c1[{i}]", "start": a0, "end": a1})
        print(f"  C1[{i}]: {res['c1'][-1]['tok_s']} tok/s", flush=True)
    for i in range(args.reps):
        a0 = snap(f"c2[{i}]:start")
        b = c2_batch(args.base, args.model)
        a1 = snap(f"c2[{i}]:end")
        res["c2"].append({"rep": i, "agg_tok_s": round(b["agg_tok_s"], 2) if b["agg_tok_s"] else None,
                          "per_stream": [round(x, 2) if x else None for x in b["per_stream"]]})
        res["acceptance"].append({"phase": f"c2[{i}]", "start": a0, "end": a1})
        print(f"  C2[{i}]: agg {res['c2'][-1]['agg_tok_s']} tok/s {res['c2'][-1]['per_stream']}", flush=True)

    # ---- cold 32K prefill probe ---------------------------------------------
    nonce = str(int(time.time()))
    prompt = f"[mia prefill probe {nonce}]" + " the" * 32000 + "\n\nReply with the single word OK."
    r = stream_chat(args.base, args.model, prompt, 8, 0, 1, False, timeout=1800)
    row = {"target_tokens": 32000, "prompt_tokens": r["pt"],
           "ttft_s": round(r["ttft_content"], 3) if r["ttft_content"] else None,
           "prefill_tok_s": round(r["pt"] / r["ttft_content"], 1) if r["ttft_content"] else None}
    res["prefill"].append(row)
    print(f"  prefill 32K: {row}", flush=True)

    # ---- summary -------------------------------------------------------------
    c1s = [x["tok_s"] for x in res["c1"] if x["tok_s"]]
    c2s = [x["agg_tok_s"] for x in res["c2"] if x["agg_tok_s"]]
    res["summary"] = {
        "sampled_medians": {r["workload"]: r["median"] for r in res["sampled"]},
        "c1_median": round(st.median(c1s), 2) if c1s else None,
        "c1_all": c1s,
        "c2_median": round(st.median(c2s), 2) if c2s else None,
        "c2_all": c2s,
        "prefill_32k_tok_s": res["prefill"][0]["prefill_tok_s"],
    }
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / f"bench-{args.label}.json").write_text(json.dumps(res, indent=1))
    print("summary:", json.dumps(res["summary"], indent=1), flush=True)
    print(f"wrote {out}/bench-{args.label}.json", flush=True)


if __name__ == "__main__":
    main()
