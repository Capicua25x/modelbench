#!/usr/bin/env python3
"""Re-judges ALREADY-generated responses with another judge, without regenerating
anything.

Why it exists: the HLE and AA-LCR runners fixed the judge to
`deepseek-v4-flash` — the same model the DS4 columns evaluate, i.e. self-
assessment. The candidate responses are saved in the .jsonl files, so changing
judge costs only the judge calls, not the generation.

Usage:
    rejudge.py hle  <file.jsonl> <judge-model>
    rejudge.py lcr  <file.jsonl> <judge-model>
    rejudge.py agree <file.jsonl> <judgeA> <judgeB>   # agreement between judges
"""
import concurrent.futures as cf
import json, os
import os
import sys
import urllib.request

from datasets import load_dataset

# Judge endpoint: any OpenAI-compatible chat/completions API. Configure via env.
JUDGE = (os.environ.get("JUDGE_URL", ""), os.environ.get("JUDGE_KEY", ""))
def _route(model):
    return JUDGE


def ask(model, prompt):
    URL, KEY = _route(model)
    # NO `temperature`: the Claude 5 models reject it with a 400 ("deprecated for
    # this model") while luna accepts it. A judge that only works with some
    # models can't be used to compare judges. The provider's defect suffices: the
    # task is answering one word with the reference in front of it.
    body = json.dumps({"model": model, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    h = {"Content-Type": "application/json", "User-Agent": "curl/8.18.0",
         "Authorization": "Bearer " + KEY}
    last = ""
    for _ in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, body, h), timeout=600) as r:
                d = json.load(r)
            if "error" in d:                       # e.g. MonthlyLimitError
                last = "__ERR__" + str(d["error"].get("type", "?"))
                continue
            return (d["choices"][0]["message"].get("content") or "").upper()
        except Exception as exc:
            last = "__ERR__" + type(exc).__name__
            continue
    raise RuntimeError(f"judge gave no answer after 4 attempts: {last}")


def prompt_for(kind, rec, ds):
    if kind == "hle":
        q = ds[rec["i"]]["question"][:1500]
        return (f"Question: {q}\n\nCorrect answer: {rec['reference']}\n\nStudent response:\n"
                f"{rec['candidate'][:2500]}\n\nDid the student's final Answer match the correct "
                "answer in substance (exact fact/choice, wording irrelevant)? "
                "Reply exactly CORRECT or INCORRECT.")
    return (f"Question: {ds[rec['i']]['question']}\n\nReference answer: {rec['reference']}\n\n"
            f"Candidate answer: {rec['candidate'][:2000]}\n\n"
            "Does the candidate convey the same essential answer as the reference? "
            "Judge substance, not wording. Reply with exactly CORRECT or INCORRECT.")


def verdicts(kind, recs, ds, model):
    def one(r):
        if r["candidate"].startswith("__ERROR__"):
            return 0
        out = ask(model, prompt_for(kind, r, ds))
        return 1 if "CORRECT" in out and "INCORRECT" not in out else 0
    with cf.ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, recs))


def main(argv):
    mode, path = argv[0], argv[1]
    kind = "hle" if mode in ("hle", "agree") and "/hle/" in path else ("hle" if mode == "hle" else "lcr")
    ds = (load_dataset("cais/hle", split="test") if kind == "hle"
          else load_dataset("ArtificialAnalysis/AA-LCR", split="test"))
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if mode == "agree":
        a, b = argv[2], argv[3]
        va, vb = verdicts(kind, recs, ds, a), verdicts(kind, recs, ds, b)
        same = sum(1 for x, y in zip(va, vb) if x == y)
        print(f"n={len(recs)}  {a}: {sum(va)}/{len(va)} = {sum(va)/len(va):.3f}")
        print(f"        {b}: {sum(vb)}/{len(vb)} = {sum(vb)/len(vb):.3f}")
        print(f"agreement: {same}/{len(recs)} = {100*same/len(recs):.1f}%")
        print(f"discrepancies: {len(recs)-same}  (>5% => the judge MATTERS, use the strong one)")
    else:
        v = verdicts(kind, recs, ds, argv[2])
        print(f"{path} judge={argv[2]}: {sum(v)}/{len(v)} = {sum(v)/max(1,len(v)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
