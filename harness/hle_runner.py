#!/usr/bin/env python3
"""HLE (subset texto) runner — plantilla oficial + juez model_graded_fact compartido.
Uso: hle_runner.py <tag> <base_url> [api_key] [n] [seed]"""
import json, os, sys, time, random, urllib.request, concurrent.futures as cf
from datasets import load_dataset

TAG, URL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 else ""
N = int(sys.argv[4]) if len(sys.argv) > 4 else 120
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 1234
OUT = os.environ.get("BENCH_OUT_DIR", "./results/hle"); os.makedirs(OUT, exist_ok=True)
SYS = ("Your response should be in the following format:\n\n"
       "Explanation: {your explanation for your answer choice}\n"
       "Answer: {your chosen answer}\n"
       "Confidence: {your confidence score between 0% and 100% for your answer}")

ds = load_dataset("cais/hle", split="test")
text_idx = [i for i in range(len(ds)) if not ds[i]["image"]]
random.Random(SEED).shuffle(text_idx)
sample = sorted(text_idx[:N])

MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")
# Ritmo. Fireworks (router HF) corta a ~140K tokens generados por ventana: a
# concurrencia 4 devuelve 429 en ráfaga y la corrida se llena de errores. Se
# auto-frena a 1 hilo —el mismo ritmo en serie que el par— salvo que se fije
# BENCH_CONC. (Con proveedores que limitan por ventana, ir en serie evita rachas de 429.)
CONC = int(os.environ.get("BENCH_CONC", "0")) or (
    1 if "router.huggingface.co" in (sys.argv[2] if len(sys.argv) > 2 else "") else 4)
# 2026-08-08: el tope estaba fijo en 12.000 y las corridas salían con 55-72% de
# __ERROR__/vacías, contadas como respuestas MALAS. La única completa (la referencia)
# promedió 8.384 tok/ítem: 12K cortaba los duros a mitad. Ahora configurable.
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "60000"))
# 2026-08-08: el timeout estaba FIJO en 3600s y se comió 6 ítems de parhle con
# `__ERROR__timed out` y 0 tokens — no fallo del modelo, el socket cerrado. A los
# 19,4 tok/s que daba el motor con 6 streams compitiendo, una hora son ~70K
# tokens; esos ítems necesitaban más. Con el motor libre un stream va a ~50 tok/s,
# pero el tope debe poder subir para presupuestos de 300K.
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "3600"))
EFFORT = os.environ.get("BENCH_EFFORT", "max")   # candidato; el juez queda fijo
EXTRA = json.loads(os.environ.get("BENCH_EXTRA_BODY", "{}"))  # p.ej. chat_template_kwargs
STREAM = os.environ.get("BENCH_STREAM", "") == "1"           # transporte streaming (evita 504 de gateway)

def _stream_read(r):
    content, usage, reasoning, fin = [], {}, [], []
    for raw in r:
        line = raw.decode(errors="ignore").strip()
        if not line.startswith("data:"): continue
        p = line[5:].strip()
        if p == "[DONE]": break
        d = json.loads(p)
        if d.get("usage"): usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content"): content.append(delta["content"])
            # 2026-08-17: record reasoning volume and finish_reason separately. Providers that
            # stream a reasoning channel can emit 30K+ reasoning tokens and zero content; without
            # this the row is just "empty" and indistinguishable from a dropped-content bug.
            # Reasoning is NEVER used as the answer -- that would let the judge grade a scratchpad.
            if delta.get("reasoning"): reasoning.append(delta["reasoning"])
            if ch.get("finish_reason"): fin.append(ch["finish_reason"])
    usage["_reasoning_chars"] = sum(len(x) for x in reasoning)
    usage["_finish_reason"] = fin[-1] if fin else None
    return "".join(content), usage

# Muestreo. El DEFECTO se deja en 0,6/0,95 a propósito: es lo que produjo TODAS
# las celdas publicadas,
# y cambiarlo en silencio movería números ya publicados.
#
# OJO — 0,6/0,95 es FUERA DE ESPECIFICACIÓN para este checkpoint. Su
# generation_config.json trae `temperature 1.0, top_p 1.0`, y el README pide
# 1.0 con top_p 1.0 salvo en escenarios agénticos. Medido en el ítem 970, tope
# 100.000: a 0,6/0,95 terminan 14/36 = 39%; a 1,0/1,0 terminan 18/18 = 100%
# (Fisher p≈4e-6). O sea que el muestreo del marcador degrada la terminación en
# ítems de razonamiento largo, y cualquier celda vieja hay que leerla así.
TEMP = float(os.environ.get("BENCH_TEMP", "0.6"))
TOPP = float(os.environ.get("BENCH_TOPP", "0.95"))


def chat(url, key, messages, max_tok, extra=None, model="deepseek-v4-flash", want_usage=False,
         temp=None, topp=None):
    payload = {"model": model,
               "temperature": TEMP if temp is None else temp,
               "top_p": TOPP if topp is None else topp,
               "seed": SEED, "max_tokens": max_tok, "messages": messages}
    if extra: payload.update(extra)
    stream = STREAM and want_usage                 # sólo llamadas candidato; el juez va sin stream
    if stream: payload.update({"stream": True, "stream_options": {"include_usage": True}})
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", "User-Agent": "curl/8.18.0"}
    if key: h["Authorization"] = "Bearer " + key
    a = 0
    while True:
        a += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(url, body, h), timeout=TIMEOUT) as r:
                if stream:
                    c, u = _stream_read(r)
                else:
                    d = json.load(r)
                    c = (d["choices"][0]["message"].get("content") or "")
                    u = d.get("usage", {})
            return (c, u) if want_usage else c
        except Exception as e:
            code = getattr(e, "code", None)
            cap = 8 if code == 429 else 3          # 429 → paciencia; 504 → 2 reintentos y null
            if a >= cap:
                return (f"__ERROR__{e}", {}) if want_usage else f"__ERROR__{e}"
            time.sleep(30 * a if code == 429 else 15)

def run_item(i):
    x = ds[i]
    cand, usage = chat(URL, KEY, [{"role": "system", "content": SYS},
                                  {"role": "user", "content": x["question"]}],
                       MAXTOK, {"reasoning_effort": EFFORT, **EXTRA}, model=MODEL, want_usage=True)
    return {"i": i, "candidate": cand, "reference": x["answer"], "type": x["answer_type"],
            "usage": usage}

outfile = f"{OUT}/{TAG}_s{SEED}.jsonl"
done = set()
if os.path.exists(outfile):
    done = {json.loads(l)["i"] for l in open(outfile)}
todo = [i for i in sample if i not in done]
print(f"{TAG}: {len(todo)} pendientes de {N} | modelo={MODEL} effort={EFFORT} tope={MAXTOK}", flush=True)
_first_err = []
with cf.ThreadPoolExecutor(CONC) as ex, open(outfile, "a") as f:
    for res in ex.map(run_item, todo):
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
        bad = str(res["candidate"]).startswith("__ERROR__")
        if bad and not _first_err:                 # la PRIMERA excepción, entera
            _first_err.append(res["candidate"][:300])
            print(f"  !! PRIMER ERROR: {_first_err[0]}", flush=True)
        print(f"  {res['i']} {'ERR' if bad else 'ok'}", flush=True)

JKEY = os.environ.get("JUDGE_KEY", "")
# 2026-08-08: el juez heredaba el `model` por defecto de chat() —o sea
# deepseek-v4-flash— así que en los brazos DS4 el modelo se juzgaba A SÍ MISMO.
# Se fija a gpt-5.6-luna vía el endpoint del juez (JUDGE_URL): juez independiente del candidato, y el mismo
# simulador que ya usa el arnés de τ². Configurable por entorno.
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gpt-5.6-sol")
def judge(rec):
    if rec["candidate"].startswith("__ERROR__"): return 0
    q = ds[rec["i"]]["question"][:1500]
    jp = (f"Question: {q}\n\nCorrect answer: {rec['reference']}\n\nStudent response:\n"
          f"{rec['candidate'][:2500]}\n\nDid the student's final Answer match the correct "
          "answer in substance (exact fact/choice, wording irrelevant)? Reply exactly CORRECT or INCORRECT.")
    # 2026-08-10: el juez comparte chat() con el candidato, así que BENCH_TEMP/
    # BENCH_TOPP se le colarían y un brazo cambiaría CANDIDATO **y** JUEZ a la
    # vez. Se fija a 0,6/0,95, que es con lo que se juzgaron todas las celdas
    # anteriores. Misma clase de fallo que el juez heredando el `model`.
    out = chat(os.environ.get("JUDGE_URL", ""), JKEY,
               [{"role": "user", "content": jp}], 300, model=JUDGE_MODEL,
               temp=0.6, topp=0.95)
    return 1 if "CORRECT" in out.upper() and "INCORRECT" not in out.upper() else 0
recs = [json.loads(l) for l in open(outfile)]
_err = sum(1 for r in recs if str(r["candidate"]).startswith("__ERROR__") or not str(r["candidate"]).strip())
_pct = 100.0 * _err / max(1, len(recs))
print(f"{TAG}: generación con {_err}/{len(recs)} fallidas ({_pct:.1f}%)", flush=True)
if _pct > 10:
    raise SystemExit(f"ABORTA: {_pct:.1f}% de generaciones fallidas — el score sería ruido, no capacidad")
with cf.ThreadPoolExecutor(CONC) as ex:
    v = list(ex.map(judge, recs))
print(f"RESULTADO {TAG} s{SEED}: {sum(v)}/{len(v)} = {sum(v)/max(1,len(v)):.3f}")
# 2026-08-11: el veredicto POR ÍTEM no se persistía — sólo el agregado. Por eso
# parhle y parhle_onspec tenían score pero ni b ni c, y el McNemar hubo que
# recuperarlo re-juzgando las 240 respuestas. El texto sobrevive, el juicio no:
# omitir esto es IRRECUPERABLE salvo pagando otra pasada de juez. Se escribe
# siempre, junto al muestreo/esfuerzo reales, para que la celda sea auditable.
with open(f"{OUT}/{TAG}_s{SEED}.verdicts.jsonl", "w") as _g:
    for _r, _v in zip(recs, v):
        _g.write(json.dumps({"i": _r["i"], "v": _v, "juez": JUDGE_MODEL,
                             "vacio": not str(_r["candidate"]).strip()
                             or str(_r["candidate"]).startswith("__ERROR__")}) + "\n")
json.dump({"tag": TAG, "seed": SEED, "score": sum(v)/max(1,len(v)), "n": len(v),
           "temp": TEMP, "top_p": TOPP, "effort": EFFORT, "max_tokens": MAXTOK,
           "timeout": TIMEOUT, "conc": CONC, "juez": JUDGE_MODEL, "modelo": MODEL},
          open(f"{OUT}/{TAG}_s{SEED}.score.json", "w"))
