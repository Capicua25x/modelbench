import os
#!/usr/bin/env python3
"""AA-LCR runner: contexto largo (~100K tok) + juez de equivalencia.
Uso: aa_lcr_runner.py <tag> <base_url> [api_key] [limit] [seed]
Salida: $BENCH_OUT_DIR (defecto ./results/aa-lcr)/<tag>_s<seed>.jsonl + resumen."""
import json
import re, os, sys, time, zipfile, urllib.request, concurrent.futures as cf
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Muestreo parametrizable. DEFECTO 0,6/0,95 a propósito: es lo que generó
# TODAS las celdas del marcador, y cambiarlo en silencio movería números ya
# publicados. OJO: 0,6/0,95 es FUERA DE ESPECIFICACIÓN para este checkpoint
# (generation_config.json trae 1,0/1,0). Medido en el item 970, tope 100.000:
# a 0,6/0,95 terminan 14/36; a 1,0/1,0 terminan 18/18 (Fisher p~4e-6).
TEMP = float(os.environ.get("BENCH_TEMP", "0.6"))
TOPP = float(os.environ.get("BENCH_TOPP", "0.95"))

TAG, URL = sys.argv[1], sys.argv[2]
KEY = sys.argv[3] if len(sys.argv) > 3 else ""
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 100
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 1234
OUT = os.environ.get("BENCH_OUT_DIR", "./results/aa-lcr"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "docs")

if not os.path.isdir(CACHE):
    z = hf_hub_download("ArtificialAnalysis/AA-LCR", "extracted_text/AA-LCR_extracted-text.zip", repo_type="dataset")
    os.makedirs(CACHE, exist_ok=True)
    zipfile.ZipFile(z).extractall(CACHE)
IDX = {}
for root, _, files in os.walk(CACHE):
    for f in files:
        IDX[f] = os.path.join(root, f)

MODEL = os.environ.get("BENCH_MODEL", "deepseek-v4-flash")
# Ritmo. Fireworks (router HF) corta a ~140K tokens generados por ventana: a
# concurrencia 4 devuelve 429 en ráfaga y la corrida se llena de errores. Se
# auto-frena a 1 hilo —el mismo ritmo en serie que el par— salvo que se fije
# BENCH_CONC. (Con proveedores que limitan por ventana, ir en serie evita rachas de 429.)
CONC = int(os.environ.get("BENCH_CONC", "0")) or (
    1 if "router.huggingface.co" in (sys.argv[2] if len(sys.argv) > 2 else "") else 4)   # candidato; el juez queda fijo
# 2026-08-08: el runner no mandaba reasoning_effort, así que «high» y «max» eran
# la MISMA corrida con distinta etiqueta. Ahora se envía de verdad (vacío = no
# mandar el campo, que es lo correcto para el par: allí vLLM lo descarta igual).
# Tope de generación. Era 6.000 FIJO (línea 80) y ahí se truncaban 8-11 ítems por
# brazo — vacías que puntúan como fallo. La tarjeta recomienda 384K para
# high/max; aquí basta con dejar de cortar la cola. Defecto 6000 para no
# invalidar en silencio las celdas ya medidas con ese tope.
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "6000"))
EFFORT = os.environ.get("BENCH_EFFORT", "")
EXTRA = json.loads(os.environ.get("BENCH_EXTRA_BODY", "{}"))  # p.ej. chat_template_kwargs

def chat(messages, max_tok, model=None):
    payload = {"model": model or MODEL, "temperature": TEMP, "top_p": TOPP, "seed": SEED,
               "max_tokens": max_tok, "messages": messages}
    if EFFORT:
        payload["reasoning_effort"] = EFFORT
    payload.update(EXTRA)
    body = json.dumps(payload).encode()
    h = {"Content-Type": "application/json", "User-Agent": "curl/8.18.0"}
    if KEY: h["Authorization"] = "Bearer " + KEY
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, body, h), timeout=1800) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            return (m.get("content") or ""), d.get("usage", {})
        except Exception as e:
            code = getattr(e, "code", None)
            # 2026-08-16 (repair at higher caps on a 131k window): vLLM answers 400 when
            # prompt + max_tokens > max_model_len, with the numbers in the message. Clamp
            # max_tokens to what fits and retry ONCE — the item then gets the largest cap its
            # own prompt allows instead of dying as __ERROR__ (window-bound, not model-bound).
            if code == 400 and not payload.get("_clamped"):
                try:
                    msg = e.read().decode(errors="ignore")
                except Exception:
                    msg = str(e)
                mm = re.search(r"maximum context length is (\d+).*?(\d+) (?:tokens )?in the messages", msg, re.S)
                if mm:
                    room = int(mm.group(1)) - int(mm.group(2)) - 64
                    if room > 256:
                        payload["max_tokens"] = room; payload["_clamped"] = True
                        body = json.dumps({k: v for k, v in payload.items() if k != "_clamped"}).encode()
                        print(f"  [clamp] max_tokens -> {room} (window-bound)", flush=True)
                        continue
            cap = 8 if code == 429 else 3          # 429=Model busy → paciencia; 504 → 2 reintentos y null
            if attempt >= cap: return f"__ERROR__{e}", {}
            time.sleep(30 * attempt if code == 429 else 20)

ds = load_dataset("ArtificialAnalysis/AA-LCR", split="test")
def run_item(i):
    x = ds[i]
    docs = []
    for fn in str(x["data_source_filenames"]).split(";"):
        p = IDX.get(fn.strip())
        if p: docs.append(f"=== DOCUMENT: {fn.strip()} ===\n" + open(p, errors="ignore").read())
    ctx = "\n\n".join(docs)
    prompt = (ctx + "\n\n=== QUESTION ===\n" + x["question"] +
              "\n\nAnswer concisely and precisely based only on the documents above.")
    cand, usage = chat([{"role": "user", "content": prompt}], MAXTOK)
    return {"i": i, "qid": x["question_id"], "set": x["document_set_id"],
            "candidate": cand, "reference": x["answer"], "usage": usage}

outfile = f"{OUT}/{TAG}_s{SEED}.jsonl"
done_i = set()
if os.path.exists(outfile):
    for line in open(outfile):
        done_i.add(json.loads(line)["i"])
todo = [i for i in range(min(LIMIT, len(ds))) if i not in done_i]
print(f"{TAG}: {len(todo)} items pendientes", flush=True)
with cf.ThreadPoolExecutor(CONC) as ex, open(outfile, "a") as f:
    for res in ex.map(run_item, todo):
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
        print(f"  item {res['i']} ok ({str(res['usage'].get('prompt_tokens','?'))} in)", flush=True)

# Juez de equivalencia (mismo juez para todas las columnas → pareado justo)
# 2026-08-08: estaba fijado a deepseek-v4-flash, o sea el MISMO modelo que se
# evalúa en las columnas DS4 — el candidato se juzgaba a sí mismo. Cambiado a
# gpt-5.6-luna: juez independiente, y el mismo que usa el arnés de τ².
# Juzgar es más fácil que responder (clasificación binaria con la referencia
# delante), así que un modelo medio sobra; lo que importa es la INDEPENDENCIA.
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gpt-5.6-sol")
JURL = os.environ.get("JUDGE_URL", "")  # OpenAI-compatible chat/completions endpoint for the judge
JKEY = os.environ.get("JUDGE_KEY", "")  # judging is skipped when JUDGE_URL/JUDGE_KEY are unset
_FALLOS_JUEZ = []   # ítems cuyo JUICIO falló (no el modelo)


def judge(rec):
    if rec["candidate"].startswith("__ERROR__"): return 0
    jp = (f"Question: {ds[rec['i']]['question']}\n\nReference answer: {rec['reference']}\n\n"
          f"Candidate answer: {rec['candidate'][:2000]}\n\n"
          "Does the candidate convey the same essential answer as the reference? "
          "Judge substance, not wording. Reply with exactly CORRECT or INCORRECT.")
    body = json.dumps({"model": JUDGE_MODEL, "max_tokens": 400,
                       "messages": [{"role": "user", "content": jp}]}).encode()
    h = {"Content-Type": "application/json", "User-Agent": "curl/8.18.0", "Authorization": "Bearer " + JKEY}
    try:
        with urllib.request.urlopen(urllib.request.Request(JURL, body, h), timeout=600) as r:
            out = (json.load(r)["choices"][0]["message"].get("content") or "")
        return 1 if "CORRECT" in out.upper() and "INCORRECT" not in out.upper() else 0
    except Exception as e:
        # 2026-08-12: devolver -1 y contar sólo v>=0 BORRABA el ítem del
        # denominador — un fallo del JUEZ encogía la corrida en silencio
        # (parlcr_p384k salió 74/99 contra los 100 del fabricante, sin aviso).
        # Un juicio fallido es una medición AUSENTE, no un ítem inexistente.
        for _int in range(3):
            time.sleep(4 * (_int + 1))
            try:
                with urllib.request.urlopen(urllib.request.Request(JURL, body, h), timeout=600) as r:
                    out = (json.load(r)["choices"][0]["message"].get("content") or "")
                return 1 if "CORRECT" in out.upper() and "INCORRECT" not in out.upper() else 0
            except Exception:
                continue
        _FALLOS_JUEZ.append((rec["i"], type(e).__name__))
        return -1
recs = [json.loads(l) for l in open(outfile)]
with cf.ThreadPoolExecutor(CONC) as ex:
    verdicts = list(ex.map(judge, recs))
ok = sum(1 for v in verdicts if v == 1); n = sum(1 for v in verdicts if v >= 0)
# El literal decía «juez: ds4-flash» y siguió diciéndolo después de cambiar el
# juez el 2026-08-08 — o sea que las líneas RESULTADO de esa noche mienten sobre
# quién juzgó. Se imprime la variable para que el log no pueda volver a desviarse.
if _FALLOS_JUEZ:
    print(f"!! AVISO: {len(_FALLOS_JUEZ)} item(s) SIN JUZGAR tras 4 intentos "
          f"-> FUERA del denominador: {_FALLOS_JUEZ}", flush=True)
print(f"RESULTADO {TAG} s{SEED}: {ok}/{n} = {ok/max(1,n):.3f} (juez: {JUDGE_MODEL})")
json.dump({"tag": TAG, "seed": SEED, "score": ok/max(1,n), "n": n, "n_generado": len(recs), "sin_juzgar": _FALLOS_JUEZ},
          open(f"{OUT}/{TAG}_s{SEED}.score.json", "w"))
