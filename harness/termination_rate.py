#!/usr/bin/env python3
"""Tasa de terminación de un run lm-eval con --log_samples.

Un ítem "termina" si el modelo cerró el thinking y emitió contenido (resp no vacía).
Uso: termination_rate.py <output_dir del run>   (busca samples_*.jsonl recursivo)
"""
import glob
import json
import sys

files = glob.glob(f"{sys.argv[1]}/**/samples_*.jsonl", recursive=True)
if not files:
    sys.exit(f"sin samples_*.jsonl bajo {sys.argv[1]} — ¿corriste con --log_samples?")
total = done = 0
for path in files:
    for line in open(path):
        s = json.loads(line)
        resp = (s.get("resps") or [[""]])[0][0] or ""
        total += 1
        done += bool(resp.strip())
print(f"{done}/{total} terminados = {done/total:.1%}  (vacíos: {total-done})")
