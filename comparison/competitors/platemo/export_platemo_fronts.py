# -*- coding: utf-8 -*-
"""Exporte les campagnes MOKPR (formulation a reparation native de PlatEMO) au
format canonique des concurrents : mokpr_fronts/raw_<algo>_<inst>.txt (points
de toutes les executions, dans l'ordre), sizes_<algo>_<inst>.txt (nombre de
points par execution) et times_<algo>_<inst>.txt (secondes par execution).

Source : les dossiers campagnes/<algo>/front_<inst>_r<k>.txt et time_... ecrits
par run_campagne.m. Une campagne incomplete est exportee telle quelle ; sizes_
en donne le nombre d'executions.

usage : python export_platemo_fronts.py <dossier campagnes>
"""
import sys
from pathlib import Path

INST = ["250.2", "250.3", "250.4", "500.2", "500.3", "500.4", "750.2", "750.3", "750.4"]
src = Path(sys.argv[1])
dst = Path(__file__).parent / "mokpr_fronts"
dst.mkdir(exist_ok=True)

for d in sorted(p for p in src.iterdir() if p.is_dir()):
    tag = d.name
    total = 0
    for inst in INST:
        runs = sorted((int(f.stem.rsplit("_r", 1)[1]), f) for f in d.glob(f"front_{inst}_r*.txt"))
        if not runs:
            continue
        raw, sizes, times = [], [], []
        for k, f in runs:
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            raw += lines
            sizes.append(str(len(lines)))
            times.append((d / f"time_{inst}_r{k}.txt").read_text().split()[0])
        (dst / f"raw_{tag}_{inst}.txt").write_text("\n".join(raw) + "\n")
        (dst / f"sizes_{tag}_{inst}.txt").write_text("\n".join(sizes) + "\n")
        (dst / f"times_{tag}_{inst}.txt").write_text("\n".join(times) + "\n")
        total += len(runs)
    print(f"{tag}: {total} executions exportees")
