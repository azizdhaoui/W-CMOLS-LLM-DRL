# Amorçage informé et configuration par instance de W-CMOLS

Code et résultats du mémoire de mastère *Utilisation des LLM et du DRL dans une approche fondée sur la
recherche locale pour résoudre le problème du sac à dos multi-objectif* (ENSI, LARIA, 2026).

Le point de départ est le solveur **W-CMOLS** (Ben Mansour, Basseur et Saubion, 2018), appliqué aux neuf
instances de Zitzler et Thiele (1999) : 250, 500 et 750 objets, 2, 3 et 4 objectifs.

- **Contribution 1 — SW-CMOLS** : l'archive initialement vide est remplacée par un *cloud* de solutions :
  glouton (**G**), construit avec un modèle de langage (**L**), ou union des deux (**U**).
- **Contribution 2 — SW-CMOLS-Q** : un agent DQN choisit les quatre paramètres (α, NBL, L, κ) avant chaque
  exécution, sous un budget de temps (**GQ**, **LQ**, **UQ**).
- **Validation** : oracle empirique, instance exclue de l'entraînement, ablation, SMAC3, et comparaison
  à treize algorithmes de la littérature (chapitre 6).

## Organisation

```
paths.py, common.py      emplacements du dépôt ; instances, graines gloutonnes, voisins, solveur, hypervolume
data/instances/          neuf instances (format Zitzler-Thiele) et fichiers de vecteurs de poids de W-CMOLS
data/reference/          bornes de normalisation de l'hypervolume et références de W-CMOLS, par instance
solver/                  W-CMOLS en Cython (moacp_noreinj : solveur de tous les résultats rapportés)
rl_agent/                calcul de l'hypervolume normalisé
rl_agent_v3/             agent DQN : réseau autorégressif à têtes Dueling, masque de budget, environnement
contribution1/           clouds G, L, U ; prompts et décisions du modèle de langage (llm/)
contribution2/           modèle de coût, entraînement et évaluation de l'agent
validation/              oracle empirique, SMAC3, temps remesurés le même jour, synthèse
comparison/              chapitre 6 : tableaux, indicateur epsilon ; lanceurs des concurrents (competitors/)
figures/                 scripts des figures du mémoire
clouds/                  clouds utilisés : greedy_cloud (G), v8b_llm (L), u2 (U), v8b_control (L sans LLM)
models/                  politiques entraînées, modèles de coût, budgets, trace de décision
results/                 résultats des campagnes (JSON) et tableaux de synthèse (Markdown)
```

## Installation

```bash
pip install -r requirements.txt
cd solver && python setup.py build_ext --inplace     # nécessite un compilateur C (MSVC, gcc ou clang)
```

## Correspondance avec le mémoire

| Mémoire | Script | Résultat |
|:---|:---|:---|
| Tab. 4.1 (nombre de directions) | `contribution1/control_n.py` | `results/screening/` |
| Tab. 4.2, 4.3, 4.5 (W-CMOLS, G, L, U) | `contribution1/final_runs.py` | `results/{baseline,G,L,U2}_nbl100_<inst>.json` |
| Tab. 4.4 (effet propre du LLM) | `contribution1/final_runs.py ctrl`, `contribution1/llm_effect.py` | `results/ctrl_nbl100_*`, `results/LLM_EFFECT.md` |
| Modèle de coût (§5.3.1) | `contribution2/fit_cost_model.py`, `calibrate_cost_model.py` | `models/cost_model_*.json` |
| Tab. 5.1, 5.2 (GQ, LQ, UQ) | `contribution2/train_v31.py`, `contribution2/c2_screen.py` | `results/c2_*_r0.8_ng_retrain_n50.json`, `results/C2_CH5_FINAL.md` |
| Contrôle sans cloud (§5.4.2) | `contribution2/c2_screen.py WQ` | `results/c2_WQ_r0.8_ng_retrain_n50.json` |
| Tab. 5.3 (oracle empirique) | `validation/val_oracle_final.py` | `results/validation/oracle_*.json` |
| Tab. 5.4 (instance exclue) | `contribution2/train_v31.py --exclude`, `c2_screen.py` | `results/c2_UQ_r0.8_ng_loo_n50.json` |
| Tab. 5.5 (ablation) | `contribution2/train_v31.py --no-dueling / --no-per` | `results/c2_GQ_r0.8_ng_ablfinal_*.json` |
| SMAC3 (§5.5.3) | `validation/val_smac_final.py` | `results/validation/smac_final*.json` |
| Temps du même jour | `validation/val_retime.py` | `results/validation/retime_sameday.json` |
| Synthèse des validations | `validation/val_summary.py` | `results/validation/VALIDATIONS_FINALES.md` |
| Tab. 6.2 à 6.7 | `comparison/ch6_v8.py`, `comparison/ch6_tables.py` | `results/CH6_V8.json`, `results/CH6_TABLES.md` |

Toutes les exécutions emploient les mêmes germes : relancer un script reproduit les hypervolumes rapportés.

## Reproduire

Commandes lancées depuis la racine du dépôt.

**Contribution 1.** Le cloud G est produit par `contribution1/gen_weighted_greedy_seeds.py`. Pour le cloud L,
`python contribution1/pipeline2.py prompts` écrit un prompt par graine centrale (45 au total, dans
`contribution1/llm/prompts/`) ; les réponses du modèle de langage (Claude) sont consignées dans
`contribution1/llm/decisions/` et rejouées à l'identique, sans nouvel appel au modèle, par
`python contribution1/pipeline2.py run llm`. Évaluation (50 exécutions par instance) :

```bash
python contribution1/final_runs.py baseline 100 250.2 250.3 250.4 500.2 500.3 500.4 750.2 750.3 750.4
python contribution1/final_runs.py U2 100 250.2 ...       # de même : G, L, ctrl
```

**Contribution 2.** Entraînement de la politique de UQ (celle de GQ : `--mode greedy` et
`models/cost_model_final_G.json`, sans `--seeds-pattern`) :

```bash
python contribution2/train_v31.py --mode union --budget-ratio 1 --suffix _final \
  --warm-start models/ckpt_v31_greedy_b1_n10/checkpoint.pt --episodes-per-instance 25 --n-reward-runs 3 \
  --epsilon-start 0.4 --nu 0.2 --no-speed-bonus --budgets-json models/budgets_final_r0.8.json \
  --solver noreinj --fresh-times models/baseline_times_final.json --lambda3 0.10 --guard if_fits \
  --seeds-pattern "clouds/u2_{inst}.txt" --cost-model models/cost_model_final_U.json
```

Évaluation (50 exécutions, budget 0,8 × temps de W-CMOLS) :

```bash
RUNS=50 NOGUARD=1 CKPT_TAG=union_b1_final TAG=_retrain python contribution2/c2_screen.py UQ 0.8
RUNS=50 NOGUARD=1 CKPT_TAG=greedy_b1_final TAG=_retrain python contribution2/c2_screen.py GQ 0.8
```

## Données non incluses

- Les fronts bruts de chaque exécution (`results/fronts/`, environ 1 Go) et ceux des concurrents
  (`competitor_fronts/`) ne sont pas versionnés ; les valeurs qui en sont tirées sont dans `results/`.
  Les scripts du chapitre 6 et des figures les lisent à ces emplacements.
- Les concurrents sont des implémentations publiées, employées sans modification algorithmique :
  PlatEMO (MATLAB), PISA, mobkp et le code C++ de MOEA/D. Seuls les lanceurs et adaptateurs d'entrée/sortie
  sont fournis (`comparison/competitors/`) ; les emplacements des binaires se règlent par variables
  d'environnement (`PISA_DIR`, `MOEAD_SRC`, `MOEAD_EXE`, `MOBKP_BUILD`).
