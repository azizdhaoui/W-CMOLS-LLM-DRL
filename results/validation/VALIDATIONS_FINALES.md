# Validations refaites sur l'agent final

Hors rapport tant que l'utilisateur n'a pas dit oui. Généré par `val_summary.py`.

## 1. Écart à l'oracle empirique (agent UQ final, cloud union)

Instances terminées : 9/9. Étape 1 : chaque configuration faisable, 3 répétitions ; étape 2 : les 12 meilleures + le choix de l'agent, 10 répétitions.

| Instance | Faisables | HV oracle | HV de l'agent | Écart | Rang de l'agent (étape 1) | Config. oracle | Config. agent |
|:---|---:|---:|---:|---:|---:|:---|:---|
| 250.2 | 51 / 432 | 0,8148 | 0,8145 | +0,04 % | 7 / 51 | α=20, NBL=30, L=10, κ=0,1 | α=20, NBL=30, L=8, κ=0,1 |
| 250.3 | 53 / 432 | 0,5595 | 0,5576 | +0,34 % | 21 / 53 | α=20, NBL=30, L=10, κ=0,05 | α=20, NBL=30, L=3, κ=0,05 |
| 250.4 | 32 / 432 | 0,3332 | 0,3301 | +0,92 % | 22 / 32 | α=10, NBL=30, L=10, κ=0,2 | α=15, NBL=30, L=8, κ=0,2 |
| 500.2 | 64 / 432 | 0,7873 | 0,7862 | +0,14 % | 16 / 64 | α=15, NBL=50, L=5, κ=0,05 | α=20, NBL=30, L=8, κ=0,2 |
| 500.3 | 51 / 432 | 0,5228 | 0,5221 | +0,14 % | 2 / 51 | α=10, NBL=50, L=10, κ=0,1 | α=10, NBL=50, L=8, κ=0,1 |
| 500.4 | 41 / 432 | 0,3119 | 0,3077 | +1,33 % | 13 / 41 | α=10, NBL=30, L=3, κ=0,2 | α=10, NBL=50, L=8, κ=0,05 |
| 750.2 | 77 / 432 | 0,7749 | 0,7716 | +0,43 % | 10 / 77 | α=15, NBL=50, L=5, κ=0,2 | α=10, NBL=70, L=5, κ=0,2 |
| 750.3 | 57 / 432 | 0,5117 | 0,5089 | +0,54 % | 38 / 57 | α=10, NBL=30, L=10, κ=0,05 | α=10, NBL=70, L=5, κ=0,1 |
| 750.4 | 43 / 432 | 0,2953 | 0,2921 | +1,08 % | 16 / 43 | α=10, NBL=30, L=8, κ=0,2 | α=10, NBL=50, L=8, κ=0,05 |

- Écart médian : **0,43 %**, moyen : 0,55 %, maximal : 1,33 % (500.4).
- Écart < 1 % sur 7 instance(s) ; l'agent retrouve exactement l'oracle sur 0 instance(s).

## 2. Instance jamais vue (validation croisée, 9 agents entraînés depuis zéro sur 8 instances)

Instances terminées : 9/9. Évaluation : 50 répétitions, budget 0,8 × W-CMOLS, Mann-Whitney U unilatéral contre W-CMOLS.

| Instance exclue | Gain d'HV | p | A12 | Configuration choisie | Temps / budget | Verdict |
|:---|---:|---:|---:|:---|---:|:---|
| 250.2 | +3,10 % | 3.5e-18 | 1,000 | α=15, NBL=30, L=5, κ=0,1 | 0,53 | gagné |
| 250.3 | +29,26 % | 3.5e-18 | 1,000 | α=20, NBL=30, L=3, κ=0,05 | 0,64 | gagné |
| 250.4 | +80,97 % | 3.5e-18 | 1,000 | α=15, NBL=30, L=3, κ=0,05 | 0,65 | gagné |
| 500.2 | +8,20 % | 3.5e-18 | 1,000 | α=25, NBL=30, L=3, κ=0,2 | 0,92 | gagné |
| 500.3 | +35,04 % | 3.5e-18 | 1,000 | α=20, NBL=30, L=3, κ=0,1 | 0,70 | gagné |
| 500.4 | +127,76 % | 3.5e-18 | 1,000 | α=20, NBL=30, L=5, κ=0,05 | 0,67 | gagné |
| 750.2 | +18,65 % | 1.0e+00 | 0,327 | α=25, NBL=30, L=3, κ=0,1 | 0,89 | non conclu |
| 750.3 | +31,65 % | 3.5e-18 | 1,000 | α=20, NBL=30, L=8, κ=0,2 | 0,78 | gagné |
| 750.4 | +125,75 % | 3.5e-18 | 1,000 | α=15, NBL=30, L=5, κ=0,1 | 0,49 | gagné |

- Bilan : **8/9** instances exclues gagnées (p < 0,05).
- Temps cumulé des 50 répétitions : 1022 s contre 1978 s pour W-CMOLS ; dépassement maximal du budget : 0,92 × (500.2).

## 3. Ablation Dueling / rejeu priorisé (cloud glouton, entraînement depuis zéro, 225 épisodes)

| Variante | Gain moyen d'HV vs W-CMOLS | Instances gagnées | HV moyen | Temps total / exécution |
|:---|---:|:---|---:|---:|
| Agent complet (Dueling + rejeu priorisé) | +46,79 % | 9 / 9 | 0,5326 | 24,0 s |
| Sans Dueling | +46,23 % | 9 / 9 | 0,5310 | 22,0 s |
| Sans rejeu priorisé | +46,60 % | 9 / 9 | 0,5318 | 23,7 s |
| *Pour mémoire : GQ final (amorcé, rapport)* | +48,09 % | 9 / 9 | 0,5340 | 25,3 s |

- Complet contre sans dueling : meilleur sur 7/9 instances, écart moyen +0,37 %.
- Complet contre sans rejeu priorisé : meilleur sur 4/9 instances, écart moyen +0,15 %.

## 4. SMAC3 dans les conditions finales, objectif HV (cloud glouton, 25 configurations faisables x 10 répétitions)

| Instance | Config. SMAC3 | HV SMAC3 | HV GQ | Gain SMAC3 vs W | Gain GQ vs W | Solutions SMAC3 / GQ | Temps/exéc. SMAC3 / GQ | Recherche |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 250.2 | α=15, NBL=50, L=5, κ=0,2 | 0,8100 | 0,8097 | +2,60 % | +2,56 % | 106 / 105 | 0,25 / 0,31 s | 1,3 min |
| 250.3 | α=10, NBL=70, L=10, κ=0,2 | 0,5493 | 0,5457 | +27,41 % | +26,58 % | 732 / 684 | 0,64 / 0,54 s | 2,6 min |
| 250.4 | α=10, NBL=30, L=10, κ=0,2 | 0,3238 | 0,3213 | +78,08 % | +76,70 % | 947 / 1219 | 0,52 / 0,94 s | 18,0 min |
| 500.2 | α=20, NBL=30, L=10, κ=0,2 | 0,7851 | 0,7845 | +8,12 % | +8,05 % | 154 / 146 | 0,93 / 0,84 s | 3,2 min |
| 500.3 | α=10, NBL=70, L=10, κ=0,1 | 0,5058 | 0,5051 | +31,48 % | +31,32 % | 1231 / 1235 | 2,02 / 1,65 s | 5,8 min |
| 500.4 | α=10, NBL=30, L=8, κ=0,1 | 0,3028 | 0,2959 | +127,60 % | +122,42 % | 1330 / 1802 | 1,20 / 2,49 s | 24,9 min |
| 750.2 | α=15, NBL=50, L=3, κ=0,1 | 0,7737 | 0,7674 | +20,56 % | +19,58 % | 327 / 290 | 1,57 / 1,47 s | 6,7 min |
| 750.3 | α=10, NBL=30, L=10, κ=0,05 | 0,5018 | 0,4986 | +29,65 % | +28,83 % | 791 / 1574 | 1,35 / 2,97 s | 10,1 min |
| 750.4 | α=10, NBL=30, L=5, κ=0,05 | 0,2844 | 0,2774 | +122,24 % | +116,79 % | 1864 / 2820 | 2,26 / 7,48 s | 36,4 min |

- SMAC3 : +49,75 % en moyenne, gagne 9/9 ; GQ sur les mêmes instances : +48,09 %. SMAC3 meilleur que GQ en HV sur 9/9.
- Solutions par exécution (somme des instances) : SMAC3 7481, GQ 9875.
- Coût de recherche de SMAC3 : 2250 exécutions, 109 min au total, max 36,4 min sur une instance.
- Temps par exécution (somme) : SMAC3 10,7 s, GQ 18,7 s (même jour).

## 5. SMAC3 avec la note de l'agent (HV + cardinalité − pénalité de temps), mêmes conditions

| Instance | Config. SMAC3 | HV SMAC3 | HV GQ | Gain SMAC3 vs W | Gain GQ vs W | Solutions SMAC3 / GQ | Temps/exéc. SMAC3 / GQ | Recherche |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 250.2 | α=25, NBL=30, L=5, κ=0,2 | 0,8102 | 0,8097 | +2,63 % | +2,56 % | 114 / 105 | 0,36 / 0,31 s | 1,5 min |
| 250.3 | α=15, NBL=50, L=8, κ=0,05 | 0,5456 | 0,5457 | +26,55 % | +26,58 % | 697 / 684 | 0,67 / 0,54 s | 2,8 min |
| 250.4 | α=20, NBL=30, L=8, κ=0,05 | 0,3175 | 0,3213 | +74,62 % | +76,70 % | 1473 / 1219 | 1,32 / 0,94 s | 18,7 min |
| 500.2 | α=20, NBL=30, L=10, κ=0,05 | 0,7849 | 0,7845 | +8,10 % | +8,05 % | 153 / 146 | 0,95 / 0,84 s | 2,8 min |
| 500.3 | α=10, NBL=70, L=5, κ=0,05 | 0,5050 | 0,5051 | +31,29 % | +31,32 % | 1242 / 1235 | 1,60 / 1,65 s | 6,6 min |
| 500.4 | α=15, NBL=50, L=5, κ=0,1 | 0,2877 | 0,2959 | +116,30 % | +122,42 % | 2500 / 1802 | 4,49 / 2,49 s | 26,8 min |
| 750.2 | α=10, NBL=70, L=5, κ=0,1 | 0,7695 | 0,7674 | +19,89 % | +19,58 % | 294 / 290 | 2,20 / 1,47 s | 6,3 min |
| 750.3 | α=10, NBL=70, L=5, κ=0,1 | 0,4986 | 0,4986 | +28,83 % | +28,83 % | 1574 / 1574 | 2,99 / 2,97 s | 12,0 min |
| 750.4 | α=10, NBL=70, L=8, κ=0,2 | 0,2784 | 0,2774 | +117,55 % | +116,79 % | 2920 / 2820 | 8,42 / 7,48 s | 37,0 min |

- SMAC3 : +47,31 % en moyenne, gagne 9/9 ; GQ sur les mêmes instances : +48,09 %. SMAC3 meilleur que GQ en HV sur 4/9.
- Solutions par exécution (somme des instances) : SMAC3 10966, GQ 9875.
- Coût de recherche de SMAC3 : 2250 exécutions, 115 min au total, max 37,0 min sur une instance.
- Temps par exécution (somme) : SMAC3 23,0 s, GQ 18,7 s (même jour).

## 6. Temps de GQ remesuré le même jour que SMAC3 (mêmes germes, 50 répétitions)

- HV identique à la campagne canonique : oui (déterminisme).
- Temps total par exécution : 18,7 s le même jour, contre 25,3 s dans la campagne canonique.

## 7. Temps des sept systèmes remesurés le même jour, en alternance (50 répétitions)

Instances terminées : 9/9 ; HV identique à la campagne canonique : 63/63 mesures.

| Système | Temps / exéc. (même jour) | Temps / exéc. (rapport) |
|:---|---:|---:|
| W | 31,7 s | 39,6 s |
| G | 36,4 s | 45,0 s |
| GQ | 19,4 s | 25,3 s |
| L | 47,4 s | 60,0 s |
| LQ | 21,6 s | 28,3 s |
| U | 44,2 s | 56,6 s |
| UQ | 20,1 s | 25,8 s |

- GQ contre G : −47 % le même jour (rapport : −44 %).
- LQ contre L : −54 % le même jour (rapport : −53 %).
- UQ contre U : −55 % le même jour (rapport : −54 %).
- UQ contre W : −36 % le même jour (rapport : −35 %).
- GQ plus rapide que W-CMOLS, même jour : 9/9 instances.
- LQ plus rapide que W-CMOLS, même jour : 9/9 instances.
- UQ plus rapide que W-CMOLS, même jour : 9/9 instances.
- UQ / budget (0,8 × W du même jour) : de 0,69 à 1,06 ; sous le budget sur 8/9.
- Instance jamais vue : 1022 s contre 1583 s pour W-CMOLS remesuré (au lieu de 1 978 s mesurés le 23 sept.).
