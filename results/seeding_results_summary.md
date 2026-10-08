# Amorçage : tests statistiques

Hypervolume moyen sur 50 répétitions. (*) : amélioration significative par rapport à W-CMOLS, Mann-Whitney U unilatéral, p < 0,05.

| Instance | W-CMOLS | SW-CMOLS-G | SW-CMOLS-L | SW-CMOLS-U |
|:---|---:|---:|---:|---:|
| 250.2 | 0,7895 | 0,8107* | 0,8144* | 0,8143* |
| 250.3 | 0,4311 | 0,5530* | 0,5575* | 0,5609* |
| 250.4 | 0,1818 | 0,3185* | 0,3245* | 0,3292* |
| 500.2 | 0,7261 | 0,7855* | 0,7822* | 0,7862* |
| 500.3 | 0,3847 | 0,5106* | 0,5260* | 0,5251* |
| 500.4 | 0,1330 | 0,2922* | 0,2992* | 0,3065* |
| 750.2 | 0,6418 | 0,7714* | 0,7708* | 0,7728* |
| 750.3 | 0,3871 | 0,4991* | 0,5065* | 0,5091* |
| 750.4 | 0,1280 | 0,2771* | 0,2815* | 0,2898* |
| Moyenne | 0,4226 | 0,5354 | 0,5403 | 0,5438 |

## Gain sur W-CMOLS

- SW-CMOLS-G : gain moyen +48,0 % (de +2,7 % à +119,6 %), significatif sur 9/9 instances.
- SW-CMOLS-L : gain moyen +50,1 % (de +3,2 % à +124,9 %), significatif sur 9/9 instances.
- SW-CMOLS-U : gain moyen +52,0 % (de +3,1 % à +130,4 %), significatif sur 9/9 instances.

## Variantes entre elles

- SW-CMOLS-L contre SW-CMOLS-G : moyenne supérieure sur 7/9, significativement supérieure sur 7/9, significativement inférieure sur 1/9 (500.2) ; Wilcoxon apparié unilatéral sur les neuf moyennes p = 0,0098.
- SW-CMOLS-U contre SW-CMOLS-G : moyenne supérieure sur 9/9, significativement supérieure sur 9/9, significativement inférieure sur 0/9 ; Wilcoxon apparié unilatéral sur les neuf moyennes p = 0,0020.
- SW-CMOLS-U contre SW-CMOLS-L : moyenne supérieure sur 7/9, significativement supérieure sur 7/9, significativement inférieure sur 1/9 (500.3) ; Wilcoxon apparié unilatéral sur les neuf moyennes p = 0,0098.

## Stabilité sur les instances à deux objectifs

Effondrement : répétition sous la moitié de la médiane de W-CMOLS sur la même instance.

| Instance | Système | Moyenne | Médiane | Écart-type | Minimum | Effondrements |
|:---|:---|---:|---:|---:|---:|---:|
| 250.2 | W-CMOLS | 0,7895 | 0,7898 | 0,0043 | 0,7676 | 0 / 50 |
|  | SW-CMOLS-L | 0,8144 | 0,8144 | 0,0003 | 0,8141 | 0 / 50 |
|  | SW-CMOLS-U | 0,8143 | 0,8143 | 0,0005 | 0,8133 | 0 / 50 |
| 500.2 | W-CMOLS | 0,7261 | 0,7703 | 0,1529 | 0,1771 | 4 / 50 |
|  | SW-CMOLS-L | 0,7822 | 0,7823 | 0,0009 | 0,7800 | 0 / 50 |
|  | SW-CMOLS-U | 0,7862 | 0,7861 | 0,0007 | 0,7850 | 0 / 50 |
| 750.2 | W-CMOLS | 0,6418 | 0,7636 | 0,2478 | 0,1210 | 10 / 50 |
|  | SW-CMOLS-L | 0,7708 | 0,7710 | 0,0017 | 0,7668 | 0 / 50 |
|  | SW-CMOLS-U | 0,7728 | 0,7729 | 0,0019 | 0,7658 | 0 / 50 |
