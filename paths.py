"""Emplacements du dépôt, partagés par tous les scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "instances"          # neuf instances de Zitzler et Thiele + fichiers de poids
REFERENCE = ROOT / "data" / "reference"     # bornes d'hypervolume et références de W-CMOLS par instance
SOLVER = ROOT / "solver"                    # W-CMOLS (Cython)
CLOUDS = ROOT / "clouds"                    # clouds G, L, U et témoin sans LLM
LLM = ROOT / "seeding" / "llm"        # prompts et décisions du modèle de langage
MODELS = ROOT / "models"                    # politiques DQN, modèles de coût, budgets
RESULTS = ROOT / "results"                  # résultats des campagnes (JSON)
FRONTS = RESULTS / "fronts"                 # fronts bruts par exécution (non versionnés)
COMPETITOR_FRONTS = ROOT / "competitor_fronts"  # fronts bruts des concurrents (non versionnés)
FIGURES = ROOT / "figures" / "images"       # figures produites
WORK = ROOT / "work"                        # fichiers temporaires du solveur (non versionnés)
