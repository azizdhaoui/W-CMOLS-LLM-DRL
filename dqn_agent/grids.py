"""
v3 action grids — FINAL, chosen from measured 1D response profiles (Phase 1,
2026-06-11, results/grid_sensitivity_profiles.json). Value-by-value justification with the
measured curves: GRID_DESIGN_V3.md.

Summary of the data:
    NBL   : dominant param. 2-obj saturates at 50-70; 3-obj rises through 200;
            4-obj jumps +41% at 130 then saturates at 160. -> extend to 130, 160.
            (200 excluded: reward-negative under lambda time penalty.)
    alpha : <=10 catastrophic on 2-obj without seeds (bimodal failures);
            flat 30->40. -> drop dead 50, keep 10 (Ben Mansour default).
    L     : second-order. L=2 dominated; L=10 best on 4-obj. -> add 10.
    kappa : flat everywhere incl. 0.3 (censoring NOT confirmed). -> unchanged.
"""

GRIDS_V3: dict[str, tuple] = {
    "alpha": (10, 15, 20, 25, 30, 40),       # 6 — was 7 (dropped dead 50)
    "NBL":   (30, 50, 70, 100, 130, 160),    # 6 — was 4 (censoring fixed: +130, +160)
    "L":     (3, 5, 8, 10),                  # 4 — was 3 (+10, lifts possible censoring)
    "kappa": (0.05, 0.1, 0.2),               # 3 — unchanged (measured flat)
}
