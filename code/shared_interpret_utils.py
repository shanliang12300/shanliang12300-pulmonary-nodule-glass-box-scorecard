from __future__ import annotations
import numpy as np
N_BOOT = 2000
SCORECARD_MAX_POINTS = 10
UNITS = {'CEA': 'ng/mL', 'CYFRA': 'ng/mL', 'SCCAg': 'ng/mL', 'CA125_GC': 'U/mL', 'AAG': 'g/L', 'CRP': 'mg/L', 'LYMPH%': '%', 'LYMM': '10⁹/L', 'APTT': 's', 'DD': 'mg/L'}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def wilson_ci(events: int, n: int, z: float=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = events / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(center - half), float(center + half))
