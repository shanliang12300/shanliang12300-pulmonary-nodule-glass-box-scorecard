from __future__ import annotations
import os
import numpy as np
import pandas as pd
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
RISK_CSV = os.path.join(OUTPUT_ROOT, 'Fig5_外部验证', '00_计算结果与原始数据', 'val_risk_by_score.csv')
BLOOD_RULES = [('LYMPH%', (20.0, 50.0), -3, '%'), ('CEA', (None, 5.0), 5, 'ng/mL'), ('APTT', (25.0, 35.0), 1, 's'), ('AU_Urea', (2.5, 7.1), -2, 'mmol/L'), ('MCHC', (320.0, 360.0), 6, 'g/L'), ('AAG', (0.55, 1.4), 2, 'g/L')]
CT_SIGN_RULES = [('CT_基线分叶', 10), ('CT_基线毛刺', 6)]
DIAMETER_RULES = [('CT_直径8–15mm', 8, 15, 4), ('CT_直径15–30mm', 15, 30, -2), ('CT_直径>30mm', 30, None, 3)]

def compute_score(blood: dict, diameter_mm: float | None, lobulation: bool, spiculation: bool) -> tuple[int, list]:
    details, total = ([], 0)
    for name, (lo, hi), pts, _unit in BLOOD_RULES:
        v = blood.get(name)
        hit = False
        if v is not None:
            hit = hi is not None and v > hi or (lo is not None and v < lo)
        got = pts if hit else 0
        total += got
        details.append((f'{name} 异常', hit, got))
    for name, pts in CT_SIGN_RULES:
        hit = lobulation if '分叶' in name else spiculation
        got = pts if hit else 0
        total += got
        details.append((name, bool(hit), got))
    for name, lo, hi, pts in DIAMETER_RULES:
        hit = False
        if diameter_mm is not None:
            hit = diameter_mm > lo and (hi is None or diameter_mm <= hi)
        got = pts if hit else 0
        total += got
        details.append((name, hit, got))
    return (total, details)

def load_risk_table() -> pd.DataFrame | None:
    if not os.path.exists(RISK_CSV):
        return None
    df = pd.read_csv(RISK_CSV, encoding='utf-8-sig')
    df = df[df['组别'] == '合并'][['score', 'events', 'n']].copy()
    df['rate'] = (df['events'] + 1) / (df['n'] + 2)
    return df.sort_values('score').reset_index(drop=True)

def risk_rate(score: int) -> float | None:
    tab = load_risk_table()
    if tab is None or tab.empty:
        return None
    x, y = (tab['score'].to_numpy(float), tab['rate'].to_numpy(float))
    return float(np.clip(np.interp(score, x, y), 0, 1))

def risk_band(rate: float) -> str:
    if rate < 0.3:
        return '低危'
    if rate < 0.6:
        return '中危'
    return '高危'
RULE_NAME_EN = {'LYMPH% 异常': 'LYMPH% abnormal', 'CEA 异常': 'CEA abnormal', 'APTT 异常': 'APTT abnormal', 'AU_Urea 异常': 'Urea abnormal', 'MCHC 异常': 'MCHC abnormal', 'AAG 异常': 'AAG abnormal', 'CT_基线分叶': 'Lobulation', 'CT_基线毛刺': 'Spiculation', 'CT_直径8–15mm': 'Diameter 8–15 mm', 'CT_直径15–30mm': 'Diameter 15–30 mm', 'CT_直径>30mm': 'Diameter >30 mm'}
BAND_EN = {'低危': 'Low', '中危': 'Intermediate', '高危': 'High'}
