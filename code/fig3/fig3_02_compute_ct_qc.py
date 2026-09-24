from __future__ import annotations
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while not _os.path.exists(_os.path.join(_root, 'common_preprocessing.py')):
    _root = _os.path.dirname(_root)
for _p in (_root, _here):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import numpy as np
import pandas as pd
from scipy import stats
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
MERGED = os.path.join(OUTPUT_ROOT, '队列合并数据', '队列_血液+影像_合并表.csv')
OUT_DIR = os.path.join(OUTPUT_ROOT, 'Fig3_CT数据QC', '00_计算结果与原始数据')
LABEL_COL = '标签_诊断表'
CONT_FEATURES = {'CT_基线最大径mm': '基线最大径 (mm)', 'CT_历史最大径mm': '历史最大径 (mm)', 'CT_报告数': '胸部CT报告数', 'CT_随访跨度天': '随访跨度 (天)'}
BOOL_FEATURES = {'CT_基线GGN': '磨玻璃结节', 'CT_基线部分实性': '部分实性', 'CT_基线多发': '多发结节', 'CT_基线毛刺': '毛刺', 'CT_基线分叶': '分叶', 'CT_基线胸膜牵拉凹陷': '胸膜牵拉/凹陷', 'CT_基线空泡空腔': '空泡/空腔', 'CT_基线血管集束': '血管集束', 'CT_基线支气管截断': '支气管截断', 'CT_基线钙化': '钙化'}
DIAM_BINS = [(0, 8, '≤8'), (8, 15, '8–15'), (15, 30, '15–30'), (30, np.inf, '>30')]

def wilson_ci(k: int, n: int, z: float=1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)

def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return pd.Series(out, index=pvals.index)

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MERGED):
        raise FileNotFoundError(f'未找到合并表：{MERGED}\n请先依次运行 radiology_01 → 02 → 03。')
    df = pd.read_csv(MERGED, encoding='utf-8-sig')
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(str).str.lower().map({'true': True, 'false': False})
    df['y'] = (df[LABEL_COL] == '肺癌').astype(int)
    df.to_csv(os.path.join(OUT_DIR, 'ct_analysis_dataset.csv'), index=False, encoding='utf-8-sig')
    ca = df[df['y'] == 1]
    bn = df[df['y'] == 0]
    rows = []
    for col, name in CONT_FEATURES.items():
        a = pd.to_numeric(ca[col], errors='coerce').dropna()
        b = pd.to_numeric(bn[col], errors='coerce').dropna()
        u = stats.mannwhitneyu(a, b, alternative='two-sided')
        rbc = 2 * u.statistic / (len(a) * len(b)) - 1
        rows.append({'feature': name, 'type': '连续', '肺癌': f'{a.median():.1f} [{a.quantile(0.25):.1f}, {a.quantile(0.75):.1f}] (n={len(a)})', '对照': f'{b.median():.1f} [{b.quantile(0.25):.1f}, {b.quantile(0.75):.1f}] (n={len(b)})', '效应量': f'r={rbc:.3f}', '检验': 'Mann-Whitney', 'p': u.pvalue})
    for col, name in BOOL_FEATURES.items():
        a1, a0 = (int(ca[col].sum()), int((~ca[col]).sum()))
        b1, b0 = (int(bn[col].sum()), int((~bn[col]).sum()))
        table = np.array([[a1, a0], [b1, b0]])
        if (table < 5).any():
            p = stats.fisher_exact(table)[1]
            test = 'Fisher'
        else:
            p = stats.chi2_contingency(table, correction=False)[1]
            test = 'Chi2'
        t = table.astype(float)
        if (t == 0).any():
            t = t + 0.5
        or_ = t[0, 0] * t[1, 1] / (t[0, 1] * t[1, 0])
        se = np.sqrt((1 / t).sum())
        lo, hi = (np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se))
        rows.append({'feature': name, 'type': '二分类', '肺癌': f'{a1}/{len(ca)} ({a1 / len(ca):.1%})', '对照': f'{b1}/{len(bn)} ({b1 / len(bn):.1%})', '效应量': f'OR={or_:.2f} ({lo:.2f}–{hi:.2f})', '检验': test, 'p': p})
    stat = pd.DataFrame(rows)
    stat['q_FDR'] = bh_fdr(stat['p']).round(4)
    stat['p'] = stat['p'].map(lambda x: f'{x:.2e}')
    stat.to_csv(os.path.join(OUT_DIR, 'ct_descriptive_stats.csv'), index=False, encoding='utf-8-sig')
    d = pd.to_numeric(df['CT_基线最大径mm'], errors='coerce')
    bin_rows = []
    for lo, hi, lab in DIAM_BINS:
        mask = (d > lo) & (d <= hi) if np.isfinite(hi) else d > lo
        n = int(mask.sum())
        k = int(df.loc[mask, 'y'].sum())
        ci_lo, ci_hi = wilson_ci(k, n)
        bin_rows.append({'分层mm': lab, 'n': n, '肺癌数': k, '恶性率': k / n if n else np.nan, 'ci_lo': ci_lo, 'ci_hi': ci_hi})
    pd.DataFrame(bin_rows).to_csv(os.path.join(OUT_DIR, 'ct_diameter_bins.csv'), index=False, encoding='utf-8-sig')
    lobe = df.loc[df['CT_基线叶位置'].fillna('') != '', ['CT_基线叶位置', 'y']]
    cross = pd.crosstab(lobe['CT_基线叶位置'], lobe['y'], margins=True, margins_name='合计')
    cross.columns = ['对照', '肺癌', '合计']
    chi2_p = stats.chi2_contingency(cross.iloc[:-1, :-1])[1]
    cross.to_csv(os.path.join(OUT_DIR, 'ct_lobe_crosstab.csv'), encoding='utf-8-sig')
    print(f'[save] {OUT_DIR}')
    print(stat[['feature', '肺癌', '对照', '效应量', 'q_FDR']].to_string(index=False))
    print(f'叶位置整体卡方 p = {chi2_p:.3e}')
if __name__ == '__main__':
    main()
