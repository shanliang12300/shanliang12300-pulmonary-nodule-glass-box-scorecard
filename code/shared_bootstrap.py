from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
MERGED = os.path.join(OUTPUT_ROOT, '队列合并数据', '队列_血液+影像_合并表.csv')
OUT_DIR = os.path.join(OUTPUT_ROOT, '联合模型探路', '00_计算结果与原始数据')
SEED = 42
N_BOOT = 2000
BLOOD_PANEL = ['CEA', 'CYFRA', 'SCCAg', 'CA125_GC', 'AAG', 'CRP', 'LYMPH%', 'LYMM', 'APTT', 'DD']
CT_BOOL = ['CT_基线GGN', 'CT_基线部分实性', 'CT_基线多发', 'CT_基线毛刺', 'CT_基线分叶', 'CT_基线胸膜牵拉凹陷', 'CT_基线空泡空腔', 'CT_基线支气管截断', 'CT_基线钙化']
CT_SIZE = 'CT_基线最大径mm'

def bootstrap_auc(y: np.ndarray, p: np.ndarray, rng) -> tuple[float, float]:
    n = len(y)
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))

def bootstrap_delta(y, p1, p2, rng) -> tuple[float, float, float]:
    n = len(y)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return (roc_auc_score(y, p1) - roc_auc_score(y, p2), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MERGED):
        raise FileNotFoundError(f'未找到合并表：{MERGED}\n请先运行 data_pipeline_run_all.py。')
    df = pd.read_csv(MERGED, encoding='utf-8-sig')
    y = (df['标签_诊断表'] == '肺癌').astype(int).to_numpy()
    for c in BLOOD_PANEL:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CT_BOOL:
        df[c] = df[c].astype(str).str.lower().map({'true': 1.0, 'false': 0.0}).fillna(0.0)
    df[CT_SIZE] = pd.to_numeric(df[CT_SIZE], errors='coerce')
    df['CT_径线缺失'] = df[CT_SIZE].isna().astype(float)
    X_blood = df[BLOOD_PANEL].to_numpy(dtype=float)
    X_ct = np.column_stack([df[CT_BOOL].to_numpy(dtype=float), df[[CT_SIZE, 'CT_径线缺失']].to_numpy(dtype=float)])
    X_size = df[[CT_SIZE]].to_numpy(dtype=float)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)
    y_tr, y_te = (y[tr], y[te])

    def impute(train_X, test_X):
        med = np.nanmedian(train_X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        inds = np.where(np.isnan(train_X))
        train_X[inds] = np.take(med, inds[1])
        inds = np.where(np.isnan(test_X))
        test_X[inds] = np.take(med, inds[1])
        return (train_X, test_X)
    sets = {}
    for name, X in [('A_blood', X_blood), ('B_CT', X_ct), ('B_size', X_size), ('C_blood+CT', np.column_stack([X_blood, X_ct]))]:
        Xtr, Xte = impute(X[tr].copy(), X[te].copy())
        sets[name] = (Xtr, Xte)
    models = {'LR': lambda: LogisticRegression(max_iter=3000, random_state=SEED), 'RF': lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)}
    rng = np.random.default_rng(SEED)
    rows, probs = ([], {'y_true': y_te})
    for sname, (Xtr, Xte) in sets.items():
        for mname, make in models.items():
            scaler = StandardScaler().fit(Xtr) if mname == 'LR' else None
            Xtr_f = scaler.transform(Xtr) if scaler else Xtr
            Xte_f = scaler.transform(Xte) if scaler else Xte
            m = make().fit(Xtr_f, y_tr)
            p = m.predict_proba(Xte_f)[:, 1]
            auc = roc_auc_score(y_te, p)
            lo, hi = bootstrap_auc(y_te, p, rng)
            rows.append({'特征组': sname, '模型': mname, 'AUC': round(auc, 3), 'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3)})
            probs[f'{sname}_{mname}'] = p
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR, 'abc_auc_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(probs).to_csv(os.path.join(OUT_DIR, 'abc_test_probs.csv'), index=False, encoding='utf-8-sig')
    drows = []
    for mname in models:
        pC = probs[f'C_blood+CT_{mname}']
        for comp in ['A_blood', 'B_CT']:
            d, lo, hi = bootstrap_delta(y_te, pC, probs[f'{comp}_{mname}'], rng)
            drows.append({'模型': mname, '对比': f'C−{comp}', 'dAUC': round(d, 3), 'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3)})
    delta = pd.DataFrame(drows)
    delta.to_csv(os.path.join(OUT_DIR, 'abc_delta_auc.csv'), index=False, encoding='utf-8-sig')
    print('\n===== 三模型测试集 AUC =====')
    print(summ.to_string(index=False))
    print('\n===== 配对 ΔAUC（CI 不含 0 = 显著增量） =====')
    print(delta.to_string(index=False))
    print(f'\n[save] {OUT_DIR}')
if __name__ == '__main__':
    main()
