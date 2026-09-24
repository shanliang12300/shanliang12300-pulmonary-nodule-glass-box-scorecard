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
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common_preprocessing import DATA_PATH, SEED, load_raw_data, quality_control
from fig2 import fig2_00_compute_selection as fs
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
MERGED = os.path.join(OUTPUT_ROOT, '队列合并数据', '队列_血液+影像_合并表.csv')
OUT_DIR = os.path.join(OUTPUT_ROOT, 'Fig3_联合特征筛选', '00_计算结果与原始数据')
CT_FEATURES = ['CT_基线最大径mm', 'CT_径线缺失', 'CT_基线GGN', 'CT_基线部分实性', 'CT_基线多发', 'CT_基线毛刺', 'CT_基线分叶', 'CT_基线胸膜牵拉凹陷', 'CT_基线空泡空腔', 'CT_基线支气管截断', 'CT_基线钙化']
CT_BOOL = [c for c in CT_FEATURES if c not in ('CT_基线最大径mm', 'CT_径线缺失')]
CT_SIZE = 'CT_基线最大径mm'
CORR_THRESHOLD = 0.8

def build_joint_dataset():
    df_blood = load_raw_data(DATA_PATH)
    X_b, y, blood_features, _ = quality_control(df_blood)
    X_b['pid'] = df_blood['pid'].astype(str).values
    mg = pd.read_csv(MERGED, encoding='utf-8-sig', dtype={'pid': str})
    for c in CT_BOOL:
        mg[c] = mg[c].astype(str).str.lower().map({'true': 1.0, 'false': 0.0})
    mg[CT_SIZE] = pd.to_numeric(mg[CT_SIZE], errors='coerce')
    mg['CT_径线缺失'] = mg[CT_SIZE].isna().astype(float)
    joint = X_b.merge(mg[['pid'] + CT_FEATURES], on='pid', how='left')
    features = blood_features + CT_FEATURES
    X = joint[features].copy()
    for c in blood_features:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in CT_BOOL + ['CT_径线缺失']:
        X[c] = X[c].fillna(0.0)
    tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=SEED, stratify=y)
    X_train, X_test = (X.iloc[tr].copy(), X.iloc[te].copy())
    y_train, y_test = (y.iloc[tr].copy(), y.iloc[te].copy())
    med_cols = blood_features + [CT_SIZE]
    meds = X_train[med_cols].median()
    X_train[med_cols] = X_train[med_cols].fillna(meds)
    X_test[med_cols] = X_test[med_cols].fillna(meds)
    return (X_train, X_test, y_train, y_test, features, blood_features)

def prune_collinear(X_train, features, blood_features):
    rho = X_train[features].corr(method='spearman').abs()
    drop = set()
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if b in drop or a in drop:
                continue
            if {a, b} <= set(blood_features):
                continue
            if rho.loc[a, b] > CORR_THRESHOLD:
                loser = b if a in blood_features else a
                drop.add(loser)
    kept = [f for f in features if f not in drop]
    rec = pd.DataFrame({'dropped': sorted(drop), 'reason': f'|rho|>{CORR_THRESHOLD} with higher-priority feature'})
    return (kept, rec)

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    X_train, X_test, y_train, y_test, features, blood_features = build_joint_dataset()
    print(f'[data] 合并候选池 {len(features)} 个（血液 {len(blood_features)} + CT {len(features) - len(blood_features)}），训练集 {len(X_train)} 例')
    features, drop_rec = prune_collinear(X_train, features, blood_features)
    drop_rec.to_csv(os.path.join(OUT_DIR, 'joint_collinearity_dropped.csv'), index=False, encoding='utf-8-sig')
    if len(drop_rec):
        print(f"[qc] 合并池剪枝剔除 {len(drop_rec)} 个: {drop_rec['dropped'].tolist()}")
    pd.DataFrame({'feature': features, 'source': ['blood' if f in blood_features else 'CT' for f in features]}).to_csv(os.path.join(OUT_DIR, 'joint_candidates.csv'), index=False, encoding='utf-8-sig')
    Xs_train = X_train[features]
    boruta_confirmed, boruta_tentative, _ = fs.run_boruta_decision(Xs_train, y_train, features)
    print(f'[Boruta] confirmed={boruta_confirmed} tentative={boruta_tentative}')
    lasso_selected, lasso_path, lasso_cv, lasso_table, best_C = fs.run_lasso(Xs_train, y_train, features)
    print(f'[LASSO] C={best_C:.4g} 选中 {len(lasso_selected)} 个')
    mrmr_selected, mrmr_table = fs.run_mrmr(Xs_train, y_train, features)
    relieff_selected, relieff_table = fs.run_relieff(Xs_train, y_train, features)
    pca_selected, pca_importance, pca_variance, pca_threshold, pca_n = fs.run_pca_selection(Xs_train, features)
    print(f'[PCA] 保留 {pca_n} PCs，选中 {len(pca_selected)} 个')
    print('[stability] 开始 B=100 五法稳定性选择 …')
    stability_frequency, stability_replicates = fs.run_bootstrap_stability(Xs_train, y_train, features, lasso_best_C=best_C)
    votes = pd.Series(0, index=features, dtype=int)
    for selected in (boruta_confirmed, lasso_selected, mrmr_selected, relieff_selected, pca_selected):
        votes.loc[selected] += 1
    selection_sets = {'Boruta': set(boruta_confirmed) | set(boruta_tentative), 'LASSO': set(lasso_selected), 'mRMR': set(mrmr_selected), 'ReliefF': set(relieff_selected), 'PCA': set(pca_selected)}
    matrix_rows = votes.sort_values(ascending=False).index.tolist()
    selection_matrix = pd.DataFrame({m: [int(f in s) for f in matrix_rows] for m, s in selection_sets.items()}, index=matrix_rows)
    selection_matrix.index.name = 'feature'
    freq_by_feature = stability_frequency.set_index('feature')['mean_frequency']
    consensus = pd.DataFrame({'feature': features, 'source': ['blood' if f in blood_features else 'CT' for f in features], 'votes': votes.values, 'mean_frequency': freq_by_feature.reindex(features).values}).sort_values(['votes', 'mean_frequency'], ascending=False)
    pd.DataFrame({'feature': features, 'decision': ['confirmed' if f in boruta_confirmed else 'tentative' if f in boruta_tentative else 'rejected' for f in features]}).to_csv(os.path.join(OUT_DIR, 'boruta_decisions.csv'), index=False, encoding='utf-8-sig')
    lasso_table.to_csv(os.path.join(OUT_DIR, 'lasso_selected.csv'), index=False, encoding='utf-8-sig')
    lasso_path.to_csv(os.path.join(OUT_DIR, 'lasso_path.csv'), index=False, encoding='utf-8-sig')
    lasso_cv.to_csv(os.path.join(OUT_DIR, 'lasso_cv.csv'), index=False, encoding='utf-8-sig')
    mrmr_table.to_csv(os.path.join(OUT_DIR, 'mrmr_scores.csv'), index=False, encoding='utf-8-sig')
    relieff_table.to_csv(os.path.join(OUT_DIR, 'relieff_scores.csv'), index=False, encoding='utf-8-sig')
    pca_importance.to_csv(os.path.join(OUT_DIR, 'pca_importance.csv'), index=False, encoding='utf-8-sig')
    pca_variance.to_csv(os.path.join(OUT_DIR, 'pca_variance.csv'), index=False, encoding='utf-8-sig')
    selection_matrix.to_csv(os.path.join(OUT_DIR, 'five_method_selection_matrix.csv'), encoding='utf-8-sig')
    stability_frequency.to_csv(os.path.join(OUT_DIR, 'stability_frequency.csv'), index=False, encoding='utf-8-sig')
    stability_replicates.to_csv(os.path.join(OUT_DIR, 'stability_replicates.csv'), index=False, encoding='utf-8-sig')
    consensus.to_csv(os.path.join(OUT_DIR, 'consensus_votes_and_stability.csv'), index=False, encoding='utf-8-sig')
    summary = pd.DataFrame([('候选池', len(features)), ('血液/CT', f'{len(blood_features)}/{len(features) - len(blood_features)}'), ('Boruta confirmed', len(boruta_confirmed)), ('LASSO 选中', len(lasso_selected)), ('mRMR Top30', len(mrmr_selected)), ('ReliefF Top30', len(relieff_selected)), ('PCA 选中', len(pca_selected)), ('5票特征', int((votes == 5).sum())), ('≥3票特征', int((votes >= 3).sum())), ('用时分钟', round((time.time() - t0) / 60, 1))], columns=['项目', '数值'])
    summary.to_csv(os.path.join(OUT_DIR, 'selection_summary.csv'), index=False, encoding='utf-8-sig')
    print('\n===== 共识总表（票数≥2） =====')
    print(consensus[consensus['votes'] >= 2].to_string(index=False))
    print(f'\n[save] {OUT_DIR}（用时 {(time.time() - t0) / 60:.1f} 分钟）')
if __name__ == '__main__':
    main()
