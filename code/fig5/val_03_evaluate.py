from __future__ import annotations
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from common_preprocessing import OUTPUT_ROOT, SEED
from fig2 import fig2_01_compute_models as f2
from fig3 import fig3_00_compute_joint_selection as j1
from fig3.fig3_03_compute_models import JOINT_PANEL, add_src_flags
from fig4 import fig4_00_compute_glassbox as f6
DATA_DIR = _os.path.join(OUTPUT_ROOT, 'Fig5_外部验证', '00_计算结果与原始数据')
WEIGHTS_CSV = _os.path.join(OUTPUT_ROOT, 'Fig4_SRC玻璃盒', '00_计算结果与原始数据', 'fig6_rule_weights.csv')

def find_weights_csv() -> str:
    if _os.path.isfile(WEIGHTS_CSV):
        return WEIGHTS_CSV
    hits = _glob.glob(_os.path.join(OUTPUT_ROOT, '**', 'fig6_rule_weights.csv'), recursive=True)
    if hits:
        print(f'[frozen] 标准路径未找到，改用: {hits[0]}')
        return hits[0]
    raise FileNotFoundError(f'未找到冻结分值表 fig6_rule_weights.csv。\n请先运行 Fig4 计算脚本：\n    cd {CODE_DIR}\n    python fig4\\fig4_run_all.py\n（会生成 Fig4_SRC玻璃盒\\00_计算结果与原始数据\\fig6_rule_weights.csv）\n然后再运行本脚本。')
MODELS = ['评分卡(11条)', '评分卡(12条)', 'LR(12规则)', 'SRC-RF']
GROUPS = [('val1', '验证组1'), ('val2', '验证组2')]

def load_frozen_training():
    X_train, _, y_train, _, features, blood_features = j1.build_joint_dataset()
    meds = X_train[blood_features + [f6.CT_SIZE]].median()
    R_train = f6.build_rule_matrix(X_train)
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(R_train[f6.RULES], y_train)
    S_train = add_src_flags(X_train[JOINT_PANEL])
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
    rf.fit(S_train, y_train)
    print(f'[frozen] 训练集 {len(X_train)} 例；LR/SRC-RF 已按冻结配置重拟合')
    return (meds, blood_features, lr, rf, S_train.columns.tolist())

def load_scorecard_points() -> pd.Series:
    w = pd.read_csv(find_weights_csv(), encoding='utf-8-sig')
    pts = w.set_index('rule')['scorecard_points'].astype(int).reindex(f6.RULES)
    if pts.isna().any():
        missing = pts[pts.isna()].index.tolist()
        raise ValueError(f'冻结分值表缺少规则: {missing}')
    print(f'[frozen] 评分卡分值读取: {dict(pts)}')
    return pts
MASK_COLS_BY_GROUP = {'val2': ['APTT']}

def build_val_features(tag: str, meds: pd.Series) -> pd.DataFrame:
    blood = pd.read_csv(_os.path.join(DATA_DIR, f'{tag}_blood_wide.csv'), encoding='utf-8-sig', dtype={'HospNo': str})
    ct = pd.read_csv(_os.path.join(DATA_DIR, 'val_ct_features.csv'), encoding='utf-8-sig', dtype={'HospNo': str})
    ct = ct.rename(columns={'max_diameter_mm': f6.CT_SIZE, '分叶': 'CT_基线分叶', '毛刺': 'CT_基线毛刺'})
    df = blood.merge(ct[['HospNo', f6.CT_SIZE, 'CT_基线分叶', 'CT_基线毛刺', 'CT_径线缺失']], on='HospNo', how='left')
    for c in ['CT_基线分叶', 'CT_基线毛刺', 'CT_径线缺失']:
        df[c] = df[c].fillna(0.0).astype(float)
    X = df[JOINT_PANEL].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    for c in MASK_COLS_BY_GROUP.get(tag, []):
        if c in X.columns:
            X[c] = np.nan
            print(f'[{tag}] {c} 回填数据不可信，整列置缺失（冻结插补为正常）')
    med_cols = [c for c in X.columns if c in meds.index]
    X[med_cols] = X[med_cols].fillna(meds[med_cols])
    n_ct_miss = int((df['CT_径线缺失'] == 1).sum())
    print(f'[{tag}] {len(df)} 人进入评估；CT 径线缺失 {n_ct_miss} 人')
    return df[['pid', 'HospNo', 'target']].join(X)

def dca_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(y)
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append({'threshold': float(t), 'net_benefit': tp / n - fp / n * t / (1 - t)})
    return pd.DataFrame(rows)

def _squash(v: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(v - v.mean()) / max(v.std(), 1e-09)))

def evaluate_group(tag: str, gname: str, df: pd.DataFrame, lr, rf, s_cols, pts) -> tuple[list, list, list, list, list]:
    y = df['target'].to_numpy()
    R = f6.build_rule_matrix(df)
    rules11 = [r for r in f6.RULES if r != f6.CT_MISS]
    score = R[pts.index].to_numpy() @ pts.to_numpy()
    score11 = R[rules11].to_numpy() @ pts.reindex(rules11).to_numpy()
    prob_lr = lr.predict_proba(R[f6.RULES])[:, 1]
    S_val = add_src_flags(df[JOINT_PANEL])
    prob_rf = rf.predict_proba(S_val[s_cols])[:, 1]
    preds = {'评分卡(11条)': score11, '评分卡(12条)': score, 'LR(12规则)': prob_lr, 'SRC-RF': prob_rf}
    prob_for = {'评分卡(11条)': _squash(score11), '评分卡(12条)': _squash(score), 'LR(12规则)': prob_lr, 'SRC-RF': prob_rf}
    summ_rows, roc_rows, risk_rows, dca_rows, score_rows = ([], [], [], [], [])
    for mname, v in preds.items():
        auc = roc_auc_score(y, v)
        lo, hi = f2.bootstrap_auc_ci(y, np.asarray(v, dtype=float))
        br = brier_score_loss(y, np.clip(prob_for[mname], 0, 1))
        summ_rows.append({'组别': gname, '模型': mname, 'n': len(y), 'n_肺癌': int(y.sum()), 'AUC': round(float(auc), 4), 'CI_lower': round(float(lo), 4), 'CI_upper': round(float(hi), 4), 'Brier': round(float(br), 4)})
        fpr, tpr, _ = roc_curve(y, v)
        roc_rows.extend(({'组别': gname, '模型': mname, 'fpr': float(a), 'tpr': float(b)} for a, b in zip(fpr, tpr)))
        print(f'[{gname}] {mname}: AUC={auc:.3f} ({lo:.3f}-{hi:.3f})')
    for s in sorted(np.unique(score11)):
        m = score11 == s
        risk_rows.append({'组别': gname, 'score': int(s), 'n': int(m.sum()), 'events': int(y[m].sum()), 'observed_rate': float(y[m].mean())})
    ths = np.linspace(0.01, 0.99, 99)
    for mname in ('评分卡(11条)', 'LR(12规则)', 'SRC-RF'):
        d = dca_curve(y, np.asarray(prob_for[mname], dtype=float), ths)
        d['组别'] = gname
        d['模型'] = mname
        dca_rows.append(d)
    score_rows.extend(({'组别': gname, 'pid': p, 'score': int(s), 'score12': int(s12), 'prob_lr': float(a), 'prob_rf': float(b), 'outcome': int(t)} for p, s, s12, a, b, t in zip(df['pid'], score11, score, prob_lr, prob_rf, y)))
    return (summ_rows, roc_rows, risk_rows, dca_rows, score_rows)

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    meds, _, lr, rf, s_cols = load_frozen_training()
    pts = load_scorecard_points()
    all_summ, all_roc, all_risk, all_dca, all_scores = ([], [], [], [], [])
    frames = {}
    for tag, gname in GROUPS:
        df = build_val_features(tag, meds)
        frames[gname] = df
        r = evaluate_group(tag, gname, df, lr, rf, s_cols, pts)
        for acc, part in zip((all_summ, all_roc, all_risk, all_dca, all_scores), r):
            acc.extend(part) if isinstance(part, list) else acc.append(part)
    pooled = pd.concat(frames.values(), ignore_index=True)
    r = evaluate_group('pooled', '合并', pooled, lr, rf, s_cols, pts)
    for acc, part in zip((all_summ, all_roc, all_risk, all_dca, all_scores), r):
        acc.extend(part) if isinstance(part, list) else acc.append(part)
    pd.DataFrame(all_summ).to_csv(_os.path.join(DATA_DIR, 'val_evaluation_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(all_roc).to_csv(_os.path.join(DATA_DIR, 'val_roc_curves.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(all_risk).to_csv(_os.path.join(DATA_DIR, 'val_risk_by_score.csv'), index=False, encoding='utf-8-sig')
    pd.concat(all_dca).to_csv(_os.path.join(DATA_DIR, 'val_dca.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(all_scores).to_csv(_os.path.join(DATA_DIR, 'val_scorecard_scores.csv'), index=False, encoding='utf-8-sig')
    print('\n======== 评估摘要 ========')
    print(pd.DataFrame(all_summ).to_string(index=False))
    print(f'\n[done] 输出目录: {DATA_DIR}')
if __name__ == '__main__':
    main()
