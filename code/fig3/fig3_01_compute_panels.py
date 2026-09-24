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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import fig3_00_compute_joint_selection as j1
from shared_bootstrap import SEED, bootstrap_auc, bootstrap_delta
from display_utils import jsource
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
DATA_DIR = os.path.join(OUTPUT_ROOT, 'Fig3_联合特征筛选', '00_计算结果与原始数据')
CONSENSUS = os.path.join(DATA_DIR, 'consensus_votes_and_stability.csv')
TIGHT_VOTES, TIGHT_FREQ = (3, 0.4)
EXT_VOTES, EXT_FREQ = (2, 0.5)
STABLE_FREQ = 0.5

def define_panels(consensus: pd.DataFrame) -> dict[str, list[str]]:
    v, f = (consensus['votes'], consensus['mean_frequency'])
    panels = {'P1_tight': consensus.loc[(v >= TIGHT_VOTES) & (f >= TIGHT_FREQ), 'feature'].tolist(), 'P2_all3': consensus.loc[v >= TIGHT_VOTES, 'feature'].tolist(), 'P3_ext': consensus.loc[(v >= TIGHT_VOTES) & (f >= TIGHT_FREQ) | (v >= EXT_VOTES) & (f >= EXT_FREQ), 'feature'].tolist(), 'P4_stable': consensus.loc[f >= STABLE_FREQ, 'feature'].tolist()}
    return panels

def evaluate(X_train, X_test, y_tr, y_te, panels, rng):
    models = {'LR': lambda: LogisticRegression(max_iter=3000, random_state=SEED), 'RF': lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)}
    rows, probs = ([], {})
    for pname, feats in panels.items():
        Xtr, Xte = (X_train[feats], X_test[feats])
        for mname, make in models.items():
            scaler = StandardScaler().fit(Xtr) if mname == 'LR' else None
            Xtr_f = scaler.transform(Xtr) if scaler else Xtr
            Xte_f = scaler.transform(Xte) if scaler else Xte
            m = make().fit(Xtr_f, y_tr)
            p = m.predict_proba(Xte_f)[:, 1]
            auc = roc_auc_score(y_te, p)
            lo, hi = bootstrap_auc(y_te, p, rng)
            rows.append({'panel': pname, '特征数': len(feats), '模型': mname, 'AUC': round(auc, 3), 'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3)})
            probs[f'{pname}_{mname}'] = p
    drows = []
    for mname in models:
        p_ref = probs[f'P1_tight_{mname}']
        for pname in panels:
            if pname == 'P1_tight':
                continue
            d, lo, hi = bootstrap_delta(y_te, probs[f'{pname}_{mname}'], p_ref, rng)
            drows.append({'模型': mname, '对比': f'{pname}−P1_tight', 'dAUC': round(d, 3), 'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3)})
    return (pd.DataFrame(rows), pd.DataFrame(drows))

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CONSENSUS):
        print('[auto] 缺少共识总表，先运行 fig3_00_compute_joint_selection.py')
        j1.main()
    consensus = pd.read_csv(CONSENSUS, encoding='utf-8-sig')
    X_train, X_test, y_train, y_test, features, _ = j1.build_joint_dataset()
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)
    panels = define_panels(consensus)
    avail = set(X_train.columns)
    for pname, feats in panels.items():
        missing = [c for c in feats if c not in avail]
        if missing:
            print(f'[warn] {pname} 有 {len(missing)} 个特征不在候选池: {missing}')
            panels[pname] = [c for c in feats if c in avail]
    rng = np.random.default_rng(SEED)
    summ, delta = evaluate(X_train, X_test, y_tr, y_te, panels, rng)
    member_rows = []
    for pname, feats in panels.items():
        sub = consensus.set_index('feature').loc[feats]
        for feat, r in sub.iterrows():
            member_rows.append({'panel': pname, 'feature': feat, 'source': jsource(feat), 'votes': int(r['votes']), 'mean_frequency': round(float(r['mean_frequency']), 3)})
    pd.DataFrame(member_rows).to_csv(os.path.join(DATA_DIR, 'candidate_panels.csv'), index=False, encoding='utf-8-sig')
    summ.to_csv(os.path.join(DATA_DIR, 'panel_auc_summary.csv'), index=False, encoding='utf-8-sig')
    delta.to_csv(os.path.join(DATA_DIR, 'panel_delta_auc.csv'), index=False, encoding='utf-8-sig')
    print('\n===== 候选最终 panel 测试集 AUC =====')
    print(summ.to_string(index=False))
    print('\n===== 配对 ΔAUC（vs P1_tight；CI 不含 0 = 显著差异） =====')
    print(delta.to_string(index=False))
    print('\n===== panel 成员 =====')
    for pname, feats in panels.items():
        print(f'{pname} ({len(feats)}): {feats}')
    print(f'\n[save] {DATA_DIR}')
if __name__ == '__main__':
    main()
