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
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from common_preprocessing import FINAL_PANEL, REFERENCE_INTERVALS, SEED
try:
    from common_preprocessing import REFERENCE_INTERVALS_EXT
except ImportError:
    REFERENCE_INTERVALS_EXT = {'AU_Urea': (2.5, 7.1), 'MCHC': (320.0, 360.0)}
from fig2 import fig2_01_compute_models as f4
import fig3_00_compute_joint_selection as j1
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
DATA_DIR = os.path.join(OUTPUT_ROOT, 'Fig3_模型比较', '00_计算结果与原始数据')
CT_SIZE = 'CT_基线最大径mm'
CT_MISS = 'CT_径线缺失'
B_CT = [CT_SIZE, CT_MISS, 'CT_基线GGN', 'CT_基线部分实性', 'CT_基线多发', 'CT_基线毛刺', 'CT_基线分叶', 'CT_基线胸膜牵拉凹陷', 'CT_基线空泡空腔', 'CT_基线支气管截断', 'CT_基线钙化']
JOINT_PANEL = ['LYMPH%', 'CEA', 'APTT', 'AU_Urea', 'MCHC', 'AAG', CT_SIZE, CT_MISS, 'CT_基线分叶', 'CT_基线毛刺']
JOINT_BLOOD = ['LYMPH%', 'CEA', 'APTT', 'AU_Urea', 'MCHC', 'AAG']
DIAMETER_THRESHOLDS = (8, 15, 30)
FEATURE_SETS = {'A_blood': FINAL_PANEL, 'B_ct': B_CT, 'C_joint': JOINT_PANEL}
ALL_INTERVALS = {**REFERENCE_INTERVALS, **REFERENCE_INTERVALS_EXT}

def add_src_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if col in ALL_INTERVALS:
            lo, hi = ALL_INTERVALS[col]
            flag = pd.Series(0, index=X.index, dtype=int)
            if hi is not None:
                flag = flag | (X[col] > hi).astype(int)
            if lo is not None:
                flag = flag | (X[col] < lo).astype(int)
            X[f'{col}_abnormal'] = flag.astype(int)
    if CT_SIZE in X.columns:
        for t in DIAMETER_THRESHOLDS:
            X[f'CT_直径>{t}mm'] = (X[CT_SIZE] > t).astype(int)
    return X

def metrics_full(y_true, probability, threshold: float=0.5) -> dict:
    pred = (np.asarray(probability) >= threshold).astype(int)
    y = np.asarray(y_true)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {'AUC': float(roc_auc_score(y, probability)), 'ACC': float(accuracy_score(y, pred)), 'sensitivity': float(recall_score(y, pred, zero_division=0)), 'specificity': float(tn / (tn + fp)) if tn + fp else np.nan, 'precision': float(precision_score(y, pred, zero_division=0)), 'F1': float(f1_score(y, pred, zero_division=0)), 'Brier': float(brier_score_loss(y, probability))}

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, _, _ = j1.build_joint_dataset()
    y_te = np.asarray(y_test)
    missing = [c for s in FEATURE_SETS.values() for c in s if c not in X_train.columns]
    if missing:
        raise KeyError(f'特征不在合并数据集中：{sorted(set(missing))}')
    perf_rows, roc_rows, probs = ([], [], {'y_true': y_te})
    prob_of = {}
    for sname, cols in FEATURE_SETS.items():
        variants = {'Basic': (X_train[cols], X_test[cols]), 'SRC': (add_src_flags(X_train[cols]), add_src_flags(X_test[cols]))}
        for mode, (Xtr, Xte) in variants.items():
            for mname in f4.MODEL_ORDER:
                model = f4.get_models()[mname]
                p = f4.fit_predict(model, Xtr, Xte, y_train, scale=mname in {'LR', 'SVC'})
                prob_of[sname, mode, mname] = p
                probs[f'{sname}|{mode}|{mname}'] = p
                lo, hi = f4.bootstrap_auc_ci(y_te, p)
                perf_rows.append({'feature_set': sname, 'mode': mode, 'model': mname, **metrics_full(y_te, p), 'AUC_CI_lower': lo, 'AUC_CI_upper': hi, 'n_features': Xtr.shape[1]})
                fpr, tpr, _ = roc_curve(y_te, p)
                roc_rows.extend(({'feature_set': sname, 'mode': mode, 'model': mname, 'fpr': float(a), 'tpr': float(b)} for a, b in zip(fpr, tpr)))
                print(f"[{sname}|{mode}] {mname}: AUC={perf_rows[-1]['AUC']:.3f} ({lo:.3f}-{hi:.3f})")
    paired_rows, boot_rows = ([], [])
    comparisons = [('C−B（主终点）', 'C_joint', 'B_ct'), ('C−A', 'C_joint', 'A_blood')]
    for mname in f4.MODEL_ORDER:
        for mode in ('Basic', 'SRC'):
            for label, s1, s2 in comparisons:
                p1 = prob_of[s1, mode, mname]
                p2 = prob_of[s2, mode, mname]
                diffs, lo, hi, pv = f4.paired_bootstrap_auc_difference(y_te, p2, p1)
                paired_rows.append({'comparison': f'{label} [{mode}]', 'model': mname, 'AUC_1': float(roc_auc_score(y_te, p1)), 'AUC_2': float(roc_auc_score(y_te, p2)), 'dAUC': float(diffs.mean()), 'CI_lower': lo, 'CI_upper': hi, 'p_two_sided': pv})
                boot_rows.extend(({'comparison': f'{label} [{mode}]', 'model': mname, 'iteration': i, 'dAUC': float(v)} for i, v in enumerate(diffs, 1)))
        for sname in FEATURE_SETS:
            pb = prob_of[sname, 'Basic', mname]
            ps = prob_of[sname, 'SRC', mname]
            diffs, lo, hi, pv = f4.paired_bootstrap_auc_difference(y_te, pb, ps)
            paired_rows.append({'comparison': f'SRC−Basic [{sname}]', 'model': mname, 'AUC_1': float(roc_auc_score(y_te, ps)), 'AUC_2': float(roc_auc_score(y_te, pb)), 'dAUC': float(diffs.mean()), 'CI_lower': lo, 'CI_upper': hi, 'p_two_sided': pv})
            boot_rows.extend(({'comparison': f'SRC−Basic [{sname}]', 'model': mname, 'iteration': i, 'dAUC': float(v)} for i, v in enumerate(diffs, 1)))
    Xtr_src = add_src_flags(X_train[JOINT_PANEL])
    Xte_src = add_src_flags(X_test[JOINT_PANEL])
    blood_flags = [f'{c}_abnormal' for c in JOINT_BLOOD]
    ct_part = [CT_SIZE, CT_MISS, 'CT_基线分叶', 'CT_基线毛刺'] + [f'CT_直径>{t}mm' for t in DIAMETER_THRESHOLDS]
    ablation_groups = {'Blood continuous (6)': JOINT_BLOOD, 'Blood flags (6)': blood_flags, 'CT features (7)': ct_part, 'Full C_joint SRC (19)': list(Xtr_src.columns)}
    abl_rows = []
    from sklearn.ensemble import RandomForestClassifier
    for gname, cols in ablation_groups.items():
        rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
        p = f4.fit_predict(rf, Xtr_src[cols], Xte_src[cols], y_train, scale=False)
        lo, hi = f4.bootstrap_auc_ci(y_te, p)
        abl_rows.append({'feature_group': gname, 'n_features': len(cols), 'AUC': float(roc_auc_score(y_te, p)), 'AUC_CI_lower': lo, 'AUC_CI_upper': hi})
        print(f"[ablation] {gname}: AUC={abl_rows[-1]['AUC']:.3f} ({lo:.3f}-{hi:.3f})")
    pd.DataFrame(perf_rows).to_csv(os.path.join(DATA_DIR, 'abc_model_performance.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(roc_rows).to_csv(os.path.join(DATA_DIR, 'abc_roc_curves.csv'), index=False, encoding='utf-8-sig')
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(os.path.join(DATA_DIR, 'abc_paired_auc.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(boot_rows).to_csv(os.path.join(DATA_DIR, 'abc_paired_bootstrap_distributions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(abl_rows).to_csv(os.path.join(DATA_DIR, 'abc_rf_modality_ablation.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(probs).to_csv(os.path.join(DATA_DIR, 'abc_test_probabilities.csv'), index=False, encoding='utf-8-sig')
    fs_rows = [{'feature_set': s, 'feature': c, 'source': 'CT' if c.startswith('CT_') else 'blood'} for s, cols in FEATURE_SETS.items() for c in cols]
    pd.DataFrame(fs_rows).to_csv(os.path.join(DATA_DIR, 'abc_feature_sets.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([('train_n', len(X_train)), ('test_n', len(X_test)), ('A_blood 特征数', len(FINAL_PANEL)), ('B_ct 特征数', len(B_CT)), ('C_joint 特征数', len(JOINT_PANEL)), ('模型族', len(f4.MODEL_ORDER)), ('bootstrap', f4.N_BOOT), ('seed', SEED)], columns=['item', 'value']).to_csv(os.path.join(DATA_DIR, 'abc_analysis_summary.csv'), index=False, encoding='utf-8-sig')
    print('\n===== 主终点：C−B 配对 ΔAUC（Basic 模式） =====')
    main_ep = paired[paired['comparison'] == 'C−B（主终点） [Basic]']
    print(main_ep[['model', 'AUC_2', 'AUC_1', 'dAUC', 'CI_lower', 'CI_upper', 'p_two_sided']].to_string(index=False))
    print(f'\n[save] {DATA_DIR}')
if __name__ == '__main__':
    main()
