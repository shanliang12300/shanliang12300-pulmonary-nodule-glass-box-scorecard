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
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common_preprocessing import DATA_PATH, FINAL_PANEL, OUTPUT_ROOT, REFERENCE_INTERVALS, SEED, add_symbolic_flags, load_raw_data, quality_control, split_and_impute
warnings.filterwarnings('ignore')
FIG4_ROOT = os.path.join(OUTPUT_ROOT, 'Fig2_血液模型比较')
DATA_DIR = os.path.join(FIG4_ROOT, '00_计算结果与原始数据')
MODEL_ORDER = ['RF', 'LR', 'SVC', 'LightGBM', 'XGBoost']
N_BOOT = 2000

def get_models(seed: int=SEED) -> Dict[str, object]:
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('缺少 lightgbm 或 xgboost，请先运行：pip install lightgbm xgboost') from exc
    return {'RF': RandomForestClassifier(n_estimators=500, max_depth=5, random_state=seed, n_jobs=-1), 'LR': LogisticRegression(max_iter=2000, random_state=seed), 'SVC': SVC(probability=True, random_state=seed), 'LightGBM': LGBMClassifier(random_state=seed, verbose=-1), 'XGBoost': XGBClassifier(eval_metric='auc', random_state=seed, n_jobs=-1)}

def fit_predict(model, X_train, X_test, y_train, scale: bool=False):
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    return probability

def classification_metrics(y_true, probability, threshold: float=0.5) -> Dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {'AUC': float(roc_auc_score(y_true, probability)), 'ACC': float(accuracy_score(y_true, pred)), 'precision': float(precision_score(y_true, pred, zero_division=0)), 'recall': float(recall_score(y_true, pred, zero_division=0)), 'F1': float(f1_score(y_true, pred, zero_division=0)), 'Brier': float(brier_score_loss(y_true, probability))}

def bootstrap_auc_ci(y_true, probability, n_boot: int=N_BOOT) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    y = np.asarray(y_true)
    p = np.asarray(probability)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return (float(lower), float(upper))

def paired_bootstrap_auc_difference(y_true, basic_probability, src_probability, n_boot: int=N_BOOT):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y_true)
    pb = np.asarray(basic_probability)
    ps = np.asarray(src_probability)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        differences.append(roc_auc_score(y[idx], ps[idx]) - roc_auc_score(y[idx], pb[idx]))
    differences = np.asarray(differences)
    lower, upper = np.percentile(differences, [2.5, 97.5])
    p_two_sided = min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0)))
    return (differences, float(lower), float(upper), float(p_two_sided))

def roc_records(mode: str, model_name: str, y_true, probability) -> List[Dict[str, float]]:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    return [{'mode': mode, 'model': model_name, 'fpr': float(f), 'tpr': float(t), 'threshold': float(th) if np.isfinite(th) else np.nan} for f, t, th in zip(fpr, tpr, thresholds)]

def symbolic_flag_table(X_src: pd.DataFrame, y: pd.Series, cohort: str) -> pd.DataFrame:
    records = []
    flag_cols = [c for c in X_src.columns if c.endswith('_abnormal')]
    for flag in flag_cols:
        cancer = y == 1
        benign = y == 0
        table = np.array([[int(X_src.loc[cancer, flag].sum()), int((1 - X_src.loc[cancer, flag]).sum())], [int(X_src.loc[benign, flag].sum()), int((1 - X_src.loc[benign, flag]).sum())]])
        _, p_value = fisher_exact(table)
        records.append({'cohort': cohort, 'flag': flag, 'cancer_abnormal_rate': float(X_src.loc[cancer, flag].mean()), 'benign_abnormal_rate': float(X_src.loc[benign, flag].mean()), 'cancer_abnormal_n': table[0, 0], 'cancer_normal_n': table[0, 1], 'benign_abnormal_n': table[1, 0], 'benign_normal_n': table[1, 1], 'fisher_exact_p': float(p_value)})
    return pd.DataFrame(records)

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    X, y, features, _ = quality_control(df)
    X_train, X_test, y_train, y_test, _ = split_and_impute(X, y)
    Xtr_basic = X_train[FINAL_PANEL].copy()
    Xte_basic = X_test[FINAL_PANEL].copy()
    Xtr_src = add_symbolic_flags(Xtr_basic)
    Xte_src = add_symbolic_flags(Xte_basic)
    performance_records = []
    roc_curve_records = []
    paired_records = []
    bootstrap_distribution_records = []
    probabilities = {}
    for model_name in MODEL_ORDER:
        model_basic = get_models()[model_name]
        model_src = get_models()[model_name]
        scale = model_name in {'LR', 'SVC'}
        prob_basic = fit_predict(model_basic, Xtr_basic, Xte_basic, y_train, scale=scale)
        prob_src = fit_predict(model_src, Xtr_src, Xte_src, y_train, scale=scale)
        probabilities[model_name, 'Basic'] = prob_basic
        probabilities[model_name, 'SRC'] = prob_src
        for mode, prob in (('Basic', prob_basic), ('SRC', prob_src)):
            metrics = classification_metrics(y_test, prob)
            auc_lower, auc_upper = bootstrap_auc_ci(y_test, prob)
            performance_records.append({'mode': mode, 'model': model_name, **metrics, 'AUC_CI_lower': auc_lower, 'AUC_CI_upper': auc_upper})
            roc_curve_records.extend(roc_records(mode, model_name, y_test, prob))
        differences, diff_lower, diff_upper, p_value = paired_bootstrap_auc_difference(y_test, prob_basic, prob_src)
        paired_records.append({'model': model_name, 'basic_AUC': float(roc_auc_score(y_test, prob_basic)), 'src_AUC': float(roc_auc_score(y_test, prob_src)), 'test_difference_SRC_minus_Basic': float(roc_auc_score(y_test, prob_src) - roc_auc_score(y_test, prob_basic)), 'bootstrap_difference_mean': float(differences.mean()), 'bootstrap_CI_lower': diff_lower, 'bootstrap_CI_upper': diff_upper, 'bootstrap_two_sided_p': p_value})
        for i, value in enumerate(differences, start=1):
            bootstrap_distribution_records.append({'model': model_name, 'bootstrap_iteration': i, 'AUC_difference_SRC_minus_Basic': float(value)})
        print(f"[model] {model_name}: Basic AUC={paired_records[-1]['basic_AUC']:.3f}, SRC AUC={paired_records[-1]['src_AUC']:.3f}")
    flag_cols = [c for c in Xtr_src.columns if c.endswith('_abnormal')]
    ablation_sets = {'Continuous only': (Xtr_basic, Xte_basic), 'Symbolic flags only': (Xtr_src[flag_cols], Xte_src[flag_cols]), 'Continuous + flags': (Xtr_src, Xte_src)}
    ablation_records = []
    ablation_roc_records = []
    for group_name, (A_train, A_test) in ablation_sets.items():
        rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
        prob = fit_predict(rf, A_train, A_test, y_train, scale=False)
        auc = float(roc_auc_score(y_test, prob))
        lower, upper = bootstrap_auc_ci(y_test, prob)
        ablation_records.append({'feature_group': group_name, 'AUC': auc, 'AUC_CI_lower': lower, 'AUC_CI_upper': upper, 'n_features': A_train.shape[1]})
        ablation_roc_records.extend(roc_records(group_name, 'RF', y_test, prob))
        print(f'[ablation] {group_name}: AUC={auc:.3f} ({lower:.3f}-{upper:.3f})')
    flag_table = pd.concat([symbolic_flag_table(Xtr_src, y_train, 'training'), symbolic_flag_table(Xte_src, y_test, 'test')], ignore_index=True)
    summary = pd.DataFrame([('train_n', len(X_train)), ('test_n', len(X_test)), ('continuous_features', len(FINAL_PANEL)), ('symbolic_flags', len(flag_cols)), ('models', len(MODEL_ORDER)), ('bootstrap_iterations', N_BOOT), ('random_seed', SEED)], columns=['item', 'value'])
    reference_table = pd.DataFrame({'variable': FINAL_PANEL, 'lower_limit': [REFERENCE_INTERVALS[v][0] for v in FINAL_PANEL], 'upper_limit': [REFERENCE_INTERVALS[v][1] for v in FINAL_PANEL]})
    pd.DataFrame(performance_records).to_csv(os.path.join(DATA_DIR, 'model_performance.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(roc_curve_records).to_csv(os.path.join(DATA_DIR, 'model_roc_curves.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(paired_records).to_csv(os.path.join(DATA_DIR, 'paired_auc_comparison.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(bootstrap_distribution_records).to_csv(os.path.join(DATA_DIR, 'paired_auc_bootstrap_distributions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(ablation_records).to_csv(os.path.join(DATA_DIR, 'rf_feature_group_ablation.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(ablation_roc_records).to_csv(os.path.join(DATA_DIR, 'rf_ablation_roc_curves.csv'), index=False, encoding='utf-8-sig')
    flag_table.to_csv(os.path.join(DATA_DIR, 'symbolic_flag_prevalence.csv'), index=False, encoding='utf-8-sig')
    reference_table.to_csv(os.path.join(DATA_DIR, 'symbolic_reference_intervals.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'final_panel_feature': FINAL_PANEL}).to_csv(os.path.join(DATA_DIR, 'final_10_biomarker_panel.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(DATA_DIR, 'fig4_analysis_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'[save] Fig 4 全部计算结果与原始数据已保存至: {DATA_DIR}')
if __name__ == '__main__':
    main()
