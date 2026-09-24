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
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from common_preprocessing import DATA_PATH, FINAL_PANEL, OUTPUT_ROOT, SEED, get_final_panel_data, load_raw_data, quality_control, split_and_impute
from fig2_01_compute_models import MODEL_ORDER, N_BOOT, get_models
warnings.filterwarnings('ignore')
FIG4_ROOT = os.path.join(OUTPUT_ROOT, 'Fig2_血液模型比较')
DATA_DIR = os.path.join(FIG4_ROOT, '00_计算结果与原始数据')
SHIFT_LEVELS = [0.05, 0.1, 0.15, 0.2]
NOISE_LEVELS = [0.05, 0.1, 0.15, 0.2]
OUTLIER_LEVELS = [0.02, 0.05, 0.1, 0.15]
MISSING_LEVELS = [0.05, 0.1, 0.2, 0.3]
ROBUST_REPEATS = 20
LC_FRACTIONS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
LC_REPEATS = 5
DCA_THRESHOLDS = np.linspace(0.01, 0.6, 120)
NRI_CUTOFFS = [0.2, 0.5]
SCALE_MODELS = {'LR', 'SVC'}

def fit_models_cached(X_train, y_train):
    Xtr_basic = get_final_panel_data(X_train, symbolic=False)
    Xtr_src = get_final_panel_data(X_train, symbolic=True)
    cache = {}
    for key, Xtr in (('Basic', Xtr_basic), ('SRC', Xtr_src)):
        models = get_models()
        scalers = {}
        for name, model in models.items():
            if name in SCALE_MODELS:
                scaler = StandardScaler().fit(Xtr)
                model.fit(scaler.transform(Xtr), y_train)
                scalers[name] = scaler
            else:
                model.fit(Xtr, y_train)
                scalers[name] = None
        cache[key] = {name: (m, scalers[name]) for name, m in models.items()}
    return cache

def perturb_test(X_test: pd.DataFrame, train_medians: pd.Series, family: str, level: float, rng: np.random.Generator, direction: int=1) -> pd.DataFrame:
    Xp = X_test.copy().astype(float)
    if family == 'shift':
        Xp = Xp * (1.0 + direction * level)
    elif family == 'noise':
        Xp = Xp * (1.0 + rng.normal(0.0, level, size=Xp.shape))
    elif family == 'outlier':
        mask = rng.random(Xp.shape) < level
        spikes = rng.uniform(3.0, 8.0, size=Xp.shape)
        Xp = Xp.where(~mask, Xp * spikes)
    elif family == 'missing':
        mask = rng.random(Xp.shape) < level
        Xp = Xp.mask(mask)
        Xp = Xp.fillna(train_medians)
    else:
        raise ValueError(family)
    return Xp

def run_robustness(models_cache, X_test_panel, y_test, train_medians):
    rng = np.random.default_rng(SEED + 11)
    records = []

    def evaluate(family, level, repeat, Xp):
        Xp_src = get_final_panel_data(Xp, symbolic=False)
        for mode in ('Basic', 'SRC'):
            if mode == 'SRC':
                Xp_src_full = get_final_panel_data(Xp, symbolic=True)
            X_in = Xp if mode == 'Basic' else Xp_src_full
            for name in MODEL_ORDER:
                model, scaler = models_cache[mode][name]
                Xin = scaler.transform(X_in) if scaler is not None else X_in
                prob = model.predict_proba(Xin)[:, 1]
                records.append({'perturbation': family, 'level': level, 'repeat': repeat, 'model': name, 'mode': mode, 'auc': float(roc_auc_score(y_test, prob))})
    evaluate('none', 0.0, 0, X_test_panel)
    for level in SHIFT_LEVELS:
        for direction in (1, -1):
            Xp = perturb_test(X_test_panel, train_medians, 'shift', level, rng, direction)
            evaluate('shift', level, direction, Xp)
    for level in NOISE_LEVELS:
        for rep in range(ROBUST_REPEATS):
            Xp = perturb_test(X_test_panel, train_medians, 'noise', level, rng)
            evaluate('noise', level, rep, Xp)
    for level in OUTLIER_LEVELS:
        for rep in range(ROBUST_REPEATS):
            Xp = perturb_test(X_test_panel, train_medians, 'outlier', level, rng)
            evaluate('outlier', level, rep, Xp)
    for level in MISSING_LEVELS:
        for rep in range(ROBUST_REPEATS):
            Xp = perturb_test(X_test_panel, train_medians, 'missing', level, rng)
            evaluate('missing', level, rep, Xp)
    table = pd.DataFrame(records)
    print(f'[robustness] {len(table)} 条 AUC 记录完成')
    return table

def run_dca(y_test, probs_basic, probs_src) -> pd.DataFrame:
    yv = np.asarray(y_test, dtype=float)
    n = len(yv)
    prevalence = yv.mean()
    records = []
    for pt in DCA_THRESHOLDS:
        w = pt / (1.0 - pt)
        records.append({'threshold': pt, 'model': 'Treat all', 'mode': 'Reference', 'net_benefit': float(prevalence - (1 - prevalence) * w)})
        records.append({'threshold': pt, 'model': 'Treat none', 'mode': 'Reference', 'net_benefit': 0.0})
        for name in MODEL_ORDER:
            for mode, probs in (('Basic', probs_basic), ('SRC', probs_src)):
                pred = (probs[name] >= pt).astype(float)
                tp = float((pred * yv).sum())
                fp = float((pred * (1 - yv)).sum())
                records.append({'threshold': pt, 'model': name, 'mode': mode, 'net_benefit': tp / n - fp / n * w})
    table = pd.DataFrame(records)
    print('[DCA] 净获益曲线完成')
    return table

def calibration_slope_intercept(y_true, probability) -> Tuple[float, float]:
    eps = 0.0001
    p = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    recal = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    recal.fit(logit_p, y_true)
    return (float(recal.intercept_[0]), float(recal.coef_[0][0]))

def run_calibration(y_test, probs_basic, probs_src, n_bins: int=10):
    point_records, metric_records = ([], [])
    for name in MODEL_ORDER:
        for mode, probs in (('Basic', probs_basic), ('SRC', probs_src)):
            p = np.asarray(probs[name], dtype=float)
            bins = pd.qcut(p, n_bins, duplicates='drop')
            grouped = pd.DataFrame({'p': p, 'y': np.asarray(y_test)}).groupby(bins, observed=True)
            for interval, g in grouped:
                point_records.append({'model': name, 'mode': mode, 'predicted_mean': float(g['p'].mean()), 'observed_fraction': float(g['y'].mean()), 'bin_n': int(len(g))})
            intercept, slope = calibration_slope_intercept(y_test, p)
            metric_records.append({'model': name, 'mode': mode, 'calibration_intercept': intercept, 'calibration_slope': slope, 'brier': float(brier_score_loss(y_test, p)), 'mean_predicted': float(p.mean()), 'observed_rate': float(np.mean(y_test))})
    points = pd.DataFrame(point_records)
    metrics = pd.DataFrame(metric_records)
    print('[calibration] 校准点与校准指标完成')
    return (points, metrics)

def idi_nri_from_probs(y_true, p_basic, p_src, cutoffs=NRI_CUTOFFS) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p0 = np.asarray(p_basic, dtype=float)
    p1 = np.asarray(p_src, dtype=float)
    events = y == 1
    diff_events = (p1[events] - p0[events]).mean()
    diff_nonevents = (p1[~events] - p0[~events]).mean()
    idi = diff_events - diff_nonevents
    up = p1 > p0
    down = p1 < p0
    nri_cont = up[events].mean() - down[events].mean() + (down[~events].mean() - up[~events].mean())
    c0 = np.digitize(p0, cutoffs)
    c1 = np.digitize(p1, cutoffs)
    move_up = c1 > c0
    move_down = c1 < c0
    nri_cat = move_up[events].mean() - move_down[events].mean() + (move_down[~events].mean() - move_up[~events].mean())
    return {'IDI': float(idi), 'NRI_continuous': float(nri_cont), 'NRI_categorical': float(nri_cat)}

def run_idi_nri(y_test, probs_basic, probs_src, n_boot: int=N_BOOT):
    rng = np.random.default_rng(SEED + 13)
    y = np.asarray(y_test)
    point_records, boot_records = ([], [])
    for name in MODEL_ORDER:
        p0 = np.asarray(probs_basic[name])
        p1 = np.asarray(probs_src[name])
        point = idi_nri_from_probs(y, p0, p1)
        boots = {'IDI': [], 'NRI_continuous': [], 'NRI_categorical': []}
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            r = idi_nri_from_probs(y[idx], p0[idx], p1[idx])
            for k in boots:
                boots[k].append(r[k])
            boot_records.append({'model': name, 'replicate': len(boot_records), **{k: r[k] for k in boots}})
        row = {'model': name}
        for k, values in boots.items():
            arr = np.asarray(values)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            p_val = 2 * min((arr <= 0).mean(), (arr > 0).mean())
            row[f'{k}_point'] = point[k]
            row[f'{k}_ci_lower'] = float(lo)
            row[f'{k}_ci_upper'] = float(hi)
            row[f'{k}_p'] = float(min(p_val, 1.0))
        point_records.append(row)
    points = pd.DataFrame(point_records)
    distributions = pd.DataFrame(boot_records)
    print('[IDI/NRI] 重分类指标完成')
    return (points, distributions)

def operating_points_from_probs(y_true, probability) -> Dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probability)
    fpr, tpr, thresholds = roc_curve(y, p)
    spec = 1 - fpr
    ok_in = spec >= 0.9
    sens_at_spec90 = float(tpr[ok_in].max()) if ok_in.any() else np.nan
    ok_out = tpr >= 0.95
    if ok_out.any():
        best = np.argmax(spec[ok_out])
        spec_at_sens95 = float(spec[ok_out][best])
        thr = thresholds[ok_out][best]
        pred_neg = p < thr
        npv_at_sens95 = float(((y == 0) & pred_neg).sum() / max(pred_neg.sum(), 1))
    else:
        spec_at_sens95 = np.nan
        npv_at_sens95 = np.nan
    return {'Sens_at_Spec90': sens_at_spec90, 'Spec_at_Sens95': spec_at_sens95, 'NPV_at_Sens95': npv_at_sens95}
OPERATING_METRICS = ['Sens_at_Spec90', 'Spec_at_Sens95', 'NPV_at_Sens95']

def run_operating_points(y_test, probs_basic, probs_src, n_boot: int=N_BOOT):
    rng = np.random.default_rng(SEED + 17)
    y = np.asarray(y_test)
    point_records, boot_records = ([], [])
    for name in MODEL_ORDER:
        p0 = np.asarray(probs_basic[name])
        p1 = np.asarray(probs_src[name])
        point0 = operating_points_from_probs(y, p0)
        point1 = operating_points_from_probs(y, p1)
        diffs = {m: [] for m in OPERATING_METRICS}
        for rep in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            b = operating_points_from_probs(y[idx], p0[idx])
            s = operating_points_from_probs(y[idx], p1[idx])
            row = {'model': name, 'replicate': rep}
            for m in OPERATING_METRICS:
                d = s[m] - b[m]
                diffs[m].append(d)
                row[f'delta_{m}'] = d
            boot_records.append(row)
        for m in OPERATING_METRICS:
            arr = np.asarray([v for v in diffs[m] if not np.isnan(v)])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            p_val = 2 * min((arr <= 0).mean(), (arr > 0).mean())
            point_records.append({'model': name, 'metric': m, 'basic': point0[m], 'src': point1[m], 'delta': point1[m] - point0[m], 'ci_lower': float(lo), 'ci_upper': float(hi), 'p': float(min(p_val, 1.0))})
    points = pd.DataFrame(point_records)
    distributions = pd.DataFrame(boot_records)
    print('[operating points] 临床工作点指标完成')
    return (points, distributions)

def run_learning_curves(X_train, X_test, y_train, y_test):
    Xte_basic = get_final_panel_data(X_test, symbolic=False)
    Xte_src = get_final_panel_data(X_test, symbolic=True)
    records = []
    t0 = time.time()
    total = len(LC_FRACTIONS) * LC_REPEATS
    for rep in range(LC_REPEATS):
        for frac in LC_FRACTIONS:
            if frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=frac, stratify=y_train, random_state=SEED + 101 * rep)
            else:
                X_sub, y_sub = (X_train, y_train)
            Xsub_basic = get_final_panel_data(X_sub, symbolic=False)
            Xsub_src = get_final_panel_data(X_sub, symbolic=True)
            models = get_models(seed=SEED + rep)
            for name, model in models.items():
                for mode, Xtr, Xte in (('Basic', Xsub_basic, Xte_basic), ('SRC', Xsub_src, Xte_src)):
                    if name in SCALE_MODELS:
                        scaler = StandardScaler().fit(Xtr)
                        model.fit(scaler.transform(Xtr), y_sub)
                        prob = model.predict_proba(scaler.transform(Xte))[:, 1]
                    else:
                        model.fit(Xtr, y_sub)
                        prob = model.predict_proba(Xte)[:, 1]
                    records.append({'model': name, 'mode': mode, 'train_fraction': frac, 'train_n': int(len(X_sub)), 'repeat': rep, 'test_auc': float(roc_auc_score(y_test, prob))})
            done = rep * len(LC_FRACTIONS) + LC_FRACTIONS.index(frac) + 1
            print(f'[learning curve] {done}/{total} 完成，已用时 {(time.time() - t0) / 60:.1f} 分钟', end='\r')
    print()
    table = pd.DataFrame(records)
    print('[learning curve] 学习曲线完成')
    return table

def run_flag_score(X_test, y_test, X_train, y_train):
    Xte_flags = get_final_panel_data(X_test, symbolic=True)
    flag_cols = [c for c in Xte_flags.columns if c.endswith('_abnormal')]
    score = Xte_flags[flag_cols].sum(axis=1)
    fpr, tpr, thresholds = roc_curve(y_test, score)
    score_auc = float(roc_auc_score(y_test, score))
    cutoff_records = []
    for k in sorted(score.unique()):
        pred = (score >= k).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        cutoff_records.append({'cutoff_k': int(k), 'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1), 'PPV': tp / max(tp + fp, 1), 'NPV': tn / max(tn + fn, 1), 'youden': tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1})
    Xtr_flags = get_final_panel_data(X_train, symbolic=True)
    lr_flags = LogisticRegression(max_iter=2000, random_state=SEED)
    lr_flags.fit(Xtr_flags[flag_cols], y_train)
    lr_prob = lr_flags.predict_proba(Xte_flags[flag_cols])[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    lr_auc = float(roc_auc_score(y_test, lr_prob))
    roc_table = pd.DataFrame({'fpr': np.concatenate([fpr, lr_fpr]), 'tpr': np.concatenate([tpr, lr_tpr]), 'score': ['Flag count'] * len(fpr) + ['LR on flags'] * len(lr_fpr)})
    dist = pd.DataFrame({'flag_count': score.values, 'outcome': np.where(y_test.values == 1, 'Cancer', 'Benign')})
    cutoffs = pd.DataFrame(cutoff_records)
    summary = pd.DataFrame({'item': ['flag_count_auc', 'lr_flags_auc', 'test_n'], 'value': [score_auc, lr_auc, len(y_test)]})
    print(f'[flag score] 计数评分 AUC={score_auc:.3f}，LR-flags AUC={lr_auc:.3f}')
    return (roc_table, dist, cutoffs, summary)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    t_start = time.time()
    df = load_raw_data(DATA_PATH)
    X, y, features, dropped_report = quality_control(df)
    X_train, X_test, y_train, y_test, _ = split_and_impute(X, y)
    Xte_basic = get_final_panel_data(X_test, symbolic=False)
    Xte_src = get_final_panel_data(X_test, symbolic=True)
    train_medians = X_train[FINAL_PANEL].median()
    print('[fit] 训练 Basic 与 SRC 模型 ...')
    cache = fit_models_cached(X_train, y_train)
    probs = {'Basic': {}, 'SRC': {}}
    for mode, Xte in (('Basic', Xte_basic), ('SRC', Xte_src)):
        for name, (model, scaler) in cache[mode].items():
            Xin = scaler.transform(Xte) if scaler is not None else Xte
            probs[mode][name] = model.predict_proba(Xin)[:, 1]
    robustness = run_robustness(cache, Xte_basic, y_test, train_medians)
    dca = run_dca(y_test, probs['Basic'], probs['SRC'])
    calib_points, calib_metrics = run_calibration(y_test, probs['Basic'], probs['SRC'])
    idi_points, idi_boot = run_idi_nri(y_test, probs['Basic'], probs['SRC'])
    op_points, op_boot = run_operating_points(y_test, probs['Basic'], probs['SRC'])
    learning = run_learning_curves(X_train, X_test, y_train, y_test)
    score_roc, score_dist, score_cutoffs, score_summary = run_flag_score(X_test, y_test, X_train, y_train)
    outputs = {'robustness_auc.csv': robustness, 'dca_net_benefit.csv': dca, 'calibration_curve_points.csv': calib_points, 'calibration_metrics.csv': calib_metrics, 'idi_nri_summary.csv': idi_points, 'idi_nri_bootstrap.csv': idi_boot, 'operating_point_summary.csv': op_points, 'operating_point_bootstrap.csv': op_boot, 'learning_curve.csv': learning, 'flag_score_roc.csv': score_roc, 'flag_score_distribution.csv': score_dist, 'flag_score_cutoffs.csv': score_cutoffs, 'flag_score_summary.csv': score_summary}
    for filename, table in outputs.items():
        table.to_csv(os.path.join(DATA_DIR, filename), index=False, encoding='utf-8-sig')
    summary = pd.DataFrame({'item': ['robust_shift_levels', 'robust_noise_levels', 'robust_outlier_levels', 'robust_missing_levels', 'robust_repeats', 'lc_fractions', 'lc_repeats', 'dca_threshold_min', 'dca_threshold_max', 'nri_cutoffs', 'bootstrap_n', 'runtime_minutes', 'random_seed'], 'value': [str(SHIFT_LEVELS), str(NOISE_LEVELS), str(OUTLIER_LEVELS), str(MISSING_LEVELS), ROBUST_REPEATS, str(LC_FRACTIONS), LC_REPEATS, float(DCA_THRESHOLDS.min()), float(DCA_THRESHOLDS.max()), str(NRI_CUTOFFS), N_BOOT, round((time.time() - t_start) / 60, 1), SEED]})
    summary.to_csv(os.path.join(DATA_DIR, 'fig4_extended_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'[save] Fig 4 扩展分析结果已保存至: {DATA_DIR}')
    print(f'[done] 总用时 {(time.time() - t_start) / 60:.1f} 分钟')
if __name__ == '__main__':
    main()
