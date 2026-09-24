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
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from boruta import BorutaPy
from skrebate import ReliefF
from common_preprocessing import DATA_PATH, FINAL_PANEL, ID_COLS, OUTPUT_ROOT, SEED, load_raw_data, quality_control, split_and_impute
FIG3_ROOT = os.path.join(OUTPUT_ROOT, 'Fig2_血液特征筛选')
DATA_DIR = os.path.join(FIG3_ROOT, '00_计算结果与原始数据')
TOP_N = 30
BORUTA_ITERATIONS = 40
N_BOOT = 2000
STABILITY_BOOTSTRAPS = 100
METHODS = ['Boruta', 'LASSO', 'mRMR', 'ReliefF', 'PCA']
BIOLOGICAL_AXES = {'CEA': 'Tumor marker', 'CYFRA': 'Tumor marker', 'SCCAg': 'Tumor marker', 'CA125_GC': 'Tumor marker', 'AAG': 'Inflammation', 'CRP': 'Inflammation', 'LYMPH%': 'Cellular immunity', 'LYMM': 'Cellular immunity', 'APTT': 'Coagulation', 'DD': 'Coagulation'}

def run_boruta_decision(X_train: pd.DataFrame, y_train: pd.Series, features: List[str]):
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators='auto', random_state=SEED, max_iter=100, verbose=0)
    boruta.fit(X_train[features].values, y_train.values)
    confirmed = [f for f, keep in zip(features, boruta.support_) if keep]
    tentative = [f for f, keep in zip(features, boruta.support_weak_) if keep]
    rejected = [f for f in features if f not in set(confirmed) | set(tentative)]
    print(f'[Boruta] confirmed ({len(confirmed)}): {confirmed}')
    print(f'[Boruta] tentative ({len(tentative)}): {tentative}')
    return (confirmed, tentative, rejected)

def run_boruta_history(X_train: pd.DataFrame, y_train: pd.Series, features: List[str], n_iterations: int=BORUTA_ITERATIONS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    Xv = X_train[features].values
    yv = y_train.values
    real_records = []
    shadow_records = []
    for iteration in range(n_iterations):
        shadows = np.column_stack([rng.permutation(Xv[:, j]) for j in range(Xv.shape[1])])
        X_aug = np.hstack([Xv, shadows])
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED + iteration, n_jobs=-1)
        rf.fit(X_aug, yv)
        importances = rf.feature_importances_
        real_imp = importances[:len(features)]
        shadow_imp = importances[len(features):]
        real_records.append(real_imp)
        shadow_records.append({'iteration': iteration + 1, 'shadowMin': float(np.min(shadow_imp)), 'shadowMean': float(np.mean(shadow_imp)), 'shadowMax': float(np.max(shadow_imp))})
        if (iteration + 1) % 10 == 0:
            print(f'[Boruta history] {iteration + 1}/{n_iterations}')
    history = pd.DataFrame(real_records, columns=features)
    history.insert(0, 'iteration', np.arange(1, n_iterations + 1))
    shadow_summary = pd.DataFrame(shadow_records)
    return (history, shadow_summary)

def run_lasso(X_train: pd.DataFrame, y_train: pd.Series, features: List[str]):
    Xs = pd.DataFrame(StandardScaler().fit_transform(X_train[features]), columns=features, index=X_train.index)
    Cs = np.logspace(-3, 1, 30)
    lasso = LogisticRegressionCV(Cs=Cs, penalty='l1', solver='saga', cv=5, scoring='roc_auc', random_state=SEED, max_iter=5000, n_jobs=-1, refit=True)
    lasso.fit(Xs, y_train)
    best_C = float(lasso.C_[0])
    selected = [f for f, coef in zip(features, lasso.coef_[0]) if coef != 0]
    coef_path = []
    for c in lasso.Cs_:
        model = LogisticRegression(C=float(c), penalty='l1', solver='saga', random_state=SEED, max_iter=5000)
        model.fit(Xs, y_train)
        coef_path.append(model.coef_[0])
    coef_path = pd.DataFrame(coef_path, columns=features)
    coef_path.insert(0, 'log10_C', np.log10(lasso.Cs_))
    coef_path.insert(0, 'C', lasso.Cs_)
    cv_scores = lasso.scores_[1]
    cv_table = pd.DataFrame({'C': lasso.Cs_, 'log10_C': np.log10(lasso.Cs_), 'cv_auc_mean': cv_scores.mean(axis=0), 'cv_auc_sd': cv_scores.std(axis=0)})
    selected_table = pd.DataFrame({'feature': features, 'coefficient': lasso.coef_[0], 'selected': [int(f in selected) for f in features], 'in_final_panel': [int(f in FINAL_PANEL) for f in features]})
    print(f'[LASSO] best C = {best_C:.4f}; selected ({len(selected)}): {selected}')
    return (selected, coef_path, cv_table, selected_table, best_C)

def mrmr_select(X: pd.DataFrame, mi: pd.Series, k: int=TOP_N):
    selected, rest, scores = ([], list(X.columns), [])
    corr = X.corr(method='spearman').abs()
    first = mi.idxmax()
    selected.append(first)
    rest.remove(first)
    scores.append(float(mi[first]))
    while len(selected) < min(k, len(X.columns)) and rest:
        step_scores = {f: float(mi[f] - corr.loc[f, selected].mean()) for f in rest}
        nxt = max(step_scores, key=step_scores.get)
        selected.append(nxt)
        rest.remove(nxt)
        scores.append(step_scores[nxt])
    return (selected, scores)

def run_mrmr(X_train: pd.DataFrame, y_train: pd.Series, features: List[str]):
    mi = pd.Series(mutual_info_classif(X_train[features], y_train, random_state=SEED), index=features)
    selected, scores = mrmr_select(X_train[features], mi, k=TOP_N)
    table = pd.DataFrame({'feature': selected, 'mrmr_score': scores, 'selection_rank': np.arange(1, len(selected) + 1), 'mutual_information': [float(mi[f]) for f in selected], 'in_final_panel': [int(f in FINAL_PANEL) for f in selected]})
    print(f'[mRMR] top {len(selected)} selected')
    return (selected, table)

def run_relieff(X_train: pd.DataFrame, y_train: pd.Series, features: List[str]):
    Xs = StandardScaler().fit_transform(X_train[features])
    relief = ReliefF(n_features_to_select=TOP_N, n_neighbors=10, n_jobs=-1)
    relief.fit(Xs, y_train.values)
    scores = pd.Series(relief.feature_importances_, index=features)
    selected = scores.nlargest(TOP_N).index.tolist()
    table = pd.DataFrame({'feature': scores.sort_values(ascending=False).index, 'relieff_weight': scores.sort_values(ascending=False).values})
    table['rank_by_weight'] = np.arange(1, len(table) + 1)
    table['selected_top30'] = (table['rank_by_weight'] <= TOP_N).astype(int)
    table['in_final_panel'] = table['feature'].isin(FINAL_PANEL).astype(int)
    print(f'[ReliefF] top {len(selected)} selected')
    return (selected, table)

def pca_select_from_array(Xs: np.ndarray, features: List[str]) -> Tuple[List[str], int, float]:
    pca = PCA(random_state=SEED).fit(Xs)
    n_retained = max(int((pca.explained_variance_ >= 1.0).sum()), 1)
    weights = np.abs(pca.components_[:n_retained]) * pca.explained_variance_ratio_[:n_retained, None]
    importance = pd.Series(weights.sum(axis=0), index=features)
    threshold = float(importance.mean())
    selected = importance[importance >= threshold].index.tolist()
    return (selected, n_retained, threshold)

def run_pca_selection(X_train: pd.DataFrame, features: List[str]):
    Xs = StandardScaler().fit_transform(X_train[features])
    pca = PCA(random_state=SEED).fit(Xs)
    eigenvalues = pca.explained_variance_
    retained = eigenvalues >= 1.0
    n_retained = int(retained.sum())
    weights = np.abs(pca.components_[:n_retained]) * pca.explained_variance_ratio_[:n_retained, None]
    importance = pd.Series(weights.sum(axis=0), index=features)
    threshold = float(importance.mean())
    selected = importance[importance >= threshold].index.tolist()
    variance_table = pd.DataFrame({'PC': np.arange(1, len(eigenvalues) + 1), 'eigenvalue': eigenvalues, 'variance_ratio': pca.explained_variance_ratio_, 'cumulative_variance': np.cumsum(pca.explained_variance_ratio_), 'retained_kaiser': retained.astype(int)})
    importance_sorted = importance.sort_values(ascending=False)
    importance_table = pd.DataFrame({'feature': importance_sorted.index, 'pca_importance': importance_sorted.values})
    importance_table['rank'] = np.arange(1, len(importance_table) + 1)
    importance_table['selected'] = (importance_table['pca_importance'] >= threshold).astype(int)
    importance_table['in_final_panel'] = importance_table['feature'].isin(FINAL_PANEL).astype(int)
    print(f'[PCA] Kaiser 保留 {n_retained} 个主成分（累计方差 {pca.explained_variance_ratio_[:n_retained].sum():.3f}），载荷阈值 {threshold:.4f}，选中 {len(selected)} 个特征')
    return (selected, importance_table, variance_table, threshold, n_retained)

def run_bootstrap_stability(X_train: pd.DataFrame, y_train: pd.Series, features: List[str], lasso_best_C: float, n_boot: int=STABILITY_BOOTSTRAPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 7)
    Xv = X_train[features].reset_index(drop=True)
    yv = y_train.reset_index(drop=True)
    counts = {m: pd.Series(0.0, index=features) for m in METHODS}
    replicate_records = []
    valid = 0
    t_start = time.time()
    while valid < n_boot:
        idx = rng.integers(0, len(Xv), len(Xv))
        yb = yv.iloc[idx]
        if yb.nunique() < 2:
            continue
        Xb = Xv.iloc[idx].reset_index(drop=True)
        yb = yb.reset_index(drop=True)
        Xs = StandardScaler().fit_transform(Xb)
        rf_lite = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED + valid, n_jobs=-1)
        boruta_lite = BorutaPy(rf_lite, n_estimators='auto', random_state=SEED + valid, max_iter=30, verbose=0)
        boruta_lite.fit(Xb.values, yb.values)
        sel_boruta = [f for f, keep in zip(features, boruta_lite.support_) if keep]
        lasso_fixed = LogisticRegression(C=lasso_best_C, penalty='l1', solver='saga', random_state=SEED, max_iter=5000)
        lasso_fixed.fit(Xs, yb)
        sel_lasso = [f for f, coef in zip(features, lasso_fixed.coef_[0]) if coef != 0]
        mi = pd.Series(mutual_info_classif(Xb, yb, random_state=SEED), index=features)
        sel_mrmr, _ = mrmr_select(Xb, mi, k=TOP_N)
        relief = ReliefF(n_features_to_select=TOP_N, n_neighbors=10, n_jobs=-1)
        relief.fit(Xs, yb.values)
        sel_relieff = pd.Series(relief.feature_importances_, index=features).nlargest(TOP_N).index.tolist()
        sel_pca, _, _ = pca_select_from_array(Xs, features)
        selections = [sel_boruta, sel_lasso, sel_mrmr, sel_relieff, sel_pca]
        for method, selected in zip(METHODS, selections):
            counts[method].loc[selected] += 1
            replicate_records.append({'replicate': valid + 1, 'method': method, 'n_selected': len(selected)})
        valid += 1
        if valid % 5 == 0 or valid == n_boot:
            elapsed = time.time() - t_start
            eta = elapsed / valid * (n_boot - valid)
            print(f'[stability] {valid}/{n_boot} 完成，已用时 {elapsed / 60:.1f} 分钟，预计剩余 {eta / 60:.1f} 分钟')
    frequency = pd.DataFrame({m: counts[m] / n_boot for m in METHODS})
    frequency.insert(0, 'feature', features)
    frequency['mean_frequency'] = frequency[METHODS].mean(axis=1)
    frequency['in_final_panel'] = frequency['feature'].isin(FINAL_PANEL).astype(int)
    frequency = frequency.sort_values('mean_frequency', ascending=False)
    replicate_sizes = pd.DataFrame(replicate_records)
    print('[stability] bootstrap 稳定性选择完成')
    return (frequency, replicate_sizes)

def bootstrap_auc_ci(y: pd.Series, values: pd.Series, n_boot: int=N_BOOT) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    yv = np.asarray(y)
    xv = np.asarray(values)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yv[idx], xv[idx]))
    lower, upper = np.percentile(np.asarray(aucs), [2.5, 97.5])
    return (float(lower), float(upper))

def run_final_panel_univariate(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    records = []
    for feature in FINAL_PANEL:
        values = X_train[feature]
        auc = float(roc_auc_score(y_train, values))
        ci_lower, ci_upper = bootstrap_auc_ci(y_train, values)
        p_value = float(mannwhitneyu(values[y_train == 1], values[y_train == 0], alternative='two-sided').pvalue)
        records.append({'feature': feature, 'biological_axis': BIOLOGICAL_AXES[feature], 'auc': auc, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'mannwhitney_p': p_value, 'direction': 'higher in cancer' if auc >= 0.5 else 'lower in cancer', 'distance_from_0.5': abs(auc - 0.5)})
    table = pd.DataFrame(records).sort_values('distance_from_0.5', ascending=False)
    print('[final panel] univariate AUC table completed')
    return table

def run_final_panel_correlation(X_train: pd.DataFrame) -> pd.DataFrame:
    corr = X_train[FINAL_PANEL].corr(method='spearman')
    corr_out = corr.copy()
    corr_out.insert(0, 'feature', corr_out.index)
    print('[final panel] Spearman correlation matrix completed')
    return corr_out

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_raw_data(DATA_PATH)
    X, y, features, dropped_report = quality_control(df)
    X_train, X_test, y_train, y_test, _ = split_and_impute(X, y)
    boruta_confirmed, boruta_tentative, boruta_rejected = run_boruta_decision(X_train, y_train, features)
    boruta_history, boruta_shadow = run_boruta_history(X_train, y_train, features)
    boruta_decisions = pd.DataFrame({'feature': features, 'decision': ['confirmed' if f in boruta_confirmed else 'tentative' if f in boruta_tentative else 'rejected' for f in features], 'in_final_panel': [int(f in FINAL_PANEL) for f in features]})
    lasso_selected, lasso_path, lasso_cv, lasso_table, best_C = run_lasso(X_train, y_train, features)
    mrmr_selected, mrmr_table = run_mrmr(X_train, y_train, features)
    relieff_selected, relieff_table = run_relieff(X_train, y_train, features)
    pca_selected, pca_importance, pca_variance, pca_threshold, pca_n_retained = run_pca_selection(X_train, features)
    stability_frequency, stability_replicates = run_bootstrap_stability(X_train, y_train, features, lasso_best_C=best_C)
    final_panel_auc = run_final_panel_univariate(X_train, y_train)
    final_panel_corr = run_final_panel_correlation(X_train)
    votes = pd.Series(0, index=features, dtype=int)
    for selected in (boruta_confirmed, lasso_selected, mrmr_selected, relieff_selected, pca_selected):
        votes.loc[selected] += 1
    votes_table = pd.DataFrame({'feature': votes.sort_values(ascending=False).index, 'votes': votes.sort_values(ascending=False).values})
    votes_table['in_final_panel'] = votes_table['feature'].isin(FINAL_PANEL).astype(int)
    boruta_matrix_set = set(boruta_confirmed) | set(boruta_tentative)
    selection_sets = {'Boruta': boruta_matrix_set, 'LASSO': set(lasso_selected), 'mRMR': set(mrmr_selected), 'ReliefF': set(relieff_selected), 'PCA': set(pca_selected)}
    other_features = votes.drop(index=FINAL_PANEL, errors='ignore').sort_values(ascending=False).index.tolist()
    matrix_rows = FINAL_PANEL + [f for f in other_features if f not in FINAL_PANEL][:8]
    selection_matrix = pd.DataFrame({method: [int(f in selected) for f in matrix_rows] for method, selected in selection_sets.items()}, index=matrix_rows)
    selection_matrix['Final panel'] = [int(f in FINAL_PANEL) for f in matrix_rows]
    selection_matrix.insert(0, 'feature', matrix_rows)
    consensus = votes_table.merge(stability_frequency[['feature', *METHODS, 'mean_frequency']], on='feature', how='left')
    consensus = consensus.sort_values(['votes', 'mean_frequency'], ascending=False).reset_index(drop=True)
    n_raw_variables = len([c for c in df.columns if c not in set(ID_COLS)])
    summary_rows = [('raw_variables', n_raw_variables), ('qc_candidate_variables', len(features)), ('train_n', len(X_train)), ('test_n', len(X_test)), ('train_cancer_n', int(y_train.sum())), ('train_benign_n', int((1 - y_train).sum())), ('test_cancer_n', int(y_test.sum())), ('test_benign_n', int((1 - y_test).sum())), ('boruta_confirmed_n', len(boruta_confirmed)), ('boruta_tentative_n', len(boruta_tentative)), ('lasso_selected_n', len(lasso_selected)), ('mrmr_top_n', len(mrmr_selected)), ('relieff_top_n', len(relieff_selected)), ('pca_selected_n', len(pca_selected)), ('pca_kaiser_pcs', pca_n_retained), ('pca_importance_threshold', pca_threshold), ('stability_bootstrap_n', STABILITY_BOOTSTRAPS), ('final_panel_n', len(FINAL_PANEL)), ('lasso_best_C', best_C), ('random_seed', SEED)]
    summary = pd.DataFrame(summary_rows, columns=['item', 'value'])
    pd.Series(features, name='candidate_variable').to_csv(os.path.join(DATA_DIR, 'candidate_variables.csv'), index=False, encoding='utf-8-sig')
    dropped_report.to_csv(os.path.join(DATA_DIR, 'collinearity_dropped_variables.csv'), index=False, encoding='utf-8-sig')
    boruta_decisions.to_csv(os.path.join(DATA_DIR, 'boruta_decisions.csv'), index=False, encoding='utf-8-sig')
    boruta_history.to_csv(os.path.join(DATA_DIR, 'boruta_iteration_history.csv'), index=False, encoding='utf-8-sig')
    boruta_shadow.to_csv(os.path.join(DATA_DIR, 'boruta_shadow_summary.csv'), index=False, encoding='utf-8-sig')
    lasso_path.to_csv(os.path.join(DATA_DIR, 'lasso_coefficient_path.csv'), index=False, encoding='utf-8-sig')
    lasso_cv.to_csv(os.path.join(DATA_DIR, 'lasso_cv_scores.csv'), index=False, encoding='utf-8-sig')
    lasso_table.to_csv(os.path.join(DATA_DIR, 'lasso_selected_features.csv'), index=False, encoding='utf-8-sig')
    mrmr_table.to_csv(os.path.join(DATA_DIR, 'mrmr_scores.csv'), index=False, encoding='utf-8-sig')
    relieff_table.to_csv(os.path.join(DATA_DIR, 'relieff_scores.csv'), index=False, encoding='utf-8-sig')
    pca_variance.to_csv(os.path.join(DATA_DIR, 'pca_variance_explained.csv'), index=False, encoding='utf-8-sig')
    pca_importance.to_csv(os.path.join(DATA_DIR, 'pca_feature_importance.csv'), index=False, encoding='utf-8-sig')
    stability_frequency.to_csv(os.path.join(DATA_DIR, 'bootstrap_selection_frequency.csv'), index=False, encoding='utf-8-sig')
    stability_replicates.to_csv(os.path.join(DATA_DIR, 'bootstrap_replicate_sizes.csv'), index=False, encoding='utf-8-sig')
    final_panel_auc.to_csv(os.path.join(DATA_DIR, 'final_panel_univariate_auc.csv'), index=False, encoding='utf-8-sig')
    final_panel_corr.to_csv(os.path.join(DATA_DIR, 'final_panel_spearman_correlation.csv'), index=False, encoding='utf-8-sig')
    votes_table.to_csv(os.path.join(DATA_DIR, 'feature_selection_votes.csv'), index=False, encoding='utf-8-sig')
    consensus.to_csv(os.path.join(DATA_DIR, 'consensus_votes_and_stability.csv'), index=False, encoding='utf-8-sig')
    selection_matrix.to_csv(os.path.join(DATA_DIR, 'five_method_selection_matrix.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame({'final_panel_feature': FINAL_PANEL}).to_csv(os.path.join(DATA_DIR, 'final_10_biomarker_panel.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(DATA_DIR, 'fig3_analysis_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'[save] Fig 3 全部计算结果与原始数据已保存至: {DATA_DIR}')
if __name__ == '__main__':
    main()
