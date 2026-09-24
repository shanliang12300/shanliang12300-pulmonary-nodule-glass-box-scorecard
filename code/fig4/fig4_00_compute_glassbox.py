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
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from common_preprocessing import DATA_PATH, SEED, load_raw_data, quality_control
from fig3 import fig3_00_compute_joint_selection as j1
from fig3.fig3_03_compute_models import ALL_INTERVALS, JOINT_BLOOD, JOINT_PANEL, add_src_flags
from shared_interpret_utils import N_BOOT, SCORECARD_MAX_POINTS, UNITS, sigmoid, wilson_ci
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
DATA_DIR = os.path.join(OUTPUT_ROOT, 'Fig4_SRC玻璃盒', '00_计算结果与原始数据')
CT_SIZE, CT_MISS = ('CT_基线最大径mm', 'CT_径线缺失')
CT_SIGNS = ['CT_基线分叶', 'CT_基线毛刺', CT_MISS]
BLOOD_RULES = [f'{c}_abnormal' for c in JOINT_BLOOD]
DIAM_BINS = ((8, 15), (15, 30), (30, None))
DIAM_RULES = [f'CT_直径{a}–{b}mm' if b is not None else f'CT_直径>{a}mm' for a, b in DIAM_BINS]
RULES = BLOOD_RULES + CT_SIGNS + DIAM_RULES
UNITS_JOINT = {**UNITS, 'AU_Urea': 'mmol/L', 'MCHC': 'g/L'}

def rule_type(rule: str) -> str:
    if rule.endswith('_abnormal'):
        return 'blood'
    if rule.startswith('CT_直径'):
        return 'diameter'
    return 'ct_sign'

def threshold_text(rule: str) -> str:
    if rule.endswith('_abnormal'):
        lo, hi = ALL_INTERVALS[rule[:-len('_abnormal')]]
        if lo is None:
            return f'>{hi}'
        if hi is None:
            return f'<{lo}'
        return f'{lo}–{hi}'
    if rule.startswith('CT_直径'):
        return rule[len('CT_直径'):]
    if rule == CT_MISS:
        return 'no diameter in report'
    return 'sign present'

def build_rule_matrix(X: pd.DataFrame) -> pd.DataFrame:
    R = pd.DataFrame(index=X.index)
    for c in JOINT_BLOOD:
        lo, hi = ALL_INTERVALS[c]
        flag = pd.Series(0, index=X.index, dtype=int)
        if hi is not None:
            flag = flag | (X[c] > hi).astype(int)
        if lo is not None:
            flag = flag | (X[c] < lo).astype(int)
        R[f'{c}_abnormal'] = flag.astype(int)
    for c in CT_SIGNS:
        R[c] = X[c].astype(int)
    d = pd.to_numeric(X[CT_SIZE], errors='coerce')
    measured = X[CT_MISS] == 0
    for (a, b), name in zip(DIAM_BINS, DIAM_RULES):
        hit = (d > a) & (d <= b) if b is not None else d > a
        R[name] = (hit & measured).astype(int)
    return R[RULES]

def load_joint_raw():
    df_blood = load_raw_data(DATA_PATH)
    X_b, y, _, _ = quality_control(df_blood)
    X_b['pid'] = df_blood['pid'].astype(str).values
    mg = pd.read_csv(j1.MERGED, encoding='utf-8-sig', dtype={'pid': str})
    for c in ['CT_基线分叶', 'CT_基线毛刺']:
        mg[c] = mg[c].astype(str).str.lower().map({'true': 1.0, 'false': 0.0})
    mg[CT_SIZE] = pd.to_numeric(mg[CT_SIZE], errors='coerce')
    joint = X_b.merge(mg[['pid', CT_SIZE, 'CT_基线分叶', 'CT_基线毛刺']], on='pid', how='left')
    return (joint, y)

def run_rule_weights(R_train, y_train):
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(R_train, y_train)
    rng = np.random.default_rng(SEED + 23)
    yv = np.asarray(y_train)
    boot = []
    Rv = R_train.values
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(Rv), len(Rv))
        if len(np.unique(yv[idx])) < 2:
            continue
        m = LogisticRegression(max_iter=2000, random_state=SEED)
        m.fit(Rv[idx], yv[idx])
        boot.append(m.coef_[0])
    boot = pd.DataFrame(boot, columns=RULES)
    max_abs = float(np.abs(lr.coef_[0]).max())
    rows = []
    for rule, coef in zip(RULES, lr.coef_[0]):
        lo, hi = np.percentile(boot[rule], [2.5, 97.5])
        rows.append({'rule': rule, 'type': rule_type(rule), 'coefficient': float(coef), 'ci_lower': float(lo), 'ci_upper': float(hi), 'odds_ratio': float(np.exp(coef)), 'or_ci_lower': float(np.exp(lo)), 'or_ci_upper': float(np.exp(hi)), 'scorecard_points': int(np.round(coef / max_abs * SCORECARD_MAX_POINTS))})
    table = pd.DataFrame(rows).sort_values('coefficient', ascending=False)
    print(f'[rule weights] 12 规则系数完成（截距 {lr.intercept_[0]:.3f}）')
    return (table, boot, float(lr.intercept_[0]), lr)

def run_scorecard(weights, intercept, R_train, R_test, S_train, S_test, y_train, y_test):
    points = weights.set_index('rule')['scorecard_points']
    score_test = R_test[points.index].values @ points.values
    scorecard_auc = float(roc_auc_score(y_test, score_test))
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(R_train[points.index], y_train)
    lr_prob = lr.predict_proba(R_test[points.index])[:, 1]
    lr_auc = float(roc_auc_score(y_test, lr_prob))
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
    rf.fit(S_train, y_train)
    rf_prob = rf.predict_proba(S_test)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_prob))
    roc_rows = []
    for name, values in (('Integer scorecard', score_test), ('LR on 12 rules', lr_prob), ('SRC-RF (full)', rf_prob)):
        fpr, tpr, _ = roc_curve(y_test, values)
        roc_rows.extend(({'model': name, 'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr, tpr)))
    yv = np.asarray(y_test)
    risk_rows = []
    for s in sorted(np.unique(score_test)):
        mask = score_test == s
        n, events = (int(mask.sum()), int(yv[mask].sum()))
        lo, hi = wilson_ci(events, n)
        risk_rows.append({'score': int(s), 'n': n, 'events': events, 'observed_rate': events / n, 'ci_lower': lo, 'ci_upper': hi})
    summary = pd.DataFrame({'item': ['scorecard_auc', 'lr_rules_auc', 'src_rf_auc', 'lr_intercept', 'score_min', 'score_max', 'n_rules'], 'value': [scorecard_auc, lr_auc, rf_auc, intercept, int(score_test.min()), int(score_test.max()), len(RULES)]})
    print(f'[scorecard] 整数评分卡 AUC={scorecard_auc:.3f}，LR-rules={lr_auc:.3f}，SRC-RF={rf_auc:.3f}')
    return (pd.DataFrame(roc_rows), pd.DataFrame(risk_rows), pd.DataFrame({'score': score_test, 'outcome': yv}), summary)

def run_audit(lr, intercept, weights, R_test, joint_raw, te_pos, y_test):
    probs = lr.predict_proba(R_test[RULES])[:, 1]
    yv = np.asarray(y_test)
    cases = {'A': int(np.argmax(np.where(yv == 1, probs, -1))), 'B': int(np.argmin(np.where(yv == 0, probs, 2))), 'C': int(np.argmax(np.where(yv == 0, probs, -1)))}
    coef_map = weights.set_index('rule')['coefficient']
    raw_test = joint_raw.iloc[te_pos]
    rows = []
    for label, pos in cases.items():
        fired = R_test.iloc[pos][RULES]
        raw = raw_test.iloc[pos]
        logit = intercept
        for rule in RULES:
            contribution = float(coef_map[rule] * fired[rule])
            logit += contribution
            rt = rule_type(rule)
            if rt == 'blood':
                var = rule[:-len('_abnormal')]
                v = raw[var]
                value_text = 'NA（中位插补）' if pd.isna(v) else f'{float(v):.3g}'
                unit = UNITS_JOINT.get(var, '')
            elif rt == 'diameter':
                v = raw[CT_SIZE]
                value_text = 'not reported' if pd.isna(v) else f'{float(v):.1f}'
                unit = 'mm'
            else:
                value_text = 'Yes' if fired[rule] == 1 else 'No'
                unit = ''
            rows.append({'case': label, 'test_position': pos, 'true_outcome': 'Cancer' if yv[pos] == 1 else 'Benign', 'rule': rule, 'type': rt, 'value_text': value_text, 'unit': unit, 'threshold_text': threshold_text(rule), 'fired': int(fired[rule]), 'coefficient': float(coef_map[rule]), 'contribution': contribution, 'case_probability': float(sigmoid(logit)), 'case_intercept': intercept})
    print('[audit] 患者级审计：A=真阳性 B=真阴性 C=假阳性')
    return pd.DataFrame(rows)

def run_rule_card(X_train, y_train):
    blood = X_train[JOINT_BLOOD].copy()
    blood['outcome'] = np.where(np.asarray(y_train) == 1, 'Cancer', 'Benign')
    long = blood.melt(id_vars='outcome', var_name='biomarker', value_name='value')
    long['unit'] = long['biomarker'].map(UNITS_JOINT)
    long['ref_lower'] = long['biomarker'].map({k: v[0] for k, v in ALL_INTERVALS.items()})
    long['ref_upper'] = long['biomarker'].map({k: v[1] for k, v in ALL_INTERVALS.items()})
    ct_rows = []
    outc = np.where(np.asarray(y_train) == 1, 'Cancer', 'Benign')
    diam = pd.DataFrame({'outcome': outc, 'diameter_mm': pd.to_numeric(X_train[CT_SIZE], errors='coerce')})
    for sign in ['CT_基线分叶', 'CT_基线毛刺']:
        for grp in ('Cancer', 'Benign'):
            mask = outc == grp
            ct_rows.append({'item': sign, 'group': grp, 'positive_rate': float(X_train.loc[mask, sign].mean()), 'n': int(mask.sum())})
    for grp in ('Cancer', 'Benign'):
        mask = outc == grp
        sub = diam.loc[mask, 'diameter_mm'].dropna()
        ct_rows.append({'item': 'diameter_median', 'group': grp, 'positive_rate': float(sub.median()), 'n': len(sub)})
    return (long, diam, pd.DataFrame(ct_rows))

def run_rule_prevalence(R_train, R_test, y_train, y_test):
    rows = []
    for cohort, R, y in (('training', R_train, y_train), ('test', R_test, y_test)):
        yv = np.asarray(y)
        for rule in RULES:
            ca, be = (yv == 1, yv == 0)
            tab = np.array([[int(R.loc[ca, rule].sum()), int((1 - R.loc[ca, rule]).sum())], [int(R.loc[be, rule].sum()), int((1 - R.loc[be, rule]).sum())]])
            _, p = fisher_exact(tab)
            rows.append({'cohort': cohort, 'rule': rule, 'type': rule_type(rule), 'cancer_positive_rate': float(R.loc[ca, rule].mean()), 'benign_positive_rate': float(R.loc[be, rule].mean()), 'fisher_exact_p': float(p)})
    return pd.DataFrame(rows)

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, _, _ = j1.build_joint_dataset()
    joint_raw, y_all = load_joint_raw()
    te_pos = train_test_split(np.arange(len(joint_raw)), test_size=0.3, random_state=SEED, stratify=y_all)[1]
    assert np.allclose(joint_raw.iloc[te_pos]['CT_基线分叶'].fillna(0).values, X_test['CT_基线分叶'].values), 'raw 与模型输入未对齐'
    R_train, R_test = (build_rule_matrix(X_train), build_rule_matrix(X_test))
    S_train = add_src_flags(X_train[JOINT_PANEL])
    S_test = add_src_flags(X_test[JOINT_PANEL])
    weights, weights_boot, intercept, lr_rules = run_rule_weights(R_train, y_train)
    sc_roc, sc_risk, sc_scores, sc_summary = run_scorecard(weights, intercept, R_train, R_test, S_train, S_test, y_train, y_test)
    audit = run_audit(lr_rules, intercept, weights, R_test, joint_raw, te_pos, y_test)
    blood_long, diam_df, ct_rates = run_rule_card(X_train, y_train)
    prevalence = run_rule_prevalence(R_train, R_test, y_train, y_test)
    rule_defs = pd.DataFrame([{'rule': r, 'type': rule_type(r), 'variable': r[:-len('_abnormal')] if r.endswith('_abnormal') else CT_SIZE if r.startswith('CT_直径>') else r, 'threshold_text': threshold_text(r), 'unit': UNITS_JOINT.get(r[:-len('_abnormal')], '') if r.endswith('_abnormal') else 'mm' if r.startswith('CT_直径>') else ''} for r in RULES])
    outputs = {'fig6_rule_definitions.csv': rule_defs, 'fig6_rule_weights.csv': weights, 'fig6_rule_weights_bootstrap.csv': weights_boot, 'fig6_rule_prevalence.csv': prevalence, 'fig6_scorecard_roc.csv': sc_roc, 'fig6_scorecard_risk.csv': sc_risk, 'fig6_scorecard_test_scores.csv': sc_scores, 'fig6_scorecard_summary.csv': sc_summary, 'fig6_patient_audit.csv': audit, 'fig6_rule_card_blood_distributions.csv': blood_long, 'fig6_rule_card_diameter.csv': diam_df, 'fig6_rule_card_ct_rates.csv': ct_rates}
    for fn, table in outputs.items():
        table.to_csv(os.path.join(DATA_DIR, fn), index=False, encoding='utf-8-sig')
    print('\n===== 规则权重（按系数排序） =====')
    print(weights[['rule', 'coefficient', 'odds_ratio', 'scorecard_points']].to_string(index=False))
    print(f'\n[save] {DATA_DIR}')
if __name__ == '__main__':
    main()
