from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
SEED = 42
DATA_PATH = os.environ.get('PULNOD_DATA_PATH', 'D:\\333\\患者原始数据.xlsx')
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
PREPROCESS_DIR = os.path.join(OUTPUT_ROOT, '01_公共预处理')
LABEL_COL = '标签_诊断表'
ID_COLS = ['Number', 'pid', '标签_诊断表', '诊断分类', 'nan', 'target']
FINAL_PANEL = ['CEA', 'CYFRA', 'SCCAg', 'CA125_GC', 'AAG', 'CRP', 'LYMPH%', 'LYMM', 'APTT', 'DD']
REFERENCE_INTERVALS = {'CEA': (None, 5.0), 'CYFRA': (None, 3.3), 'SCCAg': (None, 1.5), 'CA125_GC': (None, 35.0), 'AAG': (0.55, 1.4), 'CRP': (None, 10.0), 'LYMPH%': (20.0, 50.0), 'LYMM': (1.1, 3.2), 'APTT': (25.0, 35.0), 'DD': (None, 0.55)}
REFERENCE_INTERVALS_EXT = {'AU_Urea': (2.5, 7.1), 'MCHC': (320.0, 360.0)}

def load_raw_data(path: str=DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到原始数据文件：{path}')
    df = pd.read_excel(path)
    if LABEL_COL not in df.columns:
        raise KeyError(f'原始数据中缺少标签列：{LABEL_COL}')
    df = df.copy()
    df['target'] = (df[LABEL_COL] == '肺癌').astype(int)
    print(f'[load] 样本量: {df.shape[0]}, 原始列数: {df.shape[1]}')
    print(f"[load] 肺癌: {(df['target'] == 1).sum()}, 良性对照: {(df['target'] == 0).sum()}")
    return df

def quality_control(df: pd.DataFrame, missing_threshold: float=0.2, corr_threshold: float=0.8) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    miss = df.isna().mean()
    candidates = [c for c in miss[miss <= missing_threshold].index if c not in ID_COLS]
    X_all = df[candidates].apply(pd.to_numeric, errors='coerce')
    corr = X_all.corr(method='spearman').abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = set()
    records = []
    for col in upper.columns:
        partners = upper.index[upper[col] > corr_threshold].tolist()
        for partner in partners:
            if partner in dropped or col in dropped:
                continue
            drop_col = col if X_all[col].isna().mean() >= X_all[partner].isna().mean() else partner
            keep_col = partner if drop_col == col else col
            dropped.add(drop_col)
            records.append({'dropped_variable': drop_col, 'retained_variable': keep_col, 'spearman_abs_rho': float(corr.loc[partner, col]), 'dropped_missing_rate': float(X_all[drop_col].isna().mean()), 'retained_missing_rate': float(X_all[keep_col].isna().mean())})
    features = [c for c in candidates if c not in dropped]
    dropped_report = pd.DataFrame(records)
    print(f'[QC] 缺失率≤{missing_threshold:.0%} 的变量: {len(candidates)}')
    print(f'[QC] 共线性剔除变量: {len(dropped)}')
    print(f'[QC] 保留候选变量: {len(features)}')
    return (X_all[features], df['target'], features, dropped_report)

def split_and_impute(X: pd.DataFrame, y: pd.Series, test_size: float=0.3, seed: int=SEED):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    medians = X_tr.median()
    X_tr = X_tr.fillna(medians)
    X_te = X_te.fillna(medians)
    print(f'[split] 训练集: {X_tr.shape}, 肺癌 {int(y_tr.sum())}, 良性 {int((1 - y_tr).sum())}')
    print(f'[split] 测试集: {X_te.shape}, 肺癌 {int(y_te.sum())}, 良性 {int((1 - y_te).sum())}')
    return (X_tr, X_te, y_tr, y_te, medians)

def add_symbolic_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, (lo, hi) in REFERENCE_INTERVALS.items():
        if col not in X.columns:
            raise KeyError(f'符号化规则所需变量不存在：{col}')
        flag = pd.Series(0, index=X.index, dtype=int)
        if hi is not None:
            flag = flag | (X[col] > hi).astype(int)
        if lo is not None:
            flag = flag | (X[col] < lo).astype(int)
        X[f'{col}_abnormal'] = flag.astype(int)
    return X

def get_final_panel_data(X: pd.DataFrame, symbolic: bool=False) -> pd.DataFrame:
    missing = [c for c in FINAL_PANEL if c not in X.columns]
    if missing:
        raise KeyError(f'最终面板变量缺失：{missing}')
    out = X[FINAL_PANEL].copy()
    return add_symbolic_flags(out) if symbolic else out

def run_preprocessing(save: bool=True) -> Dict[str, object]:
    df = load_raw_data(DATA_PATH)
    X, y, features, dropped_report = quality_control(df)
    X_tr, X_te, y_tr, y_te, medians = split_and_impute(X, y)
    result = {'raw_data': df, 'X': X, 'y': y, 'features': features, 'X_train': X_tr, 'X_test': X_te, 'y_train': y_tr, 'y_test': y_te, 'train_medians': medians, 'dropped_report': dropped_report}
    if save:
        os.makedirs(PREPROCESS_DIR, exist_ok=True)
        train_out = X_tr.copy()
        train_out['target'] = y_tr
        test_out = X_te.copy()
        test_out['target'] = y_te
        train_out.to_csv(os.path.join(PREPROCESS_DIR, 'train_imputed.csv'), index=True, encoding='utf-8-sig')
        test_out.to_csv(os.path.join(PREPROCESS_DIR, 'test_imputed.csv'), index=True, encoding='utf-8-sig')
        pd.Series(features, name='candidate_variable').to_csv(os.path.join(PREPROCESS_DIR, 'candidate_variables_after_QC.csv'), index=False, encoding='utf-8-sig')
        dropped_report.to_csv(os.path.join(PREPROCESS_DIR, 'collinearity_dropped_variables.csv'), index=False, encoding='utf-8-sig')
        medians.rename('train_median').to_csv(os.path.join(PREPROCESS_DIR, 'train_median_imputation_values.csv'), encoding='utf-8-sig')
        pd.DataFrame({'variable': FINAL_PANEL, 'lower_limit': [REFERENCE_INTERVALS[v][0] for v in FINAL_PANEL], 'upper_limit': [REFERENCE_INTERVALS[v][1] for v in FINAL_PANEL]}).to_csv(os.path.join(PREPROCESS_DIR, 'clinical_reference_intervals.csv'), index=False, encoding='utf-8-sig')
        summary = '\n'.join(['公共预处理摘要', '=' * 40, f'原始数据: {DATA_PATH}', f'总样本量: {len(df)}', f'肺癌: {int((y == 1).sum())}', f'良性对照: {int((y == 0).sum())}', f'质控后候选变量数: {len(features)}', f'训练集: {X_tr.shape[0]} 例（肺癌 {int(y_tr.sum())}, 良性 {int((1 - y_tr).sum())}）', f'测试集: {X_te.shape[0]} 例（肺癌 {int(y_te.sum())}, 良性 {int((1 - y_te).sum())}）', '缺失值填补: 训练集中位数（同一中位数用于测试集，避免数据泄漏）', f'随机种子: {SEED}'])
        with open(os.path.join(PREPROCESS_DIR, 'preprocessing_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(summary + '\n')
        print(f'[save] 公共预处理结果已保存至: {PREPROCESS_DIR}')
    return result
if __name__ == '__main__':
    run_preprocessing(save=True)
