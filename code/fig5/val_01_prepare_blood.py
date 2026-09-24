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
from common_preprocessing import OUTPUT_ROOT
VAL_DIR = _os.path.join(OUTPUT_ROOT, '验证组')
BLOOD_PATH = _os.path.join(VAL_DIR, '验证组血液检查数据.xlsx')
VAL_IDMAP = _os.path.join(VAL_DIR, 'pid↔HospNo 对照表.xlsx')
TRAIN_IDMAP = _os.path.join(OUTPUT_ROOT, 'pid↔HospNo 对照表.xlsx')
OUT_DIR = _os.path.join(OUTPUT_ROOT, 'Fig5_外部验证', '00_计算结果与原始数据')
SHEETS = ['验证组1', '验证组2']
REQUIRED_ITEMS = ['CEA', 'CYFRA', 'SCCAg', 'CA125_GC', 'AAG', 'CRP', 'LYMPH%', 'LYMM', 'APTT', 'DD', 'AU_Urea', 'MCHC']
LABEL_MAP = {'肺癌': 1, '恶性': 1, '对照': 0, '良性': 0, '对照(肺部阴影/占位)': 0, '对照（肺部阴影/占位）': 0}
PID_CANDIDATES = ['病员号', '病人号', 'pid', '患者号']
ITEM_CANDIDATES = ['英文缩写', '检验项目', '项目缩写', '项目']
RES_CANDIDATES = ['检验结果', '结果', '结果值']
DATE_CANDIDATES = ['样本日期', '检验日期', '报告日期', '采样日期', '日期']

def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _clean_pid(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace('\\.0$', '', regex=True), errors='coerce')

def load_sheet(path: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    pid_col = _pick(raw.columns, PID_CANDIDATES)
    n_need = sum((c in REQUIRED_ITEMS for c in raw.columns))
    if pid_col is not None and n_need >= 3:
        print(f'[{sheet}] 检测到宽表格式（含 {n_need}/12 个必需项目列），直读')
        df = raw.copy()
        df = df.rename(columns={pid_col: 'pid'})
        df['pid'] = _clean_pid(df['pid'])
        df = df.dropna(subset=['pid'])
        df['pid'] = df['pid'].astype(int)
        if 'target' in df.columns:
            df['_target'] = pd.to_numeric(df['target'], errors='coerce')
        elif '标签' in df.columns:
            df['_target'] = df['标签'].astype(str).str.strip().map(LABEL_MAP)
        else:
            df['_target'] = np.nan
        df = df.drop(columns=[c for c in ['标签', 'target', 'label'] if c in df.columns])
        for c in df.columns:
            if c == 'pid':
                continue
            v = df[c].astype(str).str.strip().str.replace('^[<>≤≥]\\s*', '', regex=True)
            df[c] = pd.to_numeric(v, errors='coerce')
        wide = df.groupby('pid').first()
        print(f'[{sheet}] 宽表 {wide.shape[0]} 人 × {wide.shape[1]} 项')
        return wide
    item_col = _pick(raw.columns, ITEM_CANDIDATES)
    res_col = _pick(raw.columns, RES_CANDIDATES)
    if pid_col is None or item_col is None or res_col is None:
        raise ValueError(f'[{sheet}] 列识别失败，现有列：{list(raw.columns)}，请发给我')
    print(f'[{sheet}] 长表格式，列: pid={pid_col}, 项目={item_col}, 结果={res_col}')
    df = raw[[c for c in {pid_col, item_col, res_col, *DATE_CANDIDATES} if c in raw.columns]].copy()
    df[pid_col] = df[pid_col].ffill()
    df['pid'] = _clean_pid(df[pid_col])
    df = df.dropna(subset=['pid'])
    df['pid'] = df['pid'].astype(int)
    df['item'] = df[item_col].astype(str).str.strip()
    cleaned = df[res_col].astype(str).str.strip().str.replace('^[<>≤≥]\\s*', '', regex=True)
    df['value'] = pd.to_numeric(cleaned, errors='coerce')
    date_col = _pick(raw.columns, DATE_CANDIDATES)
    if date_col:
        df['_d'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values('_d', kind='stable')
        print(f'[{sheet}] 按日期列 [{date_col}] 排序取基线')
    else:
        print(f'[{sheet}] 警告：无日期列，按文件顺序取首次记录')
    first = df.dropna(subset=['value']).drop_duplicates(['pid', 'item'])
    wide = first.pivot(index='pid', columns='item', values='value')
    print(f'[{sheet}] {len(raw)} 行 → {wide.shape[0]} 人 × {wide.shape[1]} 项')
    return wide

def load_val_idmap() -> pd.DataFrame:
    xl = pd.ExcelFile(VAL_IDMAP)
    sheets = [s for s in xl.sheet_names if s in ('验证组1', '验证组2')] or xl.sheet_names[:1]
    frames = []
    for s in sheets:
        m = xl.parse(s)
        pid_col = _pick(m.columns, ['pid', '病员号', '病人号'])
        hosp_col = _pick(m.columns, ['HospNo', '住院号'])
        lab_col = _pick(m.columns, ['标签_诊断表', '标签', '诊断', '分组', '诊断分类'])
        if pid_col is None or lab_col is None:
            raise ValueError(f'对照表[{s}]列识别失败：{list(m.columns)}，请发给我')
        t = pd.DataFrame({'pid': _clean_pid(m[pid_col]).astype('Int64'), 'label_raw': m[lab_col].astype(str).str.strip(), '组别': s if s in ('验证组1', '验证组2') else ''})
        if hosp_col:
            t['HospNo'] = m[hosp_col].astype(str).str.strip().str.replace('\\.0$', '', regex=True).str.zfill(10)
        else:
            t['HospNo'] = pd.NA
        frames.append(t)
    out = pd.concat(frames, ignore_index=True).dropna(subset=['pid'])
    out['pid'] = out['pid'].astype(int)
    dup = out['pid'].duplicated().sum()
    if dup:
        print(f'[idmap] 警告：{dup} 个 pid 在两个 sheet 重复，保留首次出现')
        out = out.drop_duplicates('pid')
    if out['HospNo'].isna().all():
        print('[warn] 对照表无 HospNo 列，CT 匹配将无法进行')
    unknown = set(out['label_raw']) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f'对照表标签出现未知写法 {unknown}，请发给我补充映射')
    out['target'] = out['label_raw'].map(LABEL_MAP).astype(int)
    for g, sub in out.groupby('组别') if (out['组别'] != '').any() else [('', out)]:
        print(f"[idmap] {g or '全部'}: {len(sub)} 人：肺癌 {(sub.target == 1).sum()} / 对照 {(sub.target == 0).sum()}")
    return out[['pid', 'HospNo', 'target']]

def leakage_check(pids: pd.Series) -> str:
    tr = pd.read_excel(TRAIN_IDMAP)
    tr_pid = set(_clean_pid(tr['pid']).dropna().astype(int))
    cur = set(pids.dropna().astype(int))
    n = len(cur & tr_pid)
    verdict = '通过' if n == 0 else f'警告：{n} 人与训练队列重合，须排除！'
    print(f'[leak] 与训练队列 pid 重合 {n} 人 → {verdict}')
    return verdict

def authenticity_check(df: pd.DataFrame, name: str) -> None:
    from sklearn.metrics import roc_auc_score
    for it in ['CEA', 'CYFRA']:
        if it not in df.columns:
            continue
        s = df.dropna(subset=[it])
        if s['target'].nunique() < 2 or len(s) < 50:
            continue
        auc = roc_auc_score(s['target'], s[it])
        flag = 'OK' if auc > 0.55 else '★异常（≈0.5 提示数据有问题，停止并联系我）'
        print(f'[{name}] {it} 对标签 AUC = {auc:.3f}  {flag}')

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    idmap = load_val_idmap()
    qc_rows = []
    for sheet, tag in [(SHEETS[0], 'val1'), (SHEETS[1], 'val2')]:
        wide = load_sheet(BLOOD_PATH, sheet)
        df = wide.reset_index().merge(idmap.rename(columns={'target': 'target_map'}), on='pid', how='left')
        df['target'] = df['_target'].fillna(df['target_map'])
        df = df.drop(columns=['_target', 'target_map'])
        n_lab = int(df['target'].notna().sum())
        n_hosp = int(df['HospNo'].notna().sum())
        print(f'[{tag}] {len(df)} 人：有标签 {n_lab}，有HospNo {n_hosp}')
        if n_lab < len(df):
            print(f'[{tag}] 警告：{len(df) - n_lab} 人无标签（sheet与对照表都没有），评估时将被剔除')
        if n_hosp == 0:
            print(f'[{tag}] 警告：本组无人能匹配 HospNo，CT 特征将全部缺失；若需联合模型验证，请提供本组 pid↔HospNo 对照')
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        verdict = leakage_check(df['pid'])
        n_pos = int((df['target'] == 1).sum())
        print(f'[{tag}] 评估用 {len(df)} 人：肺癌 {n_pos} / 对照 {len(df) - n_pos}')
        authenticity_check(df, tag)
        for it in REQUIRED_ITEMS:
            qc_rows.append({'组别': tag, '项目': it, '有记录人数': int(df[it].notna().sum()) if it in df.columns else 0, '总人数': len(df), '泄漏检查': verdict})
        out = _os.path.join(OUT_DIR, f'{tag}_blood_wide.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'[save] {out}')
    qc = pd.DataFrame(qc_rows)
    qc['覆盖率'] = (qc['有记录人数'] / qc['总人数']).round(4)
    qc.to_csv(_os.path.join(OUT_DIR, 'val_blood_qc.csv'), index=False, encoding='utf-8-sig')
    print('\n======== 必需项目覆盖率 ========')
    print(qc.pivot(index='项目', columns='组别', values='覆盖率').to_string())
    print(f'\n[done] 输出目录: {OUT_DIR}')
    print('下一步：运行 val_02_prepare_ct.py')
if __name__ == '__main__':
    main()
