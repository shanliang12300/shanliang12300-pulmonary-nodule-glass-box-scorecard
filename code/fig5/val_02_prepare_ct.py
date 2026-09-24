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
pd.set_option('future.no_silent_downcasting', True)
from common_preprocessing import OUTPUT_ROOT
from radiology_01_clean_extract import extract_report_features, CHEST_LUNG, CT_HINT
VAL_DIR = _os.path.join(OUTPUT_ROOT, '验证组')
CT_PATH = _os.path.join(VAL_DIR, '验证组CT报告.xlsx')
VAL_IDMAP = _os.path.join(VAL_DIR, 'pid↔HospNo 对照表.xlsx')
OUT_DIR = _os.path.join(OUTPUT_ROOT, 'Fig5_外部验证', '00_计算结果与原始数据')
HOSP_CANDIDATES = ['HospNo', '住院号', '病人号', '病员号']
TIME_CANDIDATES = ['PublicTime', '检查时间', '报告时间', '检查日期', '报告日期']
FIND_CANDIDATES = ['Findings', '检查所见', '影像所见', '所见', 'ItemResult']
IMP_CANDIDATES = ['Impression', '诊断结论', '结论', '诊断意见', '印象', 'ItemResult.1']

def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _norm_hosp(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s.astype(str).str.strip().str.replace('\\.0$', '', regex=True), errors='coerce')
    return x.astype('Int64').astype(str).str.zfill(10)

def load_ct(path: str) -> pd.DataFrame:
    sheets = pd.ExcelFile(path).sheet_names
    frames = []
    for s in sheets:
        d = pd.read_excel(path, sheet_name=s, dtype=str)
        if _pick(d.columns, HOSP_CANDIDATES):
            d['_sheet'] = s
            frames.append(d)
    if not frames:
        raise ValueError(f'所有 sheet 均无住院号列，sheet={sheets}，请发给我列名')
    ct = pd.concat(frames, ignore_index=True)
    print(f'[load] {len(sheets)} 个 sheet 合并：{len(ct)} 行')
    h = _pick(ct.columns, HOSP_CANDIDATES)
    f = _pick(ct.columns, FIND_CANDIDATES)
    i = _pick(ct.columns, IMP_CANDIDATES)
    t = _pick(ct.columns, TIME_CANDIDATES)
    if f is None or i is None:
        obj = [c for c in ct.columns if c not in {h, t, '_sheet'}]
        ranked = sorted(obj, key=lambda c: ct[c].astype(str).str.len().mean(), reverse=True)
        f, i = (ranked[0], ranked[1] if len(ranked) > 1 else ranked[0])
        print(f'[load] 所见/结论列按文本长度兜底：所见={f}, 结论={i}')
    print(f'[load] 列: HospNo={h}, 所见={f}, 结论={i}, 时间={t}')
    out = pd.DataFrame({'HospNo': _norm_hosp(ct[h]), 'Findings': ct[f].fillna(''), 'Impression': ct[i].fillna(''), 'PublicTime': pd.to_datetime(ct[t], errors='coerce') if t else pd.NaT})
    return out.dropna(subset=['HospNo'])

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    xl = pd.ExcelFile(VAL_IDMAP)
    sheets = [s for s in xl.sheet_names if s in ('验证组1', '验证组2')] or xl.sheet_names[:1]
    idmap = pd.concat([xl.parse(s) for s in sheets], ignore_index=True)
    print(f'[idmap] 读取 sheet: {sheets}')
    hosp_col = _pick(idmap.columns, ['HospNo', '住院号'])
    if hosp_col is None:
        raise ValueError(f'对照表无 HospNo 列：{list(idmap.columns)}')
    keep = set(_norm_hosp(idmap[hosp_col]).dropna())
    print(f'[idmap] 验证组住院号 {len(keep)} 个')
    ct = load_ct(CT_PATH)
    in_val = ct['HospNo'].isin(keep)
    print(f"[filter] 总报告 {len(ct)} 条 → 验证组 {int(in_val.sum())} 条 （剔除多余 {int((~in_val).sum())} 条）；覆盖病人 {ct.loc[in_val, 'HospNo'].nunique()} / {len(keep)} 人")
    no_ct = keep - set(ct.loc[in_val, 'HospNo'])
    if no_ct:
        print(f'[warn] {len(no_ct)} 个住院号无任何 CT 报告（将按 CT_径线缺失处理）')
    rep = ct[in_val].copy()
    if rep.empty:
        raise ValueError('CT 报告中没有任何验证组住院号的记录，请核对对照表 HospNo 与 CT 文件的住院号格式')
    MRI_KW = 'T1W|T2W|DWI|FLAIR|磁共振|MRI|MR平扫|MR增强'
    full_text = rep['Findings'].astype(str) + '\n' + rep['Impression'].astype(str)
    chest = full_text.str.contains(CHEST_LUNG) & ~full_text.str.contains(MRI_KW, case=False, regex=True)
    print(f'[filter] 其中胸部影像 {int(chest.sum())} 条（剔除非胸部/MRI {int((~chest).sum())} 条）')
    rep = rep[chest].copy()
    feats = rep.apply(lambda r: extract_report_features(r['Findings'], r['Impression']), axis=1)
    rep = pd.concat([rep, pd.DataFrame(feats.tolist(), index=rep.index)], axis=1)
    rep.to_csv(_os.path.join(OUT_DIR, 'val_ct_report_level.csv'), index=False, encoding='utf-8-sig')
    print(f'[save] 逐报告特征 {rep.shape}')
    rep['_ord'] = range(len(rep))
    rep = rep.sort_values(['HospNo', 'PublicTime', '_ord'], kind='stable', na_position='last')
    nod = rep[rep['nodule_mentioned']]
    b1 = nod[nod['max_diameter_mm'].notna()].groupby('HospNo').first()
    b2 = nod.groupby('HospNo').first()
    b3 = rep.groupby('HospNo').first()
    all_h = pd.Index(sorted(keep), name='HospNo')
    base = b1.reindex(all_h)
    src = pd.Series('①最早含径线结节报告', index=all_h)
    miss = base['Findings'].isna()
    base.loc[miss] = b2.reindex(all_h).loc[miss]
    src.loc[miss] = '②最早提及结节报告'
    miss = base['Findings'].isna()
    base.loc[miss] = b3.reindex(all_h).loc[miss]
    src.loc[miss] = '③最早胸部CT（无结节描述）'
    pat = pd.DataFrame(index=all_h)
    pat['max_diameter_mm'] = base['max_diameter_mm']
    for c in ['ggn', 'part_solid', 'multi_nodules', '毛刺', '分叶', '胸膜牵拉凹陷', '空泡空腔', '血管集束', '支气管截断', '钙化']:
        if c in base.columns:
            pat[c] = base[c].reindex(all_h).fillna(False).infer_objects(copy=False).astype(bool)
    pat['CT_基线来源'] = src
    pat['n_ct'] = rep.groupby('HospNo').size().reindex(all_h).fillna(0).astype(int)
    pat['CT_径线缺失'] = pat['max_diameter_mm'].isna().astype(int)
    pat = pat.reset_index()
    pat.to_csv(_os.path.join(OUT_DIR, 'val_ct_features.csv'), index=False, encoding='utf-8-sig')
    print(f'[save] 患者级特征 {pat.shape}')
    print(f"[qc] 基线来源: {pat['CT_基线来源'].value_counts().to_dict()}")
    print(f"[qc] 基线CT提取到径线: {int((pat['CT_径线缺失'] == 0).sum())} 人; 毛刺 {int(pat['毛刺'].sum())}, 分叶 {int(pat['分叶'].sum())}, GGN {int(pat['ggn'].sum())}")
    print(f'\n[done] 输出目录: {OUT_DIR}')
    print('下一步：把 val_01 与 val_02 的运行日志发我，确认后给评估脚本')
if __name__ == '__main__':
    main()
