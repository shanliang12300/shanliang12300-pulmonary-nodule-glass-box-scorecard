from __future__ import annotations
import os
import numpy as np
import pandas as pd
DATA_PATH = os.environ.get('PULNOD_DATA_PATH', 'D:\\333\\患者原始数据.xlsx')
IDMAP_PATH = os.environ.get('PULNOD_IDMAP_PATH', 'D:\\333\\pid↔HospNo 对照表.xlsx')
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
RADIO_DIR = os.path.join(OUTPUT_ROOT, '放射数据清洗')
OUT_DIR = os.path.join(OUTPUT_ROOT, '队列合并数据')
CT_COLS = ['max_diameter_mm', 'ggn', 'part_solid', 'multi_nodules', 'lobe_first', '毛刺', '分叶', '胸膜牵拉凹陷', '空泡空腔', '血管集束', '支气管截断', '钙化']

def resolve_idmap(path: str) -> str:
    if os.path.exists(path):
        return path
    search_dirs = ['D:\\333', os.path.join(os.path.expanduser('~'), 'Downloads'), os.path.join(os.path.expanduser('~'), 'Desktop'), os.path.dirname(os.path.abspath(__file__))]
    hits = []
    for d in search_dirs:
        if os.path.isdir(d):
            hits += [os.path.join(d, f) for f in os.listdir(d) if '对照表' in f and f.lower().endswith('.xlsx')]
    if len(hits) == 1:
        print(f'[auto] 自动使用对照表: {hits[0]}')
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError('找到多个对照表，请用 PULNOD_IDMAP_PATH 指定其一：\n' + '\n'.join(hits))
    raise FileNotFoundError(f'未找到对照表：{path}\n已搜索 D:\\333、下载、桌面、脚本目录均无“*对照表*.xlsx”。')

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    blood = pd.read_excel(DATA_PATH, dtype=str)
    idmap = pd.read_excel(resolve_idmap(IDMAP_PATH), dtype=str)
    ct = pd.read_csv(os.path.join(RADIO_DIR, '胸部CT_结节特征_逐报告.csv'), dtype={'HospNo': str}, encoding='utf-8-sig')
    ct['PublicTime'] = pd.to_datetime(ct['PublicTime'], errors='coerce')
    for c in ['nodule_mentioned', 'ggn', 'part_solid', 'multi_nodules', '毛刺', '分叶', '胸膜牵拉凹陷', '空泡空腔', '血管集束', '支气管截断', '钙化']:
        ct[c] = ct[c].astype(bool)
    assert idmap['pid'].nunique() == len(idmap) == 635, '对照表 pid 不唯一或行数≠635'
    assert set(idmap['pid']) == set(blood['pid']), '对照表 pid 与血检数据不一致'
    lab_a = idmap.set_index('pid')['标签_诊断表']
    lab_b = blood.set_index('pid')['标签_诊断表']
    agree = float((lab_a == lab_b).mean())
    summary.append(('对照表-血检标签一致率', f'{agree:.1%}'))
    ct = ct.sort_values(['HospNo', 'PublicTime'])
    nod = ct[ct['nodule_mentioned']]
    b1 = nod[nod['max_diameter_mm'].notna()].groupby('HospNo').first()
    b2 = nod.groupby('HospNo').first()
    b3 = ct.groupby('HospNo').first()
    base = b1.reindex(b3.index)
    src = pd.Series('①最早含径线结节报告', index=b3.index)
    miss = base['Findings'].isna()
    base.loc[miss] = b2.loc[miss]
    src.loc[miss] = '②最早提及结节报告'
    miss = base['Findings'].isna()
    base.loc[miss] = b3.loc[miss]
    src.loc[miss] = '③最早胸部CT'
    base = base.reset_index()
    base['CT_基线来源'] = src.values
    base['hkey'] = base['HospNo'].str.lstrip('0')
    agg = ct.groupby('HospNo').agg(CT_报告数=('PublicTime', 'size'), CT_首次检查=('PublicTime', 'min'), CT_末次检查=('PublicTime', 'max'), CT_历史最大径mm=('max_diameter_mm', 'max'), CT_曾有GGN=('ggn', 'max'), CT_曾有部分实性=('part_solid', 'max'), CT_曾有多发=('multi_nodules', 'max'), CT_曾有毛刺=('毛刺', 'max'), CT_曾有分叶=('分叶', 'max'), CT_曾有胸膜牵拉凹陷=('胸膜牵拉凹陷', 'max'), CT_曾有血管集束=('血管集束', 'max'), CT_曾有支气管截断=('支气管截断', 'max')).reset_index()
    agg['CT_随访跨度天'] = (agg['CT_末次检查'] - agg['CT_首次检查']).dt.days
    agg['hkey'] = agg['HospNo'].str.lstrip('0')
    idmap['hkey'] = idmap['HospNo'].str.lstrip('0')
    base_feat = base[['hkey', 'PublicTime', 'CT_基线来源'] + CT_COLS].rename(columns={'PublicTime': 'CT_基线日期', 'max_diameter_mm': 'CT_基线最大径mm', 'ggn': 'CT_基线GGN', 'part_solid': 'CT_基线部分实性', 'multi_nodules': 'CT_基线多发', 'lobe_first': 'CT_基线叶位置', '毛刺': 'CT_基线毛刺', '分叶': 'CT_基线分叶', '胸膜牵拉凹陷': 'CT_基线胸膜牵拉凹陷', '空泡空腔': 'CT_基线空泡空腔', '血管集束': 'CT_基线血管集束', '支气管截断': 'CT_基线支气管截断', '钙化': 'CT_基线钙化'})
    radio = idmap[['pid', 'hkey']].merge(base_feat, on='hkey', how='left').merge(agg.drop(columns=['HospNo']), on='hkey', how='left')
    grp_path = os.path.join(RADIO_DIR, '患者级_合并与分组.csv')
    if os.path.exists(grp_path):
        grp = pd.read_csv(grp_path, dtype={'HospNo': str}, encoding='utf-8-sig')
        grp['hkey'] = grp['HospNo'].str.lstrip('0')
        radio = radio.merge(grp[['hkey', '分组']].rename(columns={'分组': 'CT_影像分组'}), on='hkey', how='left')
    merged = blood.merge(radio.drop(columns=['hkey']), on='pid', how='left')
    assert len(merged) == 635, '合并后行数≠635'
    n_ct = int(merged['CT_报告数'].notna().sum())
    n_base_size = int(merged['CT_基线最大径mm'].notna().sum())
    summary.append(('队列人数', 635))
    summary.append(('有胸部CT', n_ct))
    summary.append(('基线CT提取到径线', n_base_size))
    summary.append(('基线GGN人数', int(merged['CT_基线GGN'].sum())))
    summary.append(('基线毛刺人数', int(merged['CT_基线毛刺'].sum())))
    out = os.path.join(OUT_DIR, '队列_血液+影像_合并表.csv')
    merged.to_csv(out, index=False, encoding='utf-8-sig')
    out_sum = os.path.join(OUT_DIR, '合并摘要.csv')
    pd.DataFrame(summary, columns=['项目', '数值']).to_csv(out_sum, index=False, encoding='utf-8-sig')
    print(f'[save] {out}')
    print(f'[save] {out_sum}')
    for k, v in summary:
        print(f'  {k}: {v}')
if __name__ == '__main__':
    main()
