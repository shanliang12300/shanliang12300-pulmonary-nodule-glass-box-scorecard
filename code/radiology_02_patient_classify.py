from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
OUT_DIR = os.path.join(OUTPUT_ROOT, '放射数据清洗')
SRC = os.path.join(OUT_DIR, '胸部CT_结节特征_逐报告.csv')
PAT_LC = re.compile('肺癌|肺\\s*[Cc][Aa]|肺.{0,4}癌可能|癌可能大|考虑.{0,3}癌|恶性.{0,3}可能大|恶性病变(?!\\s*待排)(?!\\s*除外)(?!\\s*排除)|警惕恶性|恶性占位|恶性肿瘤|恶性可能|[上中下]叶.{0,2}癌|肺\\s*MT|MT可能|肺.{0,2}恶性|肺癌术后|肺癌化疗|肺癌放疗|肿瘤性病变')
PAT_OTHER_TUMOR = re.compile('转移瘤|转移癌|多发转移|癌性淋巴管|胸腺瘤|胸腺癌|胸腺肿瘤|纵隔肿瘤|神经源性肿瘤|肉瘤|间质瘤')
PAT_BENIGN = re.compile('炎性|炎症|感染|结核|错构瘤|肉芽肿|纤维灶|纤维化|硬结灶|良性|钙化灶|肺大泡|气肿|支扩|支气管扩张')
PAT_NORMAL = re.compile('未见明显异常|未见异常|未见确切异常|大致正常')

def classify_patient(texts: pd.Series) -> tuple[str, str]:
    all_text = ' '.join((t for t in texts if isinstance(t, str)))
    if PAT_LC.search(all_text):
        return ('肺癌提示', '肺癌词:' + '|'.join(sorted(set(PAT_LC.findall(all_text)))[:3]))
    if PAT_OTHER_TUMOR.search(all_text):
        return ('其他肿瘤提示', '转移词:' + '|'.join(sorted(set(PAT_OTHER_TUMOR.findall(all_text)))[:3]))
    if PAT_BENIGN.search(all_text):
        return ('良性倾向', '良性词:' + '|'.join(sorted(set(PAT_BENIGN.findall(all_text)))[:3]))
    if PAT_NORMAL.search(all_text):
        return ('未见异常', '')
    return ('未确定', '')

def main() -> None:
    ct = pd.read_csv(SRC, dtype={'HospNo': str}, encoding='utf-8-sig')
    ct['PublicTime'] = pd.to_datetime(ct['PublicTime'], errors='coerce')
    for c in ['ggn', 'part_solid', 'multi_nodules', 'nodule_mentioned', '毛刺', '分叶', '胸膜牵拉凹陷', '空泡空腔', '血管集束', '支气管截断', '钙化']:
        if c in ct.columns:
            ct[c] = ct[c].astype(bool)
    agg = ct.groupby('HospNo').agg(n_chest_ct=('PublicTime', 'size'), first_ct=('PublicTime', 'min'), last_ct=('PublicTime', 'max'), max_diameter_mm=('max_diameter_mm', 'max'), ever_ggn=('ggn', 'max'), ever_part_solid=('part_solid', 'max'), ever_multi=('multi_nodules', 'max'), ever_毛刺=('毛刺', 'max'), ever_分叶=('分叶', 'max'), ever_胸膜牵拉凹陷=('胸膜牵拉凹陷', 'max'), ever_支气管截断=('支气管截断', 'max')).reset_index()
    cls = ct.groupby('HospNo')['Impression'].apply(classify_patient).apply(pd.Series)
    cls.columns = ['分组', '命中证据']
    pat = agg.merge(cls.reset_index(), on='HospNo', how='left')
    out_pat = os.path.join(OUT_DIR, '患者级_合并与分组.csv')
    pat.to_csv(out_pat, index=False, encoding='utf-8-sig')
    summ = pat['分组'].value_counts().rename_axis('分组').reset_index(name='人数')
    out_sum = os.path.join(OUT_DIR, '分组摘要.csv')
    summ.to_csv(out_sum, index=False, encoding='utf-8-sig')
    rng = np.random.default_rng(42)
    rows = []
    for g in ['肺癌提示', '其他肿瘤提示', '良性倾向', '未见异常', '未确定']:
        sub = pat[pat['分组'] == g]
        if len(sub) == 0:
            continue
        samp = sub.sample(min(20, len(sub)), random_state=rng)
        for _, r in samp.iterrows():
            imp = ct[ct['HospNo'] == r['HospNo']]['Impression']
            imp = imp[imp.astype(str).str.len() > 0]
            rows.append({'分组': g, 'HospNo': r['HospNo'], '结论摘录': ' ‖ '.join(imp.astype(str).head(2))[:300]})
    out_evi = os.path.join(OUT_DIR, '分组抽样证据.csv')
    pd.DataFrame(rows).to_csv(out_evi, index=False, encoding='utf-8-sig')
    print(f'[save] {out_pat}')
    print(f'[save] {out_sum}')
    print(f'[save] {out_evi}')
    print(summ.to_string(index=False))
if __name__ == '__main__':
    main()
