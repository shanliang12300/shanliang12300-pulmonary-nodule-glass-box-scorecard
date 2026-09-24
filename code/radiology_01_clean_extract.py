from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
RADIO_PATH = os.environ.get('PULNOD_RADIO_PATH', 'D:\\333\\放射0808.csv')
OUTPUT_ROOT = os.environ.get('PULNOD_OUTPUT_ROOT', 'D:\\333')
OUT_DIR = os.path.join(OUTPUT_ROOT, '放射数据清洗')

def resolve_radio_path(path: str) -> str:
    if os.path.exists(path):
        return path
    search_dirs = ['D:\\333', os.path.join(os.path.expanduser('~'), 'Downloads'), os.path.join(os.path.expanduser('~'), 'Desktop'), os.path.dirname(os.path.abspath(__file__))]
    hits = []
    for d in search_dirs:
        if os.path.isdir(d):
            hits += [os.path.join(d, f) for f in os.listdir(d) if f.startswith('放射') and f.lower().endswith('.csv')]
    if len(hits) == 1:
        print(f'[auto] 默认路径未找到，自动使用: {hits[0]}')
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError('找到多个放射CSV，请用环境变量 PULNOD_RADIO_PATH 指定其一：\n' + '\n'.join(hits))
    raise FileNotFoundError(f'未找到放射数据文件：{path}\n已自动搜索 D:\\333、下载、桌面、脚本所在目录均无“放射*.csv”。\n请把文件放进 D:\\333，或运行前设置：set PULNOD_RADIO_PATH=文件完整路径')
CHEST_LUNG = re.compile('肺|纵隔|胸膜|胸腔')
CT_HINT = re.compile('CT|平扫|增强|薄层|层厚|肺窗|纵隔窗', re.I)
NODULE_KW = re.compile('结节|肿块|占位|病灶|团块|团片|软组织.{0,3}影|密度增高影|磨玻璃.{0,3}影|实质块影|块影|阴影')
RESET_KW = re.compile('淋巴结|胸腔|积液|椎体|肋骨|胆囊|肾上腺|肝脏|脾脏|肾脏|心影|主动脉|起搏器|术后')
MEAS_PAIR = re.compile('(\\d+(?:\\.\\d+)?)\\s*[×xX\\*]\\s*(\\d+(?:\\.\\d+)?)\\s*(mm|cm|MM|CM|毫米|厘米)')
MEAS_SINGLE = re.compile('(?:最大径|长径|短径|直径|大小|径线|大者|径)\\s*(?:大致约|约为|大致|约|为|：)?\\s*(\\d+(?:\\.\\d+)?)\\s*(mm|cm|MM|CM|毫米|厘米)')
DENSITY_GGN = re.compile('磨玻璃|GGN|GGO', re.I)
DENSITY_PART = re.compile('部分实性|混合.{0,2}磨玻璃|混杂密度')
MARGIN_FEATS = {'毛刺': '毛刺', '分叶': '分叶', '胸膜牵拉凹陷': '胸膜牵拉|胸膜凹陷|胸膜皱缩', '空泡空腔': '空泡|空腔', '血管集束': '血管集束|血管穿行', '支气管截断': '支气管.{0,3}截断', '钙化': '钙化'}
LOBE = re.compile('(左|右)(?:肺|侧)?(上|中|下)叶')
MULTI = re.compile('多发结节|散在.{0,3}结节|两肺.{0,4}结节')

def _to_mm(v: float, unit: str) -> float:
    return v * 10.0 if unit.lower() in ('cm', '厘米') else v

def extract_sizes_mm(text: str) -> list[float]:
    sizes: list[float] = []
    active = False
    for cl in re.split('[。；;，,\\n]', text):
        if NODULE_KW.search(cl):
            active = True
        elif RESET_KW.search(cl):
            active = False
        if not active or '淋巴结' in cl:
            continue
        for m in MEAS_PAIR.finditer(cl):
            sizes.append(_to_mm(float(m.group(1)), m.group(3)))
            sizes.append(_to_mm(float(m.group(2)), m.group(3)))
        for m in MEAS_SINGLE.finditer(cl):
            sizes.append(_to_mm(float(m.group(1)), m.group(2)))
    return sizes

def extract_report_features(findings: str, impression: str) -> dict:
    text = f'{findings}\n{impression}'
    sizes = extract_sizes_mm(text)
    lobe = LOBE.search(text)
    out = {'nodule_mentioned': bool(NODULE_KW.search(text)), 'max_diameter_mm': max(sizes) if sizes else np.nan, 'n_measurements': len(sizes), 'ggn': bool(DENSITY_GGN.search(text)), 'part_solid': bool(DENSITY_PART.search(text)), 'multi_nodules': bool(MULTI.search(text)), 'lobe_first': f'{lobe.group(1)}{lobe.group(2)}叶' if lobe else ''}
    for name, pat in MARGIN_FEATS.items():
        out[name] = bool(re.search(pat, text))
    return out

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    df = pd.read_csv(RADIO_PATH, dtype=str, encoding='utf-8-sig')
    df.columns = ['HospNo', 'PublicTime', 'Findings', 'Impression']
    summary.append(('原始行数', len(df)))
    df['HospNo'] = df['HospNo'].str.strip()
    df['PublicTime'] = pd.to_datetime(df['PublicTime'], errors='coerce')
    df['Findings'] = df['Findings'].fillna('').str.strip()
    df['Impression'] = df['Impression'].fillna('').str.strip()
    df = df[~((df['Findings'] == '') & (df['Impression'] == ''))]
    summary.append(('剔除双空行后', len(df)))
    df = df.drop_duplicates(subset=['HospNo', 'PublicTime', 'Findings', 'Impression'])
    summary.append(('完全去重后', len(df)))
    summary.append(('唯一患者数', df['HospNo'].nunique()))
    full_text = df['Findings'] + '\n' + df['Impression']
    is_chest_ct = full_text.str.contains(CHEST_LUNG) & full_text.str.contains(CT_HINT)
    chest = df[is_chest_ct].copy()
    summary.append(('胸部CT报告数', len(chest)))
    summary.append(('胸部CT患者数', chest['HospNo'].nunique()))
    feats = chest.apply(lambda r: extract_report_features(r['Findings'], r['Impression']), axis=1, result_type='expand')
    chest = pd.concat([chest.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    summary.append(('提及结节/肿块/占位的报告', int(chest['nodule_mentioned'].sum())))
    summary.append(('成功提取径线的报告', int(chest['max_diameter_mm'].notna().sum())))
    summary.append(('磨玻璃结节报告', int(chest['ggn'].sum())))
    out_all = os.path.join(OUT_DIR, '放射报告_清洗总表.csv')
    out_ct = os.path.join(OUT_DIR, '胸部CT_结节特征_逐报告.csv')
    out_sum = os.path.join(OUT_DIR, '提取摘要.csv')
    df.to_csv(out_all, index=False, encoding='utf-8-sig')
    chest.to_csv(out_ct, index=False, encoding='utf-8-sig')
    pd.DataFrame(summary, columns=['项目', '数值']).to_csv(out_sum, index=False, encoding='utf-8-sig')
    print(f'[save] {out_all}')
    print(f'[save] {out_ct}')
    print(f'[save] {out_sum}')
    for k, v in summary:
        print(f'  {k}: {v}')
if __name__ == '__main__':
    main()
