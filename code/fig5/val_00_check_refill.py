from __future__ import annotations
import os
import numpy as np
import pandas as pd
VAL_DIR = 'D:\\333\\验证组'
BLOOD_FILE = os.path.join(VAL_DIR, '验证组血液检查数据.xlsx')
CHECK_COLS = ['AAG', 'APTT', 'AU_Urea']
TRAIN_MEDIAN = {'AAG': 0.69, 'APTT': 28.15, 'AU_Urea': 5.5}

def check_group(gname: str, d: pd.DataFrame) -> None:
    print(f"\n{'=' * 64}\n【{gname}】 n={len(d)}（肺癌 {int(d['target'].sum())}）\n{'=' * 64}")
    y = d['target']
    v = {c: pd.to_numeric(d[c], errors='coerce') for c in CHECK_COLS + ['CRP']}
    print('1. 三列独立性（|r|>0.95 = 仍是拷贝， FAIL）')
    for lab, mask in [('肺癌', y == 1), ('对照', y == 0)]:
        pairs = []
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = (CHECK_COLS[i], CHECK_COLS[j])
                r = v[a][mask].corr(v[b][mask])
                flag = '  <<< FAIL' if abs(r) > 0.95 else ''
                pairs.append(f'{a}~{b} r={r:+.3f}{flag}')
        print(f'   {lab}: ' + ' | '.join(pairs))
    r_ac = v['AAG'].corr(v['CRP'])
    print(f'2. AAG~CRP 相关: r={r_ac:+.3f}' + ('   <<< FAIL（急性期蛋白应与CRP正相关）' if r_ac < 0.1 else '   OK'))
    print('3. 量纲对比训练集（比值>2 或 <0.5 = 单位错位， FAIL）')
    for c in CHECK_COLS:
        med = v[c].median()
        ratio = med / TRAIN_MEDIAN[c]
        flag = '   <<< FAIL' if ratio > 2 or ratio < 0.5 else ''
        print(f'   {c:8s} 本组中位 {med:8.2f} | 训练 {TRAIN_MEDIAN[c]:6.2f} | 比值 {ratio:5.2f}{flag}')
    print('4. 按标签分布（中位数 [P25, P75]）')
    for c in CHECK_COLS:
        v1, v0 = (v[c][y == 1].dropna(), v[c][y == 0].dropna())
        print(f'   {c:8s} 肺癌 {v1.median():7.2f} [{v1.quantile(0.25):.2f}, {v1.quantile(0.75):.2f}] | 对照 {v0.median():7.2f} [{v0.quantile(0.25):.2f}, {v0.quantile(0.75):.2f}]')
    print('5. 覆盖率')
    for c in CHECK_COLS:
        n = v[c].notna().sum()
        print(f'   {c:8s} {n}/{len(d)} ({n / len(d):.0%})')

def main() -> None:
    xl = pd.ExcelFile(BLOOD_FILE)
    for sheet in xl.sheet_names:
        d = xl.parse(sheet)
        if 'target' not in d.columns:
            print(f'[{sheet}] 缺 target 列，跳过')
            continue
        check_group(sheet, d)
    print('\n[提示] 全部无 FAIL 后，依次重跑：val_01 → val_03（CT 未变，val_02 不用重跑）')
if __name__ == '__main__':
    main()
