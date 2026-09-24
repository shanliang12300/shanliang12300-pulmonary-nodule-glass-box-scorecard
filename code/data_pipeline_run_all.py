from __future__ import annotations
import radiology_01_clean_extract
import radiology_02_patient_classify
import radiology_03_merge_cohort

def main() -> None:
    steps = [('放射报告清洗 + 胸部CT结节特征提取', radiology_01_clean_extract.main), ('患者级合并 + 影像文字分组', radiology_02_patient_classify.main), ('635例队列 血液+影像 合并总表', radiology_03_merge_cohort.main)]
    total = len(steps)
    for i, (desc, func) in enumerate(steps, start=1):
        print(f'[数据管道] {i}/{total} {desc}')
        func()
    print('[数据管道] 队列合并总表已生成')
if __name__ == '__main__':
    main()
