from __future__ import annotations
from nc_figure_style import C, pub
CT_PUB = {'CT_基线最大径mm': 'Diameter', 'CT_径线缺失': 'Diameter missing', 'CT_基线GGN': 'GGN', 'CT_基线部分实性': 'Part-solid', 'CT_基线多发': 'Multiple nodules', 'CT_基线毛刺': 'Spiculation', 'CT_基线分叶': 'Lobulation', 'CT_基线胸膜牵拉凹陷': 'Pleural retraction', 'CT_基线空泡空腔': 'Vacuole sign', 'CT_基线支气管截断': 'Bronchus cutoff', 'CT_基线钙化': 'Calcification', 'CT_基线血管集束': 'Vascular convergence'}
SOURCE_COLORS = {'blood': '#333333', 'CT': C['green']}

def jpub(name: str) -> str:
    return CT_PUB.get(name, pub(name))

def jsource(name: str) -> str:
    return 'CT' if name.startswith('CT_') else 'blood'

def jcolor(name: str) -> str:
    return SOURCE_COLORS[jsource(name)]
RULE_TYPE_COLORS = {'blood': '#333333', 'ct_sign': C['green'], 'diameter': C['orange']}

def rule_pub(rule: str) -> str:
    if rule.endswith('_abnormal'):
        return f"{pub(rule[:-len('_abnormal')])} abnormal"
    if rule.startswith('CT_直径'):
        return 'Diameter ' + rule[len('CT_直径'):].replace('mm', ' mm')
    if rule == 'CT_径线缺失':
        return 'No diameter reported'
    return CT_PUB.get(rule, rule)
